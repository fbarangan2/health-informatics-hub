"""
Unit tests for ML forecasting models - Health Informatics Hub
"""

import numpy as np
import pandas as pd
import pytest

from src.models.demand_forecaster import EnsembleDemandForecaster, ProphetForecaster
from src.models.infrastructure import InfrastructurePredictor, STAFFING_RATIOS
from src.pipeline.transform import DataCleaner, HospitalDemandFeatureEngineer


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def sample_census_df():
    """Small synthetic census DataFrame for fast tests."""
    dates = pd.date_range("2022-01-01", periods=400, freq="D")
    np.random.seed(42)
    n = len(dates)
    patient_count = (
        80
        + 10 * np.sin(2 * np.pi * np.arange(n) / 365)
        + 5 * np.sin(2 * np.pi * np.arange(n) / 7)
        + np.random.normal(0, 3, n)
    ).clip(0).round(0).astype(int)

    return pd.DataFrame({
        "date": dates,
        "patient_count": patient_count,
        "bed_count": 120,
        "occupancy_rate": (patient_count / 120).round(4),
        "nurses": (patient_count * 0.25 + np.random.normal(0, 1, n)).round(0).astype(int),
        "is_weekend": (dates.dayofweek >= 5).astype(int),
        "is_flu_season": dates.month.isin([10, 11, 12, 1, 2, 3]).astype(int),
    })


@pytest.fixture
def featured_df(sample_census_df):
    """Feature-engineered DataFrame for model tests."""
    eng = HospitalDemandFeatureEngineer(target_col="patient_count")
    return eng.fit_transform(sample_census_df)


# ─────────────────────────────────────────────
# Feature Engineering Tests
# ─────────────────────────────────────────────

class TestHospitalDemandFeatureEngineer:

    def test_output_has_calendar_features(self, sample_census_df):
        eng = HospitalDemandFeatureEngineer()
        result = eng.fit_transform(sample_census_df)
        expected = ["day_of_week", "month", "is_weekend", "is_flu_season",
                    "day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos"]
        for col in expected:
            assert col in result.columns, f"Missing expected column: {col}"

    def test_lag_features_created(self, sample_census_df):
        eng = HospitalDemandFeatureEngineer()
        result = eng.fit_transform(sample_census_df)
        for lag in [1, 7, 30]:
            assert f"patient_count_lag_{lag}d" in result.columns

    def test_capacity_features_created(self, sample_census_df):
        eng = HospitalDemandFeatureEngineer()
        result = eng.fit_transform(sample_census_df)
        assert "occupancy_rate" in result.columns
        assert "is_near_capacity" in result.columns
        assert result["occupancy_rate"].between(0, 1).all()

    def test_no_all_nan_columns(self, featured_df):
        all_nan = featured_df.columns[featured_df.isna().all()].tolist()
        assert len(all_nan) == 0, f"Columns with all NaN: {all_nan}"

    def test_transform_matches_fit_transform_shape(self, sample_census_df):
        eng = HospitalDemandFeatureEngineer()
        train = sample_census_df.iloc[:300]
        test = sample_census_df.iloc[300:]
        fitted = eng.fit_transform(train)
        transformed = eng.transform(test)
        assert fitted.shape[1] == transformed.shape[1]


# ─────────────────────────────────────────────
# Data Cleaner Tests
# ─────────────────────────────────────────────

class TestDataCleaner:

    def test_removes_exact_duplicates(self):
        cleaner = DataCleaner()
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        result = cleaner._remove_duplicates(df)
        assert len(result) == 2

    def test_caps_extreme_outliers(self):
        cleaner = DataCleaner()
        df = pd.DataFrame({"value": [10, 10, 10, 10, 1000]})
        result = cleaner._remove_outliers(df, z_threshold=2.0)
        assert result["value"].max() < 1000

    def test_deidentify_hashes_patient_id(self):
        cleaner = DataCleaner()
        df = pd.DataFrame({"patient_id": ["P12345", "P67890"], "metric": [1, 2]})
        result = cleaner._deidentify(df)
        assert not result["patient_id"].isin(["P12345", "P67890"]).any()
        assert result["patient_id"].notna().all()

    def test_deidentify_generalizes_zip(self):
        cleaner = DataCleaner()
        df = pd.DataFrame({"zip_code": ["90210", "10001"], "x": [1, 2]})
        result = cleaner._deidentify(df)
        assert result["zip_code"].str.endswith("XX").all()
        assert result["zip_code"].str.len().eq(5).all()


# ─────────────────────────────────────────────
# Prophet Forecaster Tests
# ─────────────────────────────────────────────

class TestProphetForecaster:

    def test_fit_and_predict_returns_correct_length(self, sample_census_df):
        forecaster = ProphetForecaster(department="test")
        df = sample_census_df.set_index("date").sort_index()
        forecaster.fit(df, target_col="patient_count")
        result = forecaster.predict(horizon_days=30)
        assert len(result) == 30

    def test_forecast_has_required_columns(self, sample_census_df):
        forecaster = ProphetForecaster(department="test")
        df = sample_census_df.set_index("date").sort_index()
        forecaster.fit(df, target_col="patient_count")
        result = forecaster.predict(horizon_days=14)
        for col in ["ds", "yhat", "yhat_lower", "yhat_upper"]:
            assert col in result.columns

    def test_confidence_intervals_are_valid(self, sample_census_df):
        forecaster = ProphetForecaster(department="test")
        df = sample_census_df.set_index("date").sort_index()
        forecaster.fit(df, target_col="patient_count")
        result = forecaster.predict(horizon_days=14)
        assert (result["yhat_upper"] >= result["yhat"]).all()
        assert (result["yhat"] >= result["yhat_lower"]).all()

    def test_predict_raises_if_not_fitted(self):
        forecaster = ProphetForecaster()
        with pytest.raises(RuntimeError, match="fitted"):
            forecaster.predict(30)


# ─────────────────────────────────────────────
# Infrastructure Predictor Tests
# ─────────────────────────────────────────────

class TestInfrastructurePredictor:

    @pytest.fixture
    def sample_forecast_df(self):
        dates = pd.date_range("2025-01-01", periods=30, freq="D")
        return pd.DataFrame({
            "date": dates,
            "forecast": np.random.randint(80, 120, 30).astype(float),
            "lower_ci": np.random.randint(70, 80, 30).astype(float),
            "upper_ci": np.random.randint(120, 140, 30).astype(float),
        })

    def test_rule_based_staffing_estimates_are_positive(self, sample_forecast_df):
        predictor = InfrastructurePredictor()
        # Without training, test rule-based fallback via _add_rule_based_estimates
        result = predictor._add_rule_based_estimates(sample_forecast_df.copy())
        assert "nurses_required" in result.columns
        assert (result["nurses_required"] > 0).all()

    def test_staffing_ratios_are_reasonable(self):
        """Verify ratios match CMS/state regulatory standards."""
        assert STAFFING_RATIOS["icu"]["nurses_per_patient"] >= 0.5, "ICU: min 1 RN per 2 patients"
        assert STAFFING_RATIOS["med_surg"]["nurses_per_patient"] >= 0.2, "Med/Surg: min 1 RN per 5 patients"
        assert STAFFING_RATIOS["ed"]["nurses_per_patient"] >= 0.25, "ED: min 1 RN per 4 patients"

    def test_recommendations_sorted_by_priority(self, sample_forecast_df):
        predictor = InfrastructurePredictor()
        sample_forecast_df["nurses_required"] = 15
        sample_forecast_df["icu_beds_required"] = 10
        recs = predictor.get_recommendations(sample_forecast_df)
        priorities = [r["priority"] for r in recs]
        order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        assert priorities == sorted(priorities, key=lambda p: order[p])
