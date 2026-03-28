"""
Data Transformation Pipeline - Health Informatics Hub
=====================================================
Cleans, engineers features, and prepares data for ML model training.

Feature engineering focuses on:
  - Temporal patterns (day of week, seasonality, holidays)
  - Lag features for time series modeling
  - Capacity utilization metrics
  - Disease burden indicators
  - Demographic pressure scores
"""

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler, LabelEncoder


class HospitalDemandFeatureEngineer:
    """
    Transforms raw hospital census/encounter data into features
    ready for demand forecasting models.
    """

    US_FEDERAL_HOLIDAYS = {
        "New Year's Day", "Martin Luther King Jr. Day", "Presidents' Day",
        "Memorial Day", "Independence Day", "Labor Day",
        "Columbus Day", "Veterans Day", "Thanksgiving Day", "Christmas Day",
    }

    def __init__(self, target_col: str = "patient_count"):
        self.target_col = target_col
        self._scaler = StandardScaler()
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler and transform training data."""
        df = self._engineer_features(df)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(self.target_col, errors="ignore")
        df[numeric_cols] = self._scaler.fit_transform(df[numeric_cols])
        self._fitted = True
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted scaler (inference time)."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform() before transform()")
        df = self._engineer_features(df)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(self.target_col, errors="ignore")
        df[numeric_cols] = self._scaler.transform(df[numeric_cols])
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full feature engineering pipeline."""
        df = df.copy()

        # Ensure datetime index
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()

        df = self._add_calendar_features(df)
        df = self._add_lag_features(df)
        df = self._add_rolling_features(df)
        df = self._add_capacity_features(df)
        df = self._handle_missing_values(df)

        logger.info(f"Feature engineering complete: {len(df.columns)} features for {len(df)} records")
        return df

    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal and calendar-based features."""
        idx = df.index

        df["day_of_week"] = idx.dayofweek              # 0=Monday
        df["day_of_month"] = idx.day
        df["month"] = idx.month
        df["quarter"] = idx.quarter
        df["week_of_year"] = idx.isocalendar().week.astype(int)
        df["year"] = idx.year
        df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
        df["is_monday"] = (idx.dayofweek == 0).astype(int)
        df["is_friday"] = (idx.dayofweek == 4).astype(int)

        # Cyclical encoding for calendar features (avoids ordinality issues)
        df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # Season (Northern Hemisphere)
        df["season"] = pd.cut(
            df["month"],
            bins=[0, 3, 6, 9, 12],
            labels=["winter", "spring", "summer", "fall"],
            include_lowest=True,
        )

        # Flu season indicator (October - March)
        df["is_flu_season"] = df["month"].isin([10, 11, 12, 1, 2, 3]).astype(int)

        return df

    def _add_lag_features(self, df: pd.DataFrame, target: str = None) -> pd.DataFrame:
        """Add lag features for autoregressive modeling."""
        target = target or self.target_col
        if target not in df.columns:
            return df

        lags = [1, 2, 3, 7, 14, 28, 30]
        for lag in lags:
            df[f"{target}_lag_{lag}d"] = df[target].shift(lag)

        # Year-over-year
        df[f"{target}_lag_365d"] = df[target].shift(365)

        return df

    def _add_rolling_features(self, df: pd.DataFrame, target: str = None) -> pd.DataFrame:
        """Add rolling window statistics."""
        target = target or self.target_col
        if target not in df.columns:
            return df

        windows = [7, 14, 30]
        for w in windows:
            df[f"{target}_roll_mean_{w}d"] = df[target].shift(1).rolling(w).mean()
            df[f"{target}_roll_std_{w}d"] = df[target].shift(1).rolling(w).std()
            df[f"{target}_roll_max_{w}d"] = df[target].shift(1).rolling(w).max()
            df[f"{target}_roll_min_{w}d"] = df[target].shift(1).rolling(w).min()

        # Trend: difference between 7-day and 30-day rolling mean
        df["demand_trend"] = df[f"{target}_roll_mean_7d"] - df[f"{target}_roll_mean_30d"]

        return df

    def _add_capacity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add hospital capacity utilization metrics if available."""
        if "bed_count" in df.columns and "patient_count" in df.columns:
            df["occupancy_rate"] = (df["patient_count"] / df["bed_count"]).clip(0, 1)
            df["beds_available"] = (df["bed_count"] - df["patient_count"]).clip(lower=0)
            df["is_near_capacity"] = (df["occupancy_rate"] >= 0.85).astype(int)
            df["is_at_capacity"] = (df["occupancy_rate"] >= 0.95).astype(int)

        if "nurses" in df.columns and "patient_count" in df.columns:
            df["patient_to_nurse_ratio"] = df["patient_count"] / df["nurses"].replace(0, np.nan)

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with appropriate strategies."""
        # Forward fill lag/rolling features (they're expected to be NaN at start)
        lag_cols = [c for c in df.columns if "lag" in c or "roll" in c]
        df[lag_cols] = df[lag_cols].fillna(method="bfill").fillna(method="ffill")

        # Fill remaining numeric NaNs with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

        return df


class DataCleaner:
    """
    Validates and cleans raw healthcare data.
    Handles duplicates, outliers, and HIPAA de-identification.
    """

    def clean(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Run full cleaning pipeline."""
        logger.info(f"Cleaning {dataset_type}: {len(df)} rows input")
        df = self._remove_duplicates(df)
        df = self._remove_outliers(df)
        df = self._deidentify(df)
        logger.info(f"Cleaning complete: {len(df)} rows output")
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        n_before = len(df)
        df = df.drop_duplicates()
        removed = n_before - len(df)
        if removed:
            logger.warning(f"Removed {removed} duplicate rows")
        return df

    def _remove_outliers(self, df: pd.DataFrame, z_threshold: float = 4.0) -> pd.DataFrame:
        """Flag and cap extreme outliers using Z-score method."""
        numeric = df.select_dtypes(include=[np.number])
        for col in numeric.columns:
            mean, std = df[col].mean(), df[col].std()
            if std == 0:
                continue
            z = (df[col] - mean) / std
            outliers = (z.abs() > z_threshold).sum()
            if outliers > 0:
                logger.warning(f"Column '{col}': {outliers} outliers capped at ±{z_threshold}σ")
                df[col] = df[col].clip(
                    lower=mean - z_threshold * std,
                    upper=mean + z_threshold * std,
                )
        return df

    def _deidentify(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove or hash direct identifiers per HIPAA Safe Harbor method."""
        phi_columns = [
            "patient_id", "mrn", "ssn", "name", "first_name", "last_name",
            "date_of_birth", "dob", "phone", "email", "address", "zip_code",
            "ip_address", "device_id", "biometric",
        ]
        for col in phi_columns:
            if col in df.columns:
                # Hash rather than drop — preserves relational joins
                df[col] = df[col].apply(
                    lambda x: str(hash(str(x)))[:12] if pd.notna(x) else x
                )
                logger.debug(f"De-identified column: {col}")

        # Generalize ZIP codes to first 3 digits
        if "zip_code" in df.columns:
            df["zip_code"] = df["zip_code"].astype(str).str[:3] + "XX"

        return df
