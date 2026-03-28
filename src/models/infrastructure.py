"""
Infrastructure Needs Predictor - Health Informatics Hub
=======================================================
Predicts healthcare infrastructure requirements based on forecasted demand.

Outputs actionable infrastructure recommendations:
  - Staffing levels (nurses, physicians, support staff) by shift
  - Bed allocation across departments (ICU, Med/Surg, ED, OR)
  - Equipment utilization forecasts (ventilators, imaging, OR suites)
  - Capital investment prioritization scores
  - Supply chain order quantities
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.config import config


INFRASTRUCTURE_TARGETS = {"nurses_required":"Registered nurses needed per shift","physicians_required":"Attending physicians needed","support_staff_required":"CNA, tech, admin staff needed","icu_beds_required":"ICU beds required","medsurg_beds_required":"Med/surgical beds required","ed_rooms_required":"ED rooms required","or_suites_required":"OR suites required","ventilators_required":"Ventilators required","infusion_pumps_required":"IV pumps required","imaging_slots_required":"Daily imaging slots required"}

STAFFING_RATIOS = {"icu":{"nurses_per_patient":0.5,"physicians_per_patient":0.1},"med_surg":{"nurses_per_patient":0.25,"physicians_per_patient":0.05},"ed":{"nurses_per_patient":0.33,"physicians_per_patient":0.1},"or":{"nurses_per_suite":2.0,"physicians_per_suite":1.0}}

SAFETY_BUFFERS = {"staffing":1.10,"beds":1.15,"equipment":1.20}


class InfrastructurePredictor:
    def __init__(self):
        cfg = config.model
        self._models = {}; self._target_cols = list(INFRASTRUCTURE_TARGETS.keys())
        self._feature_cols = []; self._is_fitted = False
    def fit(self, feature_df, target_df):
        self._feature_cols = feature_df.columns.tolist(); X = feature_df.values
        for target in self._target_cols:
            if target not in target_df.columns: continue
            y = target_df[target].values
            pl = Pipeline([("scaler",StandardScaler()),("model",GradientBoostingRegressor(n_estimators=config.model.infra_n_estimators,max_depth=config.model.infra_max_depth,learning_rate=0.05,subsample=0.8,random_state=config.model.random_seed))])
            pl.fit(X,y)
            self._models[target] = pl
        self._is_fitted = True; return self
    def predict(self, forecast_df):
        if not self._is_fitted: raise RuntimeError("Model must be fitted before predicting")
        features = self._build_features(forecast_df); X = features[self._feature_cols].values
        results = forecast_df[["date","forecast"]].copy()
        for target,model in self._models.items():
            buf = "staffing" if any(k in target for k in ["nurses","physicians","staff"]) else "beds" if any(k in target for k in ["beds","rooms","suites"]) else "equipment"
            results[target] = np.ceil(model.predict(X)*SAFETY_BUFFERS[buf]).astype(int)
        results = self._add_rule_based_estimates(results)
        return results
    def _build_features(self, df):
        df = df.copy(); df["date"] = pd.to_datetime(df["date"])
        df["day_of_week"] = df["date"].dt.dayofweek; df["month"] = df["date"].dt.month
        df["is_weekend"] = (df["day_of_week"]>=5).astype(int); df["is_flu_season"] = df["month"].isin([10,11,12,1,2,3]).astype(int)
        df["forecast_7d_avg"] = df["forecast"].rolling(7,min_periods=1).mean()
        df["forecast_growth"] = df["forecast"].pct_change().fillna(0)
        df["ci_width"] = df.get("upper_ci",df["forecast"])-df.get("lower_ci",df["forecast"])
        return df
    def _add_rule_based_estimates(self, df):
        if "forecast" in df.columns and "nurses_required" not in df.columns:
            df["nurses_required"] = np.ceil(df["forecast"]*STAFFING_RATIOS["med_surg"]["nurses_per_patient"]*SAFETY_BUFFERS["staffing"]).astype(int)
        if "forecast" in df.columns and "icu_beds_required" not in df.columns:
            df["icu_beds_required"] = np.ceil(df["forecast"]*0.10*SAFETY_BUFFERS["beds"]).astype(int)
        return df
    def get_recommendations(self, infra_df):
        recs = []; peak_row = infra_df.loc[infra_df["forecast"].idxmax()]
        avg_f = infra_df["forecast"].mean(); peak_f = infra_df["forecast"].max(); pdate = str(peak_row["date"])[:10]
        if peak_f > avg_f*1.25: recs.append({"priority":"HIGH","category":"Surge Capacity","finding":f"Peak demand on {pdate} ({peak_f:.0f} patients) exceeds baseline by {(peak_f/avg_f-1)*100:.0f}%","action":"Activate surge protocols: open overflow units, defer elective procedures","timeframe":"Immediate (within 48h of surge date)"})
        if "nurses_required" in infra_df.columns: recs.append({"priority":"MEDIUM","category":"Staffing","finding":f"Peak staffing: {infra_df['nurses_required'].max()} nurses on {pdate}","action":"Pre-schedule PRN pool and staffing agency","timeframe":"2 weeks prior"})
        if "icu_beds_required" in infra_df.columns: recs.append({"priority":"HIGH","category":"ICU Capacity","finding":f"Peak ICU requirement: {infra_df['icu_beds_required'].max()} beds","action":"Coordinate step-down unit and regional transfer agreements","timeframe":"1 week prior"})
        recs.sort(key=lambda x:{"HIGH":0,"MEDIUM":1,"LOW":2}[x["priority"]]); return recs
