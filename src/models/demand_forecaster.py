"""
Hospital Demand Forecaster - Health Informatics Hub
===================================================
Ensemble of Prophet + LSTM models for multi-horizon hospital demand forecasting.

Forecasts:
  - Daily patient census (inpatient count)
  - ED visit volume
  - ICU occupancy
  - Surgical case load
  - Appointment demand by specialty

Model Architecture:
  - Prophet: captures trend, weekly/annual seasonality, holidays
  - LSTM: captures complex nonlinear temporal patterns
  - Ensemble: weighted average with dynamic confidence intervals
"""

import json
import pickle
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from src.utils.config import config


class LSTMForecastModel(nn.Module):
    def __init__(self, n_features: int, hidden_size: int=128, num_layers: int=2, dropout: float=0.2, forecast_horizon: int=30, bidirectional: bool=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.bidirectional = bidirectional
        directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout if num_layers>1 else 0.0, batch_first=True, bidirectional=bidirectional)
        self.attention = nn.Sequential(nn.Linear(hidden_size*directions, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1), nn.Softmax(dim=1))
        self.output_layer = nn.Sequential(nn.Linear(hidden_size*directions, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, forecast_horizon))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        attn_weights = self.attention(lstm_out)
        context = (attn_weights * lstm_out).sum(dim=1)
        return self.output_layer(context)


class ProphetForecaster:
    HOSPITAL_HOLIDAYS = pd.DataFrame({"holiday": ["Christmas","Christmas","New Year","New Year","Thanksgiving","Thanksgiving","Independence Day","Independence Day","Labor Day","Labor Day"],"ds": pd.to_datetime(["2022-12-25","2023-12-25","2022-01-01","2023-01-01","2022-11-24","2023-11-23","2022-07-04","2023-07-04","2022-09-05","2023-09-04"]),"lower_window":[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],"upper_window":[1,1,1,1,1,1,1,1,1,1]})
    def __init__(self, department: str="all"):
        self.department = department
        self.model = None
        self._is_fitted = False
        cfg = config.model
        self._model_params = dict(changepoint_prior_scale=cfg.prophet_changepoint_prior_scale, seasonality_mode=cfg.prophet_seasonality_mode, yearly_seasonality=cfg.prophet_yearly_seasonality, weekly_seasonality=cfg.prophet_weekly_seasonality, daily_seasonality=False, holidays=self.HOSPITAL_HOLIDAYS, interval_width=0.80)
    def fit(self, df, target_col="patient_count"):
        prophet_df = df[[target_col]].copy()
        prophet_df.index = pd.to_datetime(prophet_df.index)
        prophet_df = prophet_df.rename_axis("ds").reset_index().rename(columns={target_col:"y"}).dropna(subset=["y"])
        self.model = Prophet(**self._model_params)
        for reg in ["is_flu_season","is_weekend","occupancy_rate"]:
            if reg in df.columns:
                self.model.add_regressor(reg)
                prophet_df[reg] = df[reg].values[:len(prophet_df)]
        logger.info(f"Fitting Prophet for '{self.department}'")
        self.model.fit(prophet_df)
        self._is_fitted = True
        return self
    def predict(self, horizon_days=30):
        if not self._is_fitted: raise RuntimeError("Model must be fitted before predicting")
        future = self.model.make_future_dataframe(periods=horizon_days, freq="D")
        return self.model.predict(future)[["ds","yhat","yhat_lower","yhat_upper"]].tail(horizon_days)


class LSTMTrainer:
    def __init__(self, n_features):
        self.n_features = n_features
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = MinMaxScaler()
        self.model = None
        cfg = config.model
        self.hidden_size = cfg.lstm_hidden_size; self.num_layers = cfg.lstm_num_layers
        self.dropout = cfg.lstm_dropout; self.lr = cfg.lstm_learning_rate
        self.epochs = cfg.lstm_epochs; self.batch_size = cfg.lstm_batch_size
        self.lookback = cfg.lookback_window_days; self.horizon = cfg.forecast_horizon_days
    def _build_sequences(self, data):
        X, y = [], []
        for i in range(len(data)-self.lookback-self.horizon+1):
            X.append(data[i:i+self.lookback]); y.append(data[i+self.lookback:i+self.lookback+self.horizon,0])
        return np.array(X), np.array(y)
    def fit(self, df):
        cfg = config.model; fcols = df.select_dtypes(include=[np.number]).columns.tolist()
        scaled = self.scaler.fit_transform(df[fcols].values)
        X, y = self._build_sequences(scaled)
        split = int(len(X)*(1-cfg.validation_size))
        Xtr, ytr = torch.FloatTensor(X[:split]).to(self.device), torch.FloatTensor(y[:split]).to(self.device)
        Xval, yval = torch.FloatTensor(X[split:]).to(self.device), torch.FloatTensor(y[split:]).to(self.device)
        self.model = LSTMForecastModel(n_features=Xtr.shape[2], hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.dropout, forecast_horizon=self.horizon).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr); crit = nn.HuberLoss()
        best_vl = float("inf"); best_st = None
        for ep in range(self.epochs):
            self.model.train()
            for i in range(0,len(Xtr), self.batch_size):
                xb,Yb=Xtr[i:i+self.batch_size],ytr[i:i+self.batch_size]; opt.zero_grad(); los=crit(self.model(xb),yb); los.backward(); opt.step()
            self.model.eval()
            with torch.no_grad(): vl = crit(self.model(Xval), yval).item()
            if vl < best_vl: best_vl = vl; best_st = self.model.state_dict().copy()
        if best_st: self.model.load_state_dict(best_st)
        return self
    def predict(self, df):
        if self.model is None: raise RuntimeError("Model must be fitted first")
        fcols = df.select_dtypes(include=[np.number]).columns.tolist()
        data = self.scaler.transform(df[fcols].values[-self.lookback:])
        X = torch.FloatTensor(data).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad(): spred = self.model(X).cpu().numpy()[0]
        dummy = np.zeros((self.horizon, len(fcols))); dummy[:,0] = spred
        return self.scaler.inverse_transform(dummy)[:,0]


class EnsembleDemandForecaster:
    def __init__(self, department="all"):
        self.department = department; self.prophet = ProphetForecaster(department=department)
        self.lstm_trainer = None; self._prophet_weight = 0.5; self._lstm_weight = 0.5; self._fitted = False
    def fit(self, df, target_col="patient_count", val_df=None):
        logger.info(f"Training EnsembleDemandForecaster for '{self.department}'")
        self.prophet.fit(df, target_col=target_col)
        fcols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.lstm_trainer = LSTMTrainer(n_features=len(fcols)); self.lstm_trainer.fit(df)
        if val_df is not None: self._calibrate_weights(val_df, target_col)
        self._fitted = True; return self
    def _calibrate_weights(self, val_df, target_col):
        horizon = min(config.model.forecast_horizon_days, len(val_df))
        pf = self.prophet.predict(horizon_days=horizon)
        pmae = mean_absolute_error(val_df[target_col].values[:horizon], pf["yhat"].values)
        lpreds = self.lstm_trainer.predict(val_df)[:horizon]
        lmae = mean_absolute_error(val_df[target_col].values[:horizon], lpreds)
        total = (1/pmae)+(1/lmae); self._prophet_weight = (1/pmae)/total; self._lstm_weight = (1/lmae)/total
    def predict(self, df, horizon_days=30):
        if not self._fitted: raise RuntimeError("Forecaster must be fitted before predicting")
        pf = self.prophet.predict(horizon_days=horizon_days); lp = self.lstm_trainer.predict(df)[:horizon_days]
        ens = self._prophet_weight*pf["yhat"].values + self._lstm_weight*lp
        cis = abs(pf["yhat_upper"].values-pf["yhat_lower"].values)/(2*pf["yhat"].values.clip(1))
        return pd.DataFrame({"date":pf["ds"].values,"forecast":ens.round(1),"lower_ci":(ens*(1-cis)).round(1),"upper_ci":(ens*(1+cis)).round(1),"model_prophet":pf["yhat"].values.round(1) ,"model_lstm":lp.round(1),"prophet_weight":self._prophet_weight,"lstm_weight":self._lstm_weight})
    def evaluate(self, actual, predicted):
        return {"mae":mean_absolute_error(actual,predicted),"rmse":np.sqrt(mean_squared_error(actual,predicted)),"mape":np.mean(np.abs((actual-predicted)/actual.clip(1)))*100,"r2":r2_score(actual,predicted)}
    def save(self, output_dir):
        output_dir.mkdir(parents=True,exist_ok=True)
        with open(output_dir/"prophet_model.pkl","wb") as f: pickle.dump(self.prophet.model,f)
        if self.lstm_trainer and self.lstm_trainer.model: torch.save(self.lstm_trainer.model.state_dict(),output_dir/"lstm_weights.pt")
        with open(output_dir/"ensemble_meta.json","w") as f: import json; json.dump({"department":self.department,"prophet_weight":self._prophet_weight,"lstm_weight":self._lstm_weight},f,indent=2)
