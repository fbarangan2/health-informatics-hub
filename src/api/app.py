"""
FastAPI Application - Health Informatics Hub
============================================
RESTful API for demand forecasts and infrastructure recommendations.
Deployed as Azure Container Apps or Azure App Service.
"""

from contextlib import asynccontextmanager
from datetime import date

from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from loguru import logger
from pydantic import BaseModel, Field

from src.utils.config import config


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != config.api.api_key: raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
    return api_key


class ForecastRequest(BaseModel):
    department: str = Field(default="all", description="Department to forecast")
    horizon_days: int = Field(default=30, ge=1, le=180)
    start_date: date = Field(default_factory=date.today)
    include_confidence_intervals: bool = Field(default=True)


class ForecastPoint(BaseModel):
    date: date
    forecast: float
    lower_ci: float | None = None
    upper_ci: float | None = None


class ForecastResponse(BaseModel):
    department: str; horizon_days: int; generated_at: str; model_version: str
    forecast: list[ForecastPoint]; metrics: dict | None = None


class InfrastructureRequest(BaseModel):
    department: str = Field(default="all")
    horizon_days: int = Field(default=30, ge=1, le=90)
    include_recommendations: bool = Field(default=True)


class Recommendation(BaseModel):
    priority: str; category: str; finding: str; action: str; timeframe: str


class InfrastructureResponse(BaseModel):
    department: str; horizon_days: int; generated_at: str
    staffing: list[dict]; capacity: list[dict]; equipment: list[dict]
    recommendations: list[Recommendation] | None = None


class HealthResponse(BaseModel):
    status: str; version: str; azure_connected: bool


MODEL_REGISTRY: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Health Informatics Hub API starting up...")
    MODEL_REGISTRY["version"] = "1.0.0"; MODEL_REGISTRY["loaded"] = False
    yield
    logger.info("Health Informatics Hub API shutting down...")
    MODEL_REGISTRY.clear()


app = FastAPI(title="Health Informatics Hub API", description="Healthcare analytics platform for forecasting hospital demand and infrastructure needs.",version="1.0.0",docs_url="/docs",redoc_url="/redoc",lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=config.api.allowed_origins,allow_credentials=True,allow_methods=["GET","POST"],allow_headers=["*"])


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(status="healthy",version="1.0.0",azure_connected=bool(config.azure.storage_account_name))


@app.post("/forecast/demand",response_model=ForecastResponse,tags=["Forecasting"],dependencies=[Depends(verify_api_key)])
async def forecast_demand(request: ForecastRequest):
    from datetime import datetime
    logger.info(f"Forecast request: dept={request.department}, horizon={request.horizon_days}d")
    if not MODEL_REGISTRY.get("loaded"): raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,detail="Models not yet loaded.")
    forecaster = MODEL_REGISTRY.get(f"forecaster_{request.department}")
    if not forecaster: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,detail=f"No model for '{request.department}'")
    result_df = forecaster.predict(df=MODEL_REGISTRY.get("latest_data"),horizon_days=request.horizon_days)
    pts = [ForecastPoint(date=row["date"].date() if hasattr(row["date"],"date") else row["date"],forecast=round(row["forecast"],2),lower_ci=round(row["lower_ci"],2) if request.include_confidence_intervals else None,upper_ci=round(row["upper_ci"],2) if request.include_confidence_intervals else None) for _,row in result_df.iterrows()]
    return ForecastResponse(department=request.department,horizon_days=request.horizon_days,generated_at=datetime.utcnow().isoformat()+"Z",model_version=MODEL_REGISTRY.get("version","unknown"),forecast=pts)


@app.post("/forecast/infrastructure",response_model=InfrastructureResponse,tags=["Infrastructure"],dependencies=[Depends(verify_api_key)])
async def forecast_infrastructure(request: InfrastructureRequest):
    from datetime import datetime
    logger.info(f"Infra request: dept={request.department}, horizon={request.horizon_days}d")
    if not MODEL_REGISTRY.get("loaded"): raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,detail="Models not yet loaded.")
    infra_predictor = MODEL_REGISTRY.get("infra_predictor"); forecaster = MODEL_REGISTRY.get(f"forecaster_{request.department}")
    if not infra_predictor or not forecaster: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,detail="Infrastructure predictor not available")
    demand_df = forecaster.predict(df=MODEL_REGISTRY.get("latest_data"),horizon_days=request.horizon_days)
    infra_df = infra_predictor.predict(demand_df)
    scols = ["date","nurses_required","physicians_required","support_staff_required"]
    ccols = ["date","icu_beds_required","medsurg_beds_required","ed_rooms_required"]
    ecols = ["date","ventilators_required","imaging_slots_required"]
    recommendations = None
    if request.include_recommendations: recommendations = [Recommendation(**r) for r in infra_predictor.get_recommendations(infra_df)]
    return InfrastructureResponse(department=request.department,horizon_days=request.horizon_days,generated_at=datetime.utcnow().isoformat()+"Z",staffing=infra_df[[c for c in scols if c in infra_df.columns]].to_dict("records"),capacity=infra_df[[c for c in ccols if c in infra_df.columns]].to_dict("records"),equipment=infra_df[[c for c in ecols if c in infra_df.columns]].to_dict("records"),recommendations=recommendations)


@app.get("/departments",tags=["System"],dependencies=[Depends(verify_api_key)])
async def list_departments():
    depts = [k.replace("forecaster_","") for k in MODEL_REGISTRY if k.startswith("forecaster_")]
    return {"departments":depts,"count":len(depts)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app",host=config.api.host,port=config.api.port,reload=config.api.debug,log_level=config.log_level.lower())
