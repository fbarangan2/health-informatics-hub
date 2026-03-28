"""
Configuration management for Health Informatics Hub.
Loads settings from environment variables and .env files.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()


@dataclass
class AzureConfig:
    """Azure service configuration."""
    subscription_id: str = field(default_factory=lambda: os.getenv("AZURE_SUBSCRIPTION_ID", ""))
    resource_group: str = field(default_factory=lambda: os.getenv("AZURE_RESOURCE_GROUP", "rg-health-informatics"))
    tenant_id: str = field(default_factory=lambda: os.getenv("AZURE_TENANT_ID", ""))
    client_id: str = field(default_factory=lambda: os.getenv("AZURE_CLIENT_ID", ""))
    client_secret: str = field(default_factory=lambda: os.getenv("AZURE_CLIENT_SECRET", ""))

    # Storage
    storage_account_name: str = field(default_factory=lambda: os.getenv("AZURE_STORAGE_ACCOUNT", "healthinfostorage"))
    storage_container_raw: str = "raw-data"
    storage_container_processed: str = "processed-data"
    storage_container_models: str = "ml-models"

    # Key Vault
    key_vault_url: str = field(default_factory=lambda: os.getenv("AZURE_KEY_VAULT_URL", ""))

    # Azure ML
    workspace_name: str = field(default_factory=lambda: os.getenv("AZUREML_WORKSPACE", "health-informatics-ml"))
    ml_compute_cluster: str = "cpu-cluster"

    # Azure SQL
    sql_server: str = field(default_factory=lambda: os.getenv("AZURE_SQL_SERVER", ""))
    sql_database: str = field(default_factory=lambda: os.getenv("AZURE_SQL_DATABASE", "HealthInformatics"))
    sql_username: str = field(default_factory=lambda: os.getenv("AZURE_SQL_USERNAME", ""))
    sql_password: str = field(default_factory=lambda: os.getenv("AZURE_SQL_PASSWORD", ""))

    @property
    def sql_connection_string(self) -> str:
        return (
            f"DRIVER={{ODBC Driver 18 for SQL Server}};"
            f"SERVER={self.sql_server};"
            f"DATABASE={self.sql_database};"
            f"UID={self.sql_username};"
            f"PWD={self.sql_password};"
            f"Encrypt=yes;TrustServerCertificate=no;"
        )


@dataclass
class ModelConfig:
    """ML model configuration."""
    # General
    random_seed: int = 42
    test_size: float = 0.2
    validation_size: float = 0.1

    # Forecasting horizon
    forecast_horizon_days: int = 30
    lookback_window_days: int = 90

    # LSTM settings
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_learning_rate: float = 0.001
    lstm_epochs: int = 100
    lstm_batch_size: int = 32

    # Prophet settings
    prophet_changepoint_prior_scale: float = 0.05
    prophet_seasonality_mode: str = "multiplicative"
    prophet_yearly_seasonality: bool = True
    prophet_weekly_seasonality: bool = True

    # Infrastructure predictor
    infra_model_type: str = "gradient_boosting"  # xgboost | gradient_boosting | random_forest
    infra_n_estimators: int = 200
    infra_max_depth: int = 6


@dataclass
class APIConfig:
    """FastAPI configuration."""
    host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    debug: bool = field(default_factory=lambda: os.getenv("API_DEBUG", "false").lower() == "true")
    api_key: str = field(default_factory=lambda: os.getenv("API_KEY", "dev-key-change-in-prod"))
    allowed_origins: list = field(default_factory=lambda: ["*"])

    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60


@dataclass
class AppConfig:
    """Root application configuration."""
    azure: AzureConfig = field(default_factory=AzureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    api: APIConfig = field(default_factory=APIConfig)

    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_format: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan> - {message}"

    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models" / "saved"

    @property
    def logs_dir(self) -> Path:
        return self.project_root / "logs"


# Singleton config instance
config = AppConfig()
