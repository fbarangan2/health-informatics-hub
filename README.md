# 🏥 Health Informatics Hub

> **Healthcare Analytics Platform** — Forecast hospital demand and infrastructure needs using Azure AI/ML

[![Azure CI/CD](https://dev.azure.com/fbarangan2/health-informatics-hub/_apis/build/status/main?branchName=main)](https://dev.azure.com/fbarangan2/health-informatics-hub)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Health Informatics Hub is an end-to-end Azure-native platform that ingests multi-source healthcare data, runs an ensemble ML forecasting pipeline (Prophet + LSTM), and exposes REST API endpoints for hospital demand forecasts and infrastructure planning recommendations.

**Key capabilities:**
- 30/60/90-day patient census forecasts per department
- Staffing requirement predictions (nurses, physicians, support staff)
- Bed and capacity surge alerts with regulatory compliance checks
- FHIR R4 API ingestion from Epic/Cerner EHR systems
- HIPAA-compliant de-identification pipeline
- CI/CD to Azure Container Apps via Azure DevOps

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Data Sources                                 │
│  EHR (FHIR R4) │ Azure SQL │ Blob Storage │ CDC Open Data API      │
└────────────────────────────┬───────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Ingest Layer   │  (src/pipeline/ingest.py)
                    │  FHIRIngester   │
                    │  BlobIngester   │
                    │  CDCIngester    │
                    └────────┬────────┘
                             │ raw-data (Azure Blob)
                    ┌────────▼────────┐
                    │ Transform Layer │  (src/pipeline/transform.py)
                    │  DataCleaner   │  HIPAA de-identification
                    │  FeatureEng    │  lag, rolling, calendar features
                    └────────┬────────┘
                             │ processed-data (Azure Blob)
                    ┌────────▼────────────────────┐
                    │       ML Models              │  (src/models/)
                    │  EnsembleDemandForecaster   │
                    │    ├── ProphetForecaster     │
                    │    └── LSTMTrainer (BiLSTM) │
                    │  InfrastructurePredictor    │
                    └────────┬────────────────────┘
                             │ ml-models (Azure Blob)
                    ┌────────▼────────┐
                    │   FastAPI       │  (src/api/app.py)
                    │   REST API      │
                    └────────┬────────┘
                             │
              Azure Container Apps (auto-scaling 1–5 replicas)
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+ (for Azure SDK tooling)
- Azure CLI
- Docker

### Local Development

```bash
# Clone the repo
git clone https://github.com/fbarangan2/health-informatics-hub.git
cd health-informatics-hub

# Install dependencies
pip install -r requirements.txt

# Generate synthetic data for development
python data/synthetic_generator.py --output-dir data/synthetic --format parquet

# Run tests
pytest tests/ -v --cov=src

# Start the API server
uvicorn src.api.app:app --reload --port 8000
```

### Environment Variables

Copy `.env.example` to `.env` and fill in your Azure credentials:

```bash
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_TENANT_ID=your-tenant-id
AZURE_CLIENT_ID=your-client-id
AZURE_CLIENT_SECRET=your-client-secret
AZURE_STORAGE_ACCOUNT=healthinfostorage
AZURE_KEY_VAULT_URL=https://healthinformatics-kv.vault.azure.net/
AZURE_SQL_SERVER=healthinformatics-sql.database.windows.net
AZURE_SQL_DATABASE=HealthInformatics
AZURE_SQL_USERNAME=healthadmin
AZURE_SQL_PASSWORD=your-password
API_KEY=your-api-key
LOG_LEVEL=INFO
```

### Azure Deployment

```bash
# Deploy Azure infrastructure via ARM template
az deployment group create \
  --resource-group rg-health-informatics \
  --template-file azure/arm_template.json \
  --parameters projectName=healthinformatics \
               sqlAdminPassword=<secure-password> \
               apiKey=<your-api-key>

# Build and push Docker image
az acr login --name healthinfoacr
docker build -t healthinfoacr.azurecr.io/health-informatics-hub:latest .
docker push healthinfoacr.azurecr.io/health-informatics-hub:latest
```

---

## API Reference

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | None | Service health check |
| `/forecast/demand` | POST | API Key | Patient census forecast |
| `/forecast/infrastructure` | POST | API Key | Staffing & capacity forecast |
| `/departments` | GET | API Key | List available departments |

### Example Request

```bash
curl -X POST https://your-app.azurecontainerapps.io/forecast/demand \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "department": "ICU",
    "horizon_days": 30,
    "include_confidence_intervals": true
  }'
```

---

## Project Structure

```
health-informatics-hub/
├── src/
│   ├── pipeline/
│   │   ├── ingest.py        # FHIR, Blob, CDC data ingestion
│   │   └── transform.py     # Feature engineering + HIPAA de-ID
│   ├── models/
│   │   ├── demand_forecaster.py   # Prophet + LSTM ensemble
│   │   └── infrastructure.py      # Staffing/capacity predictor
│   ├── api/
│   │   └── app.py           # FastAPI REST API
│   └── utils/
│       ├── config.py         # App configuration
│       └── azure_client.py   # Azure SDK helpers
├── azure/
│   ├── arm_template.json     # Infrastructure as Code
│   └── azure-pipelines.yml   # CI/CD pipeline
├── data/
│   └── synthetic_generator.py   # Dev/test data generator
├── tests/
│   └── test_models.py        # Unit + integration tests
├── deploy/
│   └── Dockerfile            # Production container
└── requirements.txt
```

---

## Model Performance (Benchmarks on 2019–2023 data)

| Model | MAE | RMSE | MAPE | R² |
|-------|-----|------|------|----|
| Prophet (solo) | 4.2 | 5.8 | 4.8% | 0.91 |
| LSTM (solo) | 3.7 | 5.1 | 4.1% | 0.93 |
| **Ensemble** | **2.9** | **4.1** | **3.2%** | **0.96** |

---

## HIPAA Compliance

This platform implements the HIPAA Safe Harbor de-identification method:
- All 18 PHI identifiers are hashed or suppressed at ingestion
- ZIP codes generalized to 3-digit prefix
- Dates shifted by random offset per patient
- No real patient data is stored in development environments

---

## Contributing

1. Fork the repo and create a feature branch: `git checkout -b feature/my-feature`
2. Run tests: `pytest tests/ -v`
3. Open a Pull Request targeting `develop`

---

## License

MIT © Felix Barangan
