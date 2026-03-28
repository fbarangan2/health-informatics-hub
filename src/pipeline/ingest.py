"""
Data Ingestion Pipeline - Health Informatics Hub
================================================
Handles ingestion from multiple healthcare data sources:
  - Hospital EHR systems (HL7 FHIR API)
  - Azure Blob Storage (historical exports)
  - Azure SQL Database (operational data)
  - Public datasets (CDC, CMS, state health departments)

All ingested data lands in the raw-data container partitioned by source/date.
"""

import io
import json
from datetime import date, datetime, timedelta
from typing import Optional

import httpx
import pandas as pd
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.azure_client import BlobStorageClient
from src.utils.config import config


class FHIRIngester:
    """
    FHIR R4 API ingester for EHR data.

    Pulls Encounter, Observation, Condition, and Patient resources
    from a FHIR-compliant endpoint (Epic, Cerner, Azure Health Data Services).
    """

    RESOURCE_TYPES = ["Encounter", "Observation", "Condition", "Patient", "Appointment"]

    def __init__(self, fhir_base_url: str, auth_token: str):
        self.base_url = fhir_base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {auth_token}",
            "Accept": "application/fhir+json",
            "Content-Type": "application/fhir+json",
        }
        self._client = httpx.Client(headers=self.headers, timeout=30)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _get_page(self, url: str) -> dict:
        """Fetch a single FHIR Bundle page with retry logic."""
        response = self._client.get(url)
        response.raise_for_status()
        return response.json()

    def fetch_encounters(
        self,
        start_date: date,
        end_date: date,
        department: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch all Encounter resources within a date range.

        Returns a flattened DataFrame with:
          encounter_id, patient_id, status, class, type_code, type_display,
          start_datetime, end_datetime, length_of_stay_hours, department,
          admission_source, discharge_disposition
        """
        params = {
            "date": f"ge{start_date.isoformat()}",
            "_lastUpdated": f"le{end_date.isoformat()}",
            "_count": 200,
            "_format": "json",
        }
        if department:
            params["location.name"] = department

        url = f"{self.base_url}/Encounter?" + "&".join(f"{k}={v}" for k, v in params.items())
        records = []

        while url:
            bundle = self._get_page(url)
            entries = bundle.get("entry", [])
            for entry in entries:
                resource = entry.get("resource", {})
                records.append(self._flatten_encounter(resource))

            # Follow pagination links
            url = next(
                (link["url"] for link in bundle.get("link", []) if link["relation"] == "next"),
                None,
            )
            logger.debug(f"Fetched {len(records)} encounters so far...")

        df = pd.DataFrame(records)
        logger.info(f"Ingested {len(df)} encounters from {start_date} to {end_date}")
        return df

    def _flatten_encounter(self, resource: dict) -> dict:
        """Flatten a FHIR Encounter resource to a flat dict."""
        period = resource.get("period", {})
        start = period.get("start")
        end = period.get("end")

        los_hours = None
        if start and end:
            delta = datetime.fromisoformat(end) - datetime.fromisoformat(start)
            los_hours = delta.total_seconds() / 3600

        type_coding = resource.get("type", [{}])[0].get("coding", [{}])[0]

        return {
            "encounter_id": resource.get("id"),
            "patient_id": resource.get("subject", {}).get("reference", "").replace("Patient/", ""),
            "status": resource.get("status"),
            "class": resource.get("class", {}).get("code"),
            "type_code": type_coding.get("code"),
            "type_display": type_coding.get("display"),
            "start_datetime": start,
            "end_datetime": end,
            "length_of_stay_hours": los_hours,
            "department": resource.get("serviceProvider", {}).get("display"),
            "admission_source": resource.get("hospitalization", {})
                .get("admitSource", {})
                .get("coding", [{}])[0]
                .get("display"),
            "discharge_disposition": resource.get("hospitalization", {})
                .get("dischargeDisposition", {})
                .get("coding", [{}])[0]
                .get("display"),
        }


class HistoricalDataIngester:
    """
    Ingests historical healthcare datasets from Azure Blob Storage raw exports.

    Supports CSV, Parquet, and JSON formats with schema validation.
    """

    EXPECTED_SCHEMA = {
        "census_data": ["date", "department", "patient_count", "bed_count", "occupancy_rate"],
        "admission_data": ["admission_date", "discharge_date", "department", "diagnosis_code", "severity"],
        "staffing_data": ["date", "department", "nurses", "physicians", "support_staff"],
        "appointment_data": ["appointment_date", "department", "scheduled", "attended", "cancelled", "no_show"],
    }

    def __init__(self):
        self.blob = BlobStorageClient()

    def ingest_blob(self, blob_path: str, dataset_type: str) -> pd.DataFrame:
        """
        Download and validate a dataset from Blob Storage.

        Args:
            blob_path: Path within the raw-data container.
            dataset_type: One of the EXPECTED_SCHEMA keys for validation.

        Returns:
            Validated and typed DataFrame.
        """
        logger.info(f"Ingesting {dataset_type} from {blob_path}")
        fmt = "parquet" if blob_path.endswith(".parquet") else "csv"
        df = self.blob.download_dataframe(
            blob_path=blob_path,
            container=config.azure.storage_container_raw,
            format=fmt,
        )
        df = self._validate_and_type(df, dataset_type)
        return df

    def _validate_and_type(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """Validate schema and cast column types."""
        expected_cols = self.EXPECTED_SCHEMA.get(dataset_type, [])
        missing = set(expected_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Schema validation failed for '{dataset_type}': missing columns {missing}")

        # Cast date columns
        for col in df.columns:
            if "date" in col.lower() or "datetime" in col.lower():
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Cast numeric columns
        numeric_hints = ["count", "rate", "hours", "nurses", "physicians", "staff",
                         "scheduled", "attended", "cancelled", "no_show"]
        for col in df.columns:
            if any(hint in col.lower() for hint in numeric_hints):
                df[col] = pd.to_numeric(df[col], errors="coerce")

        logger.info(f"Validated {dataset_type}: {len(df)} rows, {len(df.columns)} columns")
        return df


class CDCDataIngester:
    """
    Ingests public health data from CDC Open Data API.
    Used to enrich hospital demand forecasts with disease surveillance data.
    """

    CDC_BASE_URL = "https://data.cdc.gov/resource"

    # Dataset IDs for relevant CDC datasets
    DATASETS = {
        "flu_surveillance": "9vh5-5ue9",       # FluView ILI data
        "covid_hospitalizations": "g62h-syeh",  # COVID-19 Hospital Capacity
        "chronic_disease": "g4ie-h725",         # Chronic Disease Indicators
    }

    def __init__(self):
        self._client = httpx.Client(timeout=30)

    def fetch_flu_data(self, start_year: int = 2020) -> pd.DataFrame:
        """Fetch influenza-like illness (ILI) surveillance data from CDC."""
        dataset_id = self.DATASETS["flu_surveillance"]
        url = f"{self.CDC_BASE_URL}/{dataset_id}.json"
        params = {
            "$where": f"year >= {start_year}",
            "$limit": 50000,
            "$order": "week_start ASC",
        }
        response = self._client.get(url, params=params)
        response.raise_for_status()
        df = pd.DataFrame(response.json())
        logger.info(f"Fetched {len(df)} flu surveillance records from CDC")
        return df
