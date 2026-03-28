"""
Azure client helpers for Health Informatics Hub.
Provides authenticated clients for Azure Blob Storage, Key Vault, and Azure ML.
"""

import io
from pathlib import Path
from typing import Optional, Union
import pandas as pd
from loguru import logger

from azure.identity import DefaultAzureCredential, ClientSecretCredential
from azure.storage.blob import BlobServiceClient, ContainerClient
from azure.keyvault.secrets import SecretClient

from src.utils.config import config


def get_credential():
    """
    Get Azure credential using DefaultAzureCredential (supports managed identity,
    env vars, CLI login, etc.) with fallback to client secret.
    """
    azure_cfg = config.azure
    if azure_cfg.client_id and azure_cfg.client_secret and azure_cfg.tenant_id:
        logger.debug("Using ClientSecretCredential")
        return ClientSecretCredential(
            tenant_id=azure_cfg.tenant_id,
            client_id=azure_cfg.client_id,
            client_secret=azure_cfg.client_secret,
        )
    logger.debug("Using DefaultAzureCredential")
    return DefaultAzureCredential()


class BlobStorageClient:
    """
    Wrapper around Azure Blob Storage for healthcare data lake operations.

    Handles upload/download of raw data, processed datasets, and ML model artifacts.
    All paths follow the pattern: <container>/<source>/<YYYY-MM-DD>/<filename>
    """

    def __init__(self):
        credential = get_credential()
        account_url = f"https://{config.azure.storage_account_name}.blob.core.windows.net"
        self._client = BlobServiceClient(account_url=account_url, credential=credential)
        logger.info(f"Initialized BlobStorageClient for {account_url}")

    def _container_client(self, container: str) -> ContainerClient:
        return self._client.get_container_client(container)

    def upload_dataframe(
        self,
        df: pd.DataFrame,
        blob_path: str,
        container: str = None,
        format: str = "parquet",
        overwrite: bool = True,
    ) -> str:
        """Upload a DataFrame to Blob Storage as parquet or CSV."""
        container = container or config.azure.storage_container_processed
        buf = io.BytesIO()

        if format == "parquet":
            df.to_parquet(buf, index=False)
            blob_path = blob_path if blob_path.endswith(".parquet") else f"{blob_path}.parquet"
        elif format == "csv":
            df.to_csv(buf, index=False)
            blob_path = blob_path if blob_path.endswith(".csv") else f"{blob_path}.csv"
        else:
            raise ValueError(f"Unsupported format: {format}")

        buf.seek(0)
        blob_client = self._client.get_blob_client(container=container, blob=blob_path)
        blob_client.upload_blob(buf, overwrite=overwrite)
        full_path = f"abfss://{container}@{config.azure.storage_account_name}.dfs.core.windows.net/{blob_path}"
        logger.info(f"Uploaded DataFrame ({len(df)} rows) → {full_path}")
        return full_path

    def download_dataframe(
        self,
        blob_path: str,
        container: str = None,
        format: str = "parquet",
    ) -> pd.DataFrame:
        """Download a DataFrame from Blob Storage."""
        container = container or config.azure.storage_container_processed
        blob_client = self._client.get_blob_client(container=container, blob=blob_path)
        data = blob_client.download_blob().readall()
        buf = io.BytesIO(data)

        if format == "parquet":
            df = pd.read_parquet(buf)
        elif format == "csv":
            df = pd.read_csv(buf)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Downloaded DataFrame ({len(df)} rows) from {container}/{blob_path}")
        return df

    def upload_file(
        self,
        local_path: Union[str, Path],
        blob_path: str,
        container: str = None,
        overwrite: bool = True,
    ) -> str:
        """Upload a local file to Blob Storage."""
        container = container or config.azure.storage_container_models
        with open(local_path, "rb") as f:
            blob_client = self._client.get_blob_client(container=container, blob=blob_path)
            blob_client.upload_blob(f, overwrite=overwrite)
        logger.info(f"Uploaded {local_path} → {container}/{blob_path}")
        return f"{container}/{blob_path}"

    def list_blobs(self, container: str, prefix: str = "") -> list[str]:
        """List all blobs in a container with optional prefix filter."""
        container_client = self._container_client(container)
        return [b.name for b in container_client.list_blobs(name_starts_with=prefix)]


class KeyVaultClient:
    """Wrapper for Azure Key Vault secret retrieval."""

    def __init__(self):
        if not config.azure.key_vault_url:
            logger.warning("KEY_VAULT_URL not set — KeyVaultClient will not function")
            self._client = None
            return
        credential = get_credential()
        self._client = SecretClient(
            vault_url=config.azure.key_vault_url,
            credential=credential,
        )
        logger.info(f"Initialized KeyVaultClient for {config.azure.key_vault_url}")

    def get_secret(self, name: str) -> Optional[str]:
        """Retrieve a secret value by name."""
        if not self._client:
            return None
        secret = self._client.get_secret(name)
        return secret.value

    def set_secret(self, name: str, value: str) -> None:
        """Store a secret in Key Vault."""
        if not self._client:
            raise RuntimeError("KeyVaultClient not initialized — KEY_VAULT_URL missing")
        self._client.set_secret(name, value)
        logger.info(f"Secret '{name}' stored in Key Vault")
