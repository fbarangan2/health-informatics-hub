# Health Informatics Hub Platform Architecture

Health Informatics Hub is a cloud-based healthcare analytics SaaS platform.

The platform is designed to be country-agnostic and support healthcare
infrastructure analytics globally.

Core Architecture Layers:

1. Data Ingestion
   - Azure Data Factory
   - Public datasets (PSA, CMS, WHO)

2. Data Lake
   - Azure Data Lake Storage
   - Raw and processed datasets

3. Data Processing
   - Azure Databricks
   - ETL and data transformation

4. Data Warehouse
   - Azure SQL / Synapse
   - Star schema data model

5. Analytics & AI
   - Machine learning demand forecasting

6. Application Layer
   - API services
   - SaaS platform dashboard
