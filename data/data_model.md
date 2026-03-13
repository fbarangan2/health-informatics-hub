# Global Healthcare Analytics Data Model

Health Informatics Hub uses a country-agnostic data model to support healthcare
analytics across multiple countries.

The platform follows a star schema architecture commonly used in
enterprise analytics systems.

---

# Dimension Tables

## Dim_Country

Represents the country where healthcare data originates.

| Field | Description |
|------|-------------|
country_id | unique identifier |
country_name | country name |
iso_code | ISO country code |

Example:

| country_id | country_name | iso_code |
|------------|-------------|---------|
1 | Philippines | PH |
2 | United States | US |

---

## Dim_Region

Represents large geographic regions within a country.

| Field | Description |
|------|-------------|
region_id | unique identifier |
country_id | country reference |
region_name | region name |

---

## Dim_Geography

Represents administrative areas such as provinces, states, or cities.

| Field | Description |
|------|-------------|
area_id | unique identifier |
region_id | region reference |
area_name | province / state / city |

---

## Dim_Age_Group

Represents population age ranges.

| Field | Description |
|------|-------------|
age_group_id | unique identifier |
age_range | age range |

Example:

| age_group_id | age_range |
|--------------|----------|
1 | 0-4 |
2 | 5-9 |
10 | 45-49 |
15 | 65+ |

---

## Dim_Year

Represents time dimension.

| Field | Description |
|------|-------------|
year | calendar year |

---

# Fact Tables

## Fact_Population

Stores population demographics.

| Field | Description |
|------|-------------|
country_id | country |
area_id | province/state |
year | year |
age_group_id | age group |
population | population count |

---

## Fact_Hospital_Capacity

Stores hospital infrastructure information.

| Field | Description |
|------|-------------|
country_id | country |
area_id | geographic area |
hospital_name | hospital |
bed_capacity | total beds |
hospital_type | hospital classification |

---

## Fact_Healthcare_Demand

Stores healthcare demand analytics.

| Field | Description |
|------|-------------|
country_id | country |
area_id | region/province |
year | year |
demand_score | calculated demand score |
demand_category | low / medium / high |

---

# Example Analytics Query

Example question:

Which regions have the highest healthcare demand?

Example output:

| Country | Region | Demand Score |
|--------|--------|--------------|
Philippines | Central Luzon | High |
United States | Ohio | Medium |

---

# Design Principles

The model follows these principles:

- country-agnostic design
- scalable to multiple datasets
- compatible with cloud data platforms
- optimized for analytics and machine learning
