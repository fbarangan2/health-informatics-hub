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

## Dim_Geography

Represents geographic administrative areas within a country.

This table supports hierarchical geographic structures such as:

Country → State → County → City  
Country → Region → Province → Municipality

| Field | Description |
|------|-------------|
geo_id | unique geographic identifier |
geo_name | name of geographic area |
geo_level | level type (country, region, state, province, city, etc) |
parent_geo_id | parent geographic level |
country_id | reference to country |

Example hierarchy:

| geo_id | geo_name | geo_level | parent_geo_id | country_id |
|------|-------------|-------------|-------------|-------------|
1 | Philippines | country | null | PH |
2 | Central Luzon | region | 1 | PH |
3 | Bulacan | province | 2 | PH |
4 | Malolos | city | 3 | PH |
5 | United States | country | null | US |
6 | Ohio | state | 5 | US |

This design allows unlimited geographic levels across countries.

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
geo_id | geographic area |
year | year |
age_group_id | age group |
population | population count |

---

## Fact_Hospital_Capacity

Stores hospital infrastructure information.

| Field | Description |
|------|-------------|
country_id | country |
geo_id | geographic area |
hospital_name | hospital |
bed_capacity | total beds |
hospital_type | hospital classification |

---

## Fact_Healthcare_Demand

Stores healthcare demand analytics.

| Field | Description |
|------|-------------|
country_id | country |
geo_id | geographic area |
year | year |
demand_score | calculated demand score |
demand_category | low / medium / high |

---

# Example Analytics Query

Example question:

Which geographic areas have the highest healthcare demand?

Example output:

| Country | Geography | Demand Score |
|--------|--------|--------------|
Philippines | Bulacan | High |
United States | Ohio | Medium |

---

# Design Principles

The model follows these principles:

- country-agnostic design
- scalable to multiple datasets
- hierarchical geographic support
- compatible with cloud data platforms
- optimized for analytics and machine learning
