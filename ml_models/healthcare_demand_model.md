# Healthcare Demand Score Model

The Healthcare Demand Score estimates healthcare demand within a geographic area.

The goal is to identify regions where healthcare infrastructure may be insufficient
relative to population needs.

This model supports:

- hospital expansion planning
- healthcare investment decisions
- public health planning
- regional healthcare analytics

---

# Model Concept

Healthcare demand is influenced by multiple factors:

1. Population size
2. Age distribution
3. Hospital infrastructure
4. Disease burden
5. Insurance coverage

The model calculates a demand score for each geographic area.

---

# Model Inputs

## Population

Total population within a geographic region.

Example:

| Area | Population |
|-----|------------|
Cebu | 5,000,000 |
Ohio | 11,000,000 |

---

## Age Distribution

Older populations typically generate higher healthcare demand.

Example age weights:

| Age Group | Demand Weight |
|----------|---------------|
0-14 | 0.4 |
15-44 | 0.7 |
45-64 | 1.2 |
65+ | 2.0 |

---

## Hospital Capacity

Healthcare infrastructure availability.

Example metrics:

- hospital bed count
- hospital density per population
- ICU bed availability

Example:

| Region | Beds | Population | Beds per 1000 |
|-------|------|-----------|---------------|
Cebu | 2500 | 5,000,000 | 0.5 |

---

## Disease Prevalence

Higher disease prevalence increases healthcare demand.

Examples:

- diabetes
- cardiovascular disease
- respiratory illness

---

## Insurance Coverage

Regions with higher insurance coverage often have higher healthcare utilization.

---

# Demand Score Formula

Example simplified scoring formula:

Demand Score =

(population_factor × age_factor × disease_factor) ÷ hospital_capacity_factor

Higher scores indicate regions with greater unmet healthcare demand.

---

# Example Output

| Country | Region | Demand Score | Demand Category |
|--------|--------|--------------|----------------|
Philippines | Cebu | 0.82 | High |
United States | Ohio | 0.55 | Medium |

Demand categories:

Low  
Medium  
High  

---

# Future Model Enhancements

Future versions may incorporate:

- machine learning models
- historical utilization data
- hospital admission rates
- predictive population growth

---

# Platform Integration

The Healthcare Demand Score will be used by:

- analytics dashboards
- Power BI reports
- regional healthcare planning tools
- investment analytics
