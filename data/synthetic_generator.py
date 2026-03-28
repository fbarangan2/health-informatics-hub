"""
Synthetic Healthcare Data Generator - Health Informatics Hub
============================================================
Generates realistic synthetic hospital datasets for development and testing.
All data is synthetic and contains no real patient information.

Simulates:
  - Daily hospital census with realistic seasonal patterns
  - ED visit volumes with day-of-week effects
  - Staffing schedules with shift patterns
  - Flu season surges and holiday dips
  - Multi-year historical data (2019-2024)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def generate_hospital_census(
    start_date: str = "2019-01-01",
    end_date: str = "2024-12-31",
    departments: list[str] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic daily hospital census data.

    Incorporates:
      - Linear capacity growth trend (~2% per year)
      - Annual seasonal pattern (higher winter, lower summer)
      - Weekly pattern (lower weekends)
      - Flu season spikes (Nov–Mar)
      - Holiday dips (Christmas, Thanksgiving, July 4th)
      - COVID-19 disruption period (2020 Q1-Q2)
      - Gaussian noise for realistic variability
    """
    rng = np.random.default_rng(seed)
    departments = departments or ["ICU", "MedSurg", "ED", "Oncology", "Orthopedics"]

    dates = pd.date_range(start_date, end_date, freq="D")
    n_days = len(dates)
    t = np.arange(n_days)

    records = []
    BASE_CENSUS = {
        "ICU": 28,
        "MedSurg": 120,
        "ED": 180,
        "Oncology": 45,
        "Orthopedics": 38,
    }
    BED_CAPACITY = {
        "ICU": 32,
        "MedSurg": 145,
        "ED": 220,
        "Oncology": 52,
        "Orthopedics": 45,
    }

    for dept in departments:
        base = BASE_CENSUS.get(dept, 50)
        capacity = BED_CAPACITY.get(dept, 60)

        # Trend: 2% annual growth
        trend = base * (1 + 0.02 * t / 365)

        # Annual seasonality (winter peak)
        annual = 0.12 * base * np.sin(2 * np.pi * (t - 30) / 365)

        # Weekly seasonality (weekend dip ~10%)
        day_of_week = dates.dayofweek.values
        weekly = -0.10 * base * (day_of_week >= 5).astype(float)

        # Flu season boost (Oct–Mar): +15%
        month = dates.month.values
        flu_season = 0.15 * base * np.isin(month, [10, 11, 12, 1, 2, 3]).astype(float)

        # Holiday dips
        holiday_mask = (
            ((dates.month == 12) & (dates.day.isin([24, 25, 26]))) |
            ((dates.month == 11) & (dates.day_of_week == 3) & (dates.day >= 22) & (dates.day <= 28)) |
            ((dates.month == 7) & (dates.day == 4)) |
            ((dates.month == 1) & (dates.day == 1))
        ).values
        holidays = -0.20 * base * holiday_mask.astype(float)

        # COVID disruption (2020-03-01 to 2020-08-31): elective procedures dropped
        covid_mask = (dates >= "2020-03-01") & (dates <= "2020-08-31")
        covid_effect = np.where(covid_mask, -0.25 * base, 0.0)

        # Noise
        noise = rng.normal(0, 0.05 * base, n_days)

        census = (trend + annual + weekly + flu_season + holidays + covid_effect + noise).round(0)
        census = np.clip(census, 0, capacity).astype(int)
        occupancy = (census / capacity).round(4)

        dept_df = pd.DataFrame({
            "date": dates,
            "department": dept,
            "patient_count": census,
            "bed_count": capacity,
            "occupancy_rate": occupancy,
            "is_flu_season": np.isin(month, [10, 11, 12, 1, 2, 3]).astype(int),
            "is_weekend": (day_of_week >= 5).astype(int),
            "is_holiday": holiday_mask.astype(int),
            "is_covid_period": covid_mask.astype(int),
        })
        records.append(dept_df)

    df = pd.concat(records, ignore_index=True)
    logger.info(f"Generated {len(df):,} census records across {len(departments)} departments")
    return df


def generate_staffing_data(census_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Generate staffing data correlated with census.
    Uses regulatory staffing ratios with realistic staffing gaps.
    """
    rng = np.random.default_rng(seed)
    NURSE_RATIO = {"ICU": 0.50, "MedSurg": 0.25, "ED": 0.33, "Oncology": 0.28, "Orthopedics": 0.25}
    PHYSICIAN_RATIO = {"ICU": 0.10, "MedSurg": 0.05, "ED": 0.10, "Oncology": 0.08, "Orthopedics": 0.06}

    df = census_df.copy()
    dept_series = df["department"]

    # Staffing correlated with census but with real-world gaps
    df["nurses"] = (
        df.apply(lambda r: r["patient_count"] * NURSE_RATIO.get(r["department"], 0.25), axis=1)
        + rng.normal(0, 1.5, len(df))
    ).round(0).astype(int).clip(lower=1)

    df["physicians"] = (
        df.apply(lambda r: r["patient_count"] * PHYSICIAN_RATIO.get(r["department"], 0.05), axis=1)
        + rng.normal(0, 0.5, len(df))
    ).round(0).astype(int).clip(lower=1)

    df["support_staff"] = (df["nurses"] * 0.6 + rng.normal(0, 1, len(df))).round(0).astype(int).clip(lower=1)

    # Staffing adequacy flag
    df["staffing_adequate"] = (
        df["nurses"] >= df.apply(
            lambda r: max(1, int(r["patient_count"] * NURSE_RATIO.get(r["department"], 0.25))),
            axis=1,
        )
    ).astype(int)

    return df[["date", "department", "nurses", "physicians", "support_staff", "staffing_adequate"]]


def generate_appointment_data(
    start_date: str = "2019-01-01",
    end_date: str = "2024-12-31",
    specialties: list[str] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic outpatient appointment data."""
    rng = np.random.default_rng(seed)
    specialties = specialties or ["Cardiology", "Orthopedics", "Oncology", "Neurology", "Primary Care"]

    dates = pd.date_range(start_date, end_date, freq="D")
    records = []

    BASE_APPOINTMENTS = {
        "Cardiology": 45, "Orthopedics": 60, "Oncology": 35,
        "Neurology": 40, "Primary Care": 150,
    }
    NO_SHOW_RATE = {
        "Cardiology": 0.08, "Orthopedics": 0.12, "Oncology": 0.05,
        "Neurology": 0.10, "Primary Care": 0.15,
    }

    for specialty in specialties:
        base = BASE_APPOINTMENTS.get(specialty, 50)
        no_show_rate = NO_SHOW_RATE.get(specialty, 0.10)

        # No appointments on weekends for most specialties
        day_of_week = dates.dayofweek.values
        weekend = day_of_week >= 5

        scheduled = np.where(
            weekend, 0,
            (base * (1 + 0.015 * np.arange(len(dates)) / 365) + rng.normal(0, 5, len(dates))).clip(0).round(0)
        ).astype(int)

        no_shows = (scheduled * no_show_rate + rng.normal(0, 1, len(dates))).clip(0).round(0).astype(int)
        cancellations = (scheduled * 0.07 + rng.normal(0, 1, len(dates))).clip(0).round(0).astype(int)
        attended = (scheduled - no_shows - cancellations).clip(0)

        records.append(pd.DataFrame({
            "date": dates,
            "specialty": specialty,
            "scheduled": scheduled,
            "attended": attended,
            "no_show": no_shows,
            "cancelled": cancellations,
            "no_show_rate": np.where(scheduled > 0, no_shows / scheduled, 0).round(4),
        }))

    df = pd.concat(records, ignore_index=True)
    logger.info(f"Generated {len(df):,} appointment records across {len(specialties)} specialties")
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic healthcare data")
    parser.add_argument("--output-dir", default="data/synthetic", type=Path)
    parser.add_argument("--start-date", default="2019-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--format", choices=["csv", "parquet"], default="parquet")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate datasets
    census = generate_hospital_census(args.start_date, args.end_date)
    staffing = generate_staffing_data(census)
    appointments = generate_appointment_data(args.start_date, args.end_date)

    datasets = {
        "hospital_census": census,
        "staffing": staffing,
        "appointments": appointments,
    }

    for name, df in datasets.items():
        out_path = args.output_dir / f"{name}.{args.format}"
        if args.format == "parquet":
            df.to_parquet(out_path, index=False)
        else:
            df.to_csv(out_path, index=False)
        logger.info(f"Saved {name}: {len(df):,} rows → {out_path}")

    logger.info("Synthetic data generation complete!")


if __name__ == "__main__":
    main()
