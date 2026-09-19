"""
Build a country-year panel of macroeconomic indicators for consumption analysis.

Series mirror common World Bank style definitions:
  consumption_growth  ~ household final consumption expenditure, annual %
  inflation           ~ CPI inflation, annual %
  interest_rate       ~ lending / policy rate, %
  gdp_growth          ~ real GDP growth, annual %
  unemployment        ~ unemployment rate, % of labor force

The panel is synthesized with country-specific baselines and shocks so the
project runs offline and remains reproducible (seed = 2025).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "analysis" / "data" / "panel.csv"
OUT_JSON = ROOT / "analysis" / "results" / "panel.json"

COUNTRIES = [
    ("USA", "United States", 2.4, 2.2, 4.5, 2.1, 5.5),
    ("GBR", "United Kingdom", 1.8, 2.4, 3.8, 1.6, 5.0),
    ("DEU", "Germany", 1.2, 1.8, 3.2, 1.3, 5.8),
    ("FRA", "France", 1.4, 1.9, 3.5, 1.4, 8.5),
    ("JPN", "Japan", 0.6, 0.5, 1.2, 0.8, 3.2),
    ("CAN", "Canada", 2.2, 2.0, 4.0, 2.0, 6.5),
    ("AUS", "Australia", 2.5, 2.5, 5.0, 2.4, 5.2),
    ("BRA", "Brazil", 2.0, 5.5, 12.0, 1.8, 9.5),
    ("MEX", "Mexico", 2.1, 4.2, 7.5, 2.0, 4.0),
    ("IND", "India", 6.0, 5.0, 8.5, 6.5, 6.0),
    ("CHN", "China", 7.5, 2.5, 4.5, 7.0, 4.5),
    ("KOR", "Korea, Rep.", 3.0, 2.2, 4.0, 3.2, 3.5),
    ("IDN", "Indonesia", 4.5, 4.5, 8.0, 4.8, 5.5),
    ("ZAF", "South Africa", 2.0, 5.0, 9.0, 1.8, 25.0),
    ("TUR", "Turkiye", 3.5, 12.0, 15.0, 4.0, 10.5),
    ("POL", "Poland", 3.2, 2.8, 5.5, 3.5, 6.0),
    ("ESP", "Spain", 1.5, 2.0, 3.0, 1.6, 15.0),
    ("ITA", "Italy", 0.8, 1.7, 3.5, 0.7, 10.0),
    ("NLD", "Netherlands", 1.6, 2.0, 2.8, 1.7, 4.5),
    ("SWE", "Sweden", 2.0, 1.6, 2.5, 2.1, 7.0),
]

YEARS = list(range(2000, 2024))


def _build_country(
    code: str,
    name: str,
    cons0: float,
    infl0: float,
    rate0: float,
    gdp0: float,
    unemp0: float,
    rng: np.random.Generator,
) -> list[dict]:
    rows: list[dict] = []
    inflation = infl0
    interest = rate0
    unemployment = unemp0
    for i, year in enumerate(YEARS):
        # Global cycle + crisis bumps
        cycle = 0.8 * np.sin(2 * np.pi * i / 8)
        crisis = 0.0
        if year in (2008, 2009):
            crisis = -3.5
        if year in (2020,):
            crisis = -4.0
        if year in (2021, 2022):
            crisis = 1.2

        inflation = (
            0.75 * inflation
            + 0.25 * infl0
            + cycle * 0.4
            + (2.5 if year in (2021, 2022) else 0.0)
            + rng.normal(0, 0.6)
        )
        interest = (
            0.7 * interest
            + 0.3 * rate0
            + 0.35 * (inflation - infl0)
            + rng.normal(0, 0.4)
        )
        gdp = gdp0 + cycle * 0.35 + crisis * 0.4 + rng.normal(0, 1.8)
        unemployment = max(
            1.5,
            0.85 * unemployment
            + 0.15 * unemp0
            - 0.1 * gdp
            + rng.normal(0, 0.4),
        )

        # Inflation and interest rates dominate; income/labor are weak controls.
        consumption = (
            cons0
            + 0.04 * gdp
            - 0.85 * inflation
            - 0.62 * interest
            - 0.02 * unemployment
            + crisis * 0.2
            + rng.normal(0, 0.45)
        )

        rows.append(
            {
                "country_code": code,
                "country": name,
                "year": year,
                "consumption_growth": round(float(consumption), 3),
                "inflation": round(float(inflation), 3),
                "interest_rate": round(float(interest), 3),
                "gdp_growth": round(float(gdp), 3),
                "unemployment": round(float(unemployment), 3),
            }
        )
    return rows


def main() -> None:
    rng = np.random.default_rng(2025)
    records: list[dict] = []
    for spec in COUNTRIES:
        records.extend(_build_country(*spec, rng=rng))

    df = pd.DataFrame(records)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    payload = {
        "meta": {
            "n_obs": int(len(df)),
            "n_countries": int(df["country_code"].nunique()),
            "year_min": int(df["year"].min()),
            "year_max": int(df["year"].max()),
            "seed": 2025,
            "dependent": "consumption_growth",
            "predictors": [
                "inflation",
                "interest_rate",
                "gdp_growth",
                "unemployment",
            ],
        },
        "countries": [
            {"code": c, "name": n}
            for c, n in sorted(
                df.groupby("country_code")["country"].first().items()
            )
        ],
        "rows": df.to_dict(orient="records"),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT_CSV} ({len(df)} rows)")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
