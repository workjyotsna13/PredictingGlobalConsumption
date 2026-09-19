"""
Fit OLS of consumption growth on macro indicators and export diagnostics.

Diagnostics:
  - Breusch-Pagan test for heteroscedasticity
  - Variance Inflation Factors for multicollinearity
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parents[1]
PANEL_CSV = ROOT / "analysis" / "data" / "panel.csv"
OUT_JSON = ROOT / "analysis" / "results" / "baseline_results.json"

PREDICTORS = ["inflation", "interest_rate", "gdp_growth", "unemployment"]
DEPENDENT = "consumption_growth"


def fit_ols(df: pd.DataFrame) -> dict:
    y = df[DEPENDENT].astype(float)
    x = sm.add_constant(df[PREDICTORS].astype(float))
    model = sm.OLS(y, x).fit()

    coefs = []
    for name in ["const", *PREDICTORS]:
        coefs.append(
            {
                "term": name,
                "estimate": float(model.params[name]),
                "std_error": float(model.bse[name]),
                "t": float(model.tvalues[name]),
                "p_value": float(model.pvalues[name]),
                "significant_05": bool(model.pvalues[name] < 0.05),
            }
        )

    # Breusch-Pagan
    bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(model.resid, model.model.exog)

    # VIF (skip constant column index 0)
    vif_rows = []
    for i, name in enumerate(PREDICTORS, start=1):
        vif_rows.append(
            {
                "term": name,
                "vif": float(variance_inflation_factor(x.values, i)),
            }
        )

    fitted = model.fittedvalues
    resid = model.resid
    residual_sample = [
        {
            "fitted": float(fitted.iloc[i]),
            "residual": float(resid.iloc[i]),
            "country": str(df.iloc[i]["country_code"]),
            "year": int(df.iloc[i]["year"]),
        }
        for i in range(0, len(df), max(1, len(df) // 200))
    ]

    # Rank by absolute coefficient size among significant predictors
    ranked = sorted(
        [c for c in coefs if c["term"] != "const" and c["significant_05"]],
        key=lambda c: abs(c["estimate"]),
        reverse=True,
    )

    return {
        "equation": (
            "consumption_growth = b0 + b1*inflation + b2*interest_rate "
            "+ b3*gdp_growth + b4*unemployment + e"
        ),
        "n_obs": int(model.nobs),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "f_statistic": float(model.fvalue),
        "f_pvalue": float(model.f_pvalue),
        "coefficients": coefs,
        "key_predictors": [c["term"] for c in ranked[:2]],
        "breusch_pagan": {
            "lm_statistic": float(bp_lm),
            "lm_pvalue": float(bp_lm_p),
            "f_statistic": float(bp_f),
            "f_pvalue": float(bp_f_p),
            "heteroscedasticity_detected_05": bool(bp_lm_p < 0.05),
        },
        "vif": vif_rows,
        "multicollinearity_flag": any(v["vif"] > 10 for v in vif_rows),
        "residuals": residual_sample,
        "summary_text": model.summary().as_text(),
    }


def main() -> None:
    if not PANEL_CSV.exists():
        raise SystemExit(f"Missing panel at {PANEL_CSV}. Run build_panel.py first.")

    df = pd.read_csv(PANEL_CSV)
    results = fit_ols(df)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"Wrote {OUT_JSON}")
    print(f"R-squared: {results['r_squared']:.3f}")
    print(f"Key predictors: {results['key_predictors']}")
    print(
        "Breusch-Pagan p-value: "
        f"{results['breusch_pagan']['lm_pvalue']:.4f}"
    )


if __name__ == "__main__":
    main()
