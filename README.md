# Predicting Global Consumption

Why inflation matters most: OLS impact evaluation of macroeconomic indicators on final consumer expenditure (03/2025 to 04/2025).

## What this project does

- Builds a country-year panel of consumption growth, inflation, interest rates, GDP growth, and unemployment
- Fits an **OLS** model of final consumer expenditure growth
- Shows **inflation** and **interest rates** as the strongest significant predictors
- Runs **Breusch-Pagan** (heteroscedasticity) and **VIF** (multicollinearity) diagnostics

## Setup

```bash
python3 -m pip install -r analysis/requirements.txt
```

## Run

```bash
python3 analysis/build_panel.py
python3 analysis/run_ols.py
```

Outputs:

- `analysis/data/panel.csv`
- `analysis/results/panel.json`
- `analysis/results/baseline_results.json`

## Model

```
consumption_growth = b0 + b1*inflation + b2*interest_rate
                   + b3*gdp_growth + b4*unemployment + e
```

The interactive website for this analysis is deployed separately on Vercel.
