"""
Forecast generation module.
Loads trained models and generates the 12-month forecast output.
Run after train.py: py model/forecast.py
Output: data/processed/forecast_output.csv, forecast_summary.txt
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.getLogger("processing.prepare_model_data").setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime, date

from processing.prepare_model_data import prepare
from model.confidence_intervals import load_ci_info, apply_ci
from utils.logger import get_logger

logger = get_logger(__name__)

SAVED_DIR  = "model/saved"
OUTPUT_DIR = "data/processed"


def generate_forecast() -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("Generating forecast")
    logger.info("=" * 60)

    models   = _load_models()
    ci_infos = _load_ci_infos()

    today    = pd.Timestamp(date.today()).to_period("M").to_timestamp()
    features = _load_features_once()

    forecast_rows = []

    for horizon_m in range(1, 13):
        forecast_date = today + pd.DateOffset(months=horizon_m)
        frequency     = "weekly" if horizon_m <= 3 else "monthly"
        model_horizon = 1 if horizon_m <= 3 else 12

        if f"point_{model_horizon}m" not in models:
            logger.warning(f"Model not found for horizon {model_horizon}m")
            continue

        model   = models[f"point_{model_horizon}m"]
        ci_info = ci_infos.get(model_horizon)

        if frequency == "weekly":
            week_dates = pd.date_range(
                start=forecast_date,
                end=forecast_date + pd.DateOffset(months=1) - pd.DateOffset(days=1),
                freq="W-MON"
            )
            for wdate in week_dates:
                row = _make_row(wdate, horizon_m, "weekly", model, ci_info, features, model_horizon)
                if row:
                    forecast_rows.append(row)
        else:
            row = _make_row(forecast_date, horizon_m, "monthly", model, ci_info, features, model_horizon)
            if row:
                forecast_rows.append(row)

    if not forecast_rows:
        logger.error("No forecast rows generated.")
        return pd.DataFrame()

    forecast_df = pd.DataFrame(forecast_rows).sort_values("date").reset_index(drop=True)
    _save_outputs(forecast_df)
    return forecast_df


def _make_row(forecast_date, horizon_m, frequency, model, ci_info, features, model_horizon):
    try:
        X = _build_feature_row(forecast_date, features, model_horizon)
    except Exception as e:
        logger.warning(f"Feature row failed for {forecast_date.date()}: {e}")
        return None

    point = float(np.clip(model.predict(X)[0], 150, 1200))

    if ci_info:
        lower, upper = apply_ci(point, ci_info)
    else:
        lower, upper = point * 0.85, point * 1.15

    return {
        "date":           forecast_date.strftime("%Y-%m-%d"),
        "horizon_months": horizon_m,
        "frequency":      frequency,
        "point_forecast": round(point, 1),
        "lower_bound":    round(lower, 1),
        "upper_bound":    round(upper, 1),
        "ci_width":       round(upper - lower, 1),
        "model_used":     f"{model_horizon}m",
    }


def _build_feature_row(forecast_date, df, model_horizon):
    _, _, _, _, feature_cols = prepare(model_horizon)
    latest = df[feature_cols].dropna(how="all").tail(1).copy()
    if latest.empty:
        raise ValueError("No complete feature row available")
    latest = latest.fillna(df[feature_cols].median())

    # Update seasonal features for the actual forecast month
    latest["month_sin"] = np.sin(2 * np.pi * forecast_date.month / 12)
    latest["month_cos"] = np.cos(2 * np.pi * forecast_date.month / 12)
    if "month" in latest.columns:
        latest["month"] = forecast_date.month
    if "year" in latest.columns:
        latest["year"] = forecast_date.year

    return latest


def _load_features_once():
    path = os.path.join(OUTPUT_DIR, "model_features.csv")
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _load_models():
    models = {}
    for horizon in [1, 12]:
        path = os.path.join(SAVED_DIR, f"model_{horizon}m_point.json")
        if os.path.exists(path):
            m = xgb.XGBRegressor()
            m.load_model(path)
            models[f"point_{horizon}m"] = m
            logger.info(f"Loaded model: model_{horizon}m_point.json")
        else:
            logger.warning(f"Model not found: {path}")
    if not models:
        raise FileNotFoundError(f"No model files in {SAVED_DIR}. Run: py model/train.py")
    return models


def _load_ci_infos():
    ci_infos = {}
    for horizon in [1, 12]:
        ci = load_ci_info(horizon)
        if ci:
            ci_infos[horizon] = ci
            logger.info(f"Loaded CI info: ci_info_{horizon}m.json (width: {ci['ci_width_median']:.0f})")
        else:
            logger.warning(f"CI info not found for {horizon}m")
    return ci_infos


def _save_outputs(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "forecast_output.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Forecast saved: {csv_path}")

    summary = _build_summary(df)
    txt_path = os.path.join(OUTPUT_DIR, "forecast_summary.txt")
    with open(txt_path, "w") as f:
        f.write(summary)
    logger.info(f"Summary saved: {txt_path}")
    print(summary)


def _build_summary(df):
    lines = [
        "=" * 65,
        "UK AMMONIUM NITRATE PRICE FORECAST",
        f"Generated: {datetime.now().strftime('%d %B %Y')}",
        "1-month model: 80% CI (10th-90th percentile of training residuals)",
        "12-month model: 85% CI (7.5th-92.5th percentile)",
        "=" * 65,
        "",
        "SHORT-TERM FORECAST (weekly, months 1-3)",
        f"{'Date':<14} {'Point (£/t)':>12} {'Low (£/t)':>10} {'High (£/t)':>10} {'Range':>8}",
        "-" * 58,
    ]

    for _, row in df[df["frequency"] == "weekly"].iterrows():
        lines.append(
            f"{row['date']:<14} {row['point_forecast']:>12.0f} "
            f"{row['lower_bound']:>10.0f} {row['upper_bound']:>10.0f} "
            f"{row['ci_width']:>7.0f}"
        )

    lines += [
        "",
        "LONG-TERM FORECAST (monthly, months 4-12)",
        f"{'Date':<14} {'Point (£/t)':>12} {'Low (£/t)':>10} {'High (£/t)':>10} {'Range':>8}",
        "-" * 58,
    ]

    for _, row in df[df["frequency"] == "monthly"].iterrows():
        lines.append(
            f"{row['date']:<14} {row['point_forecast']:>12.0f} "
            f"{row['lower_bound']:>10.0f} {row['upper_bound']:>10.0f} "
            f"{row['ci_width']:>7.0f}"
        )

    lines += [
        "",
        "=" * 65,
        "NOTES",
        "-" * 65,
        "- Weekly forecasts (months 1-3): 1-month model, seasonal adjustment applied.",
        "- Monthly forecasts (months 4-12): 12-month model, directional only.",
        "- CI from empirical training residuals. Assumes market conditions persist.",
        "- Add gas forward curve in v2 to generate a true price path.",
        "- Model trained on AHDB UK AN bulk prices (34.5%N).",
        "=" * 65,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    forecast = generate_forecast()
    if not forecast.empty:
        logger.info(f"\nForecast complete: {len(forecast)} rows")
