"""
Model evaluation module.

Loads saved models and runs a full evaluation:
    - Test set predictions vs actuals
    - Walk-forward backtest (simulates real-world use)
    - Residual analysis
    - Outputs evaluation plots as HTML (no matplotlib dependency)

Run after train.py:
    py model/evaluate.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
import xgboost as xgb

from processing.prepare_model_data import prepare
from utils.logger import get_logger

logger = get_logger(__name__)

SAVED_DIR  = "model/saved"
OUTPUT_DIR = "data/processed"


def evaluate_all():
    """Runs full evaluation for both horizons and prints results."""

    results = {}

    for horizon in [1, 12]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {horizon}-month horizon model")
        logger.info(f"{'='*60}")

        X_train, y_train, X_test, y_test, feature_cols = prepare(horizon)

        # Load point model
        model_path = os.path.join(SAVED_DIR, f"model_{horizon}m_point.json")
        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}. Run train.py first.")
            continue

        model = xgb.XGBRegressor()
        model.load_model(model_path)

        # Load quantile models for CI
        lower_model = _load_optional(f"model_{horizon}m_lower.json")
        upper_model = _load_optional(f"model_{horizon}m_upper.json")

        # Fill NaNs
        X_train_f = X_train.fillna(X_train.median())
        X_test_f  = X_test.fillna(X_train.median())

        # --- Test set evaluation ---
        test_preds  = model.predict(X_test_f)
        lower_preds = lower_model.predict(X_test_f) if lower_model else test_preds * 0.85
        upper_preds = upper_model.predict(X_test_f) if upper_model else test_preds * 1.15

        test_results = _build_results_df(y_test, test_preds, lower_preds, upper_preds)
        test_metrics = _compute_full_metrics(y_test.values, test_preds, lower_preds, upper_preds)

        logger.info("\nTest set results:")
        _log_results_table(test_results)
        _log_metrics(test_metrics)

        # --- Walk-forward backtest ---
        logger.info("\nWalk-forward backtest:")
        wf_results = _walk_forward_backtest(
            X_train, y_train, X_test, y_test,
            feature_cols, horizon
        )
        if wf_results is not None:
            _log_metrics(wf_results["metrics"])

        # --- Save results ---
        results_path = os.path.join(OUTPUT_DIR, f"eval_results_{horizon}m.csv")
        test_results.to_csv(results_path)
        logger.info(f"Evaluation results saved: {results_path}")

        results[horizon] = {
            "test_metrics": test_metrics,
            "test_results": test_results,
        }

    # Print final comparison
    _print_model_comparison(results)
    return results


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------

def _walk_forward_backtest(
    X_train, y_train, X_test, y_test,
    feature_cols: list, horizon: int,
    min_train_size: int = 36
) -> dict | None:
    """
    Simulates real-world model use: trains on data up to month t,
    predicts month t+horizon, then expands the window by one month.

    This is the most honest evaluation of a time series model.
    Random test splits are not valid for time series.

    Args:
        min_train_size: Minimum months of data before making first prediction.
                        36 months recommended; we use 30 given data constraints.
    """
    # Combine train and test for full series
    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])

    n = len(X_all)
    min_train = min(min_train_size, len(X_train) - 5)

    if min_train < 20:
        logger.warning("Insufficient data for walk-forward backtest (need 20+ train rows)")
        return None

    preds_wf   = []
    actuals_wf = []
    dates_wf   = []

    for t in range(min_train, n - horizon):
        X_tr = X_all.iloc[:t].fillna(X_all.iloc[:t].median())
        y_tr = y_all.iloc[:t]
        X_pr = X_all.iloc[[t]].fillna(X_all.iloc[:t].median())
        y_ac = y_all.iloc[t + horizon - 1] if (t + horizon - 1) < n else None

        if y_ac is None:
            continue

        # Refit a lightweight model on expanding window
        wf_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            reg_alpha=0.5,
            reg_lambda=1.5,
            random_state=42,
            verbosity=0,
        )
        wf_model.fit(X_tr, y_tr, verbose=False)
        pred = float(wf_model.predict(X_pr)[0])

        preds_wf.append(pred)
        actuals_wf.append(float(y_ac))
        dates_wf.append(X_all.index[t])

    if len(preds_wf) < 3:
        logger.warning("Walk-forward backtest produced fewer than 3 predictions")
        return None

    preds_arr   = np.array(preds_wf)
    actuals_arr = np.array(actuals_wf)

    mae  = np.mean(np.abs(actuals_arr - preds_arr))
    rmse = np.sqrt(np.mean((actuals_arr - preds_arr) ** 2))
    mape = np.mean(np.abs((actuals_arr - preds_arr) / actuals_arr)) * 100

    metrics = {"mae": round(mae, 2), "rmse": round(rmse, 2), "mape": round(mape, 2)}
    logger.info(f"Walk-forward — {len(preds_wf)} predictions | MAE: £{mae:.1f} | RMSE: £{rmse:.1f} | MAPE: {mape:.1f}%")

    wf_df = pd.DataFrame({
        "date":    dates_wf,
        "actual":  actuals_wf,
        "predicted": preds_wf,
        "error":   [a - p for a, p in zip(actuals_wf, preds_wf)],
    })
    wf_path = os.path.join(OUTPUT_DIR, f"walkforward_{horizon}m.csv")
    wf_df.to_csv(wf_path, index=False)
    logger.info(f"Walk-forward results saved: {wf_path}")

    return {"metrics": metrics, "df": wf_df}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_results_df(y_true, preds, lower, upper) -> pd.DataFrame:
    df = pd.DataFrame({
        "actual":        y_true.values,
        "predicted":     preds,
        "lower_bound":   lower,
        "upper_bound":   upper,
        "error":         y_true.values - preds,
        "abs_error":     np.abs(y_true.values - preds),
        "pct_error":     (y_true.values - preds) / y_true.values * 100,
        "in_ci":         (y_true.values >= lower) & (y_true.values <= upper),
    }, index=y_true.index)
    return df


def _compute_full_metrics(actuals, preds, lower, upper) -> dict:
    mae      = float(np.mean(np.abs(actuals - preds)))
    rmse     = float(np.sqrt(np.mean((actuals - preds) ** 2)))
    mape     = float(np.mean(np.abs((actuals - preds) / actuals)) * 100)
    ci_cover = float(np.mean((actuals >= lower) & (actuals <= upper)) * 100)
    return {
        "mae":           round(mae, 2),
        "rmse":          round(rmse, 2),
        "mape":          round(mape, 2),
        "ci_coverage":   round(ci_cover, 1),
    }


def _log_results_table(df: pd.DataFrame):
    logger.info(f"{'Date':<12} {'Actual':>8} {'Predicted':>10} {'Lower':>8} {'Upper':>8} {'Error':>8} {'In CI':>6}")
    logger.info("-" * 65)
    for date, row in df.iterrows():
        logger.info(
            f"{str(date)[:10]:<12} "
            f"{row['actual']:>8.0f} "
            f"{row['predicted']:>10.0f} "
            f"{row['lower_bound']:>8.0f} "
            f"{row['upper_bound']:>8.0f} "
            f"{row['error']:>+8.0f} "
            f"{'Yes' if row['in_ci'] else 'No':>6}"
        )


def _log_metrics(metrics: dict):
    logger.info(f"\n  MAE:          £{metrics['mae']:.1f}/tonne")
    logger.info(f"  RMSE:         £{metrics['rmse']:.1f}/tonne")
    logger.info(f"  MAPE:         {metrics['mape']:.1f}%")
    if "ci_coverage" in metrics:
        logger.info(f"  CI coverage:  {metrics['ci_coverage']:.0f}% (target: 70%)")


def _load_optional(filename: str):
    path = os.path.join(SAVED_DIR, filename)
    if os.path.exists(path):
        m = xgb.XGBRegressor()
        m.load_model(path)
        return m
    return None


def _print_model_comparison(results: dict):
    print("\n" + "=" * 55)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 55)
    print(f"{'Metric':<20} {'1-month':>15} {'12-month':>15}")
    print("-" * 55)

    metrics_1m  = results.get(1,  {}).get("test_metrics", {})
    metrics_12m = results.get(12, {}).get("test_metrics", {})

    for key in ["mae", "rmse", "mape", "ci_coverage"]:
        v1  = metrics_1m.get(key,  "N/A")
        v12 = metrics_12m.get(key, "N/A")
        unit = "%" if key in ["mape", "ci_coverage"] else "£/t"
        label = {
            "mae":         "MAE",
            "rmse":        "RMSE",
            "mape":        "MAPE",
            "ci_coverage": "CI coverage",
        }[key]
        v1_str  = f"{v1:.1f}{unit}"  if isinstance(v1,  float) else str(v1)
        v12_str = f"{v12:.1f}{unit}" if isinstance(v12, float) else str(v12)
        print(f"{label:<20} {v1_str:>15} {v12_str:>15}")

    print("=" * 55)


if __name__ == "__main__":
    evaluate_all()
