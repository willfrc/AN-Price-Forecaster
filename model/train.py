"""
Gradient Boosting model for UK AN price forecasting.
Uses residual-based CI estimation — more stable than quantile regression on small datasets.

Run:
    py model/train.py

Outputs:
    model/saved/model_1m_point.json
    model/saved/model_12m_point.json
    model/saved/ci_info_1m.json
    model/saved/ci_info_12m.json
    model/saved/metrics.json
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from processing.prepare_model_data import prepare
from model.confidence_intervals import compute_residual_ci, save_ci_info, apply_ci
from utils.logger import get_logger

logger = get_logger(__name__)
SAVED_DIR = "model/saved"

XGB_PARAMS_1M = {
    "n_estimators": 300, "learning_rate": 0.05, "max_depth": 3,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.5, "reg_lambda": 1.5, "min_child_weight": 3,
    "random_state": 42, "early_stopping_rounds": 20,
}

XGB_PARAMS_12M = {
    "n_estimators": 200, "learning_rate": 0.03, "max_depth": 2,
    "subsample": 0.7, "colsample_bytree": 0.6,
    "reg_alpha": 1.0, "reg_lambda": 2.0, "min_child_weight": 5,
    "random_state": 42, "early_stopping_rounds": 15,
}

CI_CONFIG = {
    1:  {"lower_pct": 10.0, "upper_pct": 90.0, "label": "80% CI"},
    12: {"lower_pct":  7.5, "upper_pct": 92.5, "label": "85% CI"},
}


def train_all():
    os.makedirs(SAVED_DIR, exist_ok=True)
    all_metrics = {}

    for horizon in [1, 12]:
        logger.info("\n" + "=" * 60)
        logger.info(f"Training {horizon}-month horizon model")
        logger.info("=" * 60)

        X_train, y_train, X_test, y_test, feature_cols = prepare(horizon)
        params = XGB_PARAMS_1M if horizon == 1 else XGB_PARAMS_12M

        baseline_metrics = _train_baseline(X_train, y_train, X_test, y_test)
        model, xgb_metrics = _train_xgboost(X_train, y_train, X_test, y_test, horizon, params)

        ci_cfg = CI_CONFIG[horizon]
        ci_info = compute_residual_ci(model, X_train, y_train,
                                      lower_pct=ci_cfg["lower_pct"],
                                      upper_pct=ci_cfg["upper_pct"])
        save_ci_info(ci_info, horizon)

        X_test_filled = X_test.fillna(X_train.median())
        test_preds = model.predict(X_test_filled)
        ci_hits = sum(
            1 for actual, pred in zip(y_test.values, test_preds)
            if apply_ci(float(pred), ci_info)[0] <= actual <= apply_ci(float(pred), ci_info)[1]
        )
        ci_coverage = ci_hits / len(y_test) * 100

        _log_feature_importance(model, feature_cols, horizon)

        logger.info(f"\nBaseline MAE:  £{baseline_metrics['mae']:.1f}/tonne")
        logger.info(f"XGBoost MAE:   £{xgb_metrics['mae']:.1f}/tonne")
        logger.info(f"Improvement:   {((baseline_metrics['mae'] - xgb_metrics['mae']) / baseline_metrics['mae'] * 100):.1f}%")
        logger.info(f"CI coverage:   {ci_coverage:.0f}% ({ci_cfg['label']})")

        all_metrics[f"horizon_{horizon}m"] = {
            "baseline": baseline_metrics, "xgboost": xgb_metrics,
            "ci_coverage": round(ci_coverage, 1), "ci_label": ci_cfg["label"],
            "n_train": len(X_train), "n_test": len(X_test), "features": feature_cols,
        }

    metrics_path = os.path.join(SAVED_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"\nMetrics saved: {metrics_path}")
    return all_metrics


def _train_baseline(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train.fillna(X_train.median()))
    X_te = scaler.transform(X_test.fillna(X_train.median()))
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr, y_train)
    preds = ridge.predict(X_te)
    metrics = _compute_metrics(y_test, preds)
    logger.info(f"Baseline (Ridge) — MAE: £{metrics['mae']:.1f}, RMSE: £{metrics['rmse']:.1f}, R2: {metrics['r2']:.3f}")
    return metrics


def _train_xgboost(X_train, y_train, X_test, y_test, horizon, params):
    X_tr_f = X_train.fillna(X_train.median())
    X_te_f = X_test.fillna(X_train.median())

    val_size = max(8, int(len(X_tr_f) * 0.20))
    X_tr, y_tr = X_tr_f.iloc[:-val_size], y_train.iloc[:-val_size]
    X_val, y_val = X_tr_f.iloc[-val_size:], y_train.iloc[-val_size:]

    model_params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    model = xgb.XGBRegressor(**model_params, early_stopping_rounds=params.get("early_stopping_rounds", 20))
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    preds = model.predict(X_te_f)
    metrics = _compute_metrics(y_test, preds)
    logger.info(f"XGBoost h={horizon}m — MAE: £{metrics['mae']:.1f}, R2: {metrics['r2']:.3f}, Best iter: {model.best_iteration}")

    path = os.path.join(SAVED_DIR, f"model_{horizon}m_point.json")
    model.save_model(path)
    logger.info(f"  Saved: {path}")
    return model, metrics


def _compute_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 4), "mape": round(mape, 2)}


def _log_feature_importance(model, feature_cols, horizon):
    fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    logger.info(f"\nTop 10 features (horizon={horizon}m):")
    for feat, score in fi.head(10).items():
        logger.info(f"  {feat:<35} {score:.4f}  {'|' * int(score * 200)}")


if __name__ == "__main__":
    metrics = train_all()
    print("\n" + "=" * 55)
    print("TRAINING COMPLETE")
    print("=" * 55)
    for hk, m in metrics.items():
        print(f"\n{hk}:")
        print(f"  Baseline MAE:  £{m['baseline']['mae']:.1f}/tonne")
        print(f"  XGBoost MAE:   £{m['xgboost']['mae']:.1f}/tonne")
        print(f"  XGBoost MAPE:  {m['xgboost']['mape']:.1f}%")
        print(f"  XGBoost R2:    {m['xgboost']['r2']:.3f}")
        print(f"  CI coverage:   {m['ci_coverage']:.0f}% ({m['ci_label']})")
