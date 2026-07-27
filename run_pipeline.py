"""
Full pipeline runner.

Chains all steps end-to-end:
    1. Data ingestion      (run_ingestion.py)
    2. Feature engineering (processing/build_features.py)
    3. Model training      (model/train.py)
    4. Model evaluation    (model/evaluate.py)
    5. Forecast generation (model/forecast.py)

Usage:
    py run_pipeline.py               # Full pipeline
    py run_pipeline.py --skip ingest # Skip ingestion (use existing data)
    py run_pipeline.py --only forecast # Only generate forecast (models must exist)
"""

import sys
import os
import argparse
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logger import get_logger

logger = get_logger("run_pipeline")

STEPS = ["ingest", "features", "train", "evaluate", "forecast"]


def run_pipeline(skip: list = None, only: list = None):
    skip = skip or []
    steps = only if only else STEPS

    start = datetime.now()
    logger.info("=" * 60)
    logger.info("AN PRICE FORECASTER — FULL PIPELINE")
    logger.info(f"Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    results = {}

    for step in steps:
        if step in skip:
            logger.info(f"\nSkipping: {step}")
            results[step] = "SKIPPED"
            continue

        logger.info(f"\n{'='*40}")
        logger.info(f"Step: {step.upper()}")
        logger.info(f"{'='*40}")
        t0 = time.time()

        try:
            if step == "ingest":
                import run_ingestion
                run_ingestion.run_all()

            elif step == "features":
                from processing.build_features import build
                build()

            elif step == "train":
                from model.train import train_all
                train_all()

            elif step == "evaluate":
                from model.evaluate import evaluate_all
                evaluate_all()

            elif step == "forecast":
                from model.forecast import generate_forecast
                forecast = generate_forecast()
                if not forecast.empty:
                    logger.info(f"Forecast generated: {len(forecast)} rows")

            elapsed = time.time() - t0
            results[step] = f"OK ({elapsed:.1f}s)"

        except Exception as e:
            elapsed = time.time() - t0
            results[step] = f"FAILED: {e}"
            logger.error(f"Step '{step}' failed: {e}", exc_info=True)
            logger.error("Pipeline halted. Fix the error above and rerun.")
            break

    # Summary
    total = (datetime.now() - start).total_seconds()
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    for step, status in results.items():
        icon = "OK" if status.startswith("OK") else ("--" if status == "SKIPPED" else "!!")
        logger.info(f"  [{icon}] {step:<12} {status}")
    logger.info(f"\nTotal time: {total:.1f}s")

    return all(v.startswith("OK") or v == "SKIPPED" for v in results.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", nargs="+", choices=STEPS)
    parser.add_argument("--only", nargs="+", choices=STEPS)
    args = parser.parse_args()

    success = run_pipeline(skip=args.skip, only=args.only)
    sys.exit(0 if success else 1)
