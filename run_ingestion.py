"""
Master ingestion runner.

Runs all data ingestion modules in sequence and logs a summary.
Run this script to populate / refresh the local SQLite database.

Usage:
    python run_ingestion.py                  # Run all sources
    python run_ingestion.py --source fred    # Run a single source
    python run_ingestion.py --skip ahdb      # Skip a source

Schedule this with cron (Linux/Mac) or Task Scheduler (Windows) to
keep data fresh. Weekly is sufficient given most sources are monthly.

Example cron (every Monday at 7am):
    0 7 * * 1 /path/to/venv/bin/python /path/to/run_ingestion.py
"""

import argparse
import sys
import time
from datetime import datetime

from utils.db import initialise_tables, read_table
from utils.logger import get_logger

# Import all ingestion modules
from ingestion import (
    fred_ingestion,
    yfinance_ingestion,
    gie_ingestion,
    worldbank_ingestion,
    ahdb_ingestion,
    # ember_ingestion,
)

logger = get_logger("run_ingestion")

# ---------------------------------------------------------------------------
# Source registry — controls run order and skip logic
# ---------------------------------------------------------------------------

SOURCES = {
    "fred":       fred_ingestion,
    "yfinance":   yfinance_ingestion,
    "gie":        gie_ingestion,
    "worldbank":  worldbank_ingestion,
    "ahdb":       ahdb_ingestion,       # Requires manual CSV download
    # "ember":      ember_ingestion,      # Requires manual CSV download
}

# Sources that require manual file downloads — warn if files missing
MANUAL_SOURCES = {"ahdb"} #, "ember"


def run_all(skip: list[str] = None, only: list[str] = None):
    """
    Runs all ingestion modules and logs a summary report.

    Args:
        skip: List of source names to skip.
        only: If provided, only run these sources.
    """
    skip = skip or []
    start_time = datetime.utcnow()

    logger.info("=" * 60)
    logger.info("AN Price Forecaster — Data Ingestion Pipeline")
    logger.info(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    logger.info("=" * 60)

    # Initialise DB tables first
    initialise_tables()

    results = {}

    sources_to_run = only if only else list(SOURCES.keys())

    for source_name in sources_to_run:
        if source_name in skip:
            logger.info(f"Skipping: {source_name}")
            results[source_name] = "SKIPPED"
            continue

        if source_name not in SOURCES:
            logger.warning(f"Unknown source: {source_name}")
            results[source_name] = "UNKNOWN"
            continue

        logger.info(f"\n--- Running: {source_name} ---")
        t0 = time.time()

        try:
            SOURCES[source_name].run()
            elapsed = time.time() - t0
            results[source_name] = f"OK ({elapsed:.1f}s)"

        except Exception as e:
            elapsed = time.time() - t0
            results[source_name] = f"FAILED: {e}"
            logger.error(f"{source_name} ingestion failed: {e}", exc_info=True)
            # Continue with remaining sources

    # ---------------------------------------------------------------------------
    # Summary report
    # ---------------------------------------------------------------------------

    elapsed_total = (datetime.utcnow() - start_time).total_seconds()

    logger.info("\n" + "=" * 60)
    logger.info("INGESTION SUMMARY")
    logger.info("=" * 60)

    for source, status in results.items():
        icon = "OK" if status.startswith("OK") else ("--" if status == "SKIPPED" else "!!")
        logger.info(f"  [{icon}] {source:<15} {status}")

    logger.info(f"\nTotal time: {elapsed_total:.1f}s")
    logger.info(f"Completed: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # Quick data availability summary
    _log_data_summary()

    failed = [s for s, r in results.items() if r.startswith("FAILED")]
    if failed:
        logger.warning(f"\n{len(failed)} source(s) failed: {', '.join(failed)}")
        logger.warning("Check logs/ingestion.log for details.")
        return False

    return True


def _log_data_summary():
    """Logs a quick count of rows per table to confirm data landed."""
    logger.info("\nDATA AVAILABILITY:")

    table_map = {
        "fred_raw":       "FRED (gas, FX, Brent, urea)",
        "yfinance_raw":   "yfinance (equities, futures)",
        "gie_raw":        "GIE AGSI+ (EU gas storage)",
        "worldbank_raw":  "World Bank (urea, DAP, ammonia)",
        "ahdb_raw":       "AHDB (UK AN spot price) [TARGET]",
        # "ember_raw":      "Ember (EU ETS carbon)",
    }

    for table, label in table_map.items():
        try:
            df = read_table(table)
            if df.empty:
                logger.info(f"  {label}: NO DATA")
            else:
                logger.info(
                    f"  {label}: {len(df):,} rows | "
                    f"{df['data_date'].min()} to {df['data_date'].max()}"
                )
        except Exception:
            logger.info(f"  {label}: TABLE NOT FOUND")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="AN Price Forecaster — Data Ingestion Pipeline"
    )
    parser.add_argument(
        "--source", "-s",
        nargs="+",
        choices=list(SOURCES.keys()),
        help="Run only specific source(s)",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        choices=list(SOURCES.keys()),
        help="Skip specific source(s)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    success = run_all(
        skip=args.skip,
        only=args.source,
    )
    sys.exit(0 if success else 1)
