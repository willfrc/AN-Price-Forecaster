"""
yfinance ingestion module.

Pulls: CF Industries, Yara, SOIL ETF, CME urea futures.
Source: Yahoo Finance via yfinance library (free, no API key needed).

Important caveats:
- These are directional/sentiment proxies, not price benchmarks.
- CME urea futures (UBU=F) via yfinance are unreliable — treat as
  supplementary signal only, not a primary input.
- Yara trades in NOK on Oslo Bors (YAR.OL) — you'll need to FX-adjust
  or use as a relative signal in the processing layer.
"""

from datetime import datetime

import pandas as pd
import yfinance as yf

from utils.config import YFINANCE_TICKERS, START_DATE, END_DATE
from utils.db import write_dataframe
from utils.logger import get_logger

logger = get_logger(__name__)


def run():
    """Pull all configured yfinance tickers and write to yfinance_raw table."""
    retrieved_at = datetime.utcnow().isoformat()
    all_rows = []

    for ticker_name, ticker_symbol in YFINANCE_TICKERS.items():
        try:
            logger.info(f"Pulling yfinance ticker: {ticker_symbol} ({ticker_name})")
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(start=START_DATE, end=END_DATE)

            if hist.empty:
                logger.warning(
                    f"No data returned for {ticker_symbol}. "
                    "Futures tickers (e.g. UBU=F) are often unavailable via yfinance."
                )
                continue

            # yfinance returns DatetimeIndex — reset and normalise
            hist = hist.reset_index()

            # yfinance column names vary slightly by version — normalise
            hist.columns = [c.lower().replace(" ", "_") for c in hist.columns]

            # Handle timezone-aware dates — strip tz info for consistent storage
            if pd.api.types.is_datetime64tz_dtype(hist["date"]):
                hist["date"] = hist["date"].dt.tz_localize(None)

            df = pd.DataFrame({
                "ticker":       ticker_symbol,
                "ticker_name":  ticker_name,
                "data_date":    hist["date"].dt.strftime("%Y-%m-%d"),
                "open":         hist.get("open"),
                "high":         hist.get("high"),
                "low":          hist.get("low"),
                "close":        hist.get("close"),
                "volume":       hist.get("volume"),
                "retrieved_at": retrieved_at,
            })

            df = df.dropna(subset=["close"])
            all_rows.append(df)
            logger.info(f"  -> {len(df)} trading days from {df['data_date'].min()} to {df['data_date'].max()}")

        except Exception as e:
            logger.error(f"Failed to pull {ticker_symbol}: {e}")

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        write_dataframe(combined, "yfinance_raw")
        logger.info(f"yfinance ingestion complete — {len(combined)} total rows written.")
    else:
        logger.warning("yfinance ingestion completed with no data written.")


if __name__ == "__main__":
    from utils.db import initialise_tables
    initialise_tables()
    run()
