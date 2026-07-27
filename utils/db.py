"""
SQLite database manager.

Design principles:
- Raw data is written once and never mutated (append-only per source table).
- Every row is stamped with `retrieved_at` (when we pulled it) and `data_date`
  (the date the data point refers to). This distinction matters when
  backtesting — you must only use data that was available at the time.
- Duplicate rows are handled via INSERT OR IGNORE on a unique constraint
  (source + data_date + series_id), so re-running ingestion is safe.
"""

import os
import sqlite3
from contextlib import contextmanager
from typing import Optional

import pandas as pd

from utils.config import DB_PATH
from utils.logger import get_logger

logger = get_logger(__name__)


def get_connection() -> sqlite3.Connection:
    """Returns a sqlite3 connection. Creates the DB file and directory if needed."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent read performance
    return conn


@contextmanager
def db_connection():
    """Context manager for safe connection handling."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error, rolling back: {e}")
        raise
    finally:
        conn.close()


def initialise_tables():
    """
    Creates all raw data tables if they don't exist.
    Schema is deliberately wide/flexible — all values stored as TEXT,
    with numeric casting happening in the processing layer.
    """
    create_statements = {

        # FRED: one row per series per date
        "fred_raw": """
            CREATE TABLE IF NOT EXISTS fred_raw (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                series_id   TEXT    NOT NULL,
                series_name TEXT    NOT NULL,
                data_date   TEXT    NOT NULL,
                value       REAL,
                retrieved_at TEXT   NOT NULL,
                UNIQUE(series_id, data_date)
            )
        """,

        # yfinance: OHLCV per ticker per date
        "yfinance_raw": """
            CREATE TABLE IF NOT EXISTS yfinance_raw (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker      TEXT    NOT NULL,
                ticker_name TEXT    NOT NULL,
                data_date   TEXT    NOT NULL,
                open        REAL,
                high        REAL,
                low         REAL,
                close       REAL,
                volume      REAL,
                retrieved_at TEXT   NOT NULL,
                UNIQUE(ticker, data_date)
            )
        """,

        # GIE AGSI+: EU gas storage aggregate
        "gie_raw": """
            CREATE TABLE IF NOT EXISTS gie_raw (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                data_date       TEXT    NOT NULL,
                country_code    TEXT    NOT NULL,
                gas_in_storage  REAL,   -- TWh
                full_pct        REAL,   -- % full
                injection       REAL,   -- TWh/day
                withdrawal      REAL,   -- TWh/day
                retrieved_at    TEXT    NOT NULL,
                UNIQUE(data_date, country_code)
            )
        """,

        # World Bank Pink Sheet: commodity prices
        "worldbank_raw": """
            CREATE TABLE IF NOT EXISTS worldbank_raw (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                commodity    TEXT    NOT NULL,
                data_date    TEXT    NOT NULL,
                value        REAL,
                unit         TEXT,
                retrieved_at TEXT    NOT NULL,
                UNIQUE(commodity, data_date)
            )
        """,

        # AHDB: UK AN spot price (manually loaded from CSV)
        "ahdb_raw": """
            CREATE TABLE IF NOT EXISTS ahdb_raw (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                data_date    TEXT    NOT NULL UNIQUE,
                price_gbp_t  REAL,
                product      TEXT    DEFAULT 'AN_34.5N_bulk',
                retrieved_at TEXT    NOT NULL
            )
        """,

        # Ember ETS: EU carbon price (manually loaded from CSV)
        "ember_raw": """
            CREATE TABLE IF NOT EXISTS ember_raw (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                data_date    TEXT    NOT NULL UNIQUE,
                price_eur_t  REAL,
                retrieved_at TEXT    NOT NULL
            )
        """,
    }

    with db_connection() as conn:
        for table_name, stmt in create_statements.items():
            conn.execute(stmt)
            logger.debug(f"Table ensured: {table_name}")

    logger.info("Database initialised — all tables ready.")


def write_dataframe(df: pd.DataFrame, table: str, if_exists: str = "append"):
    """
    Writes a DataFrame to the specified table.
    Uses INSERT OR IGNORE semantics to prevent duplicates on re-runs.

    Args:
        df:         DataFrame to write. Must contain a `retrieved_at` column.
        table:      Target table name.
        if_exists:  pandas to_sql behaviour — default 'append'.
                    Pass 'replace' only when explicitly refreshing a table.
    """
    if df.empty:
        logger.warning(f"Empty DataFrame passed to write_dataframe for table: {table}")
        return

    with db_connection() as conn:
        # to_sql doesn't support INSERT OR IGNORE natively, so we use a
        # temp table approach for clean duplicate handling
        temp_table = f"_temp_{table}"
        df.to_sql(temp_table, conn, if_exists="replace", index=False)

        # Get column list from the actual target table
        cursor = conn.execute(f"PRAGMA table_info({table})")
        target_cols = [row[1] for row in cursor.fetchall() if row[1] != "id"]

        # Only insert columns that exist in the target table
        df_cols = [c for c in df.columns if c in target_cols]
        cols_str = ", ".join(df_cols)

        conn.execute(f"""
            INSERT OR IGNORE INTO {table} ({cols_str})
            SELECT {cols_str} FROM {temp_table}
        """)
        conn.execute(f"DROP TABLE IF EXISTS {temp_table}")

    logger.debug(f"Wrote {len(df)} rows to {table} (duplicates silently skipped)")


def read_table(table: str, start_date: Optional[str] = None) -> pd.DataFrame:
    """
    Reads a raw table into a DataFrame.

    Args:
        table:      Table name to read.
        start_date: Optional ISO date string to filter from (inclusive).
    """
    with db_connection() as conn:
        query = f"SELECT * FROM {table}"
        if start_date:
            query += f" WHERE data_date >= '{start_date}'"
        query += " ORDER BY data_date"
        df = pd.read_sql_query(query, conn)

    logger.debug(f"Read {len(df)} rows from {table}")
    return df
