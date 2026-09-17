#!/usr/bin/env python3
"""
Handles getting data into SQL and back out again. Using SQLite here so
there's nothing to install/configure - it's just a file. If this ever
needs to point at a real Postgres/MySQL instance, only get_connection()
needs to change.

    from db import load_raw_file_to_sql, run_query, read_table

    load_raw_file_to_sql("sales.csv", table_name="raw_sales")
    df = read_table("raw_sales")
    df2 = run_query("SELECT * FROM raw_sales WHERE amount > 100")
"""

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent / "pipeline.db"


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def load_raw_file_to_sql(file_path: str, table_name: str, db_path: Path = DB_PATH,
                          if_exists: str = "replace") -> dict:
    """
    Step 1 + 2: Read a raw data file (CSV/Excel/JSON/Parquet) and load it
    into a SQL table. Returns a small manifest describing what happened.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(file_path)
    elif suffix == ".json":
        df = pd.read_json(file_path)
    elif suffix == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file type for SQL ingestion: {suffix}")

    conn = get_connection(db_path)
    try:
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
    finally:
        conn.close()

    return {
        "source_file": str(file_path),
        "table_name": table_name,
        "rows_loaded": len(df),
        "columns_loaded": list(df.columns),
        "db_path": str(db_path),
    }


def list_tables(db_path: Path = DB_PATH) -> list:
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        ).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


def read_table(table_name: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Read an entire SQL table into a DataFrame — plain 'SELECT * FROM table'."""
    conn = get_connection(db_path)
    try:
        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    finally:
        conn.close()


def run_query(sql: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Run any arbitrary SQL SELECT query and return the result as a DataFrame."""
    conn = get_connection(db_path)
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


def write_df_to_sql(df: pd.DataFrame, table_name: str, db_path: Path = DB_PATH,
                     if_exists: str = "replace") -> dict:
    """Step 4: Write a (typically cleaned) DataFrame back into a SQL table."""
    conn = get_connection(db_path)
    try:
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
    finally:
        conn.close()
    return {"table_name": table_name, "rows_written": len(df), "db_path": str(db_path)}


def table_row_count(table_name: str, db_path: Path = DB_PATH) -> int:
    conn = get_connection(db_path)
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    finally:
        conn.close()
