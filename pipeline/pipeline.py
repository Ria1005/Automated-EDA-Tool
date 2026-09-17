#!/usr/bin/env python3
"""
Glues everything together: read the file, load it into SQL, clean it,
write the clean version back to SQL, then profile it into a report.

    python pipeline.py --input sales.csv --project sales_analysis

    # start from a table that's already in the db:
    python pipeline.py --sql-table raw_sales --db pipeline.db --project sales_analysis

    # or just hand it a query:
    python pipeline.py --sql-query "SELECT * FROM raw_sales WHERE amount > 0" --project sales_analysis
"""

import argparse
import json
from pathlib import Path

import db
import eda_tool


def run_pipeline(input_file: str = None, sql_table: str = None, sql_query: str = None,
                  project: str = "project", db_path: Path = None, target: str = None,
                  output_dir: str = "output"):

    db_path = Path(db_path) if db_path else db.DB_PATH
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"project": project}

    # ---- Step 1 + 2: Collect data & load into SQL ----------------------
    raw_table = f"{project}_raw"
    if input_file:
        print(f"[1/6] Collecting data from {input_file} ...")
        print(f"[2/6] Loading into SQL table '{raw_table}' ...")
        load_info = db.load_raw_file_to_sql(input_file, raw_table, db_path=db_path)
        manifest["load"] = load_info
        raw_df = db.read_table(raw_table, db_path=db_path)
        source_note = f"Loaded from file '{input_file}' into SQL table '{raw_table}'"
    elif sql_query:
        print(f"[1-2/6] Sourcing data directly via SQL query ...")
        raw_df = db.run_query(sql_query, db_path=db_path)
        source_note = f"Sourced via SQL query: {sql_query}"
    elif sql_table:
        print(f"[1-2/6] Sourcing data from existing SQL table '{sql_table}' ...")
        raw_df = db.read_table(sql_table, db_path=db_path)
        source_note = f"Sourced from existing SQL table '{sql_table}'"
    else:
        raise ValueError("Provide one of --input, --sql-table, or --sql-query")

    print(f"       -> {raw_df.shape[0]} rows x {raw_df.shape[1]} columns")

    # ---- Step 3: Clean with pandas --------------------------------------
    print("[3/6] Cleaning data with pandas ...")
    cleaned_df, cleaning_report = clean_and_report(raw_df)
    for entry in cleaning_report:
        print(f"       - {entry['step']}: {entry['detail']}")

    # ---- Step 4: Write cleaned data back to SQL --------------------------
    cleaned_table = f"{project}_cleaned"
    print(f"[4/6] Writing cleaned data back to SQL table '{cleaned_table}' ...")
    write_info = db.write_df_to_sql(cleaned_df, cleaned_table, db_path=db_path)
    manifest["cleaned_table"] = write_info

    # ---- Step 5 + 6: Profile + standardized report ------------------------
    print("[5/6] Running automated EDA / profiling ...")
    html, overview, columns = eda_tool.render_report(
        cleaned_df, dataset_name=f"{project} (cleaned)", target=target,
        cleaning_report=cleaning_report,
    )

    print("[6/6] Writing standardized HTML report ...")
    report_path = out_dir / f"{project}_report.html"
    report_path.write_text(html, encoding="utf-8")

    json_path = out_dir / f"{project}_summary.json"
    summary = {
        "manifest": manifest,
        "overview": overview,
        "columns": columns,
        "cleaning_report": cleaning_report,
        "source_note": source_note,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nDone. Report: {report_path.resolve()}")
    print(f"      Summary: {json_path.resolve()}")
    print(f"      SQL DB:  {db_path.resolve()}  (tables: {db.list_tables(db_path)})")

    return {
        "report_path": str(report_path),
        "json_path": str(json_path),
        "db_path": str(db_path),
        "raw_table": raw_table if input_file else None,
        "cleaned_table": cleaned_table,
    }


def clean_and_report(df):
    from clean import clean_dataframe
    return clean_dataframe(df)


def main():
    parser = argparse.ArgumentParser(description="Run the full collect -> SQL -> clean -> profile -> report pipeline.")
    parser.add_argument("--input", "-i", default=None, help="Raw input file (CSV/Excel/JSON/Parquet)")
    parser.add_argument("--sql-table", default=None, help="Read from an existing SQL table instead of a file")
    parser.add_argument("--sql-query", default=None, help="Read from an arbitrary SQL query instead of a file")
    parser.add_argument("--project", "-p", default="project", help="Project name (used for table names / output files)")
    parser.add_argument("--db", default=None, help="Path to SQLite DB file (default: ./pipeline.db)")
    parser.add_argument("--target", "-t", default=None, help="Optional target column for extra analysis")
    parser.add_argument("--output-dir", "-o", default="output", help="Directory for report/summary output")
    args = parser.parse_args()

    run_pipeline(
        input_file=args.input,
        sql_table=args.sql_table,
        sql_query=args.sql_query,
        project=args.project,
        db_path=args.db,
        target=args.target,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
