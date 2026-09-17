#!/usr/bin/env python3
"""
Takes a CSV/Excel/Parquet/JSON file and spits out a single HTML report
with summary stats, missing value counts, distributions and a
correlation heatmap. Basically automates the first hour of poking
around a new dataset.

    python eda_tool.py --input data.csv --output report.html
    python eda_tool.py --input data.csv --output report.html --target price
"""

import argparse
import base64
import io
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend, safe for CLI / servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

MAX_CATEGORIES_TO_PLOT = 15
MAX_NUMERIC_COLS_FOR_CORR = 30


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_dataset(path: str) -> pd.DataFrame:
    """Load a CSV or Excel file into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif suffix == ".tsv":
        df = pd.read_csv(path, sep="\t")
    elif suffix == ".json":
        df = pd.read_json(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return df


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def fig_to_base64(fig) -> str:
    """Convert a matplotlib figure to a base64 PNG string for HTML embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def safe_col_name(name) -> str:
    return str(name).replace("<", "&lt;").replace(">", "&gt;")


# --------------------------------------------------------------------------- #
# Profiling sections
# --------------------------------------------------------------------------- #
def profile_overview(df: pd.DataFrame) -> dict:
    mem = df.memory_usage(deep=True).sum()
    dup_rows = int(df.duplicated().sum())
    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "memory": human_bytes(mem),
        "duplicate_rows": dup_rows,
        "duplicate_pct": round(100 * dup_rows / len(df), 2) if len(df) else 0,
        "n_numeric": df.select_dtypes(include=np.number).shape[1],
        "n_categorical": df.select_dtypes(include=["object", "category"]).shape[1],
        "n_datetime": df.select_dtypes(include=["datetime", "datetimetz"]).shape[1],
        "n_bool": df.select_dtypes(include="bool").shape[1],
    }


def profile_columns(df: pd.DataFrame) -> list:
    """Per-column summary: dtype, missing, unique, and type-specific stats."""
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        missing = int(s.isna().sum())
        entry = {
            "column": safe_col_name(col),
            "dtype": str(s.dtype),
            "missing": missing,
            "missing_pct": round(100 * missing / n, 2) if n else 0,
            "unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            desc = s.describe()
            entry.update({
                "mean": round(desc.get("mean", np.nan), 4),
                "std": round(desc.get("std", np.nan), 4),
                "min": round(desc.get("min", np.nan), 4),
                "25%": round(desc.get("25%", np.nan), 4),
                "median": round(desc.get("50%", np.nan), 4),
                "75%": round(desc.get("75%", np.nan), 4),
                "max": round(desc.get("max", np.nan), 4),
                "skew": round(float(s.skew()), 4) if s.count() > 2 else None,
            })
        elif pd.api.types.is_datetime64_any_dtype(s):
            entry.update({
                "min": str(s.min()),
                "max": str(s.max()),
            })
        else:
            top = s.value_counts(dropna=True).head(1)
            entry.update({
                "top_value": safe_col_name(top.index[0]) if len(top) else None,
                "top_freq": int(top.iloc[0]) if len(top) else None,
            })
        rows.append(entry)
    return rows


def plot_missing_values(df: pd.DataFrame):
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if miss.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, max(2, 0.35 * len(miss))))
    sns.barplot(x=miss.values, y=miss.index.astype(str), ax=ax, color="#4C72B0")
    ax.set_xlabel("Missing count")
    ax.set_title("Missing values by column")
    return fig_to_base64(fig)


def plot_numeric_distributions(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    plots = {}
    for col in numeric_cols:
        data = df[col].dropna()
        if data.empty or data.nunique() <= 1:
            continue
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.histplot(data, kde=True, ax=ax, color="#4C72B0")
        ax.set_title(f"Distribution: {col}")
        plots[safe_col_name(col)] = fig_to_base64(fig)
    return plots


def plot_categorical_distributions(df: pd.DataFrame) -> dict:
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    plots = {}
    for col in cat_cols:
        vc = df[col].value_counts(dropna=True).head(MAX_CATEGORIES_TO_PLOT)
        if vc.empty:
            continue
        fig, ax = plt.subplots(figsize=(5, max(2.5, 0.35 * len(vc))))
        sns.barplot(x=vc.values, y=vc.index.astype(str), ax=ax, color="#55A868")
        ax.set_title(f"Top categories: {col}")
        ax.set_xlabel("Count")
        plots[safe_col_name(col)] = fig_to_base64(fig)
    return plots


def plot_correlation_heatmap(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2 or numeric_df.shape[1] > MAX_NUMERIC_COLS_FOR_CORR:
        return None
    corr = numeric_df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(max(5, 0.6 * len(corr)), max(4, 0.6 * len(corr))))
    sns.heatmap(corr, annot=len(corr) <= 12, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, square=True, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation heatmap (numeric columns)")
    return fig_to_base64(fig)


def top_correlated_pairs(df: pd.DataFrame, n=10) -> list:
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        return []
    corr = numeric_df.corr(numeric_only=True).abs()
    pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .sort_values(ascending=False)
    )
    out = []
    for (a, b), v in pairs.head(n).items():
        out.append({"col_a": safe_col_name(a), "col_b": safe_col_name(b), "corr": round(float(v), 4)})
    return out


def target_analysis(df: pd.DataFrame, target: str) -> dict:
    """Optional: relationship of each column to a target variable."""
    if target not in df.columns:
        return {}
    result = {"target": target, "plot": None}
    fig, ax = plt.subplots(figsize=(5, 3))
    if pd.api.types.is_numeric_dtype(df[target]):
        sns.histplot(df[target].dropna(), kde=True, ax=ax, color="#C44E52")
    else:
        df[target].value_counts().head(MAX_CATEGORIES_TO_PLOT).plot(kind="bar", ax=ax, color="#C44E52")
    ax.set_title(f"Target distribution: {target}")
    result["plot"] = fig_to_base64(fig)
    return result


# --------------------------------------------------------------------------- #
# Report rendering
# --------------------------------------------------------------------------- #
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>EDA Report — {dataset_name}</title>
<style>
  :root {{
    --bg: #0f1216; --panel: #171b21; --text: #e7eaee; --muted: #9aa4b2;
    --accent: #4C72B0; --border: #262c35; --good:#55A868; --warn:#C44E52;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); margin: 0; padding: 0 0 60px 0;
  }}
  header {{
    padding: 32px 40px; border-bottom: 1px solid var(--border);
    background: linear-gradient(135deg, #171b21, #10131a);
  }}
  header h1 {{ margin: 0 0 6px 0; font-size: 26px; }}
  header p {{ margin: 0; color: var(--muted); font-size: 14px; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 0 40px; }}
  .section {{ margin-top: 40px; }}
  .section h2 {{
    font-size: 18px; border-left: 4px solid var(--accent); padding-left: 12px;
    margin-bottom: 16px;
  }}
  .cards {{ display: flex; gap: 16px; flex-wrap: wrap; }}
  .card {{
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    padding: 16px 20px; min-width: 150px; flex: 1;
  }}
  .card .num {{ font-size: 24px; font-weight: 600; }}
  .card .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  th, td {{ border: 1px solid var(--border); padding: 6px 10px; text-align: left; }}
  th {{ background: #1d222a; position: sticky; top: 0; }}
  tr:nth-child(even) {{ background: #14171d; }}
  .table-wrap {{ overflow-x: auto; border: 1px solid var(--border); border-radius: 8px; }}
  .grid-plots {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
  .plot-box {{ background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 12px; }}
  .plot-box img {{ width: 100%; height: auto; border-radius: 6px; }}
  .plot-box h4 {{ margin: 0 0 8px 0; font-size: 13px; color: var(--muted); font-weight: 500; }}
  .badge {{ display:inline-block; padding: 2px 8px; border-radius: 20px; font-size: 11px; }}
  .badge.good {{ background: rgba(85,168,104,.15); color: var(--good); }}
  .badge.warn {{ background: rgba(196,78,82,.15); color: var(--warn); }}
  footer {{ text-align: center; color: var(--muted); font-size: 12px; margin-top: 60px; }}
  .empty {{ color: var(--muted); font-style: italic; }}
</style>
</head>
<body>

<header>
  <h1>Exploratory Data Analysis Report</h1>
  <p>Dataset: <strong>{dataset_name}</strong> &nbsp;|&nbsp; Generated {timestamp}</p>
</header>

<div class="container">

  <div class="section">
    <h2>Overview</h2>
    <div class="cards">
      <div class="card"><div class="num">{n_rows}</div><div class="label">Rows</div></div>
      <div class="card"><div class="num">{n_cols}</div><div class="label">Columns</div></div>
      <div class="card"><div class="num">{memory}</div><div class="label">Memory</div></div>
      <div class="card"><div class="num">{duplicate_rows} ({duplicate_pct}%)</div><div class="label">Duplicate rows</div></div>
      <div class="card"><div class="num">{n_numeric}</div><div class="label">Numeric cols</div></div>
      <div class="card"><div class="num">{n_categorical}</div><div class="label">Categorical cols</div></div>
      <div class="card"><div class="num">{n_datetime}</div><div class="label">Datetime cols</div></div>
    </div>
  </div>

  {cleaning_log_section}

  <div class="section">
    <h2>Column Summary</h2>
    <div class="table-wrap">
      {column_table}
    </div>
  </div>

  <div class="section">
    <h2>Missing Values</h2>
    {missing_plot_html}
  </div>

  <div class="section">
    <h2>Numeric Distributions</h2>
    <div class="grid-plots">
      {numeric_plots_html}
    </div>
  </div>

  <div class="section">
    <h2>Categorical Distributions</h2>
    <div class="grid-plots">
      {categorical_plots_html}
    </div>
  </div>

  <div class="section">
    <h2>Correlation Analysis</h2>
    {correlation_html}
    {top_corr_table}
  </div>

  {target_section}

</div>

<footer>Generated automatically by eda_tool.py &mdash; reusable dataset profiling tool</footer>

</body>
</html>
"""


def dict_list_to_html_table(rows: list) -> str:
    if not rows:
        return "<p class='empty'>No data.</p>"
    cols = list(rows[0].keys())
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = ""
    for r in rows:
        cells = "".join(f"<td>{'' if r.get(c) is None else r.get(c)}</td>" for c in cols)
        body += f"<tr>{cells}</tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_report(df: pd.DataFrame, dataset_name: str, target: str = None,
                   cleaning_report: list = None, source_note: str = None) -> str:
    overview = profile_overview(df)
    columns = profile_columns(df)

    missing_b64 = plot_missing_values(df)
    missing_html = (
        f'<div class="plot-box"><img src="data:image/png;base64,{missing_b64}"></div>'
        if missing_b64 else "<p class='empty'>No missing values detected.</p>"
    )

    numeric_plots = plot_numeric_distributions(df)
    numeric_html = "".join(
        f'<div class="plot-box"><h4>{col}</h4><img src="data:image/png;base64,{b64}"></div>'
        for col, b64 in numeric_plots.items()
    ) or "<p class='empty'>No numeric columns.</p>"

    cat_plots = plot_categorical_distributions(df)
    cat_html = "".join(
        f'<div class="plot-box"><h4>{col}</h4><img src="data:image/png;base64,{b64}"></div>'
        for col, b64 in cat_plots.items()
    ) or "<p class='empty'>No categorical columns.</p>"

    corr_b64 = plot_correlation_heatmap(df)
    corr_html = (
        f'<div class="plot-box"><img src="data:image/png;base64,{corr_b64}"></div>'
        if corr_b64 else "<p class='empty'>Not enough numeric columns for correlation analysis.</p>"
    )
    top_corr = top_correlated_pairs(df)
    top_corr_html = ""
    if top_corr:
        top_corr_html = "<h4 style='color:var(--muted);font-weight:500;'>Top correlated pairs</h4>" + dict_list_to_html_table(top_corr)

    target_section = ""
    if target:
        t = target_analysis(df, target)
        if t.get("plot"):
            target_section = f"""
            <div class="section">
              <h2>Target Variable: {target}</h2>
              <div class="plot-box"><img src="data:image/png;base64,{t['plot']}"></div>
            </div>
            """

    cleaning_log_section = ""
    if cleaning_report:
        rows_html = "".join(
            f"<tr><td>{r['step']}</td><td>{r['detail']}</td></tr>" for r in cleaning_report
        )
        cleaning_log_section = f"""
        <div class="section">
          <h2>Data Cleaning Log</h2>
          <div class="table-wrap">
            <table><thead><tr><th>Step</th><th>Detail</th></tr></thead>
            <tbody>{rows_html}</tbody></table>
          </div>
        </div>
        """

    html = HTML_TEMPLATE.format(
        dataset_name=dataset_name,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        n_rows=overview["n_rows"],
        n_cols=overview["n_cols"],
        memory=overview["memory"],
        duplicate_rows=overview["duplicate_rows"],
        duplicate_pct=overview["duplicate_pct"],
        n_numeric=overview["n_numeric"],
        n_categorical=overview["n_categorical"],
        n_datetime=overview["n_datetime"],
        column_table=dict_list_to_html_table(columns),
        missing_plot_html=missing_html,
        numeric_plots_html=numeric_html,
        categorical_plots_html=cat_html,
        correlation_html=corr_html,
        top_corr_table=top_corr_html,
        target_section=target_section,
        cleaning_log_section=cleaning_log_section,
    )
    return html, overview, columns


def export_json_summary(overview: dict, columns: list, out_path: Path):
    summary = {
        "overview": overview,
        "columns": columns,
        "generated_at": datetime.now().isoformat(),
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Automated EDA tool: dataset profiling, summary statistics, "
                    "and standardized HTML documentation for any dataset."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input CSV/Excel/Parquet/JSON file")
    parser.add_argument("--output", "-o", default="eda_report.html", help="Path to output HTML report")
    parser.add_argument("--target", "-t", default=None, help="Optional target column to analyze specifically")
    parser.add_argument("--json-summary", default=None, help="Optional path to also export a JSON summary")
    args = parser.parse_args()

    print(f"Loading dataset from {args.input} ...")
    df = load_dataset(args.input)
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns.")

    dataset_name = Path(args.input).name
    print("Profiling dataset...")
    html, overview, columns = render_report(df, dataset_name, target=args.target)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Report written to: {out_path.resolve()}")

    if args.json_summary:
        json_path = Path(args.json_summary)
        export_json_summary(overview, columns, json_path)
        print(f"JSON summary written to: {json_path.resolve()}")


if __name__ == "__main__":
    main()
