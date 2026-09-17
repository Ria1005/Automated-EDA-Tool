#!/usr/bin/env python3
"""
Cleaning step of the pipeline. Takes a raw dataframe (usually just
pulled from SQL) and runs it through a consistent set of fixes -
whitespace, dtypes, duplicates, missing values, outliers.

Each function logs what it changed so we can print a summary at the
end instead of just trusting it worked.

    from clean import clean_dataframe
    cleaned_df, report = clean_dataframe(raw_df)
"""

import numpy as np
import pandas as pd


def _log(report: list, step: str, detail: str):
    report.append({"step": step, "detail": detail})


def strip_whitespace_and_standardize_text(df: pd.DataFrame, report: list) -> pd.DataFrame:
    """Trim whitespace and standardize casing on object/string columns."""
    text_cols = df.select_dtypes(include="object").columns
    changed = []
    for col in text_cols:
        before = df[col].copy()
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})
        if not before.equals(df[col]):
            changed.append(col)
    if changed:
        _log(report, "Text standardization", f"Trimmed whitespace / normalized blanks in: {changed}")
    return df


def coerce_dtypes(df: pd.DataFrame, report: list) -> pd.DataFrame:
    """Try to convert object columns that are actually numeric or datetime."""
    converted_numeric, converted_dates = [], []
    for col in df.select_dtypes(include="object").columns:
        as_numeric = pd.to_numeric(df[col], errors="coerce")
        # If >90% of non-null values convert cleanly, treat it as numeric
        non_null = df[col].notna().sum()
        if non_null > 0 and as_numeric.notna().sum() / non_null > 0.9:
            df[col] = as_numeric
            converted_numeric.append(col)
            continue
        as_date = pd.to_datetime(df[col], errors="coerce", format=None)
        if non_null > 0 and as_date.notna().sum() / non_null > 0.9:
            df[col] = as_date
            converted_dates.append(col)
    if converted_numeric:
        _log(report, "Type coercion", f"Converted to numeric: {converted_numeric}")
    if converted_dates:
        _log(report, "Type coercion", f"Converted to datetime: {converted_dates}")
    return df


def handle_missing_values(df: pd.DataFrame, report: list,
                           numeric_strategy: str = "median",
                           categorical_strategy: str = "mode",
                           drop_high_missing_thresh: float = 0.6) -> pd.DataFrame:
    """
    Standardized missing-value policy:
    - Drop columns missing above `drop_high_missing_thresh` (default 60%)
    - Fill numeric columns with median (robust to outliers)
    - Fill categorical columns with mode (most frequent value)
    """
    n = len(df)
    dropped_cols = []
    for col in df.columns:
        missing_frac = df[col].isna().sum() / n if n else 0
        if missing_frac > drop_high_missing_thresh:
            dropped_cols.append(col)
    if dropped_cols:
        df = df.drop(columns=dropped_cols)
        _log(report, "Missing values", f"Dropped columns >{int(drop_high_missing_thresh*100)}% missing: {dropped_cols}")

    filled_numeric, filled_categorical = [], []
    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            if numeric_strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif numeric_strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            filled_numeric.append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            continue  # leave date gaps as-is; filling dates is rarely safe
        else:
            mode = df[col].mode(dropna=True)
            if not mode.empty:
                df[col] = df[col].fillna(mode.iloc[0])
                filled_categorical.append(col)

    if filled_numeric:
        _log(report, "Missing values", f"Filled numeric ({numeric_strategy}): {filled_numeric}")
    if filled_categorical:
        _log(report, "Missing values", f"Filled categorical (mode): {filled_categorical}")
    return df


def remove_duplicates(df: pd.DataFrame, report: list) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        _log(report, "Duplicates", f"Removed {removed} duplicate rows ({before} -> {len(df)})")
    return df


def cap_outliers(df: pd.DataFrame, report: list, method: str = "iqr", factor: float = 1.5) -> pd.DataFrame:
    """
    Cap (winsorize) numeric outliers using the IQR method rather than
    dropping rows, so no data is lost outright.
    """
    capped_cols = []
    for col in df.select_dtypes(include=np.number).columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_outliers > 0:
            df[col] = df[col].clip(lower, upper)
            capped_cols.append((col, int(n_outliers)))
    if capped_cols:
        detail = ", ".join(f"{c} ({n} values)" for c, n in capped_cols)
        _log(report, "Outliers", f"Capped via IQR method: {detail}")
    return df


def clean_dataframe(df: pd.DataFrame,
                     fill_missing: bool = True,
                     drop_duplicates: bool = True,
                     cap_outliers_flag: bool = True,
                     coerce_types: bool = True) -> tuple:
    """
    Runs the full standardized cleaning pipeline (Step 3) and returns
    (cleaned_df, cleaning_report) where cleaning_report is a list of
    dicts describing every change made — this becomes part of the
    standardized project documentation.
    """
    df = df.copy()
    report = []
    _log(report, "Start", f"Input shape: {df.shape}")

    df = strip_whitespace_and_standardize_text(df, report)
    if coerce_types:
        df = coerce_dtypes(df, report)
    if drop_duplicates:
        df = remove_duplicates(df, report)
    if fill_missing:
        df = handle_missing_values(df, report)
    if cap_outliers_flag:
        df = cap_outliers(df, report)

    _log(report, "Finish", f"Output shape: {df.shape}")
    return df, report
