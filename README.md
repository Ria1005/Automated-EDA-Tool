# Automated EDA Tool

A small tool I built to automate the repetitive `.info()`, `.isna().sum()`, and `.describe()` workflow every time I get a new dataset.

Point it at a CSV or Excel file and it identifies missing values, duplicates, and outliers, cleans the data, and generates a report with statistics and charts that would normally have to be created manually.

**[Live Demo](https://ria1005.github.io/Automated-EDA-Tool/)** — there's also a sample dataset button if you don't want to upload your own file.

## What's in here

- `index.html` — the browser-based demo. Runs entirely in the browser using:
  - PapaParse for CSV files
  - SheetJS for Excel files
  - Chart.js for visualizations
  - No uploaded data is sent to a server

- `pipeline/` — the Python version used for real datasets:
  - `db.py` — loads data into SQLite and allows direct querying
  - `clean.py` — cleaning logic for missing values, duplicates, data types, and outlier capping
  - `eda_tool.py` — generates an HTML EDA report from a cleaned dataframe
  - `pipeline.py` — runs the complete workflow in one pipeline

## Running it

```bash
cd pipeline
pip install -r requirements.txt
python pipeline.py --input titanic.csv --project titanic --target survived