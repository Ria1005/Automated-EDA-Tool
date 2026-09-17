# Sift

A small tool I built to stop rewriting the same `.info()` / `.isna().sum()` /
`.describe()` combo every time I get a new dataset. Point it at a CSV or
Excel file and it finds missing values, duplicates and outliers, cleans
them, and spits out a report with the stats and charts I'd normally
have to make by hand.

**[Live demo](https://YOUR-USERNAME.github.io/sift/)** — there's a sample
dataset button if you don't want to dig up your own file.

## What's in here

- `index.html` — the demo above. Runs entirely in the browser (CSV via
  PapaParse, Excel via SheetJS, charts via Chart.js), so nothing gets
  uploaded anywhere.
- `pipeline/` — the actual Python version I use for real work:
  - `db.py` — loads a file into SQLite, lets you query it directly
  - `clean.py` — the cleaning logic (missing values, duplicates, dtype
    fixes, outlier capping), with a log of what it changed
  - `eda_tool.py` — turns a cleaned dataframe into an HTML report
  - `pipeline.py` — runs all of the above in one go

## Running it

```bash
cd pipeline
pip install -r requirements.txt
python pipeline.py --input titanic.csv --project titanic --target survived
```

That drops a report + a `pipeline.db` SQLite file in an `output/` folder.

## Stack

Python, pandas, SQLite, matplotlib/seaborn for the plots. The demo page
is just plain HTML/CSS/JS, no framework.

## Notes to self / possible next steps

- outlier clipping can create duplicate rows that didn't exist before
  (learned this the hard way on the Titanic dataset — a bunch of ages
  got clipped to the same upper bound)
- would be nice to add a proper Postgres option instead of just SQLite
- report styling could use a dark mode toggle
