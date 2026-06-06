"""
Merge Kaggle Techmap and JSearch job datasets for JobPilot.

Inputs:
    data/raw/kaggle_jobs_30000.csv
    data/processed/jobs_clean.csv

Output:
    data/processed/jobs_final.csv
    data/processed/jobs_final.db

Run:
    python merge_job_datasets.py

Optional:
    python merge_job_datasets.py --kaggle data/raw/kaggle_jobs_30000.csv --jsearch data/processed/jobs_clean.csv --output data/processed/jobs_final.csv
"""

import argparse
import re
import sqlite3
from pathlib import Path

import pandas as pd


FINAL_COLUMNS = [
    "job_id",
    "title",
    "company",
    "location",
    "description",
    "skills",
    "seniority",
    "employment_type",
    "salary_min",
    "salary_max",
    "salary_period",
    "posted_at",
    "is_remote",
    "apply_url",
    "source",
    "ingested_at",
    "search_role",
    "search_location",
    "job_text",
    "clean_job_text",
]


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y"}


def ensure_schema(df: pd.DataFrame, default_source: str) -> pd.DataFrame:
    out = df.copy()

    for col in FINAL_COLUMNS:
        if col not in out.columns:
            out[col] = ""

    for col in [
        "job_id",
        "title",
        "company",
        "location",
        "description",
        "skills",
        "seniority",
        "employment_type",
        "salary_period",
        "posted_at",
        "apply_url",
        "source",
        "ingested_at",
        "search_role",
        "search_location",
        "job_text",
        "clean_job_text",
    ]:
        out[col] = out[col].apply(clean_text)

    out["source"] = out["source"].replace("", default_source)
    out["is_remote"] = out["is_remote"].apply(normalize_bool)

    missing_job_text = out["job_text"].str.len() == 0
    out.loc[missing_job_text, "job_text"] = (
        out.loc[missing_job_text, "title"]
        + ". "
        + out.loc[missing_job_text, "company"]
        + ". "
        + out.loc[missing_job_text, "location"]
        + ". "
        + out.loc[missing_job_text, "description"]
    ).apply(clean_text)

    missing_clean_text = out["clean_job_text"].str.len() == 0
    out.loc[missing_clean_text, "clean_job_text"] = (
        "Title: "
        + out.loc[missing_clean_text, "title"]
        + ". Company: "
        + out.loc[missing_clean_text, "company"]
        + ". Location: "
        + out.loc[missing_clean_text, "location"]
        + ". Skills: "
        + out.loc[missing_clean_text, "skills"]
        + ". Description: "
        + out.loc[missing_clean_text, "description"]
    ).apply(lambda x: clean_text(x)[:3500])

    return out[FINAL_COLUMNS]


def make_dedup_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["title"].str.lower().str.strip()
        + "|"
        + df["company"].str.lower().str.strip()
        + "|"
        + df["location"].str.lower().str.strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle", default="data/raw/kaggle_jobs_30000.csv")
    parser.add_argument("--jsearch", default="data/processed/jobs_clean.csv")
    parser.add_argument("--output", default="data/processed/jobs_final.csv")
    parser.add_argument("--db-output", default="data/processed/jobs_final.db")
    parser.add_argument("--keep", choices=["jsearch_first", "kaggle_first"], default="jsearch_first")
    args = parser.parse_args()

    kaggle_path = Path(args.kaggle)
    jsearch_path = Path(args.jsearch)
    output_path = Path(args.output)
    db_output_path = Path(args.db_output)

    if not kaggle_path.exists():
        raise FileNotFoundError(f"Missing Kaggle file: {kaggle_path}")
    if not jsearch_path.exists():
        raise FileNotFoundError(f"Missing JSearch file: {jsearch_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    db_output_path.parent.mkdir(parents=True, exist_ok=True)

    kaggle_raw = pd.read_csv(kaggle_path)
    jsearch_raw = pd.read_csv(jsearch_path)

    kaggle = ensure_schema(kaggle_raw, "Kaggle_Techmap_2021")
    jsearch = ensure_schema(jsearch_raw, "JSearch")

    if args.keep == "jsearch_first":
        combined = pd.concat([jsearch, kaggle], ignore_index=True)
    else:
        combined = pd.concat([kaggle, jsearch], ignore_index=True)

    before_quality = len(combined)

    combined = combined[combined["title"].str.len() > 0]
    combined = combined[combined["company"].str.len() > 0]
    combined = combined[combined["description"].str.len() > 50]
    combined = combined[combined["clean_job_text"].str.len() > 80]

    after_quality = len(combined)

    combined["dedup_key"] = make_dedup_key(combined)
    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=["dedup_key"], keep="first")
    combined = combined.drop(columns=["dedup_key"]).reset_index(drop=True)

    combined.to_csv(output_path, index=False, encoding="utf-8-sig")

    con = sqlite3.connect(db_output_path)
    combined.to_sql("jobs", con, if_exists="replace", index=False)
    con.close()

    print("Merge summary")
    print(f"Kaggle input rows: {len(kaggle):,}")
    print(f"JSearch input rows: {len(jsearch):,}")
    print(f"Combined rows before quality filters: {before_quality:,}")
    print(f"Rows after quality filters: {after_quality:,}")
    print(f"Duplicate rows removed: {before_dedup - len(combined):,}")
    print(f"Final rows: {len(combined):,}")
    print("Rows by source:")
    print(combined["source"].value_counts(dropna=False).to_string())
    print(f"Saved CSV: {output_path}")
    print(f"Saved SQLite: {db_output_path}")


if __name__ == "__main__":
    main()
