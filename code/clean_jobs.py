"""
Clean raw JSearch JSONL into structured CSV and SQLite for JobPilot.

Run:
    python clean_jobs.py

Outputs:
    data/processed/jobs_clean.csv
    data/processed/jobs.db

Notes:
    - Uses API-collected raw JSONL from data/raw/jsearch_jobs.jsonl.
    - Cleans HTML, HTML entities, mojibake characters, bullets, and spacing.
    - Saves CSV with utf-8-sig so Excel displays text correctly.
"""

import html
import json
import re
import sqlite3
from pathlib import Path
from typing import Dict

import pandas as pd

RAW_PATH = Path("data/raw/jsearch_jobs.jsonl")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SKILLS = [
    "python", "sql", "r", "excel", "tableau", "power bi", "machine learning", "deep learning",
    "aws", "azure", "gcp", "spark", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
    "nlp", "statistics", "a/b testing", "java", "javascript", "typescript", "react", "node",
    "docker", "kubernetes", "git", "linux", "airflow", "dbt", "snowflake", "looker", "salesforce",
    "financial modeling", "forecasting", "accounting", "risk", "cybersecurity", "cloud", "etl",
    "data visualization", "communication", "stakeholder", "product management", "agile", "scrum",
    "figma", "ux", "ui", "marketing", "seo", "consulting", "powerpoint", "gaap", "audit",
]


def clean_text(value) -> str:
    """Clean text returned by job APIs.

    JSearch descriptions often contain HTML, bullet symbols, smart quotes, and mojibake
    such as '鈥?' or 'â€¢'. This function normalizes those artifacts before the text is
    used for CSV display, SQLite storage, and embeddings.
    """
    if value is None:
        return ""

    text = str(value)

    # Decode common HTML entities such as &amp; and &nbsp;.
    text = html.unescape(text)

    # Remove HTML tags while keeping word separation.
    text = re.sub(r"<[^>]+>", " ", text)

    # Fix common mojibake / encoding artifacts from copied job descriptions.
    replacements = {
        "鈥?": " • ",
        "鈥�": " • ",
        "鈥¢": " • ",
        "â€¢": " • ",
        "â€™": "'",
        "â€˜": "'",
        "â€œ": '"',
        "â€": '"',
        "â€“": "-",
        "â€”": "-",
        "â€¦": "...",
        "\u2022": " • ",
        "\u00a0": " ",
        "\ufeff": "",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # Remove non-printable control characters but keep normal whitespace.
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)

    # Normalize excessive repeated bullets/spaces.
    text = re.sub(r"(?:\s*•\s*){2,}", " • ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def shorten_for_embedding(text: str, max_chars: int = 3500) -> str:
    """Keep embedding input focused and not dominated by long legal/EEO text."""
    text = clean_text(text)
    return text[:max_chars].strip()


def extract_skills(text: str) -> str:
    text_l = clean_text(text).lower()
    found = sorted({skill.title() for skill in SKILLS if skill in text_l})
    return ", ".join(found)


def detect_seniority(title: str, description: str) -> str:
    text = f"{title} {description}".lower()
    if any(term in text for term in ["intern", "internship"]):
        return "internship"
    if any(term in text for term in ["entry level", "new grad", "graduate", "junior", "jr.", "jr "]):
        return "entry_or_junior"
    if any(term in text for term in ["director", "vp", "vice president", "head of"]):
        return "executive"
    if any(term in text for term in ["senior", "sr.", "sr ", "staff", "principal", "lead"]):
        return "senior"
    return "mid_or_unspecified"


def detect_employment_type(raw: Dict) -> str:
    text = " ".join([
        str(raw.get("job_employment_type") or ""),
        str(raw.get("job_title") or ""),
        str(raw.get("job_description") or ""),
    ]).lower()
    if "intern" in text:
        return "internship"
    if "contract" in text or "contractor" in text:
        return "contract"
    if "part-time" in text or "part time" in text:
        return "part-time"
    if "temporary" in text or " temp " in f" {text} ":
        return "temporary"
    if "full-time" in text or "full time" in text:
        return "full-time"
    return clean_text(raw.get("job_employment_type"))


def normalize(wrapper: Dict) -> Dict:
    raw = wrapper.get("raw", {}) or {}

    title = clean_text(raw.get("job_title"))
    company = clean_text(raw.get("employer_name"))
    description = clean_text(raw.get("job_description"))
    location = clean_text(
        raw.get("job_location")
        or ", ".join([clean_text(x) for x in [raw.get("job_city"), raw.get("job_state"), raw.get("job_country")] if x])
    )
    apply_url = clean_text(raw.get("job_apply_link") or raw.get("job_google_link") or "")

    full_text = clean_text(f"{title}. {company}. {location}. {description}")
    skills = extract_skills(full_text)

    # Use this cleaner and shorter field for embeddings / matching.
    clean_job_text = shorten_for_embedding(
        f"Title: {title}. Company: {company}. Location: {location}. "
        f"Skills: {skills}. Description: {description}",
        max_chars=3500,
    )

    return {
        "job_id": clean_text(raw.get("job_id")),
        "title": title,
        "company": company,
        "location": location,
        "description": description,
        "skills": skills,
        "seniority": detect_seniority(title, description),
        "employment_type": detect_employment_type(raw),
        "salary_min": raw.get("job_min_salary"),
        "salary_max": raw.get("job_max_salary"),
        "salary_period": clean_text(raw.get("job_salary_period")),
        "posted_at": clean_text(raw.get("job_posted_at_datetime_utc") or raw.get("job_posted_at_timestamp") or ""),
        "is_remote": bool(raw.get("job_is_remote")),
        "apply_url": apply_url,
        "source": clean_text(wrapper.get("source_api", "JSearch")),
        "ingested_at": clean_text(wrapper.get("ingested_at")),
        "search_role": clean_text(wrapper.get("search_role")),
        "search_location": clean_text(wrapper.get("search_location")),
        "job_text": full_text,
        "clean_job_text": clean_job_text,
    }


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing {RAW_PATH}. Run collect_jsearch_jobs.py first.")

    records = []
    skipped = 0
    with RAW_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(normalize(json.loads(line)))
                except Exception:
                    skipped += 1

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("No records found in raw JSONL.")

    before = len(df)

    # Basic quality filters.
    df = df[df["title"].str.len() > 0]
    df = df[df["company"].str.len() > 0]
    df = df[df["description"].str.len() > 50]
    df = df[df["clean_job_text"].str.len() > 80]
    after_quality = len(df)

    # Deduplicate by stable business key. job_id can vary across sources, so this is safer.
    df = df.drop_duplicates(subset=["title", "company", "location"], keep="first").reset_index(drop=True)
    after_dedup = len(df)

    csv_path = OUT_DIR / "jobs_clean.csv"
    db_path = OUT_DIR / "jobs.db"

    # utf-8-sig makes Excel on Windows display the file correctly.
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    con = sqlite3.connect(db_path)
    df.to_sql("jobs", con, if_exists="replace", index=False)
    con.close()

    print("Cleaning summary")
    print(f"Raw records parsed: {before:,}")
    print(f"Skipped invalid JSON lines: {skipped:,}")
    print(f"After quality filters: {after_quality:,}")
    print(f"After deduplication: {after_dedup:,}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved SQLite: {db_path}")
    print("Use column 'clean_job_text' for embeddings and recommendation.")


if __name__ == "__main__":
    main()
