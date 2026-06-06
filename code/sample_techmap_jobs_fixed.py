"""
Sample 30,000 records from the instructor-provided Techmap Kaggle job postings dataset
and convert them into a clean CSV for JobPilot.

Dataset:
    https://www.kaggle.com/datasets/techmap/international-job-postings-september-2021

Why this script:
    The Kaggle dataset is very large and stored as JSON / JSONL, often compressed.
    This script reads it in streaming mode, extracts useful job fields, applies basic
    quality filters, deduplicates records, and saves a 30,000-row CSV snapshot.

Supported input formats:
    .jsonl
    .json
    .jsonl.gz
    .json.gz

Recommended folder:
    Put the downloaded Kaggle file in:
        data/raw/

Example runs:
    python sample_techmap_jobs_fixed.py --input data/raw/jobs.jsonl.gz --output data/raw/kaggle_jobs_30000.csv --target 30000

    python sample_techmap_jobs_fixed.py --input data/raw/jobs.json.gz --output data/raw/kaggle_jobs_30000.csv --target 30000

Output:
    data/raw/kaggle_jobs_30000.csv

After this:
    You can merge this CSV with your cleaned JSearch data:
        data/processed/jobs_clean.csv
"""

import argparse
import csv
import gzip
import html
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


DEFAULT_OUTPUT_COLUMNS = [
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


SKILLS = [
    "python", "sql", "r", "excel", "tableau", "power bi", "machine learning", "deep learning",
    "aws", "azure", "gcp", "spark", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
    "nlp", "statistics", "a/b testing", "java", "javascript", "typescript", "react", "node",
    "docker", "kubernetes", "git", "linux", "airflow", "dbt", "snowflake", "looker", "salesforce",
    "financial modeling", "forecasting", "accounting", "risk", "cybersecurity", "cloud", "etl",
    "data visualization", "communication", "stakeholder", "product management", "agile", "scrum",
    "figma", "ux", "ui", "marketing", "seo", "consulting", "powerpoint", "gaap", "audit",
    "kafka", "kubernetes", "microservices", "c++", "hadoop", "bigquery", "databricks",
]


def clean_text(value: Any) -> str:
    """Clean text for CSV display and embedding input."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return clean_scalar_value(value)
    if isinstance(value, list):
        return clean_text(", ".join(clean_scalar_value(item) for item in value if clean_scalar_value(item)))

    text = str(value)
    text = html.unescape(text)

    # Remove HTML tags.
    text = re.sub(r"<[^>]+>", " ", text)

    # Fix common mojibake / copied web artifacts.
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

    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_scalar_value(value: Any) -> str:
    """Return useful scalar text from nested location/schema values."""
    if value is None:
        return ""
    if isinstance(value, dict):
        preferred_keys = [
            "name",
            "addressLocality",
            "addressRegion",
            "addressCountry",
            "country",
            "countryCode",
            "city",
            "state",
        ]
        parts = []
        for key in preferred_keys:
            if key in value:
                cleaned = clean_scalar_value(value.get(key))
                if cleaned and cleaned not in parts:
                    parts.append(cleaned)
        if parts:
            return ", ".join(parts)
        return ""
    if isinstance(value, list):
        parts = [clean_scalar_value(item) for item in value]
        return ", ".join([part for part in parts if part])
    return clean_text(str(value))


def get_nested(obj: Dict[str, Any], path: Iterable[str], default: str = "") -> Any:
    """Safely read nested dictionary values."""
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def first_nonempty(*values: Any) -> str:
    for value in values:
        cleaned = clean_text(value)
        if cleaned:
            return cleaned
    return ""


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


def detect_employment_type(title: str, description: str) -> str:
    text = f"{title} {description}".lower()
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
    return "full-time_or_unspecified"


def detect_remote(location: str, title: str, description: str) -> bool:
    text = f"{location} {title} {description}".lower()
    return "remote" in text or "work from home" in text or "wfh" in text


def make_location(job: Dict[str, Any]) -> str:
    """Try several Techmap and schema.org location fields."""
    direct = first_nonempty(
        job.get("job_location"),
        job.get("address"),
        get_nested(job, ["orgAddress", "addressLine"]),
        get_nested(job, ["json", "schemaOrg", "jobLocation", "address", "addressLocality"]),
    )
    if direct and not direct.startswith("{"):
        region_for_direct = first_nonempty(
            get_nested(job, ["orgAddress", "state"]),
            get_nested(job, ["json", "schemaOrg", "jobLocation", "address", "addressRegion"]),
        )
        country_for_direct = first_nonempty(
            get_nested(job, ["orgAddress", "countryCode"]),
            get_nested(job, ["json", "schemaOrg", "jobLocation", "address", "addressCountry"]),
            job.get("sourceCC"),
        )
        if region_for_direct and region_for_direct.lower() not in direct.lower():
            direct = f"{direct}, {region_for_direct}"
        if country_for_direct and country_for_direct.lower() not in direct.lower():
            direct = f"{direct}, {country_for_direct}"
        return direct

    city = first_nonempty(
        get_nested(job, ["orgAddress", "city"]),
        get_nested(job, ["location", "address", "addressLocality"]),
        get_nested(job, ["json", "schemaOrg", "jobLocation", "address", "addressLocality"]),
    )
    region = first_nonempty(
        get_nested(job, ["orgAddress", "state"]),
        get_nested(job, ["location", "address", "addressRegion"]),
        get_nested(job, ["json", "schemaOrg", "jobLocation", "address", "addressRegion"]),
    )
    country = first_nonempty(
        get_nested(job, ["orgAddress", "countryCode"]),
        get_nested(job, ["location", "address", "addressCountry"]),
        get_nested(job, ["json", "schemaOrg", "jobLocation", "address", "addressCountry"]),
        job.get("sourceCC"),
    )

    parts = [x for x in [city, region, country] if x]
    return ", ".join(parts)


def normalize_job(job: Dict[str, Any], row_number: int) -> Optional[Dict[str, Any]]:
    """Convert a raw Techmap record into the same schema used by JobPilot."""
    schema_org = job.get("json", {}).get("schemaOrg", {}) if isinstance(job.get("json"), dict) else {}

    title = first_nonempty(
        job.get("title"),
        job.get("name"),
        job.get("job_title"),
        get_nested(job, ["position", "name"]),
        schema_org.get("title"),
    )

    company = first_nonempty(
        job.get("company"),
        job.get("companyName"),
        job.get("employer_name"),
        get_nested(job, ["company", "name"]),
        get_nested(job, ["hiringOrganization", "name"]),
        get_nested(job, ["organization", "name"]),
        get_nested(job, ["orgCompany", "name"]),
        get_nested(job, ["orgCompany", "nameOrg"]),
        get_nested(job, ["json", "schemaOrg", "hiringOrganization", "name"]),
        schema_org.get("hiringOrganization", {}).get("name") if isinstance(schema_org.get("hiringOrganization"), dict) else "",
    )

    description = first_nonempty(
        job.get("description"),
        job.get("text"),
        job.get("html"),
        job.get("job_description"),
        job.get("body"),
        get_nested(job, ["json", "schemaOrg", "description"]),
        schema_org.get("description"),
    )

    location = make_location(job)

    apply_url = first_nonempty(
        job.get("url"),
        job.get("apply_url"),
        job.get("job_apply_link"),
        job.get("sourceURL"),
        job.get("sourceUrl"),
        get_nested(job, ["json", "schemaOrg", "url"]),
        schema_org.get("url"),
    )

    posted_at = first_nonempty(
        job.get("datePosted"),
        job.get("date_posted"),
        get_nested(job, ["dateCreated", "$date"]),
        get_nested(job, ["dateScraped", "$date"]),
        get_nested(job, ["dateUploaded", "$date"]),
        get_nested(job, ["json", "schemaOrg", "datePosted"]),
        schema_org.get("datePosted"),
        job.get("created_at"),
    )

    salary_min = first_nonempty(
        job.get("salary_min"),
        job.get("min_salary"),
        get_nested(job, ["baseSalary", "value", "minValue"]),
        get_nested(job, ["json", "schemaOrg", "baseSalary", "value", "minValue"]),
    )

    salary_max = first_nonempty(
        job.get("salary_max"),
        job.get("max_salary"),
        get_nested(job, ["baseSalary", "value", "maxValue"]),
        get_nested(job, ["json", "schemaOrg", "baseSalary", "value", "maxValue"]),
    )

    salary_period = first_nonempty(
        job.get("salary_period"),
        get_nested(job, ["baseSalary", "value", "unitText"]),
        get_nested(job, ["json", "schemaOrg", "baseSalary", "value", "unitText"]),
    )

    employment_type = first_nonempty(
        job.get("employment_type"),
        job.get("job_employment_type"),
        get_nested(job, ["position", "workType"]),
        get_nested(job, ["json", "schemaOrg", "employmentType"]),
        schema_org.get("employmentType"),
    )

    if not title or not company or len(description) < 50:
        return None

    full_text = clean_text(f"{title}. {company}. {location}. {description}")
    skills = extract_skills(full_text)

    clean_job_text = clean_text(
        f"Title: {title}. Company: {company}. Location: {location}. "
        f"Skills: {skills}. Description: {description}"
    )[:3500]

    if len(clean_job_text) < 80:
        return None

    stable_id = first_nonempty(
        get_nested(job, ["_id", "$oid"]),
        job.get("id"),
        job.get("idInSource"),
        job.get("job_id"),
    )
    if not stable_id:
        stable_id = f"kaggle_techmap_{row_number}"

    return {
        "job_id": stable_id,
        "title": title,
        "company": company,
        "location": location,
        "description": description,
        "skills": skills,
        "seniority": detect_seniority(title, description),
        "employment_type": employment_type or detect_employment_type(title, description),
        "salary_min": salary_min,
        "salary_max": salary_max,
        "salary_period": salary_period,
        "posted_at": posted_at,
        "is_remote": detect_remote(location, title, description),
        "apply_url": apply_url,
        "source": "Kaggle_Techmap_2021",
        "ingested_at": "",
        "search_role": "",
        "search_location": "",
        "job_text": full_text,
        "clean_job_text": clean_job_text,
    }

def open_text_file(path: Path):
    """Open normal or gzip-compressed text file."""
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("r", encoding="utf-8", errors="ignore")


def iter_json_records(path: Path) -> Iterable[Dict[str, Any]]:
    """
    Yield JSON records from either:
    1. JSON Lines: one JSON object per line
    2. JSON array: [ {...}, {...} ]
    3. Single JSON object containing a list under common keys
    """
    # First try line-by-line. This is best for huge JSONL files.
    with open_text_file(path) as f:
        first_nonempty_line = ""
        for line in f:
            if line.strip():
                first_nonempty_line = line.lstrip()
                break

    if not first_nonempty_line:
        return

    if first_nonempty_line.startswith("["):
        # JSON array. This requires loading the sampled file in memory, so only use this
        # when Kaggle provides a normal JSON array instead of JSONL.
        with open_text_file(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    yield item
        return

    if first_nonempty_line.startswith("{"):
        # Try JSONL streaming first.
        parsed_any = False
        with open_text_file(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    parsed_any = True
                    if isinstance(item, dict):
                        yield item
                except json.JSONDecodeError:
                    # If the file is pretty-printed JSON, fall back below.
                    parsed_any = False
                    break

        if parsed_any:
            return

        # Fallback for a single large JSON object.
        with open_text_file(path) as f:
            data = json.load(f)

        if isinstance(data, dict):
            for key in ["data", "jobs", "records", "results"]:
                value = data.get(key)
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            yield item
                    return

            # If it is one job object.
            yield data



class SimpleProgress:
    """Lightweight progress reporter used when tqdm is unavailable or disabled."""

    def __init__(self, total: int, enabled: bool = True, interval: float = 2.0) -> None:
        self.total = total
        self.enabled = enabled
        self.interval = interval
        self.start_time = time.time()
        self.last_update = 0.0

    def update(self, scanned: int, collected: int, quality_skipped: int, duplicate_skipped: int, random_skipped: int) -> None:
        if not self.enabled:
            return
        now = time.time()
        if now - self.last_update < self.interval and scanned < self.total:
            return
        self.last_update = now
        pct = min(scanned / self.total, 1.0) * 100 if self.total else 0.0
        elapsed = max(now - self.start_time, 0.001)
        rate = scanned / elapsed
        message = (
            f"\rScanned {scanned:,}/{self.total:,} ({pct:5.1f}%) | "
            f"collected {collected:,} | quality_skip {quality_skipped:,} | "
            f"dup_skip {duplicate_skipped:,} | random_skip {random_skipped:,} | "
            f"{rate:,.0f} rec/s"
        )
        print(message, end="", flush=True)

    def close(self) -> None:
        if self.enabled:
            print()


def make_progress(total: int, disabled: bool = False):
    """Return tqdm progress bar when installed; otherwise use SimpleProgress."""
    if disabled:
        return None
    try:
        from tqdm import tqdm

        return tqdm(total=total, desc="Scanning raw jobs", unit="records")
    except Exception:
        return SimpleProgress(total=total, enabled=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the Techmap Kaggle JSON/JSONL/GZ file.")
    parser.add_argument("--output", default="data/raw/kaggle_jobs_30000.csv", help="Output CSV path.")
    parser.add_argument("--target", type=int, default=30000, help="Number of cleaned unique jobs to save.")
    parser.add_argument("--max-read", type=int, default=2000000, help="Safety limit for raw records scanned.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when --random-skip is enabled.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress display.")
    parser.add_argument(
        "--random-skip",
        type=float,
        default=0.0,
        help=(
            "Optional random skip probability before cleaning. Example: 0.8 means skip about 80%% "
            "of raw records, useful if you want less first-page bias. Keep 0.0 for fastest collection."
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    random.seed(args.seed)

    seen_keys = set()
    rows = []
    raw_scanned = 0
    quality_skipped = 0
    duplicate_skipped = 0
    random_skipped = 0

    progress = make_progress(args.max_read, disabled=args.no_progress)

    for raw_job in iter_json_records(input_path):
        raw_scanned += 1

        if raw_scanned > args.max_read:
            print(f"\nReached --max-read={args.max_read:,}. Stopping early.")
            break

        if args.random_skip > 0 and random.random() < args.random_skip:
            random_skipped += 1
            if progress is not None:
                if hasattr(progress, "set_postfix"):
                    progress.update(1)
                    progress.set_postfix({"collected": len(rows), "quality_skip": quality_skipped, "dup_skip": duplicate_skipped})
                else:
                    progress.update(raw_scanned, len(rows), quality_skipped, duplicate_skipped, random_skipped)
            continue

        normalized = normalize_job(raw_job, raw_scanned)
        if normalized is None:
            quality_skipped += 1
            if progress is not None:
                if hasattr(progress, "set_postfix"):
                    progress.update(1)
                    progress.set_postfix({"collected": len(rows), "quality_skip": quality_skipped, "dup_skip": duplicate_skipped})
                else:
                    progress.update(raw_scanned, len(rows), quality_skipped, duplicate_skipped, random_skipped)
            continue

        dedup_key = (
            normalized["title"].lower().strip(),
            normalized["company"].lower().strip(),
            normalized["location"].lower().strip(),
        )
        if dedup_key in seen_keys:
            duplicate_skipped += 1
            if progress is not None:
                if hasattr(progress, "set_postfix"):
                    progress.update(1)
                    progress.set_postfix({"collected": len(rows), "quality_skip": quality_skipped, "dup_skip": duplicate_skipped})
                else:
                    progress.update(raw_scanned, len(rows), quality_skipped, duplicate_skipped, random_skipped)
            continue

        seen_keys.add(dedup_key)
        rows.append(normalized)

        if progress is not None:
            if hasattr(progress, "set_postfix"):
                progress.update(1)
                progress.set_postfix({"collected": len(rows), "quality_skip": quality_skipped, "dup_skip": duplicate_skipped})
            else:
                progress.update(raw_scanned, len(rows), quality_skipped, duplicate_skipped, random_skipped)

        if len(rows) >= args.target:
            break

    if progress is not None:
        progress.close()

    if not rows:
        raise RuntimeError(
            "No usable records were extracted. The JSON schema may be different. "
            "Open the first few lines of the file and adjust normalize_job()."
        )

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=DEFAULT_OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print("\nSampling summary")
    print(f"Raw records scanned: {raw_scanned:,}")
    print(f"Random skipped: {random_skipped:,}")
    print(f"Quality skipped: {quality_skipped:,}")
    print(f"Duplicate skipped: {duplicate_skipped:,}")
    print(f"Final clean unique jobs saved: {len(rows):,}")
    print(f"Output CSV: {output_path}")

    if len(rows) < args.target:
        print(
            f"\nWarning: saved only {len(rows):,} rows, below target {args.target:,}. "
            "Try increasing --max-read or reducing --random-skip."
        )


if __name__ == "__main__":
    main()
