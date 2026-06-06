"""
Reusable JobPilot ingestion script for JSearch / RapidAPI.

What this does:
1. Calls the JSearch API for role + location combinations.
2. Saves raw JSONL records.
3. Can resume / append without deleting previous data.
4. Skips duplicates using a stable dedup key.
5. Handles 403 and 429 more safely with slower pacing and backoff.

Setup:
    pip install -r requirements.txt
    copy .env.example .env
    # put your RapidAPI key in .env

Test:
    python collect_jsearch_jobs.py --target 50 --pages-per-query 1 --sleep 6

Recommended free-plan run:
    python collect_jsearch_jobs.py --target 1000 --pages-per-query 1 --sleep 8

Reuse later:
    python collect_jsearch_jobs.py --target 1000 --append --sleep 8
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Set

import requests
from dotenv import load_dotenv
from tqdm import tqdm

from roles_locations import ROLES, LOCATIONS

API_HOST = "jsearch.p.rapidapi.com"
SEARCH_URL = "https://jsearch.p.rapidapi.com/search"


def stable_text(value) -> str:
    return str(value or "").lower().strip()


def make_key_from_raw(job: Dict) -> str:
    """Stable key for deduping across repeated runs and similar API calls."""
    job_id = stable_text(job.get("job_id"))
    title = stable_text(job.get("job_title"))
    employer = stable_text(job.get("employer_name"))
    location = stable_text(job.get("job_location") or job.get("job_city") or "")
    if job_id:
        return f"id::{job_id}"
    return f"text::{title}|{employer}|{location}"


def load_existing_keys(path: Path) -> Set[str]:
    keys: Set[str] = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                wrapper = json.loads(line)
                raw = wrapper.get("raw", {})
                keys.add(make_key_from_raw(raw))
            except Exception:
                continue
    return keys


def search_jobs(query: str, page: int, rapidapi_key: str, country: str = "us") -> Iterable[Dict]:
    headers = {
        "X-RapidAPI-Key": rapidapi_key,
        "X-RapidAPI-Host": API_HOST,
    }
    params = {
        "query": query,
        "page": page,
        "num_pages": 1,
        "country": country,
        "date_posted": "month",
    }
    response = requests.get(SEARCH_URL, headers=headers, params=params, timeout=40)

    if response.status_code == 403:
        raise PermissionError(
            "403 Forbidden. Check that you subscribed to JSearch on RapidAPI and that RAPIDAPI_KEY is correct."
        )
    if response.status_code == 429:
        raise RuntimeError("429 Too Many Requests. Slow down, lower target, or wait for quota reset.")

    response.raise_for_status()
    payload = response.json()
    return payload.get("data", []) or []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=1000, help="Target number of unique jobs to collect in this run.")
    parser.add_argument("--pages-per-query", type=int, default=1, help="Use 1 for free plans to avoid page-limit issues.")
    parser.add_argument("--sleep", type=float, default=8.0, help="Seconds to wait between API calls. Use 6-10 for free plans.")
    parser.add_argument("--country", default="us")
    parser.add_argument("--out", default="data/raw/jsearch_jobs.jsonl")
    parser.add_argument("--append", action="store_true", help="Append to existing raw file and skip old duplicates.")
    parser.add_argument("--max-errors", type=int, default=20, help="Stop after this many API errors.")
    args = parser.parse_args()

    load_dotenv()
    rapidapi_key = os.getenv("RAPIDAPI_KEY")
    if not rapidapi_key or rapidapi_key == "your_rapidapi_key_here":
        raise RuntimeError("Missing RAPIDAPI_KEY. Create .env and add RAPIDAPI_KEY=your_key_here")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_keys = load_existing_keys(out_path) if args.append else set()
    seen = set(existing_keys)
    mode = "a" if args.append else "w"

    if args.append:
        print(f"Append mode: loaded {len(existing_keys):,} existing unique job keys.")
    else:
        print("Fresh mode: existing raw file will be overwritten. Use --append to reuse it.")

    collected_this_run = 0
    api_errors = 0
    total_calls = len(ROLES) * len(LOCATIONS) * args.pages_per_query

    with out_path.open(mode, encoding="utf-8") as f:
        pbar = tqdm(total=total_calls, desc="API calls")
        for role in ROLES:
            for location in LOCATIONS:
                query = f"{role} in {location}"
                for page in range(1, args.pages_per_query + 1):
                    if collected_this_run >= args.target:
                        pbar.close()
                        print(f"Done. New jobs collected this run: {collected_this_run:,}")
                        print(f"Raw file: {out_path}")
                        return

                    try:
                        jobs = list(search_jobs(query, page, rapidapi_key, country=args.country))
                    except PermissionError as exc:
                        pbar.close()
                        print(f"\n[STOP] {exc}", file=sys.stderr)
                        print("Open RapidAPI > JSearch > Endpoints > Test Endpoint. If it fails there too, the key/subscription is the issue.")
                        return
                    except Exception as exc:
                        api_errors += 1
                        print(f"\n[WARN] query={query!r}, page={page}: {exc}")
                        if api_errors >= args.max_errors:
                            pbar.close()
                            print(f"Stopped after {api_errors} API errors. New jobs collected: {collected_this_run:,}")
                            return
                        # Longer backoff on quota/rate issues.
                        time.sleep(max(args.sleep * 3, 20))
                        pbar.update(1)
                        continue

                    if not jobs:
                        pbar.update(1)
                        time.sleep(args.sleep)
                        break

                    added_from_call = 0
                    for job in jobs:
                        key = make_key_from_raw(job)
                        if key in seen:
                            continue
                        seen.add(key)
                        record = {
                            "source_api": "JSearch",
                            "ingested_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "search_role": role,
                            "search_location": location,
                            "search_query": query,
                            "raw": job,
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f.flush()
                        collected_this_run += 1
                        added_from_call += 1
                        if collected_this_run >= args.target:
                            break

                    pbar.set_postfix({"new": collected_this_run, "last_add": added_from_call})
                    pbar.update(1)
                    time.sleep(args.sleep)

        pbar.close()

    print(f"Finished all queries. New jobs collected this run: {collected_this_run:,}")
    print(f"Raw file: {out_path}")


if __name__ == "__main__":
    main()
