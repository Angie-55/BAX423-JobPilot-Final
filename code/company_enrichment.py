"""Cache-first company enrichment for JobPilot.

The app must work offline by default. Online providers are optional, and every
lookup result is cached so repeated runs do not spend credits on the same company.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import requests
try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - app still works if python-dotenv is unavailable
    load_dotenv = None

DATA_DIR = Path("data/processed")
DEFAULT_CACHE_PATH = DATA_DIR / "company_enrichment.csv"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)

CACHE_COLUMNS = [
    "company",
    "normalized_company",
    "company_size_bucket",
    "employee_count_estimate",
    "meets_100_employee_threshold",
    "meets_500_employee_threshold",
    "is_large_company",
    "is_possible_tiny_startup",
    "is_known_h1b_sponsor",
    "is_research_lab",
    "is_public_company",
    "source_url",
    "confidence",
    "notes",
    "last_checked",
]

BUCKETS = [
    (1, 10, "1-10"),
    (11, 50, "11-50"),
    (51, 99, "51-99"),
    (100, 499, "100-499"),
    (500, 999, "500-999"),
    (1000, float("inf"), "1000+"),
]

COMPANY_SUFFIX_PATTERN = re.compile(
    r"\b(?:services\s+llc|llc|incorporated|inc|ltd|limited|co|company|corp|corporation)\b\.?",
    flags=re.I,
)

NORMALIZED_COMPANY_OVERRIDES = {
    "amazon com": "amazon",
    "amazon com services": "amazon",
    "amazon com services llc": "amazon",
    "amazon services": "amazon",
    "amazoncom services": "amazon",
    "amazoncom services llc": "amazon",
    "amazon web services aws": "amazon",
    "salesforce slack": "salesforce",
    "slack": "salesforce",
    "zendesk": "zendesk",
    "zendesk inc": "zendesk",
    "snap inc": "snap",
    "redfin corporation": "redfin",
    "rocket companies inc": "rocket companies",
}

KNOWN_H1B_SPONSOR_HINTS = {
    "amazon", "google", "microsoft", "meta", "apple", "nvidia", "adobe", "oracle",
    "salesforce", "ibm", "intel", "cisco", "uber", "lyft", "netflix", "doordash",
    "capital one", "jpmorgan", "goldman", "bloomberg", "servicenow", "databricks",
    "snowflake", "openai", "anthropic", "hugging face", "united airlines",
    "rocket", "redfin", "geico", "expedia", "red hat", "paramount", "snap",
    "cribl", "zendesk", "rockstar", "rockstar games", "the hartford",
}

RESEARCH_LAB_TERMS = [
    "research lab", "ai lab", "labs", "university", "college", "research institute",
    "institute of technology", "national laboratory", "medical center",
]

PUBLIC_COMPANY_TERMS = ["public company", "publicly traded", "nasdaq", "nyse"]

TINY_COMPANY_TERMS = [
    "small team", "founding team", "seed-stage", "seed stage", "early-stage startup",
    "early stage startup", "stealth startup", "startup environment", "start-up environment",
]


def normalize_company_name(company: str) -> str:
    raw = str(company or "").lower().strip()
    text = raw
    text = text.replace("&", " and ")
    text = re.sub(r"\.com\b", " com", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = COMPANY_SUFFIX_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        text = re.sub(r"\s+", " ", raw).strip()
    text = NORMALIZED_COMPANY_OVERRIDES.get(text, text)
    if text.startswith("university of illinois"):
        return "university of illinois"
    return text


def enrichment_quality_key(row: pd.Series | Dict) -> Tuple[int, int, int, int]:
    bucket = str(row.get("company_size_bucket", "") or "").strip().lower()
    confidence = str(row.get("confidence", "") or "").strip().lower()
    confidence_rank = {"high": 3, "medium": 2, "low": 1}.get(confidence, 0)
    has_source = 1 if str(row.get("source_url", "") or "").strip() else 0
    notes_len = len(str(row.get("notes", "") or "").strip())
    return (
        1 if bucket and bucket != "unknown" else 0,
        confidence_rank,
        has_source,
        notes_len,
    )


def best_enrichment_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return empty_cache_df()
    out = df.copy()
    for col in CACHE_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out["normalized_company"] = out.apply(
        lambda row: normalize_company_name(row.get("company") or row.get("normalized_company")),
        axis=1,
    )
    out["_quality"] = out.apply(enrichment_quality_key, axis=1)
    out = (
        out.sort_values("_quality")
        .drop_duplicates("normalized_company", keep="last")
        .drop(columns=["_quality"])
        .reset_index(drop=True)
    )
    return out[CACHE_COLUMNS].copy()


def empty_cache_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CACHE_COLUMNS)


def load_company_enrichment_cache(path: str | Path = DEFAULT_CACHE_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        df = empty_cache_df()
        save_company_enrichment_cache(df, path)
        return df
    try:
        df = pd.read_csv(path).fillna("")
    except Exception:
        df = empty_cache_df()
    for col in CACHE_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return best_enrichment_rows(df[CACHE_COLUMNS])


def save_company_enrichment_cache(df: pd.DataFrame, path: str | Path = DEFAULT_CACHE_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for col in CACHE_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    best_enrichment_rows(out[CACHE_COLUMNS]).to_csv(path, index=False, encoding="utf-8-sig")


def get_cached_company_enrichment(company: str, cache_df: pd.DataFrame) -> Dict | None:
    normalized = normalize_company_name(company)
    if not normalized or cache_df.empty or "normalized_company" not in cache_df.columns:
        return None
    matches = cache_df[cache_df["normalized_company"].astype(str) == normalized]
    if matches.empty:
        return None
    return best_enrichment_rows(matches).iloc[-1].to_dict()


def parse_number(value) -> float | None:
    text = str(value or "").strip().replace(",", "").replace("$", "")
    if text.lower() in {"", "nan", "none", "null"}:
        return None
    try:
        return float(text)
    except Exception:
        return None


def bucket_for_count(employee_count: float | int | None) -> str:
    if employee_count is None:
        return "unknown"
    for low, high, bucket in BUCKETS:
        if low <= float(employee_count) <= high:
            return bucket
    return "unknown"


def threshold_from_bucket(bucket: str, threshold: int) -> bool | str:
    if bucket == "unknown":
        return ""
    if threshold == 100:
        return bucket in {"100-499", "500-999", "1000+"}
    if threshold == 500:
        return bucket in {"500-999", "1000+"}
    return ""


def bool_value(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def now_date() -> str:
    return datetime.now().date().isoformat()


def parse_employee_count_from_text(text: str) -> Tuple[float | None, str]:
    text_l = str(text or "").lower().replace(",", "")
    patterns = [
        r"\b(?:over|more than|above|at least)\s+(\d{2,6})\+?\s+(?:employees|people|staff|team members)\b",
        r"\b(\d{2,6})\+\s+(?:employees|people|staff|team members)\b",
        r"\b(\d{2,6})\s+(?:employees|people|staff|team members)\b",
        r"\bteam of\s+(\d{1,5})\b",
        r"\b(\d{1,5})[- ]person team\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text_l)
        if match:
            return float(match.group(1)), match.group(0)
    return None, ""


def build_enrichment_row(
    company: str,
    employee_count: float | int | None = None,
    source_url: str = "",
    confidence: str = "low",
    notes: str = "no reliable enrichment found",
    is_possible_tiny_startup: bool = False,
    is_known_h1b_sponsor: bool = False,
    is_research_lab: bool = False,
    is_public_company: bool = False,
) -> Dict:
    bucket = bucket_for_count(employee_count)
    return {
        "company": company,
        "normalized_company": normalize_company_name(company),
        "company_size_bucket": bucket,
        "employee_count_estimate": int(employee_count) if employee_count is not None else "",
        "meets_100_employee_threshold": threshold_from_bucket(bucket, 100),
        "meets_500_employee_threshold": threshold_from_bucket(bucket, 500),
        "is_large_company": bucket == "1000+" or (bucket == "500-999" and confidence == "high"),
        "is_possible_tiny_startup": bool(is_possible_tiny_startup or bucket in {"1-10", "11-50", "51-99"}),
        "is_known_h1b_sponsor": bool(is_known_h1b_sponsor),
        "is_research_lab": bool(is_research_lab),
        "is_public_company": bool(is_public_company),
        "source_url": source_url,
        "confidence": confidence,
        "notes": notes,
        "last_checked": now_date(),
    }


def update_company_enrichment_cache(company: str, result: Dict, cache_path: str | Path = DEFAULT_CACHE_PATH) -> pd.DataFrame:
    cache_df = load_company_enrichment_cache(cache_path)
    row = {col: result.get(col, "") for col in CACHE_COLUMNS}
    row["company"] = row.get("company") or company
    row["normalized_company"] = row.get("normalized_company") or normalize_company_name(company)
    row["last_checked"] = row.get("last_checked") or now_date()
    cache_df = cache_df[cache_df["normalized_company"].astype(str) != row["normalized_company"]]
    cache_df = pd.concat([cache_df, pd.DataFrame([row])], ignore_index=True)
    save_company_enrichment_cache(cache_df, cache_path)
    return cache_df


def infer_company_profile_from_job_text(row: pd.Series | Dict) -> Dict:
    company = str(row.get("company", "") or "")
    normalized = normalize_company_name(company)
    text = " ".join(
        str(row.get(col, "") or "")
        for col in ["company", "title", "description", "clean_job_text", "job_text"]
    ).lower()
    employee_count, evidence = parse_employee_count_from_text(text)
    is_possible_tiny = any(term in text for term in TINY_COMPANY_TERMS)
    is_known_sponsor = normalized in KNOWN_H1B_SPONSOR_HINTS
    is_research_lab = any(term in text for term in RESEARCH_LAB_TERMS)
    is_public = any(term in text for term in PUBLIC_COMPANY_TERMS)
    if employee_count is not None:
        return build_enrichment_row(
            company,
            employee_count=employee_count,
            confidence="medium",
            notes=evidence,
            is_possible_tiny_startup=is_possible_tiny,
            is_known_h1b_sponsor=is_known_sponsor,
            is_research_lab=is_research_lab,
            is_public_company=is_public,
        )
    if is_possible_tiny or is_known_sponsor or is_research_lab or is_public:
        return build_enrichment_row(
            company,
            confidence="medium",
            notes="company profile signal inferred from job text",
            is_possible_tiny_startup=is_possible_tiny,
            is_known_h1b_sponsor=is_known_sponsor,
            is_research_lab=is_research_lab,
            is_public_company=is_public,
        )
    return build_enrichment_row(company)


def parse_company_api_payload(company: str, payload: Dict, provider: str) -> Dict:
    def nested(data: Dict, path: str):
        cur = data
        for part in path.split("."):
            if not isinstance(cur, dict):
                return None
            cur = cur.get(part)
        return cur

    employee_count = (
        parse_number(payload.get("employee_count"))
        or parse_number(payload.get("employees"))
        or parse_number(payload.get("number_of_employees"))
        or parse_number(payload.get("size"))
        or parse_number(nested(payload, "about.totalEmployees"))
        or parse_number(nested(payload, "about.employeeCount"))
        or parse_number(nested(payload, "metrics.employees"))
    )
    source_url = str(
        payload.get("source_url")
        or payload.get("website")
        or payload.get("url")
        or nested(payload, "domain.domain")
        or nested(payload, "about.website")
        or ""
    )
    notes = f"{provider} enrichment"
    confidence = "high" if employee_count is not None else "low"
    text = str(payload).lower()
    normalized = normalize_company_name(company)
    return build_enrichment_row(
        company,
        employee_count=employee_count,
        source_url=source_url,
        confidence=confidence,
        notes=notes if employee_count is not None else "no reliable enrichment found",
        is_known_h1b_sponsor=normalized in KNOWN_H1B_SPONSOR_HINTS,
        is_research_lab=any(term in text for term in RESEARCH_LAB_TERMS),
        is_public_company=any(term in text for term in PUBLIC_COMPANY_TERMS),
    )


def enrich_company_online(company: str, provider: str = "companies_api") -> Dict:
    api_key = os.getenv("COMPANY_ENRICHMENT_API_KEY", "").strip()
    provider = (provider or "companies_api").strip().lower()
    if not api_key:
        return build_enrichment_row(company, notes="no company enrichment API key configured")

    try:
        if provider == "tavily_search":
            response = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": f"{company} employees company size H-1B sponsor research lab",
                    "search_depth": "basic",
                    "max_results": 5,
                },
                timeout=12,
            )
            response.raise_for_status()
            data = response.json()
            joined = " ".join(
                f"{item.get('title', '')} {item.get('content', '')} {item.get('url', '')}"
                for item in data.get("results", [])
            )
            inferred = infer_company_profile_from_job_text({"company": company, "description": joined})
            inferred["source_url"] = data.get("results", [{}])[0].get("url", "") if data.get("results") else ""
            inferred["confidence"] = "medium" if inferred["company_size_bucket"] != "unknown" else "low"
            inferred["notes"] = "tavily_search evidence" if inferred["company_size_bucket"] != "unknown" else "no reliable enrichment found"
            return inferred

        if provider == "companies_api":
            response = requests.get(
                "https://api.thecompaniesapi.com/v2/companies/by-name",
                params={"name": company, "size": 1, "exactWordsMatch": "false"},
                headers={"Authorization": f"Basic {api_key}"},
                timeout=12,
            )
            if response.status_code == 403 and "noCreditsRemaining" in response.text:
                result = build_enrichment_row(company, notes="online enrichment unavailable: no credits remaining")
                result["_stop_online"] = True
                return result
            response.raise_for_status()
            data = response.json()
            companies = data.get("companies", [])
            if not companies:
                return build_enrichment_row(company, notes="no reliable enrichment found")
            return parse_company_api_payload(company, companies[0], provider)

        base_url_env = "PEOPLE_DATA_LABS_BASE_URL" if provider == "people_data_labs" else "COMPANIES_API_BASE_URL"
        base_url = os.getenv(base_url_env, "").strip()
        if not base_url:
            return build_enrichment_row(company, notes=f"{provider} endpoint is not configured")
        response = requests.get(base_url, params={"q": company, "api_key": api_key}, timeout=12)
        response.raise_for_status()
        return parse_company_api_payload(company, response.json(), provider)
    except Exception as exc:
        return build_enrichment_row(company, notes=f"online enrichment failed: {exc.__class__.__name__}")


def merge_failed_online_with_offline(online_result: Dict, offline_result: Dict) -> Dict:
    if online_result.get("company_size_bucket") != "unknown":
        return online_result
    merged = dict(online_result)
    if offline_result.get("company_size_bucket") != "unknown":
        for col in [
            "company_size_bucket",
            "employee_count_estimate",
            "meets_100_employee_threshold",
            "meets_500_employee_threshold",
            "is_large_company",
            "is_possible_tiny_startup",
        ]:
            merged[col] = offline_result.get(col, merged.get(col, ""))
        merged["confidence"] = offline_result.get("confidence", merged.get("confidence", "low"))
    for col in ["is_known_h1b_sponsor", "is_research_lab", "is_public_company", "is_possible_tiny_startup"]:
        merged[col] = bool_value(merged.get(col, "")) or bool_value(offline_result.get(col, ""))
    online_notes = str(online_result.get("notes", "") or "")
    offline_notes = str(offline_result.get("notes", "") or "")
    if offline_notes and offline_notes != "no reliable enrichment found":
        merged["notes"] = f"{online_notes}; offline fallback: {offline_notes}".strip("; ")
        if merged.get("confidence") == "low":
            merged["confidence"] = offline_result.get("confidence", "medium")
    return merged


def enrich_companies_for_candidates(
    candidates_df: pd.DataFrame,
    online_enabled: bool = False,
    force_refresh: bool = False,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    provider: str | None = None,
) -> Tuple[pd.DataFrame, Dict]:
    cache_df = load_company_enrichment_cache(cache_path)
    provider = provider or os.getenv("COMPANY_ENRICHMENT_PROVIDER", "companies_api")
    api_key = os.getenv("COMPANY_ENRICHMENT_API_KEY", "").strip()
    status = {"companies_checked": 0, "cache_hits": 0, "api_calls_made": 0, "unknown_cached": 0, "errors": 0}
    online_available = bool(online_enabled and api_key)
    if candidates_df.empty or "company" not in candidates_df.columns:
        return cache_df, status

    for company in sorted({str(x).strip() for x in candidates_df["company"].dropna() if str(x).strip()}):
        status["companies_checked"] += 1
        cached = get_cached_company_enrichment(company, cache_df)
        if cached is not None and not force_refresh:
            status["cache_hits"] += 1
            continue

        rows = candidates_df[candidates_df["company"].astype(str) == company]
        offline_result = infer_company_profile_from_job_text(rows.iloc[0]) if not rows.empty else build_enrichment_row(company)

        if online_available:
            result = enrich_company_online(company, provider)
            status["api_calls_made"] += 1
            if str(result.get("notes", "")).startswith("online enrichment failed") or result.get("_stop_online"):
                status["errors"] += 1
                if result.get("_stop_online"):
                    online_available = False
            result = merge_failed_online_with_offline(result, offline_result)
        else:
            result = offline_result

        if result.get("company_size_bucket") == "unknown":
            status["unknown_cached"] += 1
        cache_df = update_company_enrichment_cache(company, result, cache_path)
    return cache_df, status


def merge_company_enrichment(jobs_df: pd.DataFrame, enrichment_df: pd.DataFrame) -> pd.DataFrame:
    out = jobs_df.copy()
    out["normalized_company"] = out["company"].apply(normalize_company_name) if "company" in out.columns else ""
    enrichment = enrichment_df.copy()
    for col in CACHE_COLUMNS:
        if col not in enrichment.columns:
            enrichment[col] = ""
    merged = out.merge(enrichment[CACHE_COLUMNS], on="normalized_company", how="left", suffixes=("", "_enriched"))
    for col in CACHE_COLUMNS:
        if col in {"company", "normalized_company"}:
            continue
        if col not in merged.columns:
            merged[col] = ""
    return merged


def parse_company_constraints(profile: Dict) -> Dict:
    text = " ".join(
        [str(profile.get("target_role", "")), str(profile.get("profile_summary", "")), str(profile.get("location_preference", ""))]
        + [str(x) for x in profile.get("dealbreakers", [])]
        + [str(x) for x in profile.get("preferred_industries", [])]
    ).lower()
    min_size = None
    size_match = re.search(r"(?:>=|at least|minimum|min|with)\s*(100|500)\+?\s*(?:employees|people|staff)", text)
    if size_match:
        min_size = int(size_match.group(1))
    if any(term in text for term in ["no companies with <100", "no companies under 100", "fewer than 100"]):
        min_size = 100
    if any(term in text for term in ["no companies with <500", "no companies under 500", "fewer than 500"]):
        min_size = 500
    return {
        "min_company_size": min_size,
        "no_tiny_startups": any(term in text for term in ["no tiny startups", "no startups", "no small companies", "no companies with <100", "no companies under 100", "fewer than 100"]),
        "prefer_large_companies": any(term in text for term in ["large companies", "large tech", "enterprise companies"]),
        "prefer_public_companies": "public companies" in text,
        "prefer_research_labs": "research labs" in text or "research lab" in text,
        "visa_sponsorship_required": any(term in text for term in ["visa sponsorship required", "h-1b sponsorship", "h1b sponsorship", "needs sponsorship"]),
        "known_sponsor_preferred": any(term in text for term in ["known h-1b sponsors", "known h1b sponsors", "known sponsors"]),
        "no_no_sponsor_companies": any(term in text for term in ["no companies that do not sponsor", "no companies that don't sponsor"]),
        "no_contract_or_temp": any(term in text for term in ["no contract", "no temp", "no temporary"]),
    }


def company_constraint_score(profile: Dict, row: pd.Series | Dict) -> Tuple[float, str]:
    constraints = parse_company_constraints(profile)
    score = 0.60
    notes = []
    bucket = str(row.get("company_size_bucket", "unknown") or "unknown")
    min_size = constraints["min_company_size"]
    if min_size:
        pass_key = "meets_500_employee_threshold" if min_size >= 500 else "meets_100_employee_threshold"
        threshold_value = row.get(pass_key, "")
        if bool_value(threshold_value):
            score = max(score, 1.0)
            notes.append("company size meets requested threshold")
        elif bucket == "unknown" or threshold_value == "":
            score = min(score, 0.48)
            notes.append("company size is unknown")
        else:
            score = 0.0
            notes.append(f"company appears below {min_size} employees")
    if constraints["no_tiny_startups"] and bool_value(row.get("is_possible_tiny_startup", "")):
        score = min(score, 0.20)
        notes.append("possible startup risk")
    if constraints["prefer_large_companies"] and bool_value(row.get("is_large_company", "")):
        score = max(score, 0.92)
        notes.append("large employer signal")
    if constraints["prefer_public_companies"] and bool_value(row.get("is_public_company", "")):
        score = max(score, 0.86)
        notes.append("public company signal")
    if constraints["prefer_research_labs"] and bool_value(row.get("is_research_lab", "")):
        score = max(score, 0.90)
        notes.append("research lab signal")
    if constraints["visa_sponsorship_required"] or constraints["known_sponsor_preferred"]:
        if bool_value(row.get("is_known_h1b_sponsor", "")) or bool_value(row.get("is_research_lab", "")) or bool_value(row.get("is_large_company", "")):
            score = max(score, 0.90)
            notes.append("likely sponsor-style employer")
        else:
            score = min(score, 0.50)
            notes.append("sponsorship signal is unknown")
    return max(0.0, min(1.0, score)), "; ".join(dict.fromkeys(notes)) or "no company constraint"


def apply_company_constraints(profile: Dict, candidates_df: pd.DataFrame) -> pd.DataFrame:
    out = candidates_df.copy()
    if out.empty:
        return out
    scores = []
    notes = []
    for _, row in out.iterrows():
        score, note = company_constraint_score(profile, row)
        scores.append(round(score, 4))
        notes.append(note)
    out["company_constraint_score"] = scores
    out["company_constraint_notes"] = notes
    return out
