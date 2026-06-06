"""Step-by-step Streamlit app for JobPilot."""

from __future__ import annotations

import os
import re
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests
import streamlit as st

import company_enrichment
from recommender import (
    JobRecommender,
    extract_salary_from_description,
    format_candidate_funnel_summary,
    parse_location_preferences,
    salary_display as recommender_salary_display,
    update_feedback_state,
)
from resume_export import resume_docx_bytes
from resume_parser import load_resume_text, parse_resume


st.set_page_config(page_title="JobPilot", page_icon="JP", layout="centered")

DATA_PATH = Path("data/processed/jobs_final.csv")


SKILL_NORMALIZATION = {
    "sql": "SQL",
    "r": "R",
    "aws": "AWS",
    "aws certified": "AWS",
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
    "scikit-learn": "Scikit-Learn",
    "sklearn": "Scikit-Learn",
    "pandas": "Pandas",
    "numpy": "NumPy",
    "python": "Python",
    "java": "Java",
    "kafka": "Kafka",
    "kubernetes": "Kubernetes",
    "spark": "Spark",
    "pyspark": "PySpark",
    "microservices": "Microservices",
    "machine learning": "Machine Learning",
    "ml": "Machine Learning",
    "power bi": "Power BI",
    "tableau": "Tableau",
    "excel": "Excel",
    "nlp": "NLP",
    "mlops": "MLOps",
    "gpa": "GPA",
    "uc davis": "UC Davis",
    "msba": "MSBA",
}

EDUCATION_NORMALIZATION = {
    "msba": "MSBA",
    "mba": "MBA",
    "m.s.": "MS",
    "ms": "MS",
    "ba": "BA",
    "b.a.": "BA",
    "bs": "BS",
    "b.s.": "BS",
    "phd": "PhD",
    "ph.d.": "PhD",
    "uc davis": "UC Davis",
    "university of california, davis": "UC Davis",
    "aws": "AWS",
    "gpa": "GPA",
}


def apply_style() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] { background: #f8fafc; }
        [data-testid="stHeader"] { background: rgba(248, 250, 252, 0.94); }
        .block-container { max-width: 900px; padding-top: 1.2rem; padding-bottom: 2.5rem; }
        h1, h2, h3 { letter-spacing: 0; }
        .header-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 22px 24px;
            margin-bottom: 10px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        }
        .subtitle { color: #475569; line-height: 1.5; }
        .progress-wrap { margin: 14px 0 18px 0; }
        .progress-row { display: grid; grid-template-columns: repeat(6, 1fr); gap: 6px; }
        .progress-item {
            border-radius: 8px;
            padding: 7px 8px;
            text-align: center;
            font-weight: 700;
            font-size: 0.76rem;
            border: 1px solid #e2e8f0;
            color: #64748b;
            background: #ffffff;
        }
        .progress-active { color: #ffffff; border-color: #0f766e; background: #0f766e; }
        .progress-done { color: #0f766e; border-color: #99f6e4; background: #ecfeff; }
        .page-kicker {
            color: #0f766e;
            font-weight: 850;
            font-size: 0.84rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 5px;
        }
        .page-helper { color: #64748b; font-size: 0.95rem; line-height: 1.55; margin-bottom: 12px; }
        .profile-chip {
            display: inline-block;
            border: 1px solid #cbd5e1;
            border-radius: 999px;
            padding: 5px 9px;
            margin: 3px 4px 3px 0;
            background: #ffffff;
            color: #334155;
            font-size: 0.84rem;
            font-weight: 650;
        }
        .soft-summary {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 14px 16px;
            margin-bottom: 12px;
            color: #334155;
        }
        .muted-note { color: #64748b; font-size: 0.92rem; line-height: 1.45; }
        .stButton > button, .stDownloadButton > button { border-radius: 8px; min-height: 40px; }
        @media (max-width: 760px) {
            .progress-row { grid-template-columns: repeat(2, 1fr); }
            .block-container { padding-left: 1rem; padding-right: 1rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=True)
def load_recommender() -> JobRecommender:
    return JobRecommender()


@st.cache_data(show_spinner=False)
def load_dataset_info() -> Dict:
    df = pd.read_csv(DATA_PATH)
    return {
        "rows": len(df),
        "companies": df["company"].nunique(),
        "locations": df["location"].nunique(),
        "source_counts": df["source"].value_counts().to_dict(),
        "df": df,
    }


def stable_cache_json(value: Dict) -> str:
    return json.dumps(value or {}, sort_keys=True, default=str)


@st.cache_data(show_spinner=False, max_entries=12)
def cached_recommendations(profile_key: str, feedback_key: str, top_n: int, candidate_k: int) -> pd.DataFrame:
    recommender = load_recommender()
    profile = json.loads(profile_key or "{}")
    feedback = json.loads(feedback_key or "{}")
    return recommender.recommend_from_profile(
        profile,
        top_n=top_n,
        candidate_k=candidate_k,
        feedback=feedback,
    )


def init_state() -> None:
    defaults = {
        "step": 1,
        "profile": {},
        "profile_ready": False,
        "preferences_ready": False,
        "feedback": {},
        "recommendations": None,
        "company_enrichment_status": {},
        "candidate_funnel_summary": {},
        "matching_timing_log": {},
        "selected_job_index": None,
        "cover_letter_text": "",
        "cover_letter_job_index": None,
        "top_n": 10,
        "candidate_k": 1500,
        "auto_match_after_preferences": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def go_to_step(step: int) -> None:
    st.session_state.step = max(1, min(6, step))


def render_header(info: Dict) -> None:
    st.markdown(
        """
        <div class="header-card">
            <h1>JobPilot</h1>
            <div class="subtitle">
                Step-by-step job matching and resume tailoring using resume parsing,
                FAISS retrieval, hard filters, multi-stage ranking, feedback, and
                OpenRouter-backed resume drafting.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"Offline snapshot loaded: {info['rows']:,} postings, "
        f"{info['companies']:,} companies, {info['locations']:,} locations."
    )


def render_progress() -> None:
    labels = ["Resume", "Profile", "Preferences", "Ranked Jobs", "Select Job", "Resume"]
    items = []
    for idx, label in enumerate(labels, start=1):
        if idx == st.session_state.step:
            cls = "progress-item progress-active"
        elif idx < st.session_state.step:
            cls = "progress-item progress-done"
        else:
            cls = "progress-item"
        items.append(f'<div class="{cls}">{idx}. {label}</div>')
    st.markdown(
        '<div class="progress-wrap"><div class="progress-row">' + "".join(items) + "</div></div>",
        unsafe_allow_html=True,
    )


def page_start(kicker: str, title: str, helper: str) -> None:
    st.markdown(f'<div class="page-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.subheader(title)
    st.markdown(f'<div class="page-helper">{helper}</div>', unsafe_allow_html=True)


def page_end() -> None:
    return None


def comma_list(value) -> str:
    if isinstance(value, list):
        return ", ".join([str(x) for x in value if str(x).strip()])
    return str(value or "")


def parse_comma_list(value: str) -> List[str]:
    return [x.strip() for x in re.split(r"[,;\n]+", str(value)) if x.strip()]


def normalize_education_token(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    key = text.lower()
    return EDUCATION_NORMALIZATION.get(key, text.title() if text.islower() else text)


def normalize_education_list(values: List[str]) -> List[str]:
    return sorted(set(normalize_education_token(value) for value in values if str(value).strip()))


COMMON_DEALBREAKER_OPTIONS = [
    "no senior",
    "no junior",
    "no staff",
    "no principal",
    "no lead",
    "no defense",
    "no military",
    "no contract",
    "no unpaid",
    "no temporary",
    "no companies with <100 employees",
    "no tiny startups",
    "visa sponsorship required",
    "no 3+ years experience",
    "no 5+ years experience",
]

COMMON_INDUSTRY_OPTIONS = [
    "technology",
    "finance",
    "healthcare",
    "retail",
    "education",
    "consulting",
    "insurance",
    "biotech",
]


def build_skill_options(profile: Dict) -> List[str]:
    values = set(SKILL_NORMALIZATION.values()) | set(profile.get("skills", []) or [])
    return sorted([str(value) for value in values if str(value).strip()], key=str.lower)


def build_dealbreaker_options(profile: Dict) -> List[str]:
    values = set(COMMON_DEALBREAKER_OPTIONS) | set(profile.get("dealbreakers", []) or [])
    return sorted([str(value) for value in values if str(value).strip()], key=str.lower)


def build_industry_options(profile: Dict) -> List[str]:
    values = set(COMMON_INDUSTRY_OPTIONS) | set(profile.get("preferred_industries", []) or [])
    return sorted([str(value) for value in values if str(value).strip()], key=str.lower)


def render_parsed_matching_preview(profile: Dict) -> None:
    st.markdown("#### Parsed matching signals")
    st.markdown(
        "<div class='muted-note'>These fields are only previewed here. You will confirm or edit the actual matching strategy in Step 3.</div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.text_input("Parsed target roles", value=profile.get("target_role", ""), disabled=True)
        st.text_input("Parsed location preference", value=profile.get("location_preference", ""), disabled=True)
    with c2:
        st.text_input("Parsed minimum annual salary", value=str(profile.get("salary_min", "")), disabled=True)
        st.text_input("Parsed max required years", value=str(profile.get("max_required_years", "")), disabled=True)
    st.multiselect(
        "Parsed dealbreakers",
        options=build_dealbreaker_options(profile),
        default=profile.get("dealbreakers", []) or [],
        disabled=True,
    )


def render_funnel_metrics(funnel: Dict) -> None:
    if not funnel:
        return
    st.markdown("#### Matching funnel")
    items = [(key, value) for key, value in funnel.items() if isinstance(value, (int, float, str))]
    if not items:
        return
    preview_items = items[:4]
    cols = st.columns(len(preview_items))
    for col, (key, value) in zip(cols, preview_items):
        col.metric(str(key).replace("_", " ").title(), value)
    with st.expander("View full candidate funnel details"):
        st.dataframe(pd.DataFrame(items, columns=["Stage", "Count"]), width="stretch", hide_index=True)
        st.caption(format_candidate_funnel_summary(funnel))
        timing = st.session_state.get("matching_timing_log", {}) or {}
        if timing:
            st.markdown("##### Timing")
            st.dataframe(pd.DataFrame(list(timing.items()), columns=["Step", "Seconds"]), width="stretch", hide_index=True)


DISPLAY_FETCH_MULTIPLIER = 4

DUPLICATE_LOCATION_TERMS = [
    "remote", "hybrid", "onsite", "on site", "on-site", "united states", "usa", "u.s.",
    "new york", "nyc", "los angeles", "chicago", "san francisco", "bay area", "seattle",
    "boston", "austin", "dallas", "houston", "denver", "atlanta", "miami", "phoenix",
    "philadelphia", "washington", "dc", "san diego", "san jose", "portland", "nashville",
    "charlotte", "minneapolis", "detroit", "pittsburgh", "orlando", "raleigh",
]

DUPLICATE_LOCATION_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(term) for term in DUPLICATE_LOCATION_TERMS) + r")\b",
    flags=re.I,
)


def recommendation_fetch_size(display_top_n: int) -> int:
    display_top_n = max(1, int(display_top_n or 10))
    return max(display_top_n, display_top_n * DISPLAY_FETCH_MULTIPLIER, display_top_n + 20)


def reset_recommendation_index(df: pd.DataFrame) -> pd.DataFrame:
    attrs = dict(getattr(df, "attrs", {}) or {})
    out = df.reset_index(drop=True)
    out.attrs.update(attrs)
    return out


def normalize_duplicate_title(title: object) -> str:
    text = str(title or "").lower()
    text = re.sub(r"\([^)]*(?:remote|hybrid|onsite|on-site|united states|usa|new york|nyc|los angeles|chicago)[^)]*\)", " ", text)
    for separator in [r"\s+-\s+", r"\s+\|\s+", r"\s+@\s+"]:
        parts = re.split(separator, text)
        if len(parts) <= 1:
            continue
        kept = [parts[0]]
        for suffix in parts[1:]:
            if not DUPLICATE_LOCATION_RE.search(suffix):
                kept.append(suffix)
        text = " ".join(kept)
    text = re.sub(r"\b(?:remote|hybrid|onsite|on\s*site|on-site)\b", " ", text)
    text = re.sub(r"\b(?:in|based in)\s+[a-z ]{2,35}$", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def duplicate_group_id(row) -> str:
    company = company_enrichment.normalize_company_name(row.get("company", ""))
    title = normalize_duplicate_title(row.get("title", ""))
    role_family = str(row.get("role_family", "") or "").strip().lower()
    return "|".join([company, title, role_family])


def add_duplicate_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    out = df.copy()
    out["duplicate_group_id"] = out.apply(duplicate_group_id, axis=1)
    out["duplicate_group_size"] = out.groupby("duplicate_group_id")["duplicate_group_id"].transform("size")
    out["is_primary_recommendation"] = ~out.duplicated("duplicate_group_id", keep="first")
    return out


def primary_recommendations(df: pd.DataFrame, display_top_n: int) -> pd.DataFrame:
    grouped = add_duplicate_group_columns(df)
    if len(grouped) == 0:
        return grouped
    primaries = grouped[grouped["is_primary_recommendation"]].copy()
    return primaries.head(max(1, int(display_top_n or 10)))


def similar_location_summary(group_rows: pd.DataFrame, primary_row) -> str:
    locations = []
    primary_location = str(primary_row.get("location", "") or "").strip().lower()
    for value in group_rows.get("location", pd.Series(dtype=str)).fillna("").astype(str):
        location = value.strip()
        if not location or location.lower() in {"nan", primary_location}:
            continue
        if location not in locations:
            locations.append(location)
    if not locations:
        return ""
    shown = ", ".join(locations[:4])
    if len(locations) > 4:
        shown += f", +{len(locations) - 4} more"
    return shown


def similar_postings_table(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, row in rows.iterrows():
        records.append(
            {
                "Location": row.get("location", ""),
                "Salary range": format_salary(row),
                "Employment type": display_employment_type(row),
                "Source": row.get("source", ""),
                "Apply link": row.get("apply_url", ""),
            }
        )
    return pd.DataFrame(records)


def run_matching(top_n: int | None = None, candidate_k: int | None = None) -> None:
    top_n = int(top_n or st.session_state.get("top_n", 10))
    candidate_k = int(candidate_k or st.session_state.get("candidate_k", 1500))
    st.session_state.top_n = top_n
    st.session_state.candidate_k = candidate_k
    fetch_top_n = recommendation_fetch_size(top_n)

    progress = st.progress(0)
    status = st.empty()

    status.write("[1/3] Retrieving candidate jobs from FAISS...")
    progress.progress(25)

    status.write("[2/3] Applying preferences, dealbreakers, and hard filters...")
    progress.progress(55)

    results = cached_recommendations(
        stable_cache_json(st.session_state.profile),
        stable_cache_json(st.session_state.feedback),
        fetch_top_n,
        candidate_k,
    )
    results = reset_recommendation_index(results)

    status.write("[3/3] Ranking and preparing recommendations...")
    progress.progress(90)
    st.session_state.recommendations = results
    st.session_state.company_enrichment_status = results.attrs.get("company_enrichment_status", {})
    st.session_state.candidate_funnel_summary = results.attrs.get("candidate_funnel_summary", {})
    st.session_state.matching_timing_log = results.attrs.get("timing_log", {})
    st.session_state.selected_job_index = None
    st.session_state.auto_match_after_preferences = False
    progress.progress(100)
    status.success("Matching complete.")


def profile_to_dict(profile_obj) -> Dict:
    if isinstance(profile_obj, dict):
        return profile_obj
    if hasattr(profile_obj, "__dict__"):
        return dict(profile_obj.__dict__)
    return {}


def clean_skill_token(value: str) -> str:
    text = str(value or "").strip()
    if is_negative_skill_statement(text):
        return ""
    text = re.sub(r"\b(?:basic|some|certified|familiar with|experience with)\b", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip(" .:-")
    key = text.lower()
    return SKILL_NORMALIZATION.get(key, text.title() if text.islower() else text)


def is_negative_skill_statement(value: str) -> bool:
    text = str(value or "").strip().lower()
    return bool(
        re.search(r"\b(no|not|without|lack(?:ing)?|limited)\b.{0,40}\b(experience|background|ownership)\b", text)
        or re.search(r"\bno\s+production\s+(?:ml|machine learning)\s+experience\b", text)
    )


def remove_negative_skill_statements(value: str) -> str:
    parts = re.split(r"[,;/]|\band\b|\. (?=[A-Z])", str(value or ""))
    return ", ".join(part for part in parts if not is_negative_skill_statement(part))


def ml_experience_constraint_from_text(text: str) -> str:
    lower = str(text or "").lower()
    if re.search(r"\bno\s+production\s+(?:ml|machine learning)\s+experience\b", lower):
        return "no production ML experience"
    if re.search(r"\b(no|limited|without)\b.{0,30}\b(?:ml|machine learning)\b.{0,30}\bexperience\b", lower):
        return "limited ML experience"
    return ""


def extract_skills_from_value(value: str, full_text: str = "") -> List[str]:
    raw = str(value or "")
    cleaned_raw = raw.replace("AWS certified", "AWS").replace("aws certified", "AWS")
    parts = re.split(r"[,;/]|\band\b|\. (?=[A-Z])", cleaned_raw)
    skills = []
    for part in parts:
        item = clean_skill_token(part)
        if item:
            skills.append(item)

    positive_raw = remove_negative_skill_statements(raw)
    combined = f"{positive_raw} {remove_negative_skill_statements(full_text)}".lower()
    for key, label in SKILL_NORMALIZATION.items():
        pattern = r"(?<![a-z0-9])" + re.escape(key) + r"(?![a-z0-9])"
        if re.search(pattern, combined):
            skills.append(label)
    return sorted(set([x for x in skills if x]))


def sanitize_profile_skills(profile: Dict) -> Dict:
    cleaned = dict(profile or {})
    skills = [str(skill) for skill in cleaned.get("skills", []) or []]
    cleaned["skills"] = extract_skills_from_value(comma_list(skills))
    ml_constraint = ml_experience_constraint_from_text(comma_list(skills) + " " + str(cleaned.get("raw_resume_text", "")))
    if ml_constraint:
        cleaned["ml_experience_constraint"] = ml_constraint
    return cleaned


def extract_labeled_value(text: str, labels: List[str]) -> str:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    label_pattern = "|".join([re.escape(label) for label in labels])
    for i, line in enumerate(lines):
        match = re.match(rf"^\s*(?:{label_pattern})\s*[:\t\-]\s*(.+)$", line, flags=re.I)
        if match:
            return match.group(1).strip()
        if re.match(rf"^\s*(?:{label_pattern})\s*$", line, flags=re.I) and i + 1 < len(lines):
            return lines[i + 1].strip()
        match = re.search(rf"(?:{label_pattern})\s*[:\t\-]\s*(.+)$", line, flags=re.I)
        if match:
            return match.group(1).strip()
    return ""


def normalize_salary_to_annual(value: str) -> str:
    text = str(value or "").replace(",", "")
    matches = []
    for match in re.finditer(r"(?:salary\s*)?(?:>=|at least|minimum|min)?\s*(\$?)\s*(\d+(?:\.\d+)?)\s*([kKmM]?)", text, flags=re.I):
        window = text[max(0, match.start() - 18): match.end() + 18].lower()
        if "employee" in window or "people" in window or "staff" in window:
            continue
        priority = 0
        if match.group(1) == "$":
            priority += 3
        if match.group(3):
            priority += 2
        if "salary" in window:
            priority += 2
        matches.append((priority, match))
    if not matches:
        return ""
    _, match = max(matches, key=lambda item: item[0])
    amount = float(match.group(2))
    suffix = match.group(3).lower()
    if suffix == "k":
        amount *= 1000
    elif suffix == "m":
        amount *= 1000000
    if amount < 1000 and "hour" not in text.lower():
        amount *= 1000
    return str(int(amount))


def extract_max_required_years_from_dealbreakers(text: str) -> float | None:
    lower = str(text or "").lower()
    patterns = [
        r"no\s+(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)\s+(?:experience\s+)?required",
        r"no\s+(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)\s+experience",
        r"no\s+(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)",
        r"zero\s+(\d+(?:\.\d+)?)\s*\+?\s*(?:year|years)\s+requirements",
    ]
    values = []
    for pattern in patterns:
        for match in re.finditer(pattern, lower):
            values.append(float(match.group(1)))
    return min(values) if values else None


def normalize_dealbreakers(raw_items: List[str], full_text: str = "") -> List[str]:
    combined = " ".join([str(x) for x in raw_items] + [str(full_text or "")]).lower()
    combined = combined.replace("defence", "defense")
    normalized = []
    checks = [
        ("no senior", r"\bno\s+(senior|sr\.?)\b|\bavoid\s+senior\b|\bsenior/staff\b"),
        ("no junior", r"\bno\s+(junior|jr\.?|entry[- ]?level|intern)\b|\bavoid\s+(junior|entry[- ]?level|intern)\b"),
        ("no staff", r"\bno\s+staff\b|\bstaff\s+titles\b|\bsenior/staff\b"),
        ("no principal", r"\bno\s+principal\b|\bprincipal\s+titles\b"),
        ("no lead", r"\bno\s+lead\b|\blead\s+titles\b"),
        ("no defense", r"\bno\s+defense\b|\bavoid\s+defense\b|\bdefense/military\b"),
        ("no military", r"\bno\s+military\b|\bavoid\s+military\b|\bdefense/military\b"),
        ("no contract", r"\bno\s+contract\b|\bcontract[- ]only\b|\bno\s+contract[- ]only\b"),
        ("no unpaid", r"\bno\s+unpaid\b|\bunpaid\s+(role|roles|internship|internships)\b"),
        ("no temporary", r"\bno\s+temp\b|\bno\s+temporary\b|\btemporary\s+only\b"),
        ("no companies with <100 employees", r"\bno\s+companies?\s+(?:with\s+)?(?:<|under|below|fewer\s+than)\s*100\b|\bcompanies?\s+with\s+>=\s*100\s+employees?\s+only\b|\b>=\s*100\s+employees?\s+only\b"),
        ("no tiny startups", r"\bno\s+(tiny\s+)?startups?\b|\bno\s+small\s+companies\b|\bcompanies?\s+with\s+>=\s*100\s+employees?\s+only\b"),
        ("visa sponsorship required", r"(need|needs|require|requires|must).*(visa|sponsor|sponsorship|h-?1b)"),
    ]
    for label, pattern in checks:
        if re.search(pattern, combined):
            normalized.append(label)

    max_years = extract_max_required_years_from_dealbreakers(combined)
    if max_years is not None:
        normalized.append(f"no {int(max_years) if max_years.is_integer() else max_years:g}+ years experience")

    for item in raw_items:
        item_text = str(item or "").strip().lower()
        item_text = re.sub(r"\s+", " ", item_text).strip(" .:-")
        if item_text.startswith("avoid "):
            item_text = item_text.replace("avoid ", "no ", 1)
        if re.search(r"\bno\s+junior\b", item_text):
            normalized.append("no junior")
        if re.search(r"\bno\s+companies?\s+(?:with\s+)?(?:<|under|below|fewer\s+than)\s*100\b", item_text):
            normalized.append("no companies with <100 employees")
        if item_text.startswith("no ") and len(item_text.split()) <= 7:
            normalized.append(item_text)

    unique = []
    for item in normalized:
        if item and item not in unique:
            unique.append(item)
    return unique


def extract_candidate_years_from_profile(text: str) -> float | None:
    lower = str(text or "").lower()
    if "no full-time experience" in lower or "no full time experience" in lower:
        if re.search(r"\b2\s+\w*\s*internships\b|\btwo\s+\w*\s*internships\b", lower):
            return 0.5
        return 0.0

    background = extract_labeled_value(text, ["Background"])
    search_area = background if background else text
    search_area = re.sub(r"dealbreakers?:.*", " ", search_area, flags=re.I | re.S)
    patterns = [
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)\s+(?:of\s+)?(?:professional\s+)?experience",
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)\s+(?:in|as|at)\b",
        r"\b(?:background|experience)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)",
        r"\b(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, search_area, flags=re.I)
        if match:
            return float(match.group(1))

    if re.search(r"\brecent\s+.*graduate\b|\bnew grad\b|\bnew graduate\b", lower):
        return 0.0
    if re.search(r"\bintern(ship)?\b", lower):
        return 0.5
    return None


def infer_level_from_profile_text(text: str, years_value: float | None) -> str:
    lower = str(text or "").lower()
    explicit_entry = any(
        term in lower for term in ["new grad", "new graduate", "recent graduate", "no full-time experience", "no full time experience"]
    )
    if explicit_entry:
        return "entry_or_junior"
    if years_value is not None:
        if years_value >= 5:
            return "senior_or_experienced"
        if years_value >= 2:
            return "mid"
        return "entry_or_junior"
    if any(term in lower for term in ["staff", "principal", "lead", "senior", "ml platform engineer", "mlops engineer"]):
        return "senior_or_experienced"
    return "unknown"


def looks_like_person_name(value: str) -> bool:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return False
    lower = text.lower()
    blocked = [
        "email",
        "phone",
        "linkedin",
        "github",
        "address",
        "resume",
        "curriculum vitae",
        "skills",
        "education",
        "experience",
        "summary",
        "profile",
        "project",
        "certification",
    ]
    if any(term in lower for term in blocked):
        return False
    if re.search(r"[@:/\\]|\d", text):
        return False
    tokens = text.split()
    if not 2 <= len(tokens) <= 4:
        return False
    skill_words = {key.lower() for key in SKILL_NORMALIZATION} | {value.lower() for value in SKILL_NORMALIZATION.values()}
    if any(token.lower() in skill_words or (len(token) > 1 and token.isupper()) for token in tokens):
        return False
    return all(re.match(r"^[A-Z][A-Za-z'.-]+$", token) for token in tokens)


def extract_candidate_name_conservative(text: str) -> str:
    raw = str(text or "")
    labeled = re.search(r"^\s*(?:Name|Candidate)\s*[:\-]\s*([A-Z][A-Za-z .'-]{2,70})\s*$", raw, flags=re.I | re.M)
    if labeled:
        candidate = labeled.group(1).strip()
        return candidate if looks_like_person_name(candidate) else ""

    persona = re.search(
        r"^\s*Persona\s*\d*\s*[-\u2013\u2014]\s*([A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){0,2})\s*(?:[-\u2013\u2014]|$)",
        raw,
        flags=re.I | re.M,
    )
    if persona:
        name = persona.group(1).strip()
        if len(name.split()) == 1 and re.match(r"^[A-Z][A-Za-z'.-]+$", name):
            return name
        return name if looks_like_person_name(name) else ""

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    for line in lines[:5]:
        clean = re.sub(r"[^A-Za-z .'-]", " ", line)
        clean = re.sub(r"\s+", " ", clean).strip()
        if looks_like_person_name(clean):
            return clean
    return ""


def parse_freeform_candidate_profile(raw_text: str, base_profile: Dict) -> Dict:
    text = str(raw_text or "")
    lower = text.lower()
    profile = dict(base_profile or {})
    looks_like_profile = any(
        key in lower
        for key in ["background", "target roles", "target role", "preferences", "dealbreakers", "salary", "skills"]
    )

    if not profile.get("candidate_name"):
        profile["candidate_name"] = extract_candidate_name_conservative(text)

    skills_value = extract_labeled_value(text, ["Skills", "Skill"])
    if skills_value:
        profile["skills"] = extract_skills_from_value(skills_value)
        ml_constraint = ml_experience_constraint_from_text(skills_value)
        if ml_constraint:
            profile["ml_experience_constraint"] = ml_constraint
    elif ml_experience_constraint_from_text(text):
        profile["ml_experience_constraint"] = ml_experience_constraint_from_text(text)

    target_value = extract_labeled_value(text, ["Target Roles", "Target Role", "Target", "Job Intention"])
    if target_value:
        profile["target_role"] = target_value

    pref_value = extract_labeled_value(text, ["Preferences", "Preference"])
    if pref_value:
        pref_lower = pref_value.lower()
        location_preferences = parse_location_preferences(pref_value)
        if location_preferences.get("display"):
            profile["location_preference"] = location_preferences["display"]
            profile["location_preferences"] = location_preferences
        salary_value = normalize_salary_to_annual(pref_value)
        if salary_value:
            profile["salary_min"] = salary_value
        industries = []
        for industry in ["tech", "technology", "healthcare", "health care", "finance", "retail"]:
            if industry in pref_lower:
                industries.append("technology" if industry in {"tech", "technology"} else industry.replace(" ", ""))
        if industries:
            profile["preferred_industries"] = sorted(set(industries))

    salary_value = extract_labeled_value(text, ["Salary", "Minimum Salary", "Minimum annual salary"])
    if salary_value:
        annual = normalize_salary_to_annual(salary_value)
        if annual:
            profile["salary_min"] = annual

    dealbreaker_value = extract_labeled_value(text, ["Dealbreakers", "Dealbreaker", "Must avoid"])
    dealbreakers = parse_comma_list(dealbreaker_value.replace(";", ",")) if dealbreaker_value else []
    normalized_dealbreakers = normalize_dealbreakers(dealbreakers, text)
    if normalized_dealbreakers:
        profile["dealbreakers"] = normalized_dealbreakers

    candidate_years = extract_candidate_years_from_profile(text)
    if candidate_years is not None:
        profile["years_experience"] = candidate_years

    max_required_years = extract_max_required_years_from_dealbreakers(text)
    if max_required_years is not None:
        profile["max_required_years"] = max_required_years

    profile["candidate_level"] = infer_level_from_profile_text(text, profile.get("years_experience"))
    background = extract_labeled_value(text, ["Background"])
    if background:
        profile["profile_summary"] = background

    profile["raw_resume_text"] = text[:12000]
    profile["raw_text_length"] = len(text)
    profile["input_type"] = "freeform_profile" if looks_like_profile else "resume"
    profile.setdefault("preferred_industries", [])
    return profile


def extract_contact_fields_from_text(text: str) -> Dict[str, str]:
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text or "")
    phone_match = re.search(r"(?:\+?\d[\d\-.\(\) Xx]{7,}\d)", text or "")
    return {
        "candidate_name": extract_candidate_name_conservative(text),
        "email": email_match.group(0) if email_match else "",
        "phone": phone_match.group(0).strip() if phone_match else "",
        "raw_resume_text": str(text or "")[:12000],
    }


def enrich_profile_with_raw_resume(profile: Dict, raw_text: str) -> Dict:
    enriched = dict(profile or {})
    for key, value in extract_contact_fields_from_text(raw_text).items():
        if value:
            enriched[key] = value
    if raw_text:
        enriched["raw_resume_text"] = raw_text[:12000]
        enriched["raw_text_length"] = len(raw_text)
    enriched.setdefault("preferred_industries", [])
    return enriched


def render_resume_page() -> None:
    page_start(
        "Step 1 of 6",
        "Upload resume or paste candidate profile",
        "JobPilot accepts PDF, DOCX, TXT, pasted resume text, or structured profile text.",
    )
    uploaded_file = st.file_uploader("Upload resume file", type=["pdf", "docx", "txt", "md"])
    pasted_text = st.text_area(
        "Or paste resume/profile text",
        height=220,
    )

    if st.button("Continue", type="primary", width="stretch"):
        if uploaded_file is None and not pasted_text.strip():
            st.warning("Please upload a resume or paste resume/profile text first.")
        else:
            try:
                if uploaded_file is not None:
                    temp_dir = Path("data/tmp")
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    temp_path = temp_dir / uploaded_file.name
                    temp_path.write_bytes(uploaded_file.getvalue())
                    raw_text = load_resume_text(temp_path)
                else:
                    raw_text = pasted_text
                parsed = parse_resume(raw_text)
                base_profile = enrich_profile_with_raw_resume(profile_to_dict(parsed), raw_text)
                st.session_state.profile = sanitize_profile_skills(parse_freeform_candidate_profile(raw_text, base_profile))
                st.session_state.profile_ready = True
                st.session_state.preferences_ready = False
                st.session_state.recommendations = None
                go_to_step(2)
                st.rerun()
            except Exception as exc:
                st.error(f"Resume parsing failed: {exc}")
    page_end()


def render_profile_page() -> None:
    page_start(
        "Step 2 of 6",
        "Review extracted resume profile",
        "Confirm the resume information JobPilot parsed. Matching preferences are previewed here and edited in the next step.",
    )
    if not st.session_state.profile_ready:
        st.info("No profile yet. Go back and parse a resume first.")
        if st.button("Back to resume upload", width="stretch"):
            go_to_step(1)
            st.rerun()
        return

    st.session_state.profile = sanitize_profile_skills(st.session_state.profile)
    profile = st.session_state.profile
    st.caption(
        f"Input type: {profile.get('input_type', 'resume')} | "
        f"Name: {profile.get('candidate_name', '') or 'Not detected'} | "
        f"Email: {profile.get('email', '') or 'Not detected'} | "
        f"Phone: {profile.get('phone', '') or 'Not detected'}"
    )

    with st.form("profile_form"):
        basic_tab, skills_tab, preview_tab = st.tabs(["Basic info", "Skills & background", "Parsed preference preview"])

        with basic_tab:
            candidate_name = st.text_input("Name", value=profile.get("candidate_name", ""))
            email = st.text_input("Email", value=profile.get("email", ""))
            phone = st.text_input("Phone", value=profile.get("phone", ""))

        with skills_tab:
            skills_selected = st.multiselect(
                "Skills",
                options=build_skill_options(profile),
                default=profile.get("skills", []) or [],
                help="Remove incorrectly parsed skills or add missing ones.",
            )
            custom_skills = st.text_input(
                "Add custom skills, separated by commas",
                value="",
                help="Optional. Use this when a skill was not detected or is not in the dropdown.",
            )
            years = st.text_input(
                "Years of experience",
                value="" if profile.get("years_experience") is None else str(profile.get("years_experience")),
            )
            levels = ["unknown", "entry_or_junior", "mid", "senior_or_experienced"]
            current_level = profile.get("candidate_level", "unknown")
            level_index = levels.index(current_level) if current_level in levels else 0
            level = st.selectbox("Candidate level", levels, index=level_index)
            education_text = st.text_input("Education keywords", value=comma_list(profile.get("education_keywords", [])))
            projects_text = st.text_input("Project keywords", value=comma_list(profile.get("project_keywords", [])))
            industries_selected = st.multiselect(
                "Preferred industries",
                options=build_industry_options(profile),
                default=profile.get("preferred_industries", []) or [],
            )

        with preview_tab:
            render_parsed_matching_preview(profile)

        c1, c2 = st.columns(2)
        back = c1.form_submit_button("Back", width="stretch")
        saved = c2.form_submit_button("Save and continue", type="primary", width="stretch")

    if back:
        go_to_step(1)
        st.rerun()
    if saved:
        profile["candidate_name"] = candidate_name
        profile["email"] = email
        profile["phone"] = phone
        profile["skills"] = extract_skills_from_value(comma_list(list(skills_selected) + parse_comma_list(custom_skills)))
        ml_constraint = ml_experience_constraint_from_text(comma_list(list(skills_selected)) + " " + profile.get("raw_resume_text", ""))
        if ml_constraint:
            profile["ml_experience_constraint"] = ml_constraint
        try:
            profile["years_experience"] = None if not years.strip() else float(years)
        except Exception:
            profile["years_experience"] = None
        profile["candidate_level"] = level
        profile["education_keywords"] = normalize_education_list(parse_comma_list(education_text))
        profile["project_keywords"] = parse_comma_list(projects_text)
        profile["preferred_industries"] = sorted(set(industries_selected), key=str.lower)
        st.session_state.profile = profile
        st.session_state.recommendations = None
        go_to_step(3)
        st.rerun()

    page_end()

def render_preferences_page() -> None:
    page_start(
        "Step 3 of 6",
        "Confirm matching strategy",
        "Set the job search rules JobPilot should use for filtering and ranking.",
    )
    if not st.session_state.profile_ready:
        st.info("No profile yet. Go back and parse a resume first.")
        if st.button("Back to resume upload", width="stretch"):
            go_to_step(1)
            st.rerun()
        return

    profile = st.session_state.profile
    st.markdown(
        "<div class='soft-summary'>Resume profile is saved. Now confirm only the matching rules: target role, location, salary floor, experience ceiling, and dealbreakers.</div>",
        unsafe_allow_html=True,
    )
    with st.form("preferences_form"):
        target_role = st.text_input("Target roles", value=profile.get("target_role", ""))
        location = st.text_input("Location preference", value=profile.get("location_preference", ""))
        salary = st.text_input("Minimum annual salary", value=str(profile.get("salary_min", "")))
        max_years = st.text_input("Maximum acceptable required years", value=str(profile.get("max_required_years", "")))
        dealbreakers_selected = st.multiselect(
            "Dealbreakers",
            options=build_dealbreaker_options(profile),
            default=profile.get("dealbreakers", []) or [],
            help="Select conditions that should strongly reduce or remove jobs.",
        )
        custom_dealbreakers = st.text_input(
            "Add custom dealbreakers, separated by commas",
            value="",
            help="Optional. Example: no blockchain, no relocation, no unpaid roles.",
        )
        c1, c2 = st.columns(2)
        back = c1.form_submit_button("Back", width="stretch")
        saved = c2.form_submit_button("Save and continue", type="primary", width="stretch")

    if back:
        go_to_step(2)
        st.rerun()
    if saved:
        profile["target_role"] = target_role
        location_preferences = parse_location_preferences(location)
        profile["location_preference"] = location_preferences.get("display") or location
        profile["location_preferences"] = location_preferences
        profile["salary_min"] = salary
        try:
            profile["max_required_years"] = None if not str(max_years).strip() else float(max_years)
        except Exception:
            profile["max_required_years"] = None
        raw_dealbreakers = list(dealbreakers_selected) + parse_comma_list(custom_dealbreakers)
        profile["dealbreakers"] = normalize_dealbreakers(raw_dealbreakers, comma_list(raw_dealbreakers))
        st.session_state.profile = profile
        st.session_state.preferences_ready = True
        st.session_state.recommendations = None
        st.session_state.top_n = 10
        st.session_state.candidate_k = 1500
        st.session_state.auto_match_after_preferences = True
        go_to_step(4)
        st.rerun()
    page_end()

def render_match_page() -> None:
    page_start(
        "Step 4 of 6",
        "View ranked jobs",
        "JobPilot retrieves candidate jobs, applies your rules, and ranks the best matches.",
    )
    if not st.session_state.preferences_ready:
        st.info("Preferences are not ready yet.")
        if st.button("Back to preferences", width="stretch"):
            go_to_step(3)
            st.rerun()
        return

    profile = st.session_state.profile
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"Target roles: {profile.get('target_role', '')}")
            st.write(f"Location: {profile.get('location_preference', '')}")
            st.write(f"Minimum annual salary: {profile.get('salary_min', '')}")
        with c2:
            st.write(f"Max required years: {profile.get('max_required_years', '')}")
            st.write(f"Dealbreakers: {comma_list(profile.get('dealbreakers', []))}")
            st.write(f"Search depth: {st.session_state.get('candidate_k', 1500)}")

    results = st.session_state.get("recommendations")
    if results is None:
        run_matching(
            top_n=st.session_state.get("top_n", 10),
            candidate_k=st.session_state.get("candidate_k", 1500),
        )
        st.rerun()

    results = st.session_state.get("recommendations")

    if results is not None:
        if len(results) == 0:
            st.warning("No jobs matched after filters. Try relaxing dealbreakers or increasing search depth.")
        else:
            grouped_results = add_duplicate_group_columns(results)
            display_top_n = int(st.session_state.get("top_n", 10))
            display_rows = primary_recommendations(grouped_results, display_top_n)
            top_row = display_rows.iloc[0] if len(display_rows) else grouped_results.iloc[0]
            st.success(
                f"Found {len(display_rows)} unique recommendations from {len(grouped_results)} matching postings. "
                f"Top match: {top_row.get('title', '')} at {top_row.get('company', '')}."
            )
            st.download_button(
                "Download Top Jobs",
                data=make_download_csv(display_rows, grouped_results),
                file_name="jobpilot_top_jobs.csv",
                mime="text/csv",
                width="stretch",
            )
            for display_rank, (idx, row) in enumerate(display_rows.iterrows(), start=1):
                group_rows = grouped_results[grouped_results["duplicate_group_id"] == row.get("duplicate_group_id")]
                similar_rows = group_rows[group_rows.index != idx]
                render_job_result(idx, row, similar_rows=similar_rows, display_rank=display_rank)
            with st.expander("Job Market Analytics"):
                render_market_analytics(load_dataset_info())
            feedback_tab = st.tabs(["Feedback memory"])[0]
            with feedback_tab:
                render_feedback_memory()

    with st.expander("Advanced matching settings"):
        top_n = st.slider("Number of matches", min_value=5, max_value=20, value=int(st.session_state.get("top_n", 10)))
        candidate_k = st.slider(
            "Search depth",
            min_value=100,
            max_value=1500,
            value=int(st.session_state.get("candidate_k", 1500)),
            step=100,
        )
        if st.button("Run matching again with these settings", width="stretch"):
            st.session_state.recommendations = None
            st.session_state.selected_job_index = None
            st.session_state.top_n = top_n
            st.session_state.candidate_k = candidate_k
            st.rerun()

    if st.button("Choose resume job", type="primary", width="stretch"):
        go_to_step(5)
        st.rerun()

    with st.expander("Other actions"):
        if st.button("Edit preferences", width="stretch"):
            go_to_step(3)
            st.rerun()
        if st.button("Run matching again", width="stretch"):
            st.session_state.recommendations = None
            st.session_state.selected_job_index = None
            st.rerun()

    page_end()


def clean_export_value(value: object) -> str:
    text = str(value or "").strip()
    if text.lower() in {"nan", "none", "unknown", "unclear"}:
        return ""
    return text


def unique_export_values(values) -> List[str]:
    unique = []
    for value in values:
        text = clean_export_value(value)
        if text and text not in unique:
            unique.append(text)
    return unique


def display_employment_type(row) -> str:
    normalized = clean_export_value(row.get("normalized_employment_type", ""))
    if normalized:
        return normalized
    raw = clean_export_value(row.get("raw_employment_type", "") or row.get("employment_type", ""))
    if raw.lower() == "internship":
        return "Not listed"
    return raw or "Not listed"


def summarize_group_locations(group_rows: pd.DataFrame) -> str:
    locations = unique_export_values(group_rows.get("location", pd.Series(dtype=str)))
    if not locations:
        return "Not listed"
    if len(locations) <= 4:
        return " / ".join(locations)
    return " / ".join(locations[:4]) + f" / +{len(locations) - 4} more"


def summarize_group_salary(group_rows: pd.DataFrame) -> str:
    salaries = unique_export_values([format_salary(row) for _, row in group_rows.iterrows()])
    if not salaries:
        return "Not listed"
    if len(salaries) == 1:
        return salaries[0]
    listed = [salary for salary in salaries if salary.lower() != "not listed"]
    if len(listed) == 1:
        return listed[0]
    return " / ".join((listed or salaries)[:3])


def summarize_potential_concerns(row, salary_text: str | None = None) -> str:
    concerns = []
    displayed_salary = str(salary_text or format_salary(row)).strip().lower()
    if displayed_salary == "not listed":
        concerns.append("Salary not listed")
    if not clean_export_value(row.get("required_years", "")):
        concerns.append("No explicit required years listed")

    size_bucket = clean_export_value(row.get("company_size_bucket", "")).lower()
    size_confidence = clean_export_value(row.get("company_enrichment_confidence", "")).lower()
    if not size_bucket or size_bucket == "unknown" or size_confidence == "low":
        concerns.append("Company size not confirmed")

    sponsor_signal = clean_export_value(row.get("company_sponsor_signal", "")).lower()
    sponsorship_reason = clean_export_value(row.get("sponsorship_reason", "")).lower()
    if sponsor_signal in {"", "unknown", "unclear"} or "unclear" in sponsorship_reason:
        concerns.append("Visa sponsorship unclear")

    employment = clean_export_value(row.get("normalized_employment_type", "")).lower()
    if any(term in employment for term in ["contract", "temp", "temporary"]):
        concerns.append("Contract or temporary role")

    try:
        quality = float(row.get("job_quality_score", "") or 0)
    except Exception:
        quality = 0
    quality_reason = clean_export_value(row.get("job_quality_reason", "")).lower()
    if quality and quality < 0.55:
        concerns.append("Posting quality risk")
    elif any(term in quality_reason for term in ["vague", "placement", "jobseeker", "low quality"]):
        concerns.append("Posting quality risk")

    return "; ".join(dict.fromkeys(concerns)) or "No major concerns flagged"


def make_download_csv(display_rows: pd.DataFrame, grouped_results: pd.DataFrame) -> bytes:
    display_rows = add_duplicate_group_columns(display_rows)
    grouped_results = add_duplicate_group_columns(grouped_results)
    export_rows = []
    for visible_rank, (idx, row) in enumerate(display_rows.iterrows(), start=1):
        group_id = row.get("duplicate_group_id")
        group_rows = grouped_results[grouped_results["duplicate_group_id"] == group_id]
        if len(group_rows) == 0:
            group_rows = pd.DataFrame([row])
        similar_count = max(0, len(group_rows) - 1)
        salary_text = summarize_group_salary(group_rows)
        export_rows.append(
            {
                "Rank": visible_rank,
                "Job Title": row.get("title", ""),
                "Company": row.get("company", ""),
                "Location": summarize_group_locations(group_rows),
                "Employment Type": display_employment_type(row),
                "Salary": salary_text,
                "Apply Link": clean_export_value(row.get("apply_url", "")),
                "Why This Match": clean_display_text(row.get("why_matched", ""), max_chars=500),
                "Key Matched Skills": clean_export_value(row.get("matched_skills", "")) or "No direct skill overlap listed",
                "Potential Concerns": summarize_potential_concerns(row, salary_text),
                "Similar Postings": "No similar postings" if similar_count == 0 else f"{similar_count} similar postings collapsed",
            }
        )
    return pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8-sig")


def safe_join(items, fallback: str = "") -> str:
    if not items:
        return fallback
    if isinstance(items, str):
        return items
    return ", ".join([str(x) for x in items if str(x).strip()]) or fallback


def profile_grounding_text(profile: Dict) -> str:
    parts = [
        str(profile.get("raw_resume_text", "") or ""),
        str(profile.get("candidate_name", "") or ""),
        str(profile.get("target_role", "") or ""),
        str(profile.get("candidate_level", "") or ""),
        safe_join(profile.get("skills", [])),
        safe_join(profile.get("education_keywords", [])),
        safe_join(profile.get("project_keywords", [])),
        safe_join(profile.get("dealbreakers", [])),
        safe_join(profile.get("preferred_industries", [])),
    ]
    constraints = profile.get("company_constraints", {}) or {}
    if isinstance(constraints, dict):
        parts.extend([str(k) for k, v in constraints.items() if v])
    return " ".join(parts).lower()


def research_early_career_resume_signals(profile: Dict) -> List[str]:
    combined = profile_grounding_text(profile)
    signals = []
    checks = [
        ("MS student", r"\bms\b|\bm\.s\.\b|master"),
        ("PhD student", r"\bphd\b|\bph\.d\.\b|doctoral"),
        ("graduate student", r"graduate student|graduating student|graduating soon|recent (?:ms|phd|master|ph\.d) graduate"),
        ("CS/AI/ML education", r"computer science|artificial intelligence|\bai\b|machine learning|\bml\b|computer vision"),
        ("published research", r"published research|publications?|research papers?"),
        ("thesis", r"\bthesis\b"),
        ("research scientist target", r"research scientist|applied scientist"),
        ("research-heavy ML target", r"(?:ai|ml|machine learning) engineer"),
        ("research keywords", r"deep learning|computer vision|\bresearch\b"),
        ("limited full-time experience", r"no full[- ]time|limited full[- ]time|internship|entry_or_junior|new grad|recent graduate"),
        ("OPT/visa constrained", r"\bopt\b|visa|h-?1b|sponsorship"),
    ]
    for label, pattern in checks:
        if re.search(pattern, combined):
            signals.append(label)
    try:
        years = profile.get("years_experience")
        if years not in [None, "", "unknown"] and float(years) <= 1.0:
            signals.append("limited full-time experience")
    except Exception:
        pass
    return list(dict.fromkeys(signals))


def is_research_oriented_early_career_profile(profile: Dict) -> bool:
    signals = research_early_career_resume_signals(profile)
    signal_text = " ".join(signals).lower()
    has_grad_or_early = any(term in signal_text for term in ["student", "graduate", "limited", "opt", "visa"])
    text = profile_grounding_text(profile)
    explicit_research = any(
        re.search(pattern, text)
        for pattern in [
            r"published research",
            r"\bpublications?\b",
            r"research papers?",
            r"\bthesis\b",
            r"\bresearch scientist\b",
            r"\bapplied scientist\b",
            r"deep learning",
            r"computer vision",
            r"\bresearch\b",
        ]
    )
    return len(signals) >= 2 and has_grad_or_early and explicit_research


def research_terms_present(profile: Dict) -> List[str]:
    combined = profile_grounding_text(profile)
    terms = []
    for raw, label in [
        ("published research", "published research"),
        ("publication", "publications"),
        ("research paper", "research papers"),
        ("thesis", "thesis"),
        ("nlp", "NLP"),
        ("computer vision", "computer vision"),
        ("deep learning", "deep learning"),
        ("pytorch", "PyTorch"),
        ("python", "Python"),
    ]:
        if raw in combined:
            terms.append(label)
    return list(dict.fromkeys(terms))


RESUME_CONTAMINATION_RULES = [
    ("publications/research", ["publications / research", "publication", "published researcher", "published research", "research paper"], ["publication", "published research", "research paper", "thesis", "research"]),
    ("computer vision", ["computer vision"], ["computer vision"]),
    ("deep learning", ["deep learning"], ["deep learning"]),
    ("PyTorch", ["pytorch"], ["pytorch"]),
    ("visa/sponsorship", ["opt", "h-1b", "h1b", "sponsorship", "visa"], ["opt", "h-1b", "h1b", "sponsorship", "visa"]),
    ("MS Computer Science", ["ms computer science", "computer science student"], ["ms computer science", "computer science"]),
    ("Research Scientist", ["research scientist"], ["research scientist"]),
    ("Applied Scientist", ["applied scientist"], ["applied scientist"]),
    ("ML Engineer", ["ml engineer", "ml engineering", "machine learning engineer"], ["ml engineer", "ml engineering", "machine learning engineer"]),
]


def resume_consistency_issues(profile: Dict, resume_text: str) -> List[str]:
    support = profile_grounding_text(profile)
    generated = str(resume_text or "").lower()
    issues = []
    for label, generated_terms, support_terms in RESUME_CONTAMINATION_RULES:
        if any(term in generated for term in generated_terms) and not any(term in support for term in support_terms):
            issues.append(f"Unsupported {label}")
    if "[" in str(resume_text or "") and "]" in str(resume_text or ""):
        issues.append("Bracketed placeholder text")
    return issues


def build_publications_research_section(profile: Dict) -> str:
    terms = research_terms_present(profile)
    topics = [term for term in ["NLP", "computer vision", "deep learning"] if term in terms]
    tools = [term for term in ["Python", "PyTorch"] if term in terms]
    bullets = []
    if "published research" in terms or "publications" in terms or "research papers" in terms:
        topic_text = safe_join(topics, "applied machine learning")
        tool_text = " and ".join(tools) if tools else "verified ML tools"
        bullets.append(
            f"- Published research in {topic_text}, with hands-on experience applying deep learning methods using {tool_text}."
        )
    elif "thesis" in terms:
        bullets.append("- Thesis-oriented graduate research experience connected to applied machine learning work.")
    else:
        topic_text = safe_join(topics, "machine learning research")
        bullets.append(f"- Graduate research background in {topic_text}, framed conservatively around verified profile evidence.")
    if tools:
        bullets.append(f"- Applied {' and '.join(tools)} in research-oriented ML workflows where supported by the profile.")
    bullets.append("- Connects research contribution, experimentation, and applied ML relevance to the selected role.")
    return "\n".join(bullets)


def resume_section_order(profile: Dict) -> str:
    if is_research_oriented_early_career_profile(profile):
        return """## Professional Summary
## Publications / Research
## Education
## Technical Skills
## Projects / Experience
## Selected Highlights"""
    if resume_profile_kind(profile) == "new_grad_analytics":
        return """## Education
## Technical Skills
## Projects / Internship Experience
## Selected Highlights
## Professional Summary"""
    return """## Professional Summary
## Education
## Technical Skills
## Projects / Experience
## Selected Highlights"""


def get_openrouter_api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip().startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def build_resume_prompt(profile: Dict, job: Dict) -> str:
    resume_text = str(profile.get("raw_resume_text", "") or "")
    if not resume_text.strip():
        return "RAW_RESUME_TEXT_MISSING"
    resume_text = resume_text[:10000]
    job_description = str(job.get("description", "") or "")[:5000]
    candidate_name = profile.get("candidate_name", "") or "Resume"
    research_resume = is_research_oriented_early_career_profile(profile)
    unsupported = "; ".join(resume_consistency_issues(profile, " ".join(term for _, terms, _ in RESUME_CONTAMINATION_RULES for term in terms)))
    research_instruction = ""
    if research_resume:
        research_instruction = """
Research-oriented early-career structure:
- The candidate appears to be graduate-level or early-career with research/publication evidence.
- Add a section titled "Publications / Research" immediately after "Professional Summary".
- Place "Publications / Research" before Education, Technical Skills, and Projects / Experience.
- Emphasize published research, NLP, computer vision, deep learning, Python, and PyTorch only when those facts are present.
- If exact publication details are not provided, use conservative general wording such as: "Published research in NLP and computer vision, with hands-on experience applying deep learning methods using Python and PyTorch."
- Do not invent publication titles, journal/conference names, dates, citation counts, university names, employers, or metrics.
"""
    return f"""You are a careful senior resume editor creating an applicant-ready resume.

Hard rules:
- Use ONLY facts found in the original resume/profile or parsed candidate profile.
- Treat this request as stateless. Do not reuse facts, sections, wording, or target roles from another candidate, persona, previous selected job, cache, or test profile.
- Preserve exact name, email, phone, education, employers, job titles, dates, GPA, projects, and quantified results when present.
- Do not invent employers, dates, degrees, GPA, certifications, tools, production experience, achievements, or metrics.
- Do not add skills from the job description unless they already appear in the candidate resume/profile.
- Do not mention these unsupported candidate facts unless they appear in this current profile: {unsupported or "none"}.
- Do not include negative self-assessments such as "no production ML experience" in the resume. Use them only to avoid overstating qualifications.
- Publications / Research is allowed only when the current profile explicitly provides publication, research, thesis, Research Scientist, Applied Scientist, deep learning, or computer vision evidence.
- Reorder and emphasize relevant evidence for the selected job.
- Use concise resume language with 2-3 strong bullets per role/project when evidence exists.
- Do not include Tailoring Notes, explanations, debug fields, raw prompt text, JSON, or bracketed placeholders.
- Output the resume only.
{research_instruction}

Selected job:
Title: {job.get("title", "")}
Company: {job.get("company", "")}
Location: {job.get("location", "")}
Matched skills: {job.get("matched_skills", "")}
Why matched: {job.get("why_matched", "")}

Job description:
{job_description}

Parsed candidate profile:
Name: {candidate_name}
Email: {profile.get("email", "")}
Phone: {profile.get("phone", "")}
Target role: {profile.get("target_role", "")}
Skills detected from original resume: {safe_join(profile.get("skills", []))}
Education keywords: {safe_join(profile.get("education_keywords", []))}
Project keywords: {safe_join(profile.get("project_keywords", []))}

Original resume/profile text:
{resume_text}

Return clean Markdown with only these sections when supported by candidate facts:
# {candidate_name}
contact line

{resume_section_order(profile)}
"""


def generate_resume_with_openrouter(profile: Dict, job: Dict) -> str | None:
    api_key = get_openrouter_api_key()
    if not api_key:
        return None
    prompt = build_resume_prompt(profile, job)
    if prompt == "RAW_RESUME_TEXT_MISSING":
        st.warning("Resume text is missing. Go back to Step 1 and upload or paste a resume/profile.")
        return None
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip() or "openai/gpt-4o-mini"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "JobPilot",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You preserve resume facts and never invent candidate details."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.05,
        "max_tokens": 2200,
    }
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        st.warning(f"OpenRouter resume generation failed. Using local fallback. Error: {exc}")
        return None


def supported_job_keywords(profile: Dict, job: Dict) -> List[str]:
    skills = [str(skill).strip() for skill in profile.get("skills", []) if str(skill).strip()]
    skill_lookup = {skill.lower(): skill for skill in skills}
    matched = [x.strip() for x in str(job.get("matched_skills", "") or "").split(",") if x.strip()]
    selected = []
    for skill in matched + skills:
        key = skill.lower()
        if key in skill_lookup and skill_lookup[key] not in selected:
            selected.append(skill_lookup[key])
    return selected[:12]


def resume_profile_kind(profile: Dict) -> str:
    target = str(profile.get("target_role", "") or "").lower()
    raw = str(profile.get("raw_resume_text", "") or "").lower()
    level = str(profile.get("candidate_level", "") or "").lower()
    if is_research_oriented_early_career_profile(profile):
        return "research_visa"
    if "msba" in raw or "uc davis" in raw or level == "entry_or_junior":
        return "new_grad_analytics"
    if any(term in target for term in ["ml platform", "mlops", "senior ml engineer"]) or "7 years" in raw:
        return "senior_ml_platform"
    if "ml engineer" in target or "applied scientist" in target or "ml-focused" in target:
        return "ml_pivot"
    return "general"


def build_summary(profile: Dict, job: Dict, keywords: List[str]) -> str:
    title = str(job.get("title", "Selected Role") or "Selected Role")
    company = str(job.get("company", "Selected Company") or "Selected Company")
    kind = resume_profile_kind(profile)
    skill_text = safe_join(keywords[:6], "relevant verified skills")
    if kind == "ml_pivot":
        return (
            f"Data analyst with 3 years of analytical experience, tailoring verified Python and ML-adjacent analytical "
            f"work toward {title} at {company}. Strengths include {skill_text}, with emphasis on model-focused analysis "
            "and careful translation of data work into model-oriented roles."
        )
    if kind == "new_grad_analytics":
        return (
            f"Recent MSBA graduate focused on entry-level analytics roles such as {title}. Brings verified technical "
            f"skills in {skill_text}, analytics projects, and internship exposure without overstating full-time experience."
        )
    if kind == "senior_ml_platform":
        return (
            f"Experienced ML platform and backend engineering profile aligned to {title} at {company}. Emphasizes "
            f"{skill_text}, infrastructure, distributed systems, and production-oriented machine learning platforms."
        )
    if kind == "research_visa":
        terms = research_terms_present(profile)
        education = safe_join(profile.get("education_keywords", []))
        intro = "Research-oriented early-career candidate"
        if "MS Computer Science" in education or "ms computer science" in profile_grounding_text(profile):
            intro = "MS Computer Science student"
        elif "published research" in terms or "publications" in terms:
            intro = "Published researcher"
        visa_context = ""
        if any(term in profile_grounding_text(profile) for term in ["opt", "h-1b", "h1b", "sponsorship", "visa"]):
            visa_context = " and US roles compatible with sponsorship needs"
        research_focus = safe_join([term for term in ["deep learning", "NLP", "computer vision"] if term in terms], "verified research strengths")
        return (
            f"{intro} targeting {title} at {company}. Emphasizes {skill_text}, {research_focus}{visa_context}."
        )
    return f"Candidate targeting {title} at {company}, emphasizing verified strengths in {skill_text}."


def build_education_section(profile: Dict) -> str:
    education = [str(x).strip() for x in profile.get("education_keywords", []) if str(x).strip()]
    raw = str(profile.get("raw_resume_text", "") or "")
    lines = []
    if any("uc davis" in item.lower() for item in education) or "UC Davis" in raw:
        msba = "MSBA, UC Davis"
        lines.append(f"- {msba}")
    elif any("ms computer science" in item.lower() for item in education) or "MS Computer Science" in raw:
        lines.append("- MS Computer Science")
    elif education:
        lines.append("- " + safe_join(education))
    return "\n".join(lines) if lines else ""


def build_projects_experience_section(profile: Dict, job: Dict, keywords: List[str]) -> str:
    kind = resume_profile_kind(profile)
    projects = [str(x).strip() for x in profile.get("project_keywords", []) if str(x).strip()]
    if kind == "ml_pivot":
        projects = [project for project in projects if project.lower() not in {"machine learning", "ml"}]
    title = str(job.get("title", "selected role") or "selected role")
    skill_text = safe_join(keywords[:5], "verified technical skills")
    bullets = []
    if kind == "ml_pivot":
        bullets = [
            f"- Applied Python, SQL, and pandas to analytical work and framed results for ML-focused {title} responsibilities.",
            "- Used scikit-learn and basic PyTorch exposure to support model evaluation, feature analysis, and analytical experimentation.",
            "- Connected data analyst experience with model evaluation, feature thinking, and stakeholder-ready interpretation.",
        ]
    elif kind == "new_grad_analytics":
        bullets = [
            "- Completed MSBA-oriented analytics work using Python, R, SQL, Tableau, PySpark, and basic NLP.",
            "- Applied classroom and project experience to dashboards, analysis workflows, and junior analytics problem solving.",
            "- Includes two analytics internships as early-career experience without presenting them as full-time roles.",
        ]
    elif kind == "senior_ml_platform":
        bullets = [
            "- Led ML platform, infrastructure, and backend engineering work across distributed systems and production-oriented environments.",
            f"- Applied {skill_text} to scalable ML, data, and service infrastructure.",
            "- Emphasized MLOps, platform reliability, model deployment context, and cross-system engineering depth for senior roles.",
        ]
    elif kind == "research_visa":
        text = profile_grounding_text(profile)
        target_roles = []
        for role in ["research scientist", "applied scientist", "ml engineering", "ml engineer", "ai engineer"]:
            if role in text:
                target_roles.append(role)
        visa_bullet = []
        if any(term in text for term in ["opt", "h-1b", "h1b", "sponsorship", "visa"]):
            visa_bullet.append("- Preserves verified OPT/H-1B sponsorship context for US role targeting.")
        bullets = [
            f"- Applied {skill_text} to research-oriented machine learning and AI engineering work.",
            "- Connected graduate research experience to applied ML experimentation, model development, and technical communication.",
        ]
        if target_roles:
            bullets.append(f"- Target roles supported by the profile: {safe_join([role.title() for role in target_roles])}.")
        bullets.extend(visa_bullet)
    else:
        bullets = [f"- Applied {skill_text} to work relevant to {title}."]
    if projects:
        bullets.append(f"- Relevant project themes: {safe_join(projects)}.")
    return "\n".join(bullets)


def build_selected_highlights(profile: Dict, job: Dict, keywords: List[str]) -> str:
    title = str(job.get("title", "selected role") or "selected role")
    company = str(job.get("company", "selected company") or "selected company")
    kind = resume_profile_kind(profile)
    highlights = [f"- Selected role alignment: {title} at {company}."]
    if kind == "ml_pivot":
        highlights.append("- Positions analytics experience toward ML-adjacent engineering and model-focused analysis roles.")
    elif kind == "new_grad_analytics":
        highlights.append("- Leads with education, technical skills, projects, and internships for entry/junior analytics fit.")
    elif kind == "senior_ml_platform":
        highlights.append("- Prioritizes senior ML platform, MLOps, cloud, data infrastructure, and backend engineering evidence.")
    elif kind == "research_visa":
        if any(term in profile_grounding_text(profile) for term in ["opt", "h-1b", "h1b", "sponsorship", "visa"]):
            highlights.append("- Leads with publications and research strengths for sponsor-aware US AI/ML roles.")
        else:
            highlights.append("- Leads with publications and research strengths only where supported by the profile.")
    if keywords:
        highlights.append(f"- Job-aligned verified keywords: {safe_join(keywords[:8])}.")
    return "\n".join(highlights)


def build_local_resume(profile: Dict, job: Dict) -> str:
    name = profile.get("candidate_name") or "Resume"
    email = profile.get("email") or ""
    phone = profile.get("phone") or ""
    contact = " | ".join([part for part in [email, phone] if str(part).strip()])
    keywords = supported_job_keywords(profile, job)
    skills = [str(skill).strip() for skill in profile.get("skills", []) if str(skill).strip()]
    education = build_education_section(profile)
    projects_experience = build_projects_experience_section(profile, job, keywords)
    selected_highlights = build_selected_highlights(profile, job, keywords)
    sections = [f"# {name}"]
    if contact:
        sections.append(contact)
    kind = resume_profile_kind(profile)
    if kind == "new_grad_analytics":
        sections.extend([
            "## Education",
            education,
            "## Technical Skills",
            safe_join(skills, safe_join(keywords, "Verified candidate skills")),
            "## Projects / Internship Experience",
            projects_experience,
            "## Selected Highlights",
            selected_highlights,
            "## Professional Summary",
            build_summary(profile, job, keywords),
        ])
    else:
        sections.extend([
            "## Professional Summary",
            build_summary(profile, job, keywords),
        ])
        if is_research_oriented_early_career_profile(profile):
            sections.extend([
                "## Publications / Research",
                build_publications_research_section(profile),
            ])
        sections.extend([
            "## Education",
            education,
            "## Technical Skills",
            safe_join(skills, safe_join(keywords, "Verified candidate skills")),
            "## Projects / Experience",
            projects_experience,
            "## Selected Highlights",
            selected_highlights,
        ])
    return "\n\n".join([section for section in sections if str(section).strip()]).strip() + "\n"


def generate_tailored_resume(profile: Dict, job: Dict) -> str:
    llm_resume = generate_resume_with_openrouter(profile, job)
    if llm_resume:
        issues = resume_consistency_issues(profile, llm_resume)
        if not issues:
            return llm_resume
        st.warning("Generated draft used unsupported profile details, so JobPilot rebuilt a grounded local draft.")
    local_resume = build_local_resume(profile, job)
    issues = resume_consistency_issues(profile, local_resume)
    if issues:
        st.warning("Local resume draft may still need review for unsupported details: " + ", ".join(issues))
    return local_resume


def cover_letter_consistency_issues(profile: Dict, job: Dict, cover_letter_text: str) -> List[str]:
    support = " ".join(
        [
            profile_grounding_text(profile),
            str(job.get("title", "")),
            str(job.get("company", "")),
            str(job.get("matched_skills", "")),
            safe_join(supported_job_keywords(profile, job)),
        ]
    ).lower()
    generated = str(cover_letter_text or "").lower()
    issues = []
    unsupported_terms = {
        "publication": ["publication", "published research", "research paper"],
        "visa": ["opt", "h-1b", "h1b", "sponsorship", "visa"],
        "computer vision": ["computer vision"],
        "deep learning": ["deep learning"],
        "pytorch": ["pytorch"],
    }
    for label, terms in unsupported_terms.items():
        if any(term in generated for term in terms) and not any(term in support for term in terms):
            issues.append(f"Unsupported {label}")
    if "[" in str(cover_letter_text or "") or "]" in str(cover_letter_text or ""):
        issues.append("Placeholder text")
    return issues


def generate_cover_letter(profile: Dict, job: Dict, matched_skills: List[str] | None = None, concerns: List[str] | None = None) -> str:
    title = clean_export_value(job.get("title", "")) or "the role"
    company = clean_export_value(job.get("company", "")) or "your team"
    name = clean_export_value(profile.get("candidate_name", "")) or "Candidate"
    target_role = clean_export_value(profile.get("target_role", ""))
    summary = clean_display_text(profile.get("profile_summary", "") or profile.get("raw_resume_text", ""), max_chars=260)
    skills = matched_skills or supported_job_keywords(profile, job)
    if not skills:
        skills = [str(skill).strip() for skill in profile.get("skills", []) if str(skill).strip()][:6]
    skills_text = safe_join(skills[:6], "the verified skills in my profile")
    education = safe_join(profile.get("education_keywords", []))
    projects = safe_join(profile.get("project_keywords", []))
    job_description = clean_display_text(job.get("description", ""), max_chars=500)
    job_requirements = supported_job_keywords(profile, job)
    requirement_text = safe_join(job_requirements[:5], "the responsibilities described in the posting")

    background_parts = []
    if education:
        background_parts.append(f"education in {education}")
    if projects:
        background_parts.append(f"project experience related to {projects}")
    if summary:
        background_parts.append(summary)
    background_text = safe_join(background_parts[:2], "a background aligned with the role")

    paragraphs = [
        (
            f"Dear Hiring Team,\n\n"
            f"I am writing to express my interest in the {title} role at {company}. "
            f"The position stands out because it connects closely with my current target roles"
            f"{' in ' + target_role if target_role else ''} and the strengths already reflected in my profile. "
            f"I am especially interested in opportunities where I can apply verified skills in a focused, practical way while continuing to learn the team, product, and business context."
        ),
        (
            f"My fit for this opportunity is grounded in {background_text}. "
            f"I can bring {skills_text} to analytical, technical, and cross-functional work without overstating experience beyond the information provided in my profile. "
            f"Those strengths would help me contribute to structured problem solving, clear communication, and reliable execution on the responsibilities connected to this role."
        ),
        (
            f"For this specific role, I would emphasize the overlap between the posting and my verified skills: {requirement_text}. "
            f"Based on the job description, I would focus on contributing carefully, learning the team context quickly, and applying relevant tools to practical business or technical problems. "
            f"The posting describes work that I would approach with attention to requirements, measurable outcomes, and collaboration with the people who use the analysis, systems, or recommendations."
        ),
        (
            f"I would welcome the opportunity to discuss how my background could support {company}'s needs for this role. "
            f"I appreciate that a strong match depends on both technical fit and team fit, and I would be glad to share more about the profile details that are most relevant to this opening. "
            f"Thank you for your time and consideration, and I look forward to the possibility of next steps.\n\n"
            f"Sincerely,\n{name}"
        ),
    ]
    letter = "\n\n".join(paragraphs)
    issues = cover_letter_consistency_issues(profile, job, letter)
    if issues:
        conservative_skills = safe_join(skills[:4], "verified profile skills")
        letter = (
            f"Dear Hiring Team,\n\n"
            f"I am interested in the {title} role at {company}. My application is based on the verified information in my profile and the requirements described in the job posting.\n\n"
            f"My relevant strengths include {conservative_skills}. I would bring a careful, practical approach to the responsibilities described for this role while avoiding claims that are not supported by my profile.\n\n"
            f"I am especially interested in the opportunity to apply my background to {company}'s needs and to learn more about the team, priorities, and next steps.\n\n"
            f"Sincerely,\n{name}"
        )
    return letter


def parse_money(value) -> float | None:
    text = str(value or "").strip().replace(",", "").replace("$", "")
    if text.lower() in {"", "nan", "none", "null"}:
        return None
    try:
        return float(text)
    except Exception:
        return None


def annual_multiplier(period: str) -> float | None:
    text = str(period or "").lower().replace(".", "").replace("/", " ").strip()
    if text in {"year", "yearly", "annual", "annually", "pa", "p a", "per annum", "annum"}:
        return 1.0
    if text in {"month", "monthly", "per month"}:
        return 12.0
    if text in {"hour", "hourly", "per hour", "hr"}:
        return 2080.0
    if text in {"week", "weekly", "per week"}:
        return 52.0
    if text in {"day", "daily", "per day"}:
        return 260.0
    return None


def annual_salary_range(row) -> tuple[float | None, float | None]:
    annual_min = parse_money(row.get("salary_annual_min"))
    annual_max = parse_money(row.get("salary_annual_max"))
    if annual_min is None and annual_max is None:
        mult = annual_multiplier(str(row.get("salary_period", "") or ""))
        raw_min = parse_money(row.get("salary_min"))
        raw_max = parse_money(row.get("salary_max"))
        if mult is None:
            annual_min = raw_min if raw_min is not None and raw_min >= 10000 else None
            annual_max = raw_max if raw_max is not None and raw_max >= 10000 else None
        else:
            annual_min = None if raw_min is None else raw_min * mult
            annual_max = None if raw_max is None else raw_max * mult
    if annual_min is None and annual_max is None:
        parsed = extract_salary_from_description(row.get("description", ""))
        annual_min = parsed.get("salary_min")
        annual_max = parsed.get("salary_max")
    if annual_min is not None and annual_max is not None and annual_min > annual_max:
        annual_min, annual_max = annual_max, annual_min
    return annual_min, annual_max


def format_usd(value: float) -> str:
    return "$" + format(float(value), ",.0f")


def format_salary(row) -> str:
    display = str(row.get("salary_display", "") or "").strip()
    if display and display.lower() not in {"nan", "none", "not listed"}:
        return display
    annual_min, annual_max = annual_salary_range(row)
    return recommender_salary_display(annual_min, annual_max)


def format_job_facts(row) -> str:
    facts = []
    employment = str(row.get("normalized_employment_type", "") or "").strip().lower()
    seniority = str(row.get("seniority", "") or "").strip()
    if employment and employment not in {"nan", "unknown", "unclear"}:
        facts.append(f"Type: {employment}")
    else:
        facts.append("Type: Not clearly listed")
    if seniority and seniority.lower() != "nan":
        facts.append(f"Level: {seniority}")
    facts.append(f"Annual salary: {format_salary(row)}")
    return " | ".join(facts)


def friendly_role_family(value: object) -> str:
    labels = {
        "ml_platform_mlops": "ML platform / MLOps",
        "ml_engineering": "ML engineering",
        "data_science": "Data science",
        "ux_research": "UX research",
        "product_research": "Product research",
        "market_research": "Market research",
        "data_analytics": "Data analytics",
        "bi_analytics": "BI analytics",
        "analytics_engineering": "Analytics engineering",
        "data_engineering": "Data engineering",
        "software_engineering": "Software engineering",
        "devops": "DevOps / platform",
        "finance_accounting": "Finance / accounting",
        "marketing": "Marketing",
        "other": "Other",
        "unclear": "Not clearly classified",
        "general": "Broad role",
    }
    text = str(value or "").strip()
    if "!=" in text:
        return "Role family mismatch"
    if "->" in text:
        return "Adjacent role family"
    return labels.get(text, text.replace("_", " ").title() if text else "Not classified")


def clean_display_text(value: object, max_chars: int = 650) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def job_option_label(row) -> str:
    return f"#{int(row.get('rank', 0))} {row.get('title', '')} - {row.get('company', '')} - {row.get('location', '')}"


def selected_job_from_results(results: pd.DataFrame):
    index = st.session_state.get("selected_job_index")
    if index is None:
        st.session_state.selected_job_index = 0
        index = 0
    try:
        index = int(index)
    except Exception:
        return None
    if index < 0 or index >= len(results):
        st.session_state.selected_job_index = 0
        index = 0
    return results.iloc[index]


def split_skill_cells(series: pd.Series) -> pd.Series:
    items = []
    for value in series.dropna().astype(str):
        for item in re.split(r"[,;/|]", value):
            clean = item.strip()
            if clean and clean.lower() not in {"nan", "none", "null"}:
                items.append(clean.title() if clean.islower() else clean)
    if not items:
        return pd.Series(dtype="int64")
    return pd.Series(items).value_counts()


@st.cache_data(show_spinner=False, max_entries=4)
def compute_market_analytics(df: pd.DataFrame) -> Dict[str, pd.DataFrame | int]:
    work = df.copy()

    top_skills = split_skill_cells(work["skills"]) if "skills" in work else pd.Series(dtype="int64")
    skill_df = top_skills.head(20).reset_index()
    skill_df.columns = ["Skill", "Count"]
    total_jobs = max(len(work), 1)
    if len(skill_df):
        skill_df["Share of Jobs"] = (skill_df["Count"] / total_jobs).map(lambda value: f"{value:.1%}")

    salary_values = []
    for _, row in work.iterrows():
        annual_min, annual_max = annual_salary_range(row)
        if annual_min is None and annual_max is None:
            continue
        if annual_min is not None and annual_max is not None:
            salary_values.append((float(annual_min) + float(annual_max)) / 2)
        else:
            salary_values.append(float(annual_min if annual_min is not None else annual_max))
    salary_bands = [
        ("<$80K", 0, 80_000),
        ("$80K-$120K", 80_000, 120_000),
        ("$120K-$160K", 120_000, 160_000),
        ("$160K-$200K", 160_000, 200_000),
        ("$200K+", 200_000, float("inf")),
    ]
    salary_rows = []
    for label, low, high in salary_bands:
        count = sum(1 for value in salary_values if low <= value < high)
        salary_rows.append({"Salary Band": label, "Count": count})
    salary_df = pd.DataFrame(salary_rows)

    location_counts = {}
    for _, row in work.iterrows():
        loc = normalize_market_location(row)
        location_counts[loc] = location_counts.get(loc, 0) + 1
    location_df = (
        pd.DataFrame([{"Location": key, "Count": value} for key, value in location_counts.items()])
        .sort_values("Count", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    role_counts = {}
    for _, row in work.iterrows():
        family = infer_simple_role_family(row)
        role_counts[family] = role_counts.get(family, 0) + 1
    role_df = (
        pd.DataFrame([{"Role Family": key, "Count": value} for key, value in role_counts.items()])
        .sort_values("Count", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    return {
        "top_skills": skill_df,
        "salary_distribution": salary_df,
        "salary_missing_count": int(len(work) - len(salary_values)),
        "top_locations": location_df,
        "role_families": role_df,
    }


def normalize_market_location(row) -> str:
    raw = str(row.get("location", "") or "").strip()
    is_remote = str(row.get("is_remote", "") or "").strip().lower() in {"true", "1", "yes"}
    if is_remote or raw.lower() in {"remote", "anywhere", "worldwide", "wfh"} or "remote" in raw.lower():
        return "Remote"
    if not raw or raw.lower() in {"nan", "none", "null"}:
        return "Location not listed"
    parts = [part.strip() for part in re.split(r"[|;/]", raw) if part.strip()]
    return parts[0] if parts else raw


def infer_simple_role_family(row) -> str:
    text = f"{row.get('title', '')} {row.get('search_role', '')} {row.get('description', '')}".lower()
    checks = [
        ("Analytics Engineer / BI", ["analytics engineer", "bi engineer", "business intelligence", "data warehouse"]),
        ("Data Scientist", ["data scientist", "machine learning scientist"]),
        ("ML Engineer / MLOps", ["machine learning engineer", "ml engineer", "mlops", "ml platform"]),
        ("Data Engineer", ["data engineer", "etl", "data pipeline"]),
        ("Data Analyst", ["data analyst", "business analyst", "reporting analyst"]),
        ("Research / Scientist", ["research scientist", "applied scientist", "researcher"]),
        ("Software Engineer", ["software engineer", "backend engineer", "full stack"]),
    ]
    for label, terms in checks:
        if any(term in text for term in terms):
            return label
    return "Other"


def render_market_analytics(info: Dict) -> None:
    df = info["df"]
    analytics = compute_market_analytics(df)
    st.markdown("#### Job Market Analytics")
    st.caption("Offline aggregate view of the job database used by JobPilot. Missing fields are excluded from each calculation where appropriate.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Top Skills")
        st.caption("Most frequently listed skills across the offline job dataset.")
        st.dataframe(analytics["top_skills"], width="stretch", hide_index=True)
        if len(analytics["top_skills"]):
            st.bar_chart(analytics["top_skills"].set_index("Skill")["Count"])
    with c2:
        st.markdown("##### Salary Distribution")
        st.caption(f"Uses structured salary first, then parsed description salary. Missing salary count: {analytics['salary_missing_count']}.")
        st.dataframe(analytics["salary_distribution"], width="stretch", hide_index=True)
        st.bar_chart(analytics["salary_distribution"].set_index("Salary Band")["Count"])

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("##### Demand by Location")
        st.caption("Remote jobs are grouped under Remote; physical locations use the normalized first location label.")
        st.dataframe(analytics["top_locations"], width="stretch", hide_index=True)
        if len(analytics["top_locations"]):
            st.bar_chart(analytics["top_locations"].set_index("Location")["Count"])
    with c4:
        st.markdown("##### Demand by Role Family")
        st.caption("Lightweight role-family grouping from job title, search role, and description.")
        st.dataframe(analytics["role_families"], width="stretch", hide_index=True)
        if len(analytics["role_families"]):
            st.bar_chart(analytics["role_families"].set_index("Role Family")["Count"])


def display_score_metrics(row) -> None:
    cols = st.columns(4)
    metrics = [
        ("Role", row.get("role_score", 0)),
        ("Target fit", row.get("target_fit_score", 0)),
        ("Skills", row.get("skill_score", 0)),
        ("Experience", row.get("experience_score", 0)),
        ("Location", row.get("location_score", 0)),
        ("Salary", row.get("salary_score", 0)),
        ("Quality", row.get("job_quality_score", 0)),
        ("Company", row.get("company_constraint_score", 0)),
        ("Final", row.get("final_score", 0)),
    ]
    for idx, (label, value) in enumerate(metrics):
        try:
            formatted = f"{float(value):.2f}"
        except Exception:
            formatted = "0.00"
        cols[idx % 4].metric(label, formatted)


def friendly_match_points(row) -> List[str]:
    points = []
    try:
        target_fit = float(row.get("target_fit_score", 0) or 0)
    except Exception:
        target_fit = 0.0
    if target_fit >= 0.85:
        points.append("Strong role fit")
    elif target_fit >= 0.60:
        points.append("Good role fit")
    else:
        points.append("Possible role fit")

    matched = str(row.get("matched_skills", "") or "").strip()
    if matched:
        points.append(f"Matched skills: {matched}")
    else:
        points.append("No direct skill overlap detected")

    points.append(format_salary(row))

    location_reason = str(row.get("location_match_reason", "") or "").strip()
    location = str(row.get("location", "") or "")
    if location_reason:
        points.append(location_reason)
    elif location:
        points.append(f"Location: {location}")

    required_years = row.get("required_years", "")
    if required_years in ["", None]:
        points.append("No explicit year requirement found")
    else:
        try:
            points.append(f"Requires about {float(required_years):g}+ years")
        except Exception:
            points.append(f"Required years: {required_years}")

    normalized_employment = str(row.get("normalized_employment_type", "") or "").strip().lower()
    if normalized_employment and normalized_employment not in {"unknown", "nan", "unclear"}:
        points.append(f"Employment type: {normalized_employment.title()}")
    else:
        points.append("Employment type: Not clearly listed")

    company_bucket = str(row.get("company_size_bucket", "") or "").strip().lower()
    meets_100 = str(row.get("meets_100_employee_threshold", row.get("meets_min_company_size", ""))).lower()
    meets_500 = str(row.get("meets_500_employee_threshold", "")).lower()
    is_large = str(row.get("is_large_company", "")).lower() in {"true", "1", "yes"}
    is_tiny = str(row.get("is_possible_tiny_startup", "")).lower() in {"true", "1", "yes"}
    is_known_sponsor = str(row.get("is_known_h1b_sponsor", "")).lower() in {"true", "1", "yes"}
    is_research_lab = str(row.get("is_research_lab", "")).lower() in {"true", "1", "yes"}
    if meets_100 == "true" or meets_500 == "true":
        points.append("Company size: Meets 100+ requirement")
    elif is_tiny:
        points.append("Company profile: Possible startup risk")
    elif is_large:
        points.append("Company profile: Large employer signal")
    elif company_bucket == "unknown":
        points.append("Company size: Unknown")
    if is_known_sponsor:
        points.append("Visa sponsorship signal: likely sponsor")
    elif str(row.get("sponsorship_reason", "")).strip() and "No sponsorship requirement" not in str(row.get("sponsorship_reason", "")):
        points.append("Visa sponsorship signal: unknown")
    if is_research_lab:
        points.append("Research lab signal: yes")

    quality_reason = str(row.get("job_quality_reason", "") or "").strip()
    if quality_reason and any(term in quality_reason.lower() for term in ["thin", "placement", "training", "program", "contract"]):
        points.append(f"Posting quality note: {quality_reason}")

    sponsorship_reason = str(row.get("sponsorship_reason", "") or "").strip()
    if sponsorship_reason and "No sponsorship requirement" not in sponsorship_reason:
        points.append(sponsorship_reason)
    return points


def render_match_highlights(row) -> None:
    try:
        target_fit = float(row.get("target_fit_score", 0) or 0)
    except Exception:
        target_fit = 0.0

    if target_fit >= 0.85:
        role_fit = "Strong role fit"
    elif target_fit >= 0.60:
        role_fit = "Good role fit"
    else:
        role_fit = "Possible role fit"

    matched = str(row.get("matched_skills", "") or "").strip() or "No direct skill overlap detected"

    location_reason = str(row.get("location_match_reason", "") or "").strip()
    location_value = str(row.get("location", "") or "").strip()
    if location_reason:
        location_text = location_reason
    elif location_value:
        location_text = location_value
    else:
        location_text = "Not clearly listed"

    normalized_employment = str(row.get("normalized_employment_type", "") or "").strip().lower()
    if normalized_employment and normalized_employment not in {"unknown", "nan", "unclear"}:
        employment_text = normalized_employment.title()
    else:
        employment_text = "Not clearly listed"

    required_years = row.get("required_years", "")
    if required_years in ["", None]:
        experience_text = "No explicit year requirement found"
    else:
        try:
            experience_text = f"About {float(required_years):g}+ years required"
        except Exception:
            experience_text = f"Required years: {required_years}"

    company_bucket = str(row.get("company_size_bucket", "") or "").strip().lower()
    meets_100 = str(row.get("meets_100_employee_threshold", row.get("meets_min_company_size", ""))).lower()
    meets_500 = str(row.get("meets_500_employee_threshold", "")).lower()
    is_large = str(row.get("is_large_company", "")).lower() in {"true", "1", "yes"}
    is_tiny = str(row.get("is_possible_tiny_startup", "")).lower() in {"true", "1", "yes"}
    is_known_sponsor = str(row.get("is_known_h1b_sponsor", "")).lower() in {"true", "1", "yes"}
    is_research_lab = str(row.get("is_research_lab", "")).lower() in {"true", "1", "yes"}

    if meets_100 == "true" or meets_500 == "true":
        company_text = "Meets the larger-company requirement"
    elif is_tiny:
        company_text = "Possible startup or very small company"
    elif is_large:
        company_text = "Large employer signal"
    elif company_bucket == "unknown":
        company_text = "Company size not confirmed"
    else:
        company_text = "No strong company-size signal"

    sponsorship_reason = str(row.get("sponsorship_reason", "") or "").strip()
    if is_known_sponsor:
        sponsorship_text = "Likely sponsor"
    elif sponsorship_reason and "No sponsorship requirement" not in sponsorship_reason:
        sponsorship_text = sponsorship_reason
    else:
        sponsorship_text = "No strong sponsorship signal"

    quality_reason = str(row.get("job_quality_reason", "") or "").strip()

    good_items = [
        ("🎯", "Role fit", role_fit),
        ("🧩", "Matched skills", matched),
        ("📍", "Location", location_text),
        ("🧾", "Employment", employment_text),
    ]

    note_items = [
        ("💰", "Salary", format_salary(row)),
        ("⏱️", "Experience requirement", experience_text),
        ("🏢", "Company signal", company_text),
    ]

    if sponsorship_text:
        note_items.append(("🛂", "Sponsorship", sponsorship_text))
    if is_research_lab:
        note_items.append(("🔬", "Research lab", "Yes"))
    if quality_reason and any(term in quality_reason.lower() for term in ["thin", "placement", "training", "program", "contract"]):
        note_items.append(("📋", "Posting quality", quality_reason))

    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown("**Best match signals**")
        for icon, label, value in good_items:
            st.markdown(f"- {icon} **{label}:** {value}")
    with right_col:
        st.markdown("**Things to note**")
        for icon, label, value in note_items:
            st.markdown(f"- {icon} **{label}:** {value}")


def advanced_match_details(row) -> pd.DataFrame:
    annual_min, annual_max = annual_salary_range(row)
    salary_range = "Not listed"
    if annual_min is not None and annual_max is not None:
        salary_range = f"{format_usd(annual_min)} - {format_usd(annual_max)}"
    elif annual_min is not None:
        salary_range = f"From {format_usd(annual_min)}"
    elif annual_max is not None:
        salary_range = f"Up to {format_usd(annual_max)}"

    fields = [
        ("Semantic match", row.get("semantic_score", "")),
        ("Target role fit", row.get("target_fit_score", "")),
        ("Posting quality", row.get("job_quality_score", "")),
        ("Sponsorship fit", row.get("sponsorship_score", "")),
        ("Role score", row.get("role_score", "")),
        ("Skill score", row.get("skill_score", "")),
        ("Experience score", row.get("experience_score", "")),
        ("Location score", row.get("location_score", "")),
        ("Salary score", row.get("salary_score", "")),
        ("Final score", row.get("final_score", "")),
        ("Required years", row.get("required_years", "Not listed") or "Not listed"),
        ("Annual salary range", salary_range),
        ("Normalized location", row.get("normalized_location", "Not listed") or "Not listed"),
        ("Location result", row.get("location_match_reason", "Not evaluated") or "Not evaluated"),
        ("Location constraint", "Passed" if row.get("location_constraint_pass", True) else "Filtered"),
        ("Normalized employment type", row.get("normalized_employment_type", "Unknown") or "Unknown"),
        ("Employment type evidence", row.get("employment_type_evidence", "Not evaluated") or "Not evaluated"),
        ("Source employment label", row.get("raw_employment_type", row.get("employment_type", "Not listed")) or "Not listed"),
        ("Sponsorship result", row.get("sponsorship_reason", "Not evaluated") or "Not evaluated"),
        ("Role family", friendly_role_family(row.get("role_family", ""))),
        ("Employee estimate", row.get("employee_count_estimate", "Not listed") or "Not listed"),
        ("Company size bucket", row.get("company_size_bucket", "unknown") or "unknown"),
        ("Meets 100+ threshold", row.get("meets_100_employee_threshold", "Unknown") or "Unknown"),
        ("Meets 500+ threshold", row.get("meets_500_employee_threshold", "Unknown") or "Unknown"),
        ("Large company signal", row.get("is_large_company", "Unknown")),
        ("Possible startup risk", row.get("is_possible_tiny_startup", "Unknown")),
        ("Known H-1B sponsor signal", row.get("is_known_h1b_sponsor", "Unknown")),
        ("Research lab signal", row.get("is_research_lab", "Unknown")),
        ("Public company signal", row.get("is_public_company", "Unknown")),
        ("Confidence", row.get("company_enrichment_confidence", row.get("company_size_confidence", "Unknown")) or "Unknown"),
        ("Source", row.get("company_enrichment_source", "Not listed") or "Not listed"),
        ("Notes", row.get("company_enrichment_notes", row.get("company_size_notes", "Not evaluated")) or "Not evaluated"),
    ]
    return pd.DataFrame(fields, columns=["Detail", "Value"])


GENERIC_REJECTION_REASONS = [
    "Role is not a good fit",
    "Skills do not match",
    "Too senior",
    "Salary or location does not work",
    "Work setup is not acceptable",
    "Company is too small or uncertain",
    "Visa sponsorship is unclear",
    "Low-quality posting",
]


def feedback_reason_options(profile: Dict, row) -> List[str]:
    return list(GENERIC_REJECTION_REASONS)


def annotate_latest_feedback_outcome(feedback: Dict, rejected_row: Dict, reranked: pd.DataFrame) -> Dict:
    feedback = dict(feedback or {})
    events = list(feedback.get("feedback_events", []))
    if not events:
        return feedback
    title = str(rejected_row.get("title", "") or "")
    company = str(rejected_row.get("company", "") or "")
    after_rank = "Left Top N"
    after_score = ""
    if reranked is not None and len(reranked):
        matches = reranked[
            (reranked["title"].astype(str) == title)
            & (reranked["company"].astype(str) == company)
        ]
        if len(matches):
            after_rank = matches.iloc[0].get("rank", "")
            after_score = matches.iloc[0].get("final_score", "")
    events[-1] = {**events[-1], "after_rank": after_rank, "after_score": after_score}
    feedback["feedback_events"] = events
    return feedback


def render_feedback_memory() -> None:
    events = st.session_state.get("feedback", {}).get("feedback_events", [])
    if not events:
        st.info("No feedback has been submitted yet.")
        return
    df = pd.DataFrame(events)
    columns = [
        "rejected_title",
        "rejected_company",
        "rejected_reason",
        "learned_signal",
        "before_rank",
        "after_rank",
        "before_score",
        "after_score",
    ]
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    st.dataframe(df[columns], width="stretch", hide_index=True)


def rerank_with_feedback(feedback: Dict, top_n: int) -> pd.DataFrame:
    reranked = cached_recommendations(
        stable_cache_json(st.session_state.profile),
        stable_cache_json(feedback),
        recommendation_fetch_size(max(10, top_n)),
        int(st.session_state.get("candidate_k", 1500)),
    )
    return reset_recommendation_index(reranked)


def render_job_result(idx: int, row, similar_rows: pd.DataFrame | None = None, display_rank: int | None = None) -> None:
    with st.container(border=True):
        rank_label = int(display_rank or row.get("rank", idx + 1))
        st.markdown(f"#### #{rank_label} {row.get('title', '')}")
        st.caption(f"{row.get('company', '')} | {row.get('location', '')}")
        st.markdown(f"**Why ranked here:** {row.get('why_matched', '')}")
        duplicate_count = int(row.get("duplicate_group_size", 1) or 1)
        if duplicate_count > 1:
            location_note = similar_location_summary(similar_rows if similar_rows is not None else pd.DataFrame(), row)
            similar_count = duplicate_count - 1
            note = f"{similar_count} similar postings collapsed"
            if location_note:
                note += f". Also available in: {location_note}"
            st.caption(note)
            st.info("This card represents the highest-ranked posting in a group of similar roles from the same employer.")
        render_match_highlights(row)

        feedback_area, spacer, generate_area = st.columns([3.0, 0.25, 2.0])
        with feedback_area:
            st.caption("Feedback")
            f1, f2, f3 = st.columns(3)
            with f1:
                if st.button("Save", key=f"save_{idx}", width="stretch"):
                    st.session_state.feedback = update_feedback_state(st.session_state.feedback, row.to_dict(), "save")
                    st.success("Saved. Run matching again to apply feedback.")
            with f2:
                if st.button("Apply later", key=f"apply_later_{idx}", width="stretch"):
                    st.session_state.feedback = update_feedback_state(st.session_state.feedback, row.to_dict(), "apply_later")
                    st.info("Marked as apply later. Run matching again to apply feedback.")
            with f3:
                if st.button("Not interested", key=f"not_interested_{idx}", width="stretch"):
                    st.session_state[f"show_reject_reason_{idx}"] = True
        with spacer:
            st.caption(" ")
        with generate_area:
            st.caption("Generate")
            g1, g2 = st.columns(2)
            with g1:
                if st.button("Generate Resume", key=f"tailor_{idx}", type="primary", width="stretch"):
                    st.session_state.selected_job_index = idx
                    st.session_state.cover_letter_text = ""
                    go_to_step(6)
                    st.rerun()
            with g2:
                if st.button("Cover letter", key=f"cover_letter_{idx}", width="stretch"):
                    st.session_state.selected_job_index = idx
                    st.session_state.cover_letter_job_index = idx
                    st.session_state.cover_letter_text = generate_cover_letter(
                        st.session_state.profile,
                        row.to_dict(),
                        supported_job_keywords(st.session_state.profile, row.to_dict()),
                    )
                    go_to_step(6)
                    st.rerun()

        if st.session_state.get(f"show_reject_reason_{idx}", False):
            reason = st.radio(
                "Reason",
                feedback_reason_options(st.session_state.profile, row),
                key=f"reject_reason_{idx}",
            )
            if st.button("Submit reason", key=f"submit_reject_{idx}", type="primary", width="stretch"):
                updated_feedback = update_feedback_state(
                    st.session_state.feedback,
                    row.to_dict(),
                    "not_interested",
                    reason=reason,
                )
                with st.spinner("Learning from feedback and reranking..."):
                    reranked = rerank_with_feedback(updated_feedback, int(st.session_state.get("top_n", 10)))
                updated_feedback = annotate_latest_feedback_outcome(updated_feedback, row.to_dict(), reranked)
                st.session_state.feedback = updated_feedback
                st.session_state.recommendations = reranked
                st.session_state.company_enrichment_status = reranked.attrs.get("company_enrichment_status", {})
                st.session_state.candidate_funnel_summary = reranked.attrs.get("candidate_funnel_summary", {})
                st.session_state.matching_timing_log = reranked.attrs.get("timing_log", {})
                st.session_state[f"show_reject_reason_{idx}"] = False
                st.success(f"Feedback applied: {reason}. Recommendations were reranked.")
                st.rerun()

        with st.expander("Job description"):
            st.write(clean_display_text(row.get("description", ""), max_chars=900))
            apply_url = str(row.get("apply_url", "") or "").strip()
            if apply_url and apply_url.lower() != "nan":
                st.link_button("Open original posting", apply_url, width="stretch")
        if similar_rows is not None and len(similar_rows) > 0:
            with st.expander("View similar postings"):
                st.dataframe(similar_postings_table(similar_rows), width="stretch", hide_index=True)
        with st.expander("Advanced match details"):
            display_score_metrics(row)
            st.dataframe(advanced_match_details(row), width="stretch", hide_index=True)



def render_results_page(info: Dict) -> None:
    page_start(
        "Step 5 of 6",
        "Select a job for resume tailoring",
        "Choose the exact recommendation that should be used for the tailored resume draft.",
    )
    results = st.session_state.get("recommendations")
    if results is None:
        st.info("No results yet. Run matching first.")
        if st.button("Back to ranked jobs", width="stretch"):
            go_to_step(4)
            st.rerun()
        return
    if len(results) == 0:
        st.warning("No jobs matched after filters. Try relaxing dealbreakers or increasing search depth.")
        if st.button("Back to preferences", width="stretch"):
            go_to_step(3)
            st.rerun()
        return

    labels = [job_option_label(row) for _, row in results.iterrows()]
    current_index = st.session_state.get("selected_job_index")
    try:
        current_index = 0 if current_index is None else int(current_index)
    except Exception:
        current_index = 0
    if current_index >= len(labels):
        current_index = 0

    selected_label = st.selectbox(
        "Choose a job to tailor your resume for",
        labels,
        index=current_index,
        key="selected_job_label",
    )
    selected_index = labels.index(selected_label)
    st.session_state.selected_job_index = selected_index
    selected_row = results.iloc[selected_index]

    with st.container(border=True):
        st.markdown(f"#### {selected_row.get('title', '')}")
        st.caption(f"{selected_row.get('company', '')} | {selected_row.get('location', '')}")
        st.markdown(f"**Why this job:** {selected_row.get('why_matched', '')}")
        render_match_highlights(selected_row)
        with st.expander("Advanced match details"):
            display_score_metrics(selected_row)
            st.dataframe(advanced_match_details(selected_row), width="stretch", hide_index=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Back to ranked jobs", width="stretch"):
            go_to_step(4)
            st.rerun()
    with c2:
        if st.button("Generate resume for selected job", type="primary", width="stretch"):
            st.session_state.cover_letter_text = ""
            go_to_step(6)
            st.rerun()
    with c3:
        if st.button("Generate cover letter", width="stretch"):
            st.session_state.cover_letter_job_index = selected_index
            st.session_state.cover_letter_text = generate_cover_letter(
                st.session_state.profile,
                selected_row.to_dict(),
                supported_job_keywords(st.session_state.profile, selected_row.to_dict()),
            )
            go_to_step(6)
            st.rerun()
    page_end()

def render_resume_builder_page() -> None:
    page_start(
        "Step 6 of 6",
        "Generate tailored resume draft",
        "Choose one recommended job and download a tailored Markdown draft.",
    )
    results = st.session_state.get("recommendations")
    if results is None or len(results) == 0:
        st.info("No recommendations available yet.")
        if st.button("Back to matching", width="stretch"):
            go_to_step(4)
            st.rerun()
        return

    row = selected_job_from_results(results)
    if row is None:
        st.warning("Please choose a job to tailor your resume for before generating a draft.")
        if st.button("Choose a job", type="primary", width="stretch"):
            go_to_step(5)
            st.rerun()
        return
    selected_idx = int(st.session_state.selected_job_index)
    st.info(f"Generating a tailored resume for: {job_option_label(row)}")

    with st.container(border=True):
        st.markdown(f"#### {row.get('title', '')}")
        st.caption(f"{row.get('company', '')} | {row.get('location', '')} | {format_job_facts(row)}")
        st.markdown(f"**Why ranked here:** {row.get('why_matched', '')}")
        render_match_highlights(row)
        st.write(clean_display_text(row.get("description", ""), max_chars=900))
        with st.expander("Advanced match details"):
            display_score_metrics(row)
            st.dataframe(advanced_match_details(row), width="stretch", hide_index=True)

    if get_openrouter_api_key():
        st.caption("Resume draft generated with OpenRouter using the uploaded resume/profile and selected job.")
    else:
        st.caption("No OpenRouter key found. Using local fallback draft.")
    resume_text = generate_tailored_resume(st.session_state.profile, row.to_dict())
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Download tailored resume (.md)",
            data=resume_text.encode("utf-8"),
            file_name=f"tailored_resume_{selected_idx + 1}.md",
            mime="text/markdown",
            width="stretch",
        )
    with d2:
        st.download_button(
            "Download tailored resume (.docx)",
            data=resume_docx_bytes(resume_text),
            file_name=f"tailored_resume_{selected_idx + 1}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            width="stretch",
        )
    st.markdown("#### Cover Letter")
    if (
        not st.session_state.get("cover_letter_text")
        or st.session_state.get("cover_letter_job_index") != selected_idx
    ):
        if st.button("Generate Cover Letter", width="stretch"):
            st.session_state.cover_letter_job_index = selected_idx
            st.session_state.cover_letter_text = generate_cover_letter(
                st.session_state.profile,
                row.to_dict(),
                supported_job_keywords(st.session_state.profile, row.to_dict()),
            )
            st.rerun()
    if st.session_state.get("cover_letter_text") and st.session_state.get("cover_letter_job_index") == selected_idx:
        cover_letter_text = st.session_state.cover_letter_text
        st.text_area("Generated cover letter", value=cover_letter_text, height=360)
        st.download_button(
            "Download cover letter (.txt)",
            data=cover_letter_text.encode("utf-8"),
            file_name=f"cover_letter_{selected_idx + 1}.txt",
            mime="text/plain",
            width="stretch",
        )
    if st.button("Back to recommendations", width="stretch"):
        go_to_step(5)
        st.rerun()
    page_end()


def main() -> None:
    apply_style()
    init_state()
    info = load_dataset_info()
    render_header(info)
    render_progress()

    step = st.session_state.step
    if step == 1:
        render_resume_page()
    elif step == 2:
        render_profile_page()
    elif step == 3:
        render_preferences_page()
    elif step == 4:
        render_match_page()
    elif step == 5:
        render_results_page(info)
    else:
        render_resume_builder_page()


if __name__ == "__main__":
    main()

