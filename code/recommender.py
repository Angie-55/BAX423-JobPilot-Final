"""
Structured JobPilot recommender.

Pipeline:
1. Dense embedding retrieval with sentence-transformers and FAISS.
2. Hard filters for dealbreakers such as seniority, contract, location, and visa constraints.
3. Multi-stage scoring and re-ranking using role, skills, experience, location, source freshness, and feedback.
4. Explanation generation for each recommended job.
5. Simple adaptive learning from accept, reject, and skip feedback.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import faiss
import numpy as np
import pandas as pd
import company_enrichment as company_enrichment

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data/processed")
JOBS_CSV = DATA_DIR / "jobs_final.csv"
FAISS_PATH = DATA_DIR / "faiss.index"
ROLE_INDEX_PATH = DATA_DIR / "role_index.pkl"
ROLE_VOCAB_PATH = DATA_DIR / "role_vocabulary.pkl"
LOCATION_VOCAB_PATH = DATA_DIR / "location_vocabulary.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
STATIC_FEATURE_VERSION = 7

SKILL_LIST = [
    "python", "sql", "r", "excel", "tableau", "power bi", "machine learning", "deep learning",
    "aws", "azure", "gcp", "spark", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
    "nlp", "statistics", "a/b testing", "java", "javascript", "typescript", "react", "node",
    "docker", "kubernetes", "git", "linux", "airflow", "dbt", "snowflake", "looker", "salesforce",
    "financial modeling", "forecasting", "accounting", "risk", "cybersecurity", "cloud", "etl",
    "data visualization", "communication", "stakeholder", "product management", "agile", "scrum",
    "figma", "ux", "ui", "marketing", "seo", "consulting", "powerpoint", "gaap", "audit",
    "analytics", "data analysis", "dashboard", "regression", "classification", "business intelligence",
    "kubernetes", "microservices", "kafka", "computer vision", "research", "optimization", "mlops",
    "sas", "stata", "vba", "jira", "confluence", "excel pivot tables", "power automate",
    "data warehousing", "redshift", "bigquery", "hadoop", "api", "rest api", "crm",
    "experimentation", "causal inference", "time series", "recommender systems",
]

SKILL_SYNONYMS = {
    "machine learning": ["machine learning", "ml", "predictive modeling", "model training"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "power bi": ["power bi", "powerbi"],
    "data visualization": ["data visualization", "visualisation", "dashboard", "dashboards"],
    "business intelligence": ["business intelligence", "bi"],
    "a/b testing": ["a/b testing", "ab testing", "experimentation"],
    "stakeholder": ["stakeholder", "cross-functional", "business partner"],
    "aws": ["aws", "amazon web services"],
    "gcp": ["gcp", "google cloud"],
    "nlp": ["nlp", "natural language processing"],
    "mlops": ["mlops", "model deployment", "model monitoring", "ml platform"],
}

ROLE_FAMILIES = {
    "ux_research": {
        "title": ["ux researcher", "user researcher", "ux research", "user research", "design researcher"],
        "text": [
            "ux research", "user research", "usability studies", "usability testing",
            "human factors", "hci", "mixed-method research", "mixed method research",
            "qualitative research", "research logistics",
        ],
        "avoid": ["machine learning", "deep learning", "pytorch", "tensorflow"],
    },
    "product_research": {
        "title": ["product researcher", "product research", "consumer insights researcher"],
        "text": ["concept studies", "concept testing", "survey research", "anthropology", "sociology", "product research"],
        "avoid": ["machine learning", "deep learning", "model training"],
    },
    "market_research": {
        "title": ["market researcher", "market research analyst", "market research"],
        "text": ["market research", "survey research", "consumer insights", "research reports", "focus groups"],
        "avoid": ["machine learning", "deep learning", "statistical modeling"],
    },
    "ml_platform_mlops": {
        "title": [
            "ml platform engineer", "machine learning platform engineer", "mlops engineer",
            "machine learning ops engineer", "machine learning infrastructure engineer",
            "ml infrastructure engineer", "ml systems engineer", "ai infrastructure engineer",
            "machine learning engineer, infrastructure", "machine learning engineer, platform",
        ],
        "text": [
            "ml platform", "mlops", "model serving", "model deployment", "model monitoring",
            "feature store", "distributed training", "training infrastructure", "inference infrastructure",
            "production ml", "ml pipeline", "llm deployment", "vllm", "kubeflow", "ray",
        ],
        "avoid": ["reporting analyst", "marketing analyst", "financial analyst"],
    },
    "ml_engineering": {
        "title": ["machine learning engineer", "ml engineer", "ai engineer", "applied ai engineer"],
        "text": [
            "machine learning", "deep learning", "pytorch", "tensorflow", "scikit-learn",
            "model training", "model deployment", "production model", "computer vision", "nlp",
        ],
        "avoid": ["business analyst", "marketing analyst", "financial analyst"],
    },
    "data_science": {
        "title": ["data scientist", "applied scientist", "decision scientist", "research scientist", "ai data scientist"],
        "text": ["machine learning", "modeling", "statistics", "experiment", "python", "scikit-learn", "causal inference"],
        "avoid": ["business analyst", "financial analyst", "marketing analyst"],
    },
    "data_analytics": {
        "title": ["data analyst", "business analyst", "analytics analyst", "product analyst", "reporting analyst"],
        "text": ["sql", "excel", "tableau", "dashboard", "reporting", "data analysis", "analytics"],
        "avoid": ["software engineer", "frontend", "backend", "full stack", "java developer"],
    },
    "bi_analytics": {
        "title": ["bi analyst", "business intelligence analyst", "bi developer", "business intelligence developer"],
        "text": ["business intelligence", "power bi", "looker", "tableau", "dashboard", "semantic layer"],
        "avoid": ["marketing analyst", "financial analyst"],
    },
    "analytics_engineering": {
        "title": [
            "analytics engineer", "data analytics engineer", "bi engineer", "business intelligence engineer",
            "data platform analyst", "data engineer - analytics", "data engineer analytics",
            "analytics data engineer", "data warehouse engineer",
        ],
        "text": [
            "dbt", "snowflake", "bigquery", "data warehouse", "semantic layer", "etl",
            "analytics engineering", "bi engineering", "data platform",
        ],
        "avoid": ["marketing analyst", "financial analyst", "business analyst", "ml engineer", "data scientist", "software engineer", "backend engineer"],
    },
    "data_engineering": {
        "title": ["data engineer", "etl engineer", "pipeline engineer"],
        "text": ["etl", "pipeline", "spark", "airflow", "dbt", "snowflake", "data warehouse", "bigquery"],
        "avoid": ["marketing analyst", "financial analyst"],
    },
    "software_engineering": {
        "title": ["software engineer", "developer", "frontend", "backend", "full stack", "java engineer"],
        "text": ["api", "microservices", "react", "node", "java", "typescript", "software development"],
        "avoid": ["data analyst", "business analyst"],
    },
    "backend_engineering": {
        "title": ["backend engineer", "back end engineer", "server engineer", "api engineer"],
        "text": ["backend", "back end", "api", "microservices", "server-side", "distributed systems"],
        "avoid": ["data analyst", "business analyst", "analytics engineer"],
    },
    "devops": {
        "title": ["devops engineer", "site reliability engineer", "sre", "platform engineer", "cloud engineer"],
        "text": ["ci/cd", "terraform", "kubernetes", "linux", "aws", "azure", "gcp", "infrastructure"],
        "avoid": ["data analyst", "business analyst", "data scientist"],
    },
    "finance_accounting": {
        "title": ["financial analyst", "finance analyst", "accounting analyst", "auditor", "tax consultant", "accountant"],
        "text": ["financial modeling", "forecasting", "accounting", "gaap", "audit", "variance analysis"],
        "avoid": ["machine learning engineer", "frontend"],
    },
    "risk_analytics": {
        "title": ["risk data scientist", "risk analyst", "risk analytics", "risk modeling analyst", "model risk analyst"],
        "text": ["risk analytics", "risk model", "risk modeling", "credit risk", "fraud risk", "portfolio risk", "loss forecasting", "stress testing", "fintech"],
        "avoid": ["marketing analyst", "seo", "sales", "audit", "accounting"],
    },
    "credit_risk": {
        "title": ["credit risk analyst", "credit analyst", "credit risk data scientist", "credit risk modeler"],
        "text": ["credit risk", "credit model", "loan", "lending", "underwriting", "fico", "default risk", "loss given default", "probability of default"],
        "avoid": ["marketing analyst", "sales", "audit", "accounting"],
    },
    "quantitative_analytics": {
        "title": ["quantitative analyst", "quant analyst", "quantitative data scientist", "quantitative researcher"],
        "text": ["quantitative analysis", "quantitative modeling", "statistical modeling", "time series", "portfolio", "trading", "risk model"],
        "avoid": ["marketing analyst", "sales", "accounting"],
    },
    "financial_analytics": {
        "title": ["financial analyst", "finance analyst", "financial data analyst", "financial data scientist"],
        "text": ["financial modeling", "forecasting", "variance analysis", "financial analytics", "finance analytics", "banking", "fintech"],
        "avoid": ["accounting", "audit", "sales"],
    },
    "marketing_analytics": {
        "title": ["marketing analyst", "growth analyst", "marketing data analyst", "seo analyst", "campaign analyst"],
        "text": ["marketing analytics", "campaign", "seo", "paid media", "attribution", "brand", "growth marketing"],
        "avoid": ["credit risk", "risk modeling", "quantitative analyst"],
    },
    "accounting": {
        "title": ["accountant", "accounting analyst", "tax accountant", "controller"],
        "text": ["accounting", "gaap", "ledger", "journal entries", "accounts payable", "accounts receivable", "tax"],
        "avoid": ["data scientist", "machine learning"],
    },
    "audit": {
        "title": ["auditor", "audit analyst", "internal audit", "it audit"],
        "text": ["audit", "sox", "internal controls", "compliance testing", "risk control"],
        "avoid": ["data scientist", "machine learning"],
    },
    "sales": {
        "title": ["sales representative", "account executive", "sales manager", "business development representative"],
        "text": ["sales quota", "pipeline", "prospecting", "cold calling", "account executive", "customer acquisition"],
        "avoid": ["data scientist", "analytics engineer"],
    },
    "operations": {
        "title": ["operations analyst", "operations manager", "business operations", "data center operations"],
        "text": ["operations", "process improvement", "vendor management", "data center operations", "facilities"],
        "avoid": ["data scientist", "machine learning"],
    },
    "executive_management": {
        "title": ["chief", "coo", "ceo", "cfo", "cto", "vp", "vice president", "director", "head of", "general manager"],
        "text": ["executive leadership", "business strategy", "operations leadership", "board", "p&l"],
        "avoid": ["individual contributor", "data analyst"],
    },
    "product_management": {
        "title": ["product manager", "technical product manager", "product owner", "program manager"],
        "text": ["roadmap", "product strategy", "requirements", "stakeholder", "go-to-market", "product launch"],
        "avoid": ["data scientist", "backend engineer"],
    },
    "consulting": {
        "title": ["consultant", "strategy consultant", "analytics consultant", "management consultant"],
        "text": ["consulting", "client engagements", "advisory", "strategy", "recommendations"],
        "avoid": [],
    },
    "marketing": {
        "title": ["marketing analyst", "growth marketing analyst", "marketing data analyst", "seo analyst"],
        "text": ["campaign", "seo", "marketing", "attribution", "brand", "paid media"],
        "avoid": ["machine learning engineer", "mlops engineer"],
    },
    "other": {"title": [], "text": [], "avoid": []},
}

FRIENDLY_ROLE_FAMILY_LABELS = {
    "ux_research": "UX research",
    "product_research": "Product research",
    "market_research": "Market research",
    "ml_platform_mlops": "ML platform / MLOps",
    "ml_engineering": "ML engineering",
    "data_science": "Data science",
    "data_analytics": "Data analytics",
    "bi_analytics": "BI analytics",
    "analytics_engineering": "Analytics engineering",
    "data_engineering": "Data engineering",
    "software_engineering": "Software engineering",
    "backend_engineering": "Backend engineering",
    "devops": "DevOps / platform",
    "finance_accounting": "Finance / accounting",
    "risk_analytics": "Risk analytics",
    "credit_risk": "Credit risk",
    "quantitative_analytics": "Quantitative analytics",
    "financial_analytics": "Financial analytics",
    "marketing_analytics": "Marketing analytics",
    "accounting": "Accounting",
    "audit": "Audit",
    "sales": "Sales",
    "operations": "Operations",
    "executive_management": "Executive management",
    "product_management": "Product management",
    "consulting": "Consulting",
    "marketing": "Marketing",
    "other": "Other",
}

SPONSOR_COMPANY_HINTS = [
    "amazon", "google", "microsoft", "meta", "apple", "nvidia", "adobe", "oracle", "salesforce",
    "ibm", "intel", "cisco", "uber", "lyft", "netflix", "doordash", "capital one", "jpmorgan",
    "goldman", "bloomberg", "servicenow", "databricks", "snowflake", "openai", "anthropic",
    "rocket", "redfin", "hugging face", "genbio", "geico", "expedia", "united airlines", "red hat",
    "paramount", "snap", "cribl", "zendesk", "rockstar", "adobe", "the hartford",
]

RESEARCH_COMPANY_HINTS = [
    "university", "college", "research institute", "research lab", "ai lab", "labs",
    "institute of technology", "national laboratory", "medical center",
]

RESEARCH_TITLE_HINTS = [
    "research scientist", "applied scientist", "research engineer", "ai scientist",
    "scientist - machine learning", "open-source machine learning",
]

RESEARCH_TEXT_HINTS = [
    "research lab", "ai lab", "applied research", "foundation model", "published research",
]

TECHNICAL_IC_TARGET_TERMS = [
    "ml engineer", "machine learning engineer", "applied scientist", "data scientist",
    "data analyst", "bi analyst", "business intelligence analyst", "analytics engineer",
    "data engineer", "mlops engineer", "ml platform engineer", "machine learning platform",
    "ml infrastructure", "ai engineer",
]

MANAGEMENT_TARGET_TERMS = [
    "manager", "management", "director", "head of", "leadership", "executive", "chief",
    "vp", "vice president", "president", "founder", "co-founder", "operations officer",
    "general manager", "strategy lead",
]

EXECUTIVE_OPERATIONS_TITLE_PATTERNS = [
    r"\bchief\b", r"\bcoo\b", r"\bceo\b", r"\bcto\b", r"\bcfo\b", r"\bcmo\b",
    r"\bfounder\b", r"\bco[- ]founder\b", r"\bpresident\b", r"\bvice president\b", r"\bvp\b",
    r"\bdirector\b", r"\bhead of\b", r"\boperations officer\b", r"\boperations manager\b",
    r"\bgeneral manager\b", r"\bbusiness operations\b", r"\bstrategy lead\b", r"\bexecutive\b",
]

DATA_CENTER_CONTEXT_PATTERNS = [
    r"\bdata cent(?:er|re)\b",
    r"\bdata cent(?:er|re) operations\b",
    r"\bdata cent(?:er|re) business\b",
]

NO_SPONSORSHIP_PATTERNS = [
    "will not be providing visa sponsorship",
    "not offering visa sponsorship",
    "will not offer visa sponsorship",
    "does not offer visa sponsorship",
    "does not provide visa sponsorship",
    "do not provide visa sponsorship",
    "does not sponsor",
    "don't sponsor",
    "do not sponsor",
    "no sponsorship",
    "without sponsorship",
    "not sponsor",
    "cannot sponsor",
    "unable to sponsor",
    "without a need for current or future visa sponsorship",
    "work visa/immigration sponsorship is not available",
    "sponsorship is not available",
    "must be authorized to work in the united states without sponsorship",
    "must be authorized to work in the us without sponsorship",
    "must be a us citizen",
    "must be us citizen",
    "us citizenship required",
    "must be a permanent resident",
    "must be permanent resident",
    "permanent resident required",
]

NO_SPONSORSHIP_REGEX_PATTERNS = [
    r"\bnot\s+(?:offering|providing|available\s+for)\s+(?:visa|immigration|work\s+visa)?\s*sponsorship\b",
    r"\b(?:must|need|required)\s+.*\b(?:u\.?\s*s\.?|united\s+states)\s+citizens?\b",
    r"\b(?:u\.?\s*s\.?|united\s+states)\s+citizenship\s+(?:is\s+)?required\b",
    r"\b(?:must|need|required)\s+.*\bpermanent\s+residents?\b",
    r"\b(?:green\s+card|permanent\s+resident)\s+(?:holder\s+)?(?:only|required)\b",
    r"\b(?:without|no)\s+(?:current\s+or\s+future\s+)?(?:visa|immigration|work\s+visa)?\s*sponsorship\b",
]

DEFENSE_HINTS = ["defense", "defence", "military", "army", "navy", "air force", "dod", "department of defense"]

REGION_ALIASES = {
    "bay_area": [
        "bay area", "ca bay area", "san francisco", "sf", "san jose", "oakland", "berkeley",
        "palo alto", "mountain view", "sunnyvale", "santa clara", "cupertino", "menlo park",
        "redwood city", "san mateo", "fremont", "hayward", "daly city", "milpitas",
        "foster city", "burlingame", "south san francisco",
    ],
    "new_york": ["new york", "nyc", "manhattan", "brooklyn", "queens", "jersey city", "hoboken", "new york city"],
    "seattle_area": ["seattle", "bellevue", "redmond", "kirkland", "renton"],
    "boston_area": ["boston", "cambridge", "somerville"],
    "la_area": ["los angeles", "santa monica", "pasadena", "irvine", "long beach", "glendale", "burbank"],
    "chicago_area": ["chicago", "evanston", "naperville", "schaumburg"],
    "austin_area": ["austin", "round rock", "cedar park"],
    "texas_region": ["texas", "tx", "austin", "dallas", "houston", "san antonio", "round rock", "cedar park"],
    "northeast": ["new york", "nyc", "new jersey", "jersey city", "newark", "boston", "cambridge", "massachusetts"],
    "midwest": ["chicago", "illinois", "evanston", "naperville", "schaumburg", "detroit", "minneapolis"],
    "south": ["texas", "austin", "dallas", "houston", "atlanta", "miami", "charlotte", "raleigh"],
    "west_coast": ["california", "san francisco", "bay area", "los angeles", "seattle", "portland"],
}

METRO_REGION_KEYS = {"new_york", "bay_area", "seattle_area", "boston_area", "la_area", "chicago_area", "austin_area"}
BROAD_REGION_KEYS = {"northeast", "midwest", "south", "west_coast", "texas_region"}

STATE_BROAD_REGIONS = {
    "NY": "northeast", "NJ": "northeast", "MA": "northeast", "CT": "northeast", "PA": "northeast",
    "IL": "midwest", "MI": "midwest", "MN": "midwest", "OH": "midwest", "WI": "midwest",
    "TX": "texas_region", "GA": "south", "FL": "south", "NC": "south", "VA": "south",
    "CA": "west_coast", "WA": "west_coast", "OR": "west_coast",
}

STATE_ALIASES = {
    "california": "CA", "ca": "CA",
    "new york": "NY", "ny": "NY",
    "washington": "WA", "wa": "WA",
    "district of columbia": "DC", "dc": "DC",
    "massachusetts": "MA", "ma": "MA",
    "texas": "TX", "tx": "TX",
}

US_STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "IA": "Iowa", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "MA": "Massachusetts", "MD": "Maryland",
    "ME": "Maine", "MI": "Michigan", "MN": "Minnesota", "MO": "Missouri", "MS": "Mississippi",
    "MT": "Montana", "NC": "North Carolina", "ND": "North Dakota", "NE": "Nebraska",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NV": "Nevada",
    "NY": "New York", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
    "TX": "Texas", "UT": "Utah", "VA": "Virginia", "VT": "Vermont", "WA": "Washington",
    "WI": "Wisconsin", "WV": "West Virginia", "WY": "Wyoming", "DC": "District of Columbia",
}

STATE_NAME_TO_CODE = {name.lower(): code for code, name in US_STATE_NAMES.items()}
STATE_NAME_TO_CODE.update({code.lower(): code for code in US_STATE_NAMES})

US_STATE_CODES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "IA", "ID", "IL", "IN",
    "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH",
    "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA",
    "VT", "WA", "WI", "WV", "WY", "DC",
}

COUNTRY_ALIASES = {
    "united states": "US", "usa": "US", "u.s.": "US", "u.s": "US", "us": "US",
    "united kingdom": "GB", "uk": "GB", "gb": "GB", "england": "GB", "scotland": "GB",
    "germany": "DE", "deutschland": "DE", "de": "DE",
    "taiwan": "TW", "taipei": "TW", "tw": "TW",
    "canada": "CA_COUNTRY", "singapore": "SG", "india": "IN", "australia": "AU",
    "france": "FR", "netherlands": "NL", "ireland": "IE",
}

NON_US_LOCATION_TERMS = {
    "sheffield": "GB", "london": "GB", "hamburg": "DE", "berlin": "DE", "munich": "DE",
    "taipei": "TW", "toronto": "CA_COUNTRY", "vancouver": "CA_COUNTRY", "singapore": "SG",
    "india": "IN", "sydney": "AU", "melbourne": "AU", "paris": "FR", "amsterdam": "NL", "dublin": "IE",
}


def parse_number(value) -> float | None:
    text = str(value or "").strip().replace(",", "").replace("$", "")
    if text.lower() in {"", "nan", "none", "null"}:
        return None
    try:
        return float(text)
    except Exception:
        return None


def salary_multiplier(period: str) -> float | None:
    text = safe_lower(period).replace(".", "").replace("/", " ").strip()
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


def annualize_salary_value(value, period: str) -> float | None:
    number = parse_number(value)
    if number is None:
        return None
    mult = salary_multiplier(period)
    if mult is None:
        # Some sources omit salary_period. Large salary numbers are usually already annual.
        if number >= 10000:
            mult = 1.0
        else:
            return None
    return number * mult


SALARY_CONTEXT_REJECT_TERMS = [
    "hour", "hourly", "/hour", "/hr", "per hour", "bonus", "signing", "sign-on",
    "equity", "stock", "option", "relocation", "allowance", "benefit", "401k",
    "paid time off", "paid holiday", "paid leave", "stipend", "commission",
]

SALARY_CONTEXT_ACCEPT_TERMS = [
    "base pay", "salary", "salary range", "compensation", "annual salary",
    "per year", "/yr", "yearly", "annually", "annual base",
]

SALARY_AMOUNT_RE = re.compile(
    r"\$\s*(\d{2,3}(?:,\d{3})?|\d+(?:\.\d+)?)\s*([kKmM]?)"
    r"(?!\s*(?:/|per\s+)?(?:hour|hr|month|week|day))"
)

SALARY_RANGE_RE = re.compile(
    r"\$\s*(\d{2,3}(?:,\d{3})?|\d+(?:\.\d+)?)\s*([kKmM]?)"
    r"(?!\s*(?:/|per\s+)?(?:hour|hr|month|week|day))"
    r"\s*(?:/yr|per\s+year|annually|yearly)?\s*(?:-|–|—|to)\s*"
    r"\$\s*(\d{2,3}(?:,\d{3})?|\d+(?:\.\d+)?)\s*([kKmM]?)"
    r"(?!\s*(?:/|per\s+)?(?:hour|hr|month|week|day))"
    r"\s*(?:/yr|per\s+year|annually|yearly)?",
    flags=re.I,
)


def normalize_salary_amount(number_text: str, suffix: str = "") -> int | None:
    try:
        value = float(str(number_text or "").replace(",", ""))
    except Exception:
        return None
    suffix = str(suffix or "").lower()
    if suffix == "k":
        value *= 1000
    elif suffix == "m":
        value *= 1_000_000
    elif value < 1000 and value >= 30:
        value *= 1000
    value = int(round(value))
    if value < 30_000 or value > 1_000_000:
        return None
    return value


def compact_salary_display(value: int) -> str:
    if value % 1000 == 0:
        return f"${int(value / 1000)}K"
    return "$" + format(int(value), ",")


def salary_display(min_value: int | float | None, max_value: int | float | None) -> str:
    if min_value is None and max_value is None:
        return "Not listed"
    if min_value is None:
        return compact_salary_display(int(max_value))
    if max_value is None or int(min_value) == int(max_value):
        return compact_salary_display(int(min_value))
    lo, hi = sorted([int(min_value), int(max_value)])
    return f"{compact_salary_display(lo)}–{compact_salary_display(hi)}"


def salary_context_is_rejected(context: str) -> bool:
    text = safe_lower(context)
    if any(term in text for term in ["per year", "/yr", "annually", "yearly", "annual salary", "base pay", "salary range"]):
        text = text.replace("annual bonus", "bonus")
        return any(term in text for term in ["hour", "hourly", "/hour", "/hr", "per hour", "signing", "sign-on", "relocation", "stipend", "commission only"])
    return any(term in text for term in SALARY_CONTEXT_REJECT_TERMS)


def extract_salary_from_description(description: str) -> Dict:
    text = " ".join(str(description or "").split())
    missing = {
        "salary_min": None,
        "salary_max": None,
        "salary_display": "Not listed",
        "salary_source": "missing",
    }
    if not text:
        return missing

    candidates: List[Tuple[int, int]] = []
    for match in SALARY_RANGE_RE.finditer(text):
        start, end = match.span()
        context = text[max(0, start - 80): min(len(text), end + 80)]
        if salary_context_is_rejected(context):
            continue
        first = normalize_salary_amount(match.group(1), match.group(2))
        second = normalize_salary_amount(match.group(3), match.group(4))
        if first is not None and second is not None:
            candidates.append(tuple(sorted([first, second])))

    if not candidates:
        for match in SALARY_AMOUNT_RE.finditer(text):
            start, end = match.span()
            context = text[max(0, start - 80): min(len(text), end + 80)]
            if salary_context_is_rejected(context):
                continue
            context_lower = safe_lower(context)
            if not any(term in context_lower for term in SALARY_CONTEXT_ACCEPT_TERMS):
                continue
            value = normalize_salary_amount(match.group(1), match.group(2))
            if value is not None:
                candidates.append((value, value))

    if not candidates:
        return missing
    salary_min, salary_max = candidates[0]
    return {
        "salary_min": salary_min,
        "salary_max": salary_max,
        "salary_display": salary_display(salary_min, salary_max),
        "salary_source": "description",
    }


def annual_salary_range(row: pd.Series | Dict) -> Tuple[float | None, float | None]:
    period = str(row.get("salary_period", "") or "")
    annual_min = annualize_salary_value(row.get("salary_min"), period)
    annual_max = annualize_salary_value(row.get("salary_max"), period)
    source_min = parse_number(row.get("salary_annual_min"))
    source_max = parse_number(row.get("salary_annual_max"))
    if annual_min is None and source_min is not None:
        annual_min = source_min
    if annual_max is None and source_max is not None:
        annual_max = source_max
    if annual_min is None and annual_max is None:
        parsed = extract_salary_from_description(row.get("description", ""))
        annual_min = parsed.get("salary_min")
        annual_max = parsed.get("salary_max")
    if annual_min is not None and annual_max is not None and annual_min > annual_max:
        annual_min, annual_max = annual_max, annual_min
    return annual_min, annual_max


def salary_source_for_row(row: pd.Series | Dict) -> str:
    cached = str(row.get("salary_source_static", "") or row.get("salary_source", "") or "").strip()
    if cached in {"structured", "description"}:
        return cached
    period = str(row.get("salary_period", "") or "")
    structured_min = annualize_salary_value(row.get("salary_min"), period)
    structured_max = annualize_salary_value(row.get("salary_max"), period)
    if structured_min is not None or structured_max is not None:
        return "structured"
    source_min = parse_number(row.get("salary_annual_min"))
    source_max = parse_number(row.get("salary_annual_max"))
    if source_min is not None or source_max is not None:
        return str(row.get("salary_source", "") or "structured")
    parsed = extract_salary_from_description(row.get("description", ""))
    return parsed.get("salary_source", "missing")


def profile_min_annual_salary(profile: Dict) -> float | None:
    return parse_number(profile.get("salary_min"))

def listed_salary_below_preference(profile: Dict, row: pd.Series | Dict) -> bool:
    requested = profile_min_annual_salary(profile)
    if requested is None:
        return False
    annual_min, annual_max = annual_salary_range(row)

    # If salary is missing, keep the job but let the salary score penalize it.
    if annual_min is None and annual_max is None:
        return False

    # If a listed range cannot reach the user's minimum, it should not pass the hard filter.
    comparable = annual_max if annual_max is not None else annual_min
    return comparable is not None and comparable < requested


def job_text_for_signals(row: pd.Series | Dict) -> str:
    return safe_lower(
        f"{row.get('title', '')} {row.get('company', '')} {row.get('description', '')} "
        f"{row.get('clean_job_text', '')} {row.get('job_text', '')}"
    )


def has_no_sponsorship_signal(row: pd.Series | Dict) -> bool:
    text = job_text_for_signals(row)
    compact_text = re.sub(r"[^a-z0-9+#]+", " ", text).strip()
    return any(pattern in text for pattern in NO_SPONSORSHIP_PATTERNS) or any(
        re.search(pattern, compact_text) for pattern in NO_SPONSORSHIP_REGEX_PATTERNS
    )


def company_sponsor_signal(row: pd.Series | Dict) -> str:
    company = safe_lower(row.get("company"))
    title = safe_lower(row.get("title"))
    text = job_text_for_signals(row)
    if any(hint in company for hint in SPONSOR_COMPANY_HINTS):
        return "known sponsor"
    if (
        any(hint in company for hint in RESEARCH_COMPANY_HINTS)
        or any(hint in title for hint in RESEARCH_TITLE_HINTS)
        or any(hint in text for hint in RESEARCH_TEXT_HINTS)
    ):
        return "research lab"
    if any(term in text for term in ["h-1b", "h1b", "visa sponsorship", "immigration sponsorship"]) and not has_no_sponsorship_signal(row):
        return "sponsorship mentioned"
    return "unknown"


def sponsorship_score_fn(profile: Dict, row: pd.Series | Dict) -> Tuple[float, str]:
    dealbreakers = parse_dealbreakers(profile)
    text = safe_lower(
        " ".join(
            [str(profile.get("target_role", "")), str(profile.get("location_preference", ""))]
            + [str(x) for x in profile.get("dealbreakers", [])]
            + [str(x) for x in profile.get("preferred_industries", [])]
        )
    )
    needs_sponsorship = dealbreakers.get("need_sponsorship") or any(
        term in text for term in ["h-1b", "h1b", "visa", "sponsorship", "international"]
    )
    if not needs_sponsorship:
        return 0.50, "No sponsorship requirement provided"
    if has_no_sponsorship_signal(row):
        return 0.0, "Does not appear compatible with visa sponsorship"
    signal = company_sponsor_signal(row)
    if signal == "known sponsor":
        return 1.0, "Known sponsor-style company"
    if signal == "research lab":
        return 0.92, "Research lab or research-focused employer"
    if signal == "sponsorship mentioned":
        return 0.85, "Sponsorship is mentioned"
    return 0.42, "Sponsorship compatibility is unclear"


COMPANY_ENRICHMENT_CSV = DATA_DIR / "company_enrichment.csv"

COMPANY_SIZE_BUCKETS = [
    (1, 10, "1-10"),
    (11, 50, "11-50"),
    (51, 99, "51-99"),
    (100, 499, "100-499"),
    (500, 999, "500-999"),
    (1000, float("inf"), "1000+"),
]

TINY_COMPANY_TERMS = [
    "small team", "founding team", "seed-stage", "seed stage", "early-stage startup",
    "early stage startup", "stealth startup", "startup environment", "start-up environment",
]


def company_size_bucket(employee_count: float | int | None) -> str:
    if employee_count is None:
        return "unknown"
    for low, high, bucket in COMPANY_SIZE_BUCKETS:
        if low <= float(employee_count) <= high:
            return bucket
    return "unknown"


def bucket_meets_min_100(bucket: str) -> bool | None:
    if bucket in {"100-499", "500-999", "1000+"}:
        return True
    if bucket in {"1-10", "11-50", "51-99"}:
        return False
    return None


def parse_employee_count_text(text: str) -> tuple[float | None, str]:
    text_l = safe_lower(text).replace(",", "")
    patterns = [
        (r"\b(?:over|more than|above|at least)\s+(\d{2,6})\+?\s+(?:employees|people|staff|team members)\b", 1.05),
        (r"\b(\d{2,6})\+\s+(?:employees|people|staff|team members)\b", 1.0),
        (r"\b(\d{2,6})\s+(?:employees|people|staff|team members)\b", 1.0),
        (r"\bteam of\s+(\d{1,5})\b", 1.0),
        (r"\b(\d{1,5})[- ]person team\b", 1.0),
    ]
    for pattern, multiplier in patterns:
        match = re.search(pattern, text_l)
        if match:
            return float(match.group(1)) * multiplier, match.group(0)
    return None, ""


def load_company_enrichment(path: Path = COMPANY_ENRICHMENT_CSV) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path).fillna("")
    except Exception:
        return {}
    if "company" not in df.columns:
        return {}
    lookup: Dict[str, Dict] = {}
    for _, row in df.iterrows():
        company = safe_lower(row.get("company")).strip()
        if company:
            lookup[company] = row.to_dict()
    return lookup


def company_size_signals(row: pd.Series | Dict, enrichment: Dict[str, Dict] | None = None) -> Dict:
    enrichment = enrichment or {}
    existing_bucket = str(row.get("company_size_bucket", "") or "").strip()
    if existing_bucket in {"1-10", "11-50", "51-99", "100-499", "500-999", "1000+", "unknown"}:
        meets_existing = row.get("meets_min_company_size", "")
        if str(meets_existing).lower() in {"true", "1", "yes"}:
            meets_existing = True
        elif str(meets_existing).lower() in {"false", "0", "no"}:
            meets_existing = False
        elif meets_existing == "":
            meets_existing = bucket_meets_min_100(existing_bucket)
        return {
            "employee_count_estimate": row.get("employee_count_estimate", ""),
            "company_size_bucket": existing_bucket,
            "meets_min_company_size": meets_existing,
            "is_large_company": str(row.get("is_large_company", "")).lower() in {"true", "1", "yes"},
            "is_possible_tiny_startup": str(row.get("is_possible_tiny_startup", "")).lower() in {"true", "1", "yes"},
            "company_size_confidence": row.get("company_size_confidence", "low") or "low",
            "company_size_notes": row.get("company_size_notes", "") or "No reliable company-size signal found",
        }

    company = safe_lower(row.get("company")).strip()
    override = enrichment.get(company, {})
    text = safe_lower(
        f"{row.get('company', '')} {row.get('title', '')} {row.get('description', '')} "
        f"{row.get('clean_job_text', '')} {row.get('job_text', '')}"
    )

    estimate = parse_number(override.get("employee_count_estimate") or override.get("employee_count"))
    note = ""
    confidence = "low"
    if estimate is not None:
        note = "Company enrichment file provided employee estimate"
        confidence = "high"
    else:
        estimate, note = parse_employee_count_text(text)
        if estimate is not None:
            confidence = "medium"

    bucket = str(override.get("company_size_bucket") or company_size_bucket(estimate))
    if bucket not in {"1-10", "11-50", "51-99", "100-499", "500-999", "1000+", "unknown"}:
        bucket = company_size_bucket(estimate)

    possible_tiny = str(override.get("is_possible_tiny_startup", "")).lower() in {"true", "1", "yes"}
    if not possible_tiny:
        possible_tiny = any(term in text for term in TINY_COMPANY_TERMS)
    if possible_tiny and not note:
        note = "Startup or small-team wording found"
        confidence = "medium"

    meets_min = bucket_meets_min_100(bucket)
    if meets_min is None and estimate is not None:
        meets_min = estimate >= 100

    is_large = bucket == "1000+" or (bucket == "500-999" and confidence in {"medium", "high"})
    if override.get("is_large_company") in {True, "true", "True", "1", 1}:
        is_large = True

    if not note:
        note = "No reliable company-size signal found"

    return {
        "employee_count_estimate": int(estimate) if estimate is not None else "",
        "company_size_bucket": bucket,
        "meets_min_company_size": "" if meets_min is None else bool(meets_min),
        "is_large_company": bool(is_large),
        "is_possible_tiny_startup": bool(possible_tiny or bucket in {"1-10", "11-50", "51-99"}),
        "company_size_confidence": confidence,
        "company_size_notes": note,
    }


def profile_requires_min_100_company(profile: Dict) -> bool:
    return company_enrichment.parse_company_constraints(profile).get("min_company_size") == 100


def company_constraint_score_fn(profile: Dict, row: pd.Series | Dict) -> Tuple[float, str]:
    return company_enrichment.company_constraint_score(profile, row)


def should_filter_company_size_for_profile(profile: Dict, row: pd.Series | Dict, enrichment: Dict[str, Dict] | None = None) -> bool:
    constraints = company_enrichment.parse_company_constraints(profile)
    min_size = constraints.get("min_company_size")
    if not min_size:
        return False
    enriched = get_company_enrichment_for_row(row, enrichment)
    key = "meets_500_employee_threshold" if int(min_size) >= 500 else "meets_100_employee_threshold"
    return str(enriched.get(key, "")).strip().lower() == "false"


def get_company_enrichment_for_row(row: pd.Series | Dict, enrichment_cache=None) -> Dict:
    company = str(row.get("company", "") or "")
    if isinstance(enrichment_cache, pd.DataFrame):
        cached = company_enrichment.get_cached_company_enrichment(company, enrichment_cache)
        if cached is not None:
            return cached
    if isinstance(enrichment_cache, dict):
        normalized = company_enrichment.normalize_company_name(company)
        if normalized in enrichment_cache:
            return enrichment_cache[normalized]
    existing_bucket = str(row.get("company_size_bucket", "") or "").strip()
    if existing_bucket:
        return {col: row.get(col, "") for col in company_enrichment.CACHE_COLUMNS}
    return company_enrichment.infer_company_profile_from_job_text(row)


def profile_has_strict_company_constraints(profile: Dict) -> bool:
    constraints = company_enrichment.parse_company_constraints(profile)
    return bool(
        constraints.get("min_company_size")
        or constraints.get("no_tiny_startups")
        or constraints.get("prefer_large_companies")
        or constraints.get("known_sponsor_preferred")
        or constraints.get("visa_sponsorship_required")
    )


def recommendation_priority_tier(profile: Dict, row: pd.Series | Dict) -> int:
    constraints = company_enrichment.parse_company_constraints(profile)
    min_company_size = constraints.get("min_company_size")
    if not min_company_size:
        return 1

    threshold_key = "meets_500_employee_threshold" if int(min_company_size) >= 500 else "meets_100_employee_threshold"
    threshold_value = str(row.get(threshold_key, row.get("meets_min_company_size", ""))).strip().lower()
    bucket = str(row.get("company_size_bucket", "") or "unknown").strip().lower()
    is_contract = normalize_employment_type(row) == "contract"
    is_tiny = company_enrichment.bool_value(row.get("is_possible_tiny_startup", ""))
    try:
        target_fit = float(row.get("target_fit_score", 0) or 0)
    except Exception:
        target_fit = 0.0

    if threshold_value == "false":
        return 5
    if is_contract:
        return 4
    if threshold_value == "true":
        return 1
    if bucket == "unknown":
        if target_fit >= 0.65 and not is_tiny:
            return 2
        return 3
    if is_tiny:
        return 3
    return 3


def format_candidate_funnel_summary(summary: Dict) -> str:
    ordered_keys = [
        "initial_retrieved_count",
        "after_seniority_filter_count",
        "after_defense_filter_count",
        "after_required_years_filter_count",
        "after_ml_related_filter_count",
        "after_basic_filters_count",
        "after_location_filter_count",
        "after_salary_filter_count",
        "after_company_size_filter_count",
        "after_contract_penalty_count",
        "tier1_confirmed_100_count",
        "tier2_unknown_count",
        "final_top_count",
        "ml_related_violation_count",
    ]
    return "; ".join(f"{key}={summary.get(key, 0)}" for key in ordered_keys if key in summary)




def salary_score_fn(profile: Dict, row: pd.Series) -> Tuple[float, str]:
    requested = profile_min_annual_salary(profile)
    annual_min, annual_max = annual_salary_range(row)
    source = salary_source_for_row(row)
    if source == "description":
        source_note = "Salary parsed from job description"
    elif source == "structured":
        source_note = "Salary listed"
    else:
        source_note = "Salary not listed in the offline snapshot"
    if requested is None:
        return 0.50, f"No salary preference provided; {source_note}"
    if annual_min is None and annual_max is None:
        return 0.42, f"{source_note}; unknown, penalized slightly"
    comparable = annual_max if annual_max is not None else annual_min
    if comparable is not None and comparable < requested:
        return 0.10, f"{source_note}; annualized salary appears below preference"
    if annual_min is not None and annual_min >= requested:
        return 1.00, f"{source_note}; annualized salary meets preference"
    return 0.75, f"{source_note}; annualized salary range may meet preference"


class JobRecommender:
    def __init__(self, jobs_csv: Path = JOBS_CSV, faiss_path: Path = FAISS_PATH) -> None:
        init_start = time.perf_counter()
        self.last_timing_log = {}
        PRECISE_ROLE_FAMILY_CACHE.clear()
        if not jobs_csv.exists():
            raise FileNotFoundError(f"Missing {jobs_csv}. Run merge_job_datasets.py first.")
        if not faiss_path.exists():
            print(f"Missing {faiss_path}. Building embeddings now. This may take several minutes.")
            from build_embeddings import main as build_embedding_index
            build_embedding_index()
        load_start = time.perf_counter()
        self.jobs = pd.read_csv(jobs_csv).fillna("").reset_index(drop=True)
        self.last_timing_log["load_jobs_seconds"] = round(time.perf_counter() - load_start, 4)
        feature_start = time.perf_counter()
        static_features = load_or_build_static_job_features(self.jobs)
        feature_cols = [col for col in static_features.columns if col not in {"job_index", "title", "company"}]
        self.jobs = self.jobs.join(static_features.set_index("job_index")[feature_cols], how="left")
        self.role_index = static_features[["job_index", "title", "company", "role_family_static", "role_profile_text"]].copy()
        self.role_vocabulary = load_or_build_role_vocabulary(self.jobs)
        self.last_timing_log["load_or_build_static_features_seconds"] = round(time.perf_counter() - feature_start, 4)
        location_vocab_start = time.perf_counter()
        self.location_vocabulary = build_location_vocabulary(jobs_csv)
        self.last_timing_log["load_or_build_location_vocabulary_seconds"] = round(time.perf_counter() - location_vocab_start, 4)
        self.company_enrichment_cache_path = company_enrichment.DEFAULT_CACHE_PATH
        company_start = time.perf_counter()
        self.company_enrichment = company_enrichment.load_company_enrichment_cache(self.company_enrichment_cache_path)
        self.last_timing_log["load_company_enrichment_seconds"] = round(time.perf_counter() - company_start, 4)
        self.last_enrichment_status = {"companies_checked": 0, "cache_hits": 0, "api_calls_made": 0, "unknown_cached": 0, "errors": 0}
        self.last_candidate_funnel_summary = {}
        index_start = time.perf_counter()
        self.index = faiss.read_index(str(faiss_path))
        self.last_timing_log["load_faiss_seconds"] = round(time.perf_counter() - index_start, 4)
        model_start = time.perf_counter()
        self.model = SentenceTransformer(MODEL_NAME)
        self.last_timing_log["load_embedding_model_seconds"] = round(time.perf_counter() - model_start, 4)
        self.last_timing_log["init_total_seconds"] = round(time.perf_counter() - init_start, 4)

    def _semantic_search(self, query_text: str, top_k: int = 250) -> List[Tuple[int, float]]:
        q = self.model.encode([query_text], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(q, min(top_k, len(self.jobs)))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def recommend_from_profile(
        self,
        profile: Dict,
        top_n: int = 10,
        candidate_k: int = 500,
        feedback: Dict | None = None,
    ) -> pd.DataFrame:
        request_start = time.perf_counter()
        timing_log = {}
        profile_start = time.perf_counter()
        profile = normalize_profile(profile)
        profile["_target_role_text"] = build_target_role_text(profile)
        profile["_target_role_families"] = infer_target_role_families(profile["_target_role_text"])
        profile["_excluded_role_families"] = sorted(parse_excluded_role_families(profile.get("dealbreakers", [])))
        profile["_location_preferences"] = parse_location_preferences(profile.get("location_preference", ""))
        profile["location_preferences"] = profile["_location_preferences"]
        timing_log["profile_features_seconds"] = round(time.perf_counter() - profile_start, 4)
        feedback = feedback or {}
        if profile_has_strict_company_constraints(profile):
            candidate_k = max(candidate_k, 800)
        elif profile_requires_ml_related_roles(profile):
            candidate_k = max(candidate_k, 500)
        candidate_k = min(candidate_k, 800)
        query_text = build_profile_query(profile, feedback)
        retrieval_start = time.perf_counter()
        retrieved = self._semantic_search(query_text, top_k=candidate_k)
        timing_log["ann_retrieval_seconds"] = round(time.perf_counter() - retrieval_start, 4)
        enrichment_pool_size = min(len(retrieved), 100)
        if enrichment_pool_size:
            enrichment_start = time.perf_counter()
            pool_indices = [idx for idx, _ in retrieved[:enrichment_pool_size]]
            pool_df = self.jobs.iloc[pool_indices].copy()
            online_enabled = os.getenv("ENABLE_ONLINE_COMPANY_ENRICHMENT", "false").strip().lower() == "true"
            force_refresh = os.getenv("FORCE_REFRESH_COMPANY_ENRICHMENT", "false").strip().lower() == "true"
            self.company_enrichment, self.last_enrichment_status = company_enrichment.enrich_companies_for_candidates(
                pool_df,
                online_enabled=online_enabled,
                force_refresh=force_refresh,
                cache_path=self.company_enrichment_cache_path,
                provider=os.getenv("COMPANY_ENRICHMENT_PROVIDER", "companies_api"),
            )
            timing_log["company_enrichment_seconds"] = round(time.perf_counter() - enrichment_start, 4)
        dealbreakers = parse_dealbreakers(profile)
        funnel_start = time.perf_counter()
        self.last_candidate_funnel_summary = self._candidate_funnel_summary(profile, retrieved, dealbreakers)
        timing_log["candidate_funnel_seconds"] = round(time.perf_counter() - funnel_start, 4)
        rows = []

        scoring_start = time.perf_counter()
        for idx, semantic_score in retrieved:
            row = self.jobs.iloc[idx]
            hard_fail, filter_reason = self._hard_filter(profile, row, dealbreakers)
            if hard_fail:
                continue
            scored = self._score_row(profile, row, semantic_score, feedback)
            scored["filter_reason"] = filter_reason
            rows.append(scored)
        timing_log["hard_filter_and_scoring_seconds"] = round(time.perf_counter() - scoring_start, 4)

        if not rows:
            empty = pd.DataFrame(columns=list(self.jobs.columns) + ["semantic_score", "final_score", "why_matched"])
            timing_log["request_total_seconds"] = round(time.perf_counter() - request_start, 4)
            self.last_timing_log = {**getattr(self, "last_timing_log", {}), **timing_log}
            empty.attrs["timing_log"] = dict(self.last_timing_log)
            return empty

        final_start = time.perf_counter()
        result = pd.DataFrame(rows)

        # Final safety filters: keep the output clean even if Streamlit cache or candidate
        # generation brings borderline rows into the pool.
        if not result.empty:
            keep_mask = []
            for _, candidate_row in result.iterrows():
                keep = True
                if listed_salary_below_preference(profile, candidate_row):
                    keep = False
                if required_years_violates_profile(profile, candidate_row):
                    keep = False
                if "no unpaid" in " ".join([str(x).lower() for x in profile.get("dealbreakers", [])]) and has_strict_unpaid_signal(candidate_row):
                    keep = False
                if should_filter_employment_for_profile(profile, candidate_row, dealbreakers):
                    keep = False
                if should_filter_location_for_profile(profile, candidate_row):
                    keep = False
                if should_filter_role_family_mismatch(profile, candidate_row):
                    keep = False
                if should_filter_excluded_role_family(profile, candidate_row):
                    keep = False
                if should_filter_low_quality_job(candidate_row):
                    keep = False
                if should_filter_internship_for_profile(profile, candidate_row):
                    keep = False
                if parse_dealbreakers(profile).get("need_sponsorship") and has_no_sponsorship_signal(candidate_row):
                    keep = False
                if should_filter_company_size_for_profile(profile, candidate_row, self.company_enrichment):
                    keep = False
                keep_mask.append(keep)
            result = result.loc[keep_mask].copy()

        if profile_has_strict_company_constraints(profile):
            result["_recommendation_tier"] = result.apply(lambda row: recommendation_priority_tier(profile, row), axis=1)
            self.last_candidate_funnel_summary["tier1_confirmed_100_count"] = int((result["_recommendation_tier"] == 1).sum())
            self.last_candidate_funnel_summary["tier2_unknown_count"] = int((result["_recommendation_tier"] == 2).sum())
            result = result.sort_values(["_recommendation_tier", "final_score"], ascending=[True, False])
        else:
            result = result.sort_values("final_score", ascending=False)
        result = result.head(top_n).reset_index(drop=True)
        self.last_candidate_funnel_summary["final_top_count"] = len(result)
        if profile_requires_ml_related_roles(profile):
            self.last_candidate_funnel_summary["ml_related_violation_count"] = (
                int((result["is_ml_related"].astype(str).str.lower() != "true").sum())
                if "is_ml_related" in result
                else 0
            )
        result.insert(0, "rank", range(1, len(result) + 1))
        if "_recommendation_tier" in result.columns:
            result = result.drop(columns=["_recommendation_tier"])
        result.attrs["company_enrichment_status"] = self.last_enrichment_status
        result.attrs["candidate_funnel_summary"] = dict(self.last_candidate_funnel_summary)
        timing_log["final_sort_and_safety_seconds"] = round(time.perf_counter() - final_start, 4)
        timing_log["request_total_seconds"] = round(time.perf_counter() - request_start, 4)
        self.last_timing_log = {**getattr(self, "last_timing_log", {}), **timing_log}
        result.attrs["timing_log"] = dict(self.last_timing_log)
        return result

    def _passes_basic_funnel_filters(self, profile: Dict, row: pd.Series, dealbreakers: Dict[str, bool]) -> bool:
        title = safe_lower(row.get("title"))
        company = safe_lower(row.get("company"))
        text = safe_lower(f"{row.get('title')} {row.get('company')} {row.get('description')} {row.get('clean_job_text')}")
        seniority = safe_lower(row.get("seniority"))

        required_years = extract_required_years(row)
        if required_years_violates_profile(profile, row):
            return False
        if dealbreakers.get("no_5_plus_years") and required_years is not None and required_years >= 5:
            return False
        if dealbreakers.get("no_unpaid") and has_strict_unpaid_signal(row):
            return False

        target_allows_senior = any(x in safe_lower(profile.get("target_role")) for x in ["senior", "sr.", "staff", "principal", "lead"])
        if safe_lower(profile.get("candidate_level")) in {"entry_or_junior", "entry", "junior", "new grad"}:
            if not target_allows_senior and has_senior_title_signal(title, seniority):
                return False
            if re.search(r"\bintermediate\b|\bmid[- ]level\b", title):
                return False

        if dealbreakers["no_senior"] and has_senior_title_signal(title, seniority):
            return False
        if dealbreakers["no_junior"] and ("junior" in title or "intern" in title or seniority in {"entry_or_junior", "internship"}):
            return False
        if dealbreakers["no_defense"] and any(x in company + " " + text for x in DEFENSE_HINTS):
            return False
        if should_filter_role_family_mismatch(profile, row):
            return False
        if should_filter_excluded_role_family(profile, row):
            return False
        if should_filter_low_quality_job(row):
            return False
        if should_filter_internship_for_profile(profile, row):
            return False
        if parse_dealbreakers(profile).get("need_sponsorship") and has_no_sponsorship_signal(row):
            return False
        return True

    def _candidate_funnel_summary(self, profile: Dict, retrieved: List[Tuple[int, float]], dealbreakers: Dict[str, bool]) -> Dict:
        sampled_retrieved = retrieved[:300]
        summary = {
            "initial_retrieved_count": len(retrieved),
            "after_basic_filters_count": 0,
            "after_location_filter_count": 0,
            "after_salary_filter_count": 0,
            "after_company_size_filter_count": 0,
            "after_contract_penalty_count": 0,
            "tier1_confirmed_100_count": 0,
            "tier2_unknown_count": 0,
            "final_top_count": 0,
        }
        if profile_requires_ml_related_roles(profile):
            senior_rows = []
            for idx, _ in sampled_retrieved:
                row = self.jobs.iloc[idx]
                title = safe_lower(row.get("title"))
                seniority = safe_lower(row.get("seniority"))
                if not has_senior_title_signal(title, seniority):
                    senior_rows.append(row)
            summary["after_seniority_filter_count"] = len(senior_rows)
            defense_rows = [
                row for row in senior_rows
                if not any(
                    x in safe_lower(f"{row.get('company', '')} {row.get('title', '')} {row.get('description', '')} {row.get('clean_job_text', '')}")
                    for x in DEFENSE_HINTS
                )
            ]
            summary["after_defense_filter_count"] = len(defense_rows)
            years_rows = []
            for row in defense_rows:
                if required_years_violates_profile(profile, row):
                    continue
                years_rows.append(row)
            summary["after_required_years_filter_count"] = len(years_rows)
            summary["after_ml_related_filter_count"] = len([row for row in years_rows if ml_related_decision(profile, row)[0]])
        basic_rows = []
        for idx, _ in sampled_retrieved:
            row = self.jobs.iloc[idx]
            if self._passes_basic_funnel_filters(profile, row, dealbreakers):
                basic_rows.append(row)
        summary["after_basic_filters_count"] = len(basic_rows)

        location_rows = [row for row in basic_rows if not should_filter_location_for_profile(profile, row)]
        summary["after_location_filter_count"] = len(location_rows)

        salary_rows = [row for row in location_rows if not listed_salary_below_preference(profile, row)]
        summary["after_salary_filter_count"] = len(salary_rows)

        company_rows = [row for row in salary_rows if not should_filter_company_size_for_profile(profile, row, self.company_enrichment)]
        summary["after_company_size_filter_count"] = len(company_rows)
        summary["after_contract_penalty_count"] = len(company_rows)

        for row in company_rows:
            company_profile = get_company_enrichment_for_row(row, self.company_enrichment)
            target_fit, _ = target_role_fit_score_fn(profile, row)
            tier_row = {**row.to_dict(), **company_profile, "target_fit_score": target_fit}
            tier = recommendation_priority_tier(profile, tier_row)
            if tier == 1:
                summary["tier1_confirmed_100_count"] += 1
            elif tier == 2:
                summary["tier2_unknown_count"] += 1
        return summary

    def _hard_filter(self, profile: Dict, row: pd.Series, dealbreakers: Dict[str, bool]) -> Tuple[bool, str]:
        title = safe_lower(row.get("title"))
        company = safe_lower(row.get("company"))
        text = safe_lower(f"{row.get('title')} {row.get('company')} {row.get('description')} {row.get('clean_job_text')}")
        seniority = safe_lower(row.get("seniority"))
        location = safe_lower(row.get("location"))
        normalized_employment = normalize_employment_type(row)
        seniority_level = job_seniority_level(row)
        if listed_salary_below_preference(profile, row):
            return True, "listed annualized salary below preference"

        required_years = extract_required_years(row)
        if required_years_violates_profile(profile, row):
            return True, "experience requirement above profile limit filtered"
        if dealbreakers.get("no_5_plus_years") and required_years is not None and required_years >= 5:
            return True, "5+ year experience requirement filtered"

        if dealbreakers.get("no_unpaid") and has_strict_unpaid_signal(row):
            return True, "unpaid role filtered"

        target_allows_senior = any(x in safe_lower(profile.get("target_role")) for x in ["senior", "sr.", "staff", "principal", "lead"])
        if safe_lower(profile.get("candidate_level")) in {"entry_or_junior", "entry", "junior", "new grad"}:
            if not target_allows_senior and seniority_level in {"senior", "executive"}:
                return True, f"{seniority_level} role filtered for entry-level profile"
            if seniority_level == "mid":
                return True, "mid-level role filtered for entry-level profile"

        if should_filter_location_for_profile(profile, row):
            return True, "outside stated location preference filtered"

        if should_filter_role_family_mismatch(profile, row):
            return True, "role family mismatch filtered"
        if should_filter_excluded_role_family(profile, row):
            return True, "excluded role family filtered"

        if should_filter_low_quality_job(row):
            return True, "low-quality or placement-like posting filtered"

        if should_filter_employment_for_profile(profile, row, dealbreakers):
            if normalized_employment == "contract":
                return True, "contract role filtered"
            if normalized_employment == "part-time":
                return True, "part-time role filtered"
            if normalized_employment == "unpaid":
                return True, "unpaid/volunteer role filtered"
            return True, "internship or unwanted employment type filtered"

        if dealbreakers["no_senior"] and seniority_level in {"senior", "executive"}:
            return True, "senior role filtered"
        if dealbreakers["no_junior"] and seniority_level in {"intern", "entry"}:
            return True, "junior role filtered"
        if dealbreakers["no_contract"] and normalized_employment == "contract":
            return True, "contract role filtered"
        if dealbreakers["no_defense"] and any(x in company + " " + text for x in DEFENSE_HINTS):
            return True, "defense role filtered"
        if dealbreakers["us_only"] and not looks_us_location(location):
            return True, "non-US location filtered"
        if dealbreakers["need_sponsorship"] and has_no_sponsorship_signal(row):
            return True, "no-sponsorship role filtered"
        if should_filter_company_size_for_profile(profile, row, self.company_enrichment):
            return True, "company below 100 employees filtered"
        return False, "passed hard filters"

    def _score_row(self, profile: Dict, row: pd.Series, semantic_score: float, feedback: Dict) -> Dict:
        user_skills = [str(x) for x in profile.get("skills", [])]
        role_score = role_score_fn(profile.get("target_role", ""), row, semantic_score)
        target_fit_score, target_fit_note = target_role_fit_score_fn(profile, row)
        guarded_target_fit, role_guardrail_note = apply_role_mismatch_guardrail(profile, row, target_fit_score)
        if guarded_target_fit < target_fit_score:
            target_fit_score = guarded_target_fit
            target_fit_note = f"{target_fit_note}; {role_guardrail_note}" if target_fit_note else role_guardrail_note
        role_score = max(role_score * 0.58 + target_fit_score * 0.42, target_fit_score)
        skill_score, matched_skills = skill_score_fn(user_skills, row)
        exp_score, exp_note = experience_score_fn(profile, row)
        location_score, location_note = location_score_fn(profile, row)
        salary_score, salary_note = salary_score_fn(profile, row)
        job_quality_score, job_quality_note = job_quality_score_fn(row)
        sponsorship_score, sponsorship_note = sponsorship_score_fn(profile, row)
        company_profile = get_company_enrichment_for_row(row, self.company_enrichment)
        company_row = {**row.to_dict(), **company_profile}
        company_constraint_score, company_constraint_note = company_constraint_score_fn(profile, company_row)
        source_score = 1.0 if safe_lower(row.get("source")) == "jsearch" else 0.70
        feedback_row = {
            **company_row,
            "sponsorship_score": sponsorship_score,
            "company_sponsor_signal": company_sponsor_signal(row),
        }
        feedback_score, feedback_note = feedback_score_fn(feedback_row, feedback)
        feedback_multiplier, feedback_additive, feedback_adjustment_note = feedback_final_score_adjustment(feedback_row, feedback)
        annual_min, annual_max = annual_salary_range(row)
        location_decision = location_constraint_decision(profile, row)
        normalized_employment = normalize_employment_type(row)
        employment_evidence = employment_type_evidence(row)
        is_ml_related, ml_related_reason = ml_related_decision(profile, row)
        role_family_mismatch = bool(profile_requires_ml_related_roles(profile) and not is_ml_related)
        disallowed_for_profile_reason = ml_related_reason if role_family_mismatch else ""
        if normalized_employment == "unknown":
            employment_note = "Employment type not listed; unknown, penalized slightly"
        elif normalized_employment == "full-time" and annual_min is None and annual_max is None:
            employment_note = "Employment type appears full-time; salary is not listed"
        else:
            employment_note = f"Employment type fit: {normalized_employment}"
        dealbreaker_note = "Hard dealbreaker checks applied before ranking"

        final_score = (
            0.30 * role_score
            + 0.21 * skill_score
            + 0.16 * exp_score
            + 0.09 * location_score
            + 0.10 * salary_score
            + 0.08 * job_quality_score
            + 0.05 * sponsorship_score
            + 0.04 * company_constraint_score
            + 0.02 * source_score
            + 0.04 * feedback_score
        )

        if target_fit_score < 0.35:
            final_score *= 0.72
        elif target_fit_score >= 0.85:
            final_score += 0.035
        if normalized_employment == "contract" and not profile_accepts_contract(profile):
            final_score *= 0.45
        if normalized_employment == "unknown":
            final_score *= 0.96
        if parse_dealbreakers(profile).get("need_sponsorship") and sponsorship_score < 0.50:
            final_score *= 0.82
        if profile_lacks_production_ml_experience(profile) and strong_production_ml_requirement(row):
            final_score *= 0.70
            exp_note = f"{exp_note}; prefers lower production-ML ownership" if exp_note else "prefers lower production-ML ownership"
        if executive_operations_title_mismatch(profile, row):
            final_score = min(final_score * 0.25, 0.18)
            mismatch_note = "Title appears executive/operations-focused, which does not match the user's technical target roles"
            exp_note = f"{exp_note}; {mismatch_note}" if exp_note else mismatch_note
        elif profile_targets_technical_ic_roles(profile) and data_center_context_without_ml_evidence(row) and skill_score < 0.25:
            final_score = min(final_score * 0.35, 0.24)
            mismatch_note = "Data center context is not enough evidence for ML/data science role relevance"
            exp_note = f"{exp_note}; {mismatch_note}" if exp_note else mismatch_note
        company_constraints = company_enrichment.parse_company_constraints(profile)
        min_company_size = company_constraints.get("min_company_size")
        if min_company_size:
            threshold_key = "meets_500_employee_threshold" if int(min_company_size) >= 500 else "meets_100_employee_threshold"
            threshold_value = str(company_profile.get(threshold_key, "")).strip().lower()
            if threshold_value == "false":
                final_score *= 0.20
            elif company_profile.get("company_size_bucket") == "unknown":
                final_score *= 0.88
            elif company_enrichment.bool_value(company_profile.get("is_possible_tiny_startup", "")):
                final_score *= 0.72
        if company_constraints.get("no_tiny_startups") and company_enrichment.bool_value(company_profile.get("is_possible_tiny_startup", "")):
            final_score *= 0.55
        if parse_dealbreakers(profile).get("need_sponsorship") and company_enrichment.bool_value(company_profile.get("is_possible_tiny_startup", "")):
            final_score *= 0.86
        location_tier = int(location_decision.get("location_tier", 1) or 1)
        location_tier_multipliers = {1: 1.00, 2: 0.94, 3: 0.86, 4: 0.78, 5: 0.66, 9: 0.82}
        final_score *= location_tier_multipliers.get(location_tier, 0.75)
        if feedback_adjustment_note:
            final_score = final_score * feedback_multiplier + feedback_additive
            feedback_note = f"{feedback_note}; {feedback_adjustment_note}" if feedback_note else feedback_adjustment_note

        if target_fit_note and target_fit_score < 0.60:
            exp_note = f"{exp_note}; {target_fit_note}" if exp_note else target_fit_note
        if company_constraint_note and company_constraint_note not in {"No company-size constraint provided"}:
            exp_note = f"{exp_note}; {company_constraint_note}" if exp_note else company_constraint_note
        why = build_explanation(
            row,
            matched_skills,
            role_score,
            skill_score,
            exp_note,
            location_note,
            salary_note,
            source_score,
            feedback_note,
            job_quality_note,
            sponsorship_note,
            employment_note,
            dealbreaker_note,
        )
        salary_source = salary_source_for_row(row)
        role_compatibility, role_compatibility_note, role_family_label = compute_role_family_compatibility(profile, row)
        target_families_cached = profile.get("_target_role_families") or infer_target_role_families(build_target_role_text(profile))
        job_families_cached = infer_job_role_families(row, use_cache=False)
        dataset_role_family = max(job_families_cached, key=job_families_cached.get) if job_families_cached else "other"
        payload = row.to_dict()
        payload.update({
            "salary_annual_min": round(float(annual_min), 2) if annual_min is not None else "",
            "salary_annual_max": round(float(annual_max), 2) if annual_max is not None else "",
            "salary_display": salary_display(annual_min, annual_max),
            "salary_source": salary_source,
            "required_years": extract_required_years(row) if extract_required_years(row) is not None else "",
            "raw_employment_type": row.get("employment_type", ""),
            "normalized_employment_type": normalized_employment,
            "employment_type_evidence": employment_evidence,
            "semantic_score": round(float(semantic_score), 4),
            "role_score": round(float(role_score), 4),
            "target_fit_score": round(float(target_fit_score), 4),
            "role_family": best_role_family_match(
                target_families_cached,
                job_families_cached,
            )[0],
            "dataset_role_family": dataset_role_family,
            "role_profile_text": build_role_profile_text(row),
            "role_family_compatibility": round(float(role_compatibility), 4),
            "role_family_compatibility_note": role_compatibility_note,
            "role_family_match_label": role_family_label,
            "is_ml_related": bool(is_ml_related),
            "ml_related_reason": ml_related_reason,
            "role_family_mismatch": bool(role_family_mismatch),
            "disallowed_for_profile_reason": disallowed_for_profile_reason,
            "skill_score": round(float(skill_score), 4),
            "experience_score": round(float(exp_score), 4),
            "location_score": round(float(location_score), 4),
            "location_tier": location_decision.get("location_tier", ""),
            "normalized_location": location_decision.get("normalized_location", ""),
            "location_match_reason": location_decision.get("reason", ""),
            "location_constraint_pass": bool(location_decision.get("pass", True)),
            "salary_score": round(float(salary_score), 4),
            "job_quality_score": round(float(job_quality_score), 4),
            "job_quality_reason": job_quality_note,
            "sponsorship_score": round(float(sponsorship_score), 4),
            "sponsorship_reason": sponsorship_note,
            "company_sponsor_signal": company_sponsor_signal(row),
            "employee_count_estimate": company_profile.get("employee_count_estimate", ""),
            "company_size_bucket": company_profile.get("company_size_bucket", "unknown"),
            "meets_100_employee_threshold": company_profile.get("meets_100_employee_threshold", ""),
            "meets_500_employee_threshold": company_profile.get("meets_500_employee_threshold", ""),
            "meets_min_company_size": company_profile.get("meets_100_employee_threshold", ""),
            "is_large_company": company_profile.get("is_large_company", False),
            "is_possible_tiny_startup": company_profile.get("is_possible_tiny_startup", False),
            "is_known_h1b_sponsor": company_profile.get("is_known_h1b_sponsor", False),
            "is_research_lab": company_profile.get("is_research_lab", False),
            "is_public_company": company_profile.get("is_public_company", False),
            "company_enrichment_confidence": company_profile.get("confidence", "low"),
            "company_enrichment_source": company_profile.get("source_url", ""),
            "company_enrichment_notes": company_profile.get("notes", ""),
            "company_size_confidence": company_profile.get("confidence", "low"),
            "company_size_notes": company_profile.get("notes", ""),
            "company_constraint_score": round(float(company_constraint_score), 4),
            "source_score": round(float(source_score), 4),
            "feedback_score": round(float(feedback_score), 4),
            "final_score": round(float(final_score), 4),
            "matched_skills": ", ".join(matched_skills[:12]),
            "why_matched": why,
        })
        return payload



def safe_lower(value) -> str:
    return str(value or "").lower()


def normalize_profile(profile: Dict) -> Dict:
    profile = dict(profile or {})
    profile.setdefault("target_role", "")
    profile.setdefault("profile_summary", "")
    profile.setdefault("skills", [])
    profile.setdefault("years_experience", None)
    profile.setdefault("candidate_level", "unknown")
    profile.setdefault("education_keywords", [])
    profile.setdefault("project_keywords", [])
    profile.setdefault("location_preference", "")
    profile.setdefault("salary_min", "")
    profile.setdefault("dealbreakers", [])
    return profile


def has_senior_title_signal(title: str, seniority: str = "") -> bool:
    text = safe_lower(f"{title} {seniority}")
    patterns = [
        r"\bsenior\b",
        r"\bsr\.?\b",
        r"\bstaff\b",
        r"\bprincipal\b",
        r"\blead\b",
        r"\bdirector\b",
        r"\bmanager\b",
        r"\bhead\b",
        r"\bvp\b|\bvice president\b",
        r"\biii\b|\biv\b|\bv\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def job_seniority_level(row: pd.Series | Dict) -> str:
    title = safe_lower(row.get("title", ""))
    seniority = safe_lower(row.get("seniority", ""))
    text = f"{title} {seniority}"
    if re.search(r"\bchief\b|\bcoo\b|\bceo\b|\bcfo\b|\bcto\b|\bvp\b|\bvice president\b|\bdirector\b|\bhead\b", title):
        return "executive"
    if re.search(r"\bmanager\b|\blead\b|\bstaff\b|\bprincipal\b|\bsenior\b|\bsr\.?\b|\biii\b|\biv\b|\bv\b", title):
        return "senior"
    if re.search(r"\bintern(ship)?\b", title):
        return "intern"
    if re.search(r"\bjunior\b|\bjr\.?\b|\bentry[- ]?level\b|\bnew grad\b|\bassociate\b", text):
        return "entry"
    if re.search(r"\bmid[- ]?level\b|\bintermediate\b|\bii\b", text):
        return "mid"
    if seniority in {"senior", "executive", "manager"}:
        return "senior"
    if seniority in {"entry_or_junior", "junior", "entry"}:
        return "entry"
    cached = str(row.get("seniority_level_static", "") or "").strip()
    if cached:
        return cached
    return "unknown"


def structured_dealbreaker_constraints(profile: Dict, extra_text: str = "") -> Dict:
    raw = profile.get("dealbreakers", [])
    parts = raw if isinstance(raw, list) else [str(raw)]
    text = " ".join([str(x) for x in parts] + [extra_text]).lower()
    max_years = profile_max_required_years(profile)
    company_constraints = company_enrichment.parse_company_constraints(profile)
    return {
        "no_senior": any(x in text for x in ["no senior", "avoid senior", "no staff", "no principal", "no lead", "no manager", "no director"]),
        "no_junior": any(x in text for x in ["no junior", "no entry", "no intern"]),
        "no_contract": any(x in text for x in ["no contract", "no temp", "no temporary", "full time only", "full-time only", "contract incompatible"]),
        "no_part_time": any(x in text for x in ["no part-time", "no part time", "full time only", "full-time only"]),
        "no_unpaid": any(x in text for x in ["no unpaid", "unpaid", "no volunteer"]),
        "no_defense": any(x in text for x in ["no defense", "no defence", "no military"]),
        "need_sponsorship": any(
            x in text
            for x in [
                "visa sponsorship required", "need sponsorship", "needs sponsorship", "h1b", "h-1b",
                "h-1b sponsorship", "companies that don't sponsor", "companies that do not sponsor",
                "no companies that don't sponsor", "no companies that do not sponsor",
            ]
        ),
        "us_only": any(x in text for x in ["us only", "united states only", "usa only"]),
        "max_required_years": max_years,
        "min_company_size": company_constraints.get("min_company_size"),
        "no_tiny_startups": company_constraints.get("no_tiny_startups"),
    }



def parse_dealbreakers(profile: Dict, extra_text: str = "") -> Dict[str, bool]:
    constraints = structured_dealbreaker_constraints(profile, extra_text)
    return {
        **constraints,
        "no_5_plus_years": constraints.get("max_required_years") == 5,
    }


def profile_lacks_production_ml_experience(profile: Dict) -> bool:
    text = safe_lower(
        " ".join(
            [
                str(profile.get("ml_experience_constraint", "")),
                str(profile.get("raw_resume_text", "")),
                str(profile.get("profile_summary", "")),
            ]
            + [str(x) for x in profile.get("dealbreakers", [])]
        )
    )
    return bool(
        re.search(r"\bno\s+production\s+(?:ml|machine learning)\s+experience\b", text)
        or re.search(r"\b(no|limited|without)\b.{0,30}\b(?:ml|machine learning)\b.{0,30}\bexperience\b", text)
    )


def strong_production_ml_requirement(row: pd.Series | Dict) -> bool:
    title = safe_lower(row.get("title"))
    text = safe_lower(f"{row.get('title', '')} {row.get('description', '')} {row.get('clean_job_text', '')}")
    strong_terms = [
        "production ml", "production machine learning", "model deployment", "model serving",
        "ml platform", "mlops", "feature store", "model monitoring", "kubeflow", "ray",
        "distributed training", "inference infrastructure", "training infrastructure",
        "machine learning infrastructure", "own production", "productionize",
    ]
    senior_title = any(term in title for term in ["senior", "sr.", "staff", "principal", "lead"])
    return any(term in text for term in strong_terms) or (
        senior_title and any(term in text for term in ["machine learning", "ml engineer", "model deployment", "production"])
    )


def profile_accepts_contract(profile: Dict) -> bool:
    text = safe_lower(
        " ".join([str(profile.get("target_role", ""))] + [str(x) for x in profile.get("dealbreakers", [])])
    )
    return any(term in text for term in ["contract ok", "open to contract", "accept contract", "contract acceptable"])


STRICT_UNPAID_PATTERNS = [
    r"\bunpaid\s+(?:role|internship|position|opportunity|work)\b",
    r"\bno\s+compensation\b",
    r"\bwithout\s+compensation\b",
    r"\bvolunteer\s+(?:position|role|opportunity)\b",
    r"\bcommission\s+only\b",
    r"\bequity\s+only\b",
]

COMPENSATION_POSITIVE_PATTERNS = [
    r"\bpaid\s+(?:time\s+off|holidays|leave|benefits|training|internship)\b",
    r"\bcompensation\s+package\b",
    r"\bsalary\b",
    r"\bwage\b",
    r"\bhourly\s+pay\b",
    r"\bbonus\b",
    r"\bbenefits\b",
]


def has_positive_compensation_signal(row: pd.Series | Dict) -> bool:
    text = safe_lower(f"{row.get('description', '')} {row.get('clean_job_text', '')}")
    if annual_salary_range(row) != (None, None):
        return True
    return any(re.search(pattern, text) for pattern in COMPENSATION_POSITIVE_PATTERNS)


def has_strict_unpaid_signal(row: pd.Series | Dict) -> bool:
    text = safe_lower(f"{row.get('title', '')} {row.get('description', '')} {row.get('clean_job_text', '')}")
    if any(re.search(pattern, text) for pattern in STRICT_UNPAID_PATTERNS):
        return True
    if re.search(r"\bstipend\s+only\b", text) and not has_positive_compensation_signal(row):
        return True
    raw = safe_lower(row.get("employment_type"))
    return any(term in raw for term in ["volunteer", "unpaid"])


def raw_employment_is_full_time(row: pd.Series | Dict) -> bool:
    raw = safe_lower(row.get("employment_type"))
    return any(
        term in raw
        for term in ["fulltime", "full-time", "full time", "full_time", "permanent", "regular employee", "employee"]
    )


def employment_type_evidence(row: pd.Series | Dict) -> str:
    raw = safe_lower(row.get("employment_type"))
    normalized = normalize_employment_type(row)
    if normalized == "unpaid":
        return "Strong unpaid evidence found in employment label or description."
    if raw_employment_is_full_time(row):
        if re.search(r"\b(unpaid|volunteer|stipend|compensation)\b", safe_lower(f"{row.get('description', '')} {row.get('clean_job_text', '')}")):
            return "Raw employment label indicates full-time; ambiguous compensation text was not treated as unpaid."
        return "Raw employment label indicates full-time."
    if normalized == "full-time":
        return "Description indicates full-time."
    if normalized == "unknown":
        return "Employment type could not be confidently normalized."
    return f"Employment type normalized from explicit {normalized} evidence."


def normalize_employment_type(row: pd.Series | Dict) -> str:
    raw = safe_lower(row.get("employment_type"))
    title = safe_lower(row.get("title"))
    description = safe_lower(row.get("description"))
    clean_text = safe_lower(row.get("clean_job_text"))
    text = f"{title} {description} {clean_text}"

    if raw_employment_is_full_time(row) and not has_strict_unpaid_signal(row):
        return "full-time"

    if has_strict_unpaid_signal(row):
        return "unpaid"

    explicit_internship = (
        re.search(r"\binternship\b|\bintern\b", title) is not None
        or re.search(r"\binternship\b|\bintern\b", description) is not None
    )
    if explicit_internship:
        return "internship"

    contract_terms = ["contract", "contractor", "contract/temp", "temporary", "temp role", "temp-to-hire", "consultant contract"]
    if any(term in raw for term in contract_terms) or any(term in title for term in ["contract", "contractor", "temporary"]):
        return "contract"
    if re.search(r"\b(contract|contractor|temporary)\b", text) and not re.search(r"\bfull[- ]?time\b", text):
        return "contract"

    if any(term in raw for term in ["parttime", "part-time", "part time"]) or re.search(r"\bpart[- ]?time\b", text):
        return "part-time"
    if raw_employment_is_full_time(row):
        return "full-time"
    if re.search(r"\bfull[- ]?time\b", text):
        return "full-time"

    if raw in {"", "nan", "none", "null", "not specified"}:
        cached = str(row.get("normalized_employment_type_static", "") or "").strip()
        return cached or "unknown"
    return "unknown"


def should_filter_employment_for_profile(profile: Dict, row: pd.Series | Dict, dealbreakers: Dict[str, bool] | None = None) -> bool:
    dealbreakers = dealbreakers or parse_dealbreakers(profile)
    normalized = normalize_employment_type(row)
    if dealbreakers.get("no_contract") and normalized == "contract":
        return True
    if dealbreakers.get("no_part_time") and normalized == "part-time":
        return True
    if dealbreakers.get("no_unpaid") and normalized == "unpaid":
        return True
    if profile_min_annual_salary(profile) is not None and normalized == "unpaid":
        return True
    if dealbreakers.get("need_sponsorship") and normalized in {"contract", "part-time"}:
        return True
    if should_filter_internship_for_profile(profile, row):
        return True
    return False


def build_profile_query(profile: Dict, feedback: Dict | None = None) -> str:
    feedback = feedback or {}
    accepted_terms = feedback.get("accepted_terms", [])
    rejected_terms = feedback.get("rejected_terms", [])
    target_role_text = build_target_role_text(profile)
    parts = [
        f"Target role profile: {target_role_text}",
        f"Target role: {profile.get('target_role', '')}",
        f"Candidate level: {profile.get('candidate_level', '')}",
        f"Skills: {', '.join(profile.get('skills', []))}",
        f"Education: {', '.join(profile.get('education_keywords', []))}",
        f"Projects: {', '.join(profile.get('project_keywords', []))}",
        f"Location preference: {profile.get('location_preference', '')}",
        f"Maximum required years allowed: {profile.get('max_required_years', '')}",
        f"Preferred feedback terms: {', '.join(accepted_terms)}",
        f"Avoid feedback terms: {', '.join(rejected_terms)}",
    ]
    if profile_lacks_production_ml_experience(profile):
        parts.append("ML experience preference: prefer entry-level or low-production-ownership ML roles; avoid heavy production ML ownership requirements")
    return ". ".join([p for p in parts if p.strip()])


def build_target_role_text(profile: Dict) -> str:
    cached = str(profile.get("_target_role_text", "") or "").strip()
    if cached:
        return cached
    parts = [
        str(profile.get("target_role", "")),
        " ".join([str(x) for x in profile.get("skills", [])]),
        " ".join([str(x) for x in profile.get("project_keywords", [])]),
        " ".join([str(x) for x in profile.get("preferred_industries", [])]),
        " ".join([str(x) for x in profile.get("dealbreakers", [])]),
        str(profile.get("candidate_level", "")),
    ]
    text = safe_lower(" ".join(parts))
    inferred = infer_target_role_families(text)
    family_terms = [FRIENDLY_ROLE_FAMILY_LABELS.get(fam, fam).lower() for fam in inferred]
    if any(term in text for term in ["risk", "credit", "fintech", "banking", "quant", "financial services"]):
        family_terms.extend(["risk analytics", "credit risk", "financial modeling", "fintech data science", "quantitative analytics"])
    if any(term in text for term in ["analytics engineer", "bi engineer", "data platform", "data warehouse"]):
        family_terms.extend(["analytics engineering", "business intelligence", "data warehouse", "data platform"])
    return ". ".join([part for part in parts + family_terms if str(part).strip()])


def build_role_profile_text(job_row: pd.Series | Dict) -> str:
    cached = str(job_row.get("role_profile_text", "") or "").strip()
    if cached:
        return cached
    families = infer_job_role_families(job_row, use_cache=False)
    family = max(families, key=families.get) if families else "other"
    return ". ".join(
        [
            str(job_row.get("title", "")),
            str(job_row.get("company", "")),
            str(job_row.get("description", "")),
            f"Skills: {job_row.get('skills', '')}",
            f"Role family: {FRIENDLY_ROLE_FAMILY_LABELS.get(family, family)}",
            f"Seniority: {job_row.get('seniority', '')}",
            f"Employment type: {job_row.get('employment_type', '')}",
            f"Location: {job_row.get('location', '')}",
        ]
    )


def build_role_vocabulary(jobs_df: pd.DataFrame, top_n: int = 300) -> Dict:
    counter = Counter()
    for title in jobs_df.get("title", pd.Series(dtype=str)).fillna("").astype(str):
        tokens = [tok for tok in re.findall(r"[a-zA-Z][a-zA-Z+#.-]+", title.lower()) if len(tok) > 1]
        for n in [1, 2, 3]:
            for idx in range(0, max(0, len(tokens) - n + 1)):
                phrase = " ".join(tokens[idx: idx + n])
                if phrase not in {"and", "the", "for", "with"}:
                    counter[phrase] += 1
    return {"title_terms": dict(counter.most_common(top_n))}


def build_role_index(jobs_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in jobs_df.iterrows():
        families = infer_job_role_families(row, use_cache=False)
        family = max(families, key=families.get) if families else "other"
        rows.append(
            {
                "job_index": idx,
                "title": row.get("title", ""),
                "company": row.get("company", ""),
                "inferred_role_family": family,
                "role_profile_text": build_role_profile_text(row),
            }
        )
    return pd.DataFrame(rows)


def build_static_job_features(jobs_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in jobs_df.iterrows():
        row_dict = row.to_dict()
        families = infer_job_role_families(row_dict, use_cache=False)
        family = max(families, key=families.get) if families else "other"
        loc = normalize_job_location(row_dict)
        annual_min, annual_max = annual_salary_range(row_dict)
        required_years = extract_required_years(row_dict)
        rows.append(
            {
                "job_index": idx,
                "title": row_dict.get("title", ""),
                "company": row_dict.get("company", ""),
                "normalized_title_static": re.sub(r"[^a-z0-9]+", " ", safe_lower(row_dict.get("title", ""))).strip(),
                "normalized_company_static": company_enrichment.normalize_company_name(row_dict.get("company", "")),
                "normalized_location_static": loc.get("normalized_location", ""),
                "location_is_remote_static": bool(loc.get("is_remote", False)),
                "location_remote_label_static": bool(loc.get("remote_label", False)),
                "location_country_static": loc.get("country", ""),
                "location_state_static": loc.get("state", ""),
                "location_city_static": loc.get("city", ""),
                "location_region_matches_json": json.dumps(loc.get("region_matches", []), sort_keys=True),
                "location_broad_region_static": loc.get("broad_region", ""),
                "seniority_level_static": job_seniority_level(row_dict),
                "required_years_static": "" if required_years is None else float(required_years),
                "normalized_employment_type_static": normalize_employment_type(row_dict),
                "salary_annual_min": "" if annual_min is None else round(float(annual_min), 2),
                "salary_annual_max": "" if annual_max is None else round(float(annual_max), 2),
                "salary_display": salary_display(annual_min, annual_max),
                "salary_source_static": salary_source_for_row(row_dict),
                "role_family_static": family,
                "role_families_json": json.dumps(families, sort_keys=True),
                "role_profile_text": build_role_profile_text({**row_dict, "role_families_json": json.dumps(families, sort_keys=True)}),
                "duplicate_group_key_static": "|".join(
                    [
                        company_enrichment.normalize_company_name(row_dict.get("company", "")),
                        re.sub(r"[^a-z0-9]+", " ", safe_lower(row_dict.get("title", ""))).strip(),
                        family,
                    ]
                ),
            }
        )
    return pd.DataFrame(rows)


def build_fast_static_job_features(jobs_df: pd.DataFrame) -> pd.DataFrame:
    df = jobs_df.fillna("").copy()
    title = df.get("title", pd.Series("", index=df.index)).astype(str).str.lower()
    desc = df.get("description", pd.Series("", index=df.index)).astype(str).str.lower()
    skills = df.get("skills", pd.Series("", index=df.index)).astype(str).str.lower()
    text = title + " " + desc + " " + skills

    role_family = pd.Series("other", index=df.index, dtype="object")
    ordered_masks = [
        ("executive_management", title.str.contains(r"\b(?:chief|coo|ceo|cfo|cto|vp|vice president|director|head of|general manager)\b", regex=True)),
        ("marketing_analytics", title.str.contains("marketing", regex=False)),
        ("credit_risk", text.str.contains("credit risk", regex=False)),
        ("risk_analytics", text.str.contains(r"\brisk\b|risk model|risk analytics|fintech", regex=True)),
        ("quantitative_analytics", title.str.contains(r"quantitative analyst|quant analyst", regex=True)),
        ("audit", title.str.contains(r"audit|auditor", regex=True)),
        ("accounting", title.str.contains(r"accounting|accountant|controller", regex=True)),
        ("sales", title.str.contains(r"sales|account executive|business development representative", regex=True)),
        ("backend_engineering", title.str.contains(r"backend|back end", regex=True)),
        ("analytics_engineering", title.str.contains(r"analytics engineer|bi engineer|business intelligence engineer|data platform|data warehouse", regex=True)),
        ("ml_platform_mlops", text.str.contains(r"mlops|ml platform|model serving|model deployment|feature store|machine learning infrastructure", regex=True)),
        ("ml_engineering", title.str.contains(r"machine learning engineer|ml engineer|ai engineer", regex=True)),
        ("data_science", title.str.contains(r"data scientist|applied scientist|research scientist|decision scientist", regex=True)),
        ("bi_analytics", title.str.contains(r"bi analyst|business intelligence|bi developer", regex=True)),
        ("data_engineering", title.str.contains(r"data engineer|etl engineer|pipeline engineer", regex=True)),
        ("data_analytics", title.str.contains(r"data analyst|business analyst|analytics analyst|reporting analyst|product analyst", regex=True)),
        ("financial_analytics", title.str.contains(r"financial analyst|finance analyst|financial data", regex=True)),
        ("operations", title.str.contains(r"operations|data center", regex=True)),
        ("product_management", title.str.contains(r"product manager|product owner|program manager", regex=True)),
        ("consulting", title.str.contains(r"consultant|consulting", regex=True)),
        ("software_engineering", title.str.contains(r"software engineer|developer|frontend|full stack|java engineer", regex=True)),
    ]
    for family, mask in ordered_masks:
        role_family = role_family.mask(mask, family)

    required_year_patterns = [
        r"experience\s+range\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)",
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)\s+(?:of\s+)?(?:relevant\s+|professional\s+|industry\s+)?experience",
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)\s+(?:in|with|required|working|building|developing)\b",
        r"minimum\s+of\s+(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)",
        r"at\s+least\s+(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)",
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)\s+(?:required|preferred|minimum)",
    ]
    required_years = pd.Series(np.nan, index=df.index, dtype="float64")
    for pattern in required_year_patterns:
        values = pd.to_numeric(text.str.extract(pattern, expand=False), errors="coerce")
        required_years = required_years.combine(values, lambda current, new: np.nanmax([current, new]) if pd.notna(new) else current)
    range_values = text.str.extract(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(?:years|yrs)", expand=True)
    if not range_values.empty:
        range_max = range_values.apply(pd.to_numeric, errors="coerce").max(axis=1)
        required_years = required_years.combine(range_max, lambda current, new: np.nanmax([current, new]) if pd.notna(new) else current)
    required_years = required_years.where((required_years >= 0) & (required_years <= 20), "")

    raw_employment = df.get("employment_type", pd.Series("", index=df.index)).astype(str).str.lower()
    normalized_employment = pd.Series("unknown", index=df.index, dtype="object")
    normalized_employment = normalized_employment.mask(raw_employment.str.contains("full|permanent|employee", regex=True), "full-time")
    normalized_employment = normalized_employment.mask(raw_employment.str.contains("contract|temporary|temp", regex=True), "contract")
    normalized_employment = normalized_employment.mask(raw_employment.str.contains("part", regex=False), "part-time")
    explicit_internship = title.str.contains(r"\b(?:intern|internship)\b", regex=True) | desc.str.contains(r"\binternship\b", regex=True)
    normalized_employment = normalized_employment.mask(explicit_internship, "internship")
    normalized_employment = normalized_employment.mask(text.str.contains(r"\b(?:unpaid role|unpaid internship|unpaid position|no compensation|without compensation|volunteer role|volunteer position|commission only|equity only)\b", regex=True), "unpaid")
    normalized_employment = normalized_employment.mask(text.str.contains(r"\b(?:contract|contractor|temporary)\b", regex=True) & ~text.str.contains(r"\bfull[- ]?time\b", regex=True), "contract")
    normalized_employment = normalized_employment.mask(text.str.contains(r"\bpart[- ]?time\b", regex=True), "part-time")
    normalized_employment = normalized_employment.mask((normalized_employment == "unknown") & text.str.contains(r"\bfull[- ]?time\b", regex=True), "full-time")

    loc = df.get("location", pd.Series("", index=df.index)).astype(str)
    loc_lower = loc.str.lower()
    source_remote = df.get("is_remote", pd.Series("", index=df.index)).astype(str).str.lower().isin(["true", "1", "yes"])
    state = loc.str.extract(r",\s*([A-Z]{2})\b", expand=False).fillna("")
    country = pd.Series("", index=df.index, dtype="object")
    for term, code in NON_US_LOCATION_TERMS.items():
        country = country.mask(loc_lower.str.contains(rf"\b{re.escape(term)}\b", regex=True), code)
    for alias, code in COUNTRY_ALIASES.items():
        if code == "US":
            continue
        country = country.mask(loc_lower.str.contains(rf"\b{re.escape(alias)}\b", regex=True), code)
    country = country.mask(state.str.upper().isin(US_STATE_CODES), "US")
    country = country.mask(loc_lower.str.contains(r"\b(?:united states|usa|us)\b", regex=True), "US")
    for alias, code in STATE_NAME_TO_CODE.items():
        if len(alias) <= 2:
            continue
        state = state.mask((state == "") & loc_lower.str.contains(rf"\b{re.escape(alias)}\b", regex=True), code)
    country = country.mask(state.str.upper().isin(US_STATE_CODES), "US")
    city = loc.str.extract(r"^\s*([^,|;/]+)", expand=False).fillna("")
    physical_city = city.where(~city.astype(str).str.lower().isin(["remote", "anywhere", "worldwide", "wfh", "work from home"]), "")
    has_physical_location = physical_city.astype(str).str.strip().ne("") | state.astype(str).str.strip().ne("") | country.astype(str).str.strip().ne("")
    remote_label = loc_lower.str.contains(r"\b(?:remote|anywhere|worldwide|wfh|work from home)\b", regex=True) | title.str.contains(r"\b(?:remote|wfh|work from home|work-from-home)\b", regex=True)
    is_remote = remote_label | (source_remote & ~has_physical_location)
    normalized_location = loc.mask(loc.str.strip() == "", "Location not listed")
    normalized_location = normalized_location.mask(is_remote & ~has_physical_location, "Remote")

    role_json = role_family.map(lambda fam: json.dumps({fam: 2.0}, sort_keys=True))
    normalized_title = title.str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    normalized_company = df.get("company", pd.Series("", index=df.index)).astype(str).map(company_enrichment.normalize_company_name)
    seniority_raw = df.get("seniority", pd.Series("", index=df.index)).astype(str).str.lower()
    seniority_level = pd.Series("unknown", index=df.index, dtype="object")
    seniority_level = seniority_level.mask(title.str.contains(r"\b(?:vp|vice president|director|chief|head of)\b", regex=True), "executive")
    seniority_level = seniority_level.mask(title.str.contains(r"\b(?:senior|sr\.?|staff|principal|lead|manager|iii|iv|v)\b", regex=True), "senior")
    seniority_level = seniority_level.mask(title.str.contains(r"\bintern(?:ship)?\b", regex=True), "intern")
    seniority_level = seniority_level.mask(title.str.contains(r"\b(?:junior|jr\.?|entry[- ]?level|new grad|associate)\b", regex=True), "entry")
    seniority_level = seniority_level.mask(title.str.contains(r"\b(?:mid[- ]?level|intermediate|ii)\b", regex=True), "mid")
    seniority_level = seniority_level.mask((seniority_level == "unknown") & seniority_raw.isin(["senior", "executive", "manager"]), "senior")
    seniority_level = seniority_level.mask((seniority_level == "unknown") & seniority_raw.isin(["entry_or_junior", "junior", "entry"]), "entry")
    role_profile = (
        df.get("title", pd.Series("", index=df.index)).astype(str)
        + ". "
        + df.get("company", pd.Series("", index=df.index)).astype(str)
        + ". "
        + df.get("description", pd.Series("", index=df.index)).astype(str).str.slice(0, 1200)
        + ". Skills: "
        + df.get("skills", pd.Series("", index=df.index)).astype(str)
        + ". Role family: "
        + role_family
    )

    return pd.DataFrame(
        {
            "job_index": df.index,
            "title": df.get("title", pd.Series("", index=df.index)).astype(str),
            "company": df.get("company", pd.Series("", index=df.index)).astype(str),
            "normalized_title_static": normalized_title,
            "static_feature_version": STATIC_FEATURE_VERSION,
            "normalized_company_static": normalized_company,
            "normalized_location_static": normalized_location,
            "location_is_remote_static": is_remote,
            "location_remote_label_static": remote_label,
            "location_country_static": country,
            "location_state_static": state,
            "location_city_static": physical_city.mask(is_remote & ~has_physical_location, ""),
            "location_region_matches_json": "[]",
            "location_broad_region_static": state.map(STATE_BROAD_REGIONS).fillna(""),
            "seniority_level_static": seniority_level,
            "required_years_static": required_years,
            "normalized_employment_type_static": normalized_employment,
            "salary_annual_min": "",
            "salary_annual_max": "",
            "salary_display": "Not listed",
            "salary_source_static": "missing",
            "role_family_static": role_family,
            "role_families_json": role_json,
            "role_profile_text": role_profile,
            "duplicate_group_key_static": normalized_company + "|" + normalized_title + "|" + role_family,
        }
    )


def load_or_build_static_job_features(jobs_df: pd.DataFrame, path: Path = ROLE_INDEX_PATH) -> pd.DataFrame:
    required_columns = {
        "job_index", "role_family_static", "role_families_json", "role_profile_text",
        "required_years_static", "normalized_employment_type_static", "salary_source_static",
        "normalized_location_static", "duplicate_group_key_static", "static_feature_version",
    }
    if path.exists():
        try:
            features = pd.read_pickle(path)
            version_ok = (
                "static_feature_version" in features.columns
                and features["static_feature_version"].dropna().astype(str).eq(str(STATIC_FEATURE_VERSION)).all()
            )
            if len(features) == len(jobs_df) and required_columns.issubset(set(features.columns)) and version_ok:
                return features
        except Exception:
            pass
    features = build_fast_static_job_features(jobs_df)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        features.to_pickle(path)
    except Exception:
        pass
    return features


def load_or_build_role_vocabulary(jobs_df: pd.DataFrame, path: Path = ROLE_VOCAB_PATH) -> Dict:
    if path.exists():
        try:
            with open(path, "rb") as fh:
                vocab = pickle.load(fh)
            if isinstance(vocab, dict) and vocab.get("title_terms"):
                return vocab
        except Exception:
            pass
    vocab = build_role_vocabulary(jobs_df)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(vocab, fh)
    except Exception:
        pass
    return vocab


def extract_skills(text: str) -> List[str]:
    text_l = safe_lower(text)
    found = []
    for skill in SKILL_LIST:
        matched = any(contains_term(text_l, variant) for variant in skill_variants(skill))
        if matched:
            found.append(normalize_skill(skill))
    return sorted(set(found))


def normalize_skill(skill: str) -> str:
    special = {"sql", "aws", "gcp", "nlp", "etl", "ux", "ui", "r"}
    skill_l = safe_lower(skill).strip()
    return skill_l.upper() if skill_l in special else skill_l.title()


def skill_variants(skill: str) -> List[str]:
    skill_l = safe_lower(skill).strip()
    return sorted(set([skill_l] + SKILL_SYNONYMS.get(skill_l, [])))


def contains_term(text: str, term: str) -> bool:
    term_l = safe_lower(term).strip()
    if not term_l:
        return False
    if len(term_l) <= 3 or re.search(r"[^a-z0-9 ]", term_l):
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(term_l) + r"(?![a-zA-Z0-9])"
        return re.search(pattern, text) is not None
    return term_l in text


def canonical_skill_key(skill: str) -> str:
    skill_l = safe_lower(skill).strip()
    for canonical, variants in SKILL_SYNONYMS.items():
        if skill_l == canonical or skill_l in variants:
            return canonical
    return skill_l


def role_score_fn(target_role: str, row: pd.Series, semantic_score: float) -> float:
    target = safe_lower(target_role)
    title = safe_lower(row.get("title"))
    text = safe_lower(row.get("clean_job_text") or row.get("job_text"))
    if not target:
        return clamp(semantic_score)
    terms = [x for x in re.split(r"[^a-z0-9]+", target) if len(x) > 2]
    if not terms:
        return clamp(semantic_score)
    title_hits = sum(1 for t in terms if t in title) / len(terms)
    text_hits = sum(1 for t in terms if t in text) / len(terms)
    return clamp(0.55 * semantic_score + 0.35 * title_hits + 0.10 * text_hits)


def skill_score_fn(user_skills: List[str], row: pd.Series) -> Tuple[float, List[str]]:
    if not user_skills:
        job_skills = extract_skills(str(row.get("clean_job_text") or row.get("job_text")))
        return 0.25 if job_skills else 0.0, []
    combined = safe_lower(f"{row.get('skills', '')} {row.get('clean_job_text') or row.get('job_text') or ''}")
    matched = []
    for skill in user_skills:
        variants = skill_variants(canonical_skill_key(skill))
        if any(contains_term(combined, variant) for variant in variants):
            matched.append(normalize_skill(canonical_skill_key(skill)))

    matched = sorted(set(matched))
    job_skill_count = len(extract_skills(combined))
    coverage = len(matched) / max(len(set(canonical_skill_key(s) for s in user_skills)), 1)
    density_bonus = min(0.12, job_skill_count * 0.01)
    return clamp(coverage + density_bonus), matched



def extract_required_years(row: pd.Series | Dict) -> float | None:
    text = safe_lower(f"{row.get('title', '')} {row.get('description', '')} {row.get('clean_job_text', '')}")
    if not text.strip():
        cached = row.get("required_years_static", "")
        try:
            if cached not in [None, "", "unknown"]:
                return float(cached)
        except Exception:
            pass

    patterns = [
        r"experience\s+range\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)",
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)\s+(?:of\s+)?(?:relevant\s+|professional\s+|industry\s+)?experience",
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)\s+(?:in|with|required|working|building|developing)\b",
        r"minimum\s+of\s+(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)",
        r"at\s+least\s+(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)",
        r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)\s+(?:required|preferred|minimum)",
        r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(?:years|yrs)",
    ]

    candidates = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            groups = [g for g in match.groups() if g is not None]
            if not groups:
                continue
            try:
                numbers = [float(g) for g in groups]
                candidates.append(max(numbers))
            except Exception:
                continue

    if not candidates:
        return None

    # Avoid unrealistic false positives from dates or salary ranges.
    candidates = [x for x in candidates if 0 <= x <= 20]
    return max(candidates) if candidates else None


def experience_requirement_exceeds_profile(profile: Dict, row: pd.Series | Dict, tolerance: float = 1.5) -> bool:
    required = extract_required_years(row)
    if required is None:
        return False
    years = profile.get("years_experience")
    try:
        years_val = None if years in [None, "", "unknown"] else float(years)
    except Exception:
        years_val = None
    if years_val is None:
        return required >= 5
    return required > years_val + tolerance


def profile_max_required_years(profile: Dict) -> float | None:
    value = profile.get("max_required_years")
    try:
        if value in [None, "", "unknown"]:
            return None
        return float(value)
    except Exception:
        pass

    raw = profile.get("dealbreakers", [])
    parts = raw if isinstance(raw, list) else [str(raw)]
    text = " ".join(
        [str(x) for x in parts]
        + [
            str(profile.get("raw_resume_text", "")),
            str(profile.get("profile_summary", "")),
            str(profile.get("preferences", "")),
        ]
    ).lower()
    matches = re.findall(
        r"(?:no|avoid|exclude)\s+(?:roles?\s+)?(?:requiring\s+)?(?:more\s+than\s+)?(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)",
        text,
    )
    if not matches:
        return None
    values = []
    for match in matches:
        try:
            values.append(float(match))
        except Exception:
            pass
    return min(values) if values else None


def required_years_violates_profile(profile: Dict, row: pd.Series | Dict) -> bool:
    required = extract_required_years(row)
    if required is None:
        return False

    raw = profile.get("dealbreakers", [])
    parts = raw if isinstance(raw, list) else [str(raw)]
    text = " ".join(
        [str(x) for x in parts]
        + [
            str(profile.get("raw_resume_text", "")),
            str(profile.get("profile_summary", "")),
            str(profile.get("preferences", "")),
        ]
    ).lower()
    no_plus_values = []
    for match in re.findall(
        r"(?:no|avoid|exclude)\s+(?:roles?\s+)?(?:requiring\s+)?(?:more\s+than\s+)?(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)",
        text,
    ):
        try:
            no_plus_values.append(float(match))
        except Exception:
            pass
    if no_plus_values and required >= min(no_plus_values):
        return True

    max_required_years = profile.get("max_required_years")
    try:
        if max_required_years not in [None, "", "unknown"]:
            return required > float(max_required_years)
    except Exception:
        pass

    parsed_limit = profile_max_required_years(profile)
    return bool(parsed_limit is not None and required >= parsed_limit)



def normalize_location_text(value: str) -> str:
    text = safe_lower(value)
    text = re.sub(r"\{[^}]+\}", " ", text)
    text = text.replace("u.s.", "us").replace("u.s", "us")
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,")


LOCATION_VOCAB_CACHE: Dict | None = None
PRECISE_ROLE_FAMILY_CACHE: Dict[object, Dict[str, float]] = {}


def canonical_location_key(value: str) -> str:
    text = normalize_location_text(value)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def display_city_state(city: str, state: str = "") -> str:
    city = re.sub(r"\s+", " ", str(city or "").strip())
    state = str(state or "").strip().upper()
    return f"{city}, {state}" if city and state else city


def add_location_alias(vocab: Dict, alias: str, display: str, city: str = "", state: str = "", region: str = "") -> None:
    alias_norm = normalize_location_text(alias)
    if not alias_norm:
        return
    display = display or alias
    city = city or display.split(",")[0].strip()
    state = state.upper() if state else ""
    key = canonical_location_key(display)
    vocab["aliases"][alias_norm] = {
        "key": key,
        "display": display,
        "city": city,
        "state": state,
        "region": region,
    }
    if city:
        vocab["city_aliases"][normalize_location_text(city)] = vocab["aliases"][alias_norm]
    if state:
        vocab["states"][state] = US_STATE_NAMES.get(state, state)


def regions_for_city_state(city: str, state: str = "") -> List[str]:
    text = normalize_location_text(f"{city} {state}")
    regions = []
    for region, aliases in REGION_ALIASES.items():
        if any(contains_term(text, alias) for alias in aliases):
            regions.append(region)
    broad = STATE_BROAD_REGIONS.get(str(state or "").upper())
    if broad:
        regions.append(broad)
    return list(dict.fromkeys(regions))


def build_location_vocabulary(jobs_csv: Path = JOBS_CSV) -> Dict:
    """Build a reusable location vocabulary from the offline job database."""
    global LOCATION_VOCAB_CACHE
    if LOCATION_VOCAB_CACHE is not None:
        return LOCATION_VOCAB_CACHE
    if LOCATION_VOCAB_PATH.exists():
        try:
            with open(LOCATION_VOCAB_PATH, "rb") as fh:
                cached_vocab = pickle.load(fh)
            if isinstance(cached_vocab, dict) and cached_vocab.get("aliases"):
                LOCATION_VOCAB_CACHE = cached_vocab
                return LOCATION_VOCAB_CACHE
        except Exception:
            pass

    vocab = {"aliases": {}, "city_aliases": {}, "states": {}, "regions": dict(REGION_ALIASES), "countries": {"US": "United States"}}
    for code, name in US_STATE_NAMES.items():
        vocab["states"][code] = name
        add_location_alias(vocab, name, name, state=code)
        add_location_alias(vocab, code, name, state=code)

    if jobs_csv.exists():
        try:
            df = pd.read_csv(jobs_csv, usecols=lambda col: col in {"location", "search_location"}).fillna("")
            values = []
            for col in [c for c in ["location", "search_location"] if c in df.columns]:
                values.extend(df[col].dropna().astype(str).unique().tolist())
            for raw in values:
                raw_text = str(raw or "").strip()
                if not raw_text:
                    continue
                for piece in re.split(r"\s*[;|/]\s*", raw_text):
                    part = piece.strip()
                    if not part:
                        continue
                    loc = normalize_location_text(part)
                    if loc in {"remote", "anywhere", "worldwide", "wfh", "work from home"}:
                        continue
                    city_state = re.match(r"^\s*([A-Za-z][A-Za-z .'-]+?)\s*,\s*([A-Z]{2})\b", part)
                    if city_state and city_state.group(2).upper() in US_STATE_CODES:
                        city = re.sub(r"\s+", " ", city_state.group(1)).strip()
                        if normalize_location_text(city) in {"remote", "anywhere", "worldwide", "wfh", "work from home"}:
                            continue
                        state = city_state.group(2).upper()
                        regions = regions_for_city_state(city, state)
                        region = next((r for r in regions if r in METRO_REGION_KEYS), "")
                        add_location_alias(vocab, city, display_city_state(city, state), city=city, state=state, region=region)
                        add_location_alias(vocab, f"{city}, {state}", display_city_state(city, state), city=city, state=state, region=region)
                        continue
                    city_country = re.match(r"^\s*([A-Za-z][A-Za-z .'-]+?)\s*,\s*(?:United States|USA|US)\b", part, flags=re.I)
                    if city_country:
                        city = re.sub(r"\s+", " ", city_country.group(1)).strip()
                        if normalize_location_text(city) in {"remote", "anywhere", "worldwide", "wfh", "work from home"}:
                            continue
                        add_location_alias(vocab, city, city, city=city)
        except Exception:
            pass

    common_aliases = [
        ("nyc", "New York, NY", "New York", "NY", "new_york"),
        ("new york", "New York, NY", "New York", "NY", "new_york"),
        ("new york city", "New York, NY", "New York", "NY", "new_york"),
        ("sf", "San Francisco, CA", "San Francisco", "CA", "bay_area"),
        ("san francisco", "San Francisco, CA", "San Francisco", "CA", "bay_area"),
        ("bay area", "San Francisco, CA / Bay Area", "San Francisco", "CA", "bay_area"),
        ("la", "Los Angeles, CA", "Los Angeles", "CA", "la_area"),
        ("los angeles", "Los Angeles, CA", "Los Angeles", "CA", "la_area"),
        ("dc", "Washington, DC", "Washington", "DC", ""),
        ("washington dc", "Washington, DC", "Washington", "DC", ""),
    ]
    for alias, display, city, state, region in common_aliases:
        add_location_alias(vocab, alias, display, city=city, state=state, region=region)

    LOCATION_VOCAB_CACHE = vocab
    try:
        LOCATION_VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOCATION_VOCAB_PATH, "wb") as fh:
            pickle.dump(vocab, fh)
    except Exception:
        pass
    return vocab


def parse_location_preferences(profile_or_text: Dict | str) -> Dict:
    """Parse flexible user location preferences into reusable constraints."""
    if isinstance(profile_or_text, dict):
        stored = profile_or_text.get("_location_preferences") or profile_or_text.get("location_preferences")
        if isinstance(stored, dict) and stored.get("display"):
            return stored
        text = safe_lower(profile_or_text.get("location_preference"))
    else:
        text = safe_lower(profile_or_text)
    normalized = normalize_location_text(text)
    vocab = build_location_vocabulary()

    constraints = {
        "remote_allowed": False,
        "remote_required": False,
        "remote_ok": False,
        "preferred_locations": [],
        "allowed_regions": [],
        "allowed_states": [],
        "allowed_countries": [],
        "excluded_locations": [],
        "strict_location": False,
        "relocation_allowed": True,
        "cities": [],
        "states": [],
        "regions": [],
        "country": None,
        "flexible_us": False,
        "international_allowed": False,
        "display": "",
    }

    if not normalized or normalized in {"any", "flexible", "open"}:
        constraints["display"] = str(text or "").strip()
        return constraints

    display_items = []
    if "remote" in normalized or "anywhere" in normalized or "wfh" in normalized:
        constraints["remote_allowed"] = True
        constraints["remote_ok"] = True
        display_items.append("Remote")
        if normalized == "remote" or re.search(r"\bremote\s+(only|required|within)\b|\bfully remote\b", normalized):
            constraints["remote_required"] = True

    flexible_us = bool(re.search(r"\b(any|open to|flexible).{0,15}\b(us|usa|united states)\b", normalized))
    us_only = bool(re.search(r"\b(us|usa|united states)\s+only\b|\bonly\s+(?:in\s+)?(?:the\s+)?(?:us|usa|united states)\b", normalized))
    if flexible_us or us_only or "remote within us" in normalized:
        constraints["allowed_countries"].append("US")
        constraints["country"] = "US"
        constraints["flexible_us"] = flexible_us
        constraints["strict_location"] = True
        display_items.append("Any US location" if flexible_us else "United States")

    if "preferred" in normalized or "preference" in normalized:
        constraints["strict_location"] = False
    if any(term in normalized for term in ["only", "within", "no relocation", "remote required", "remote only", "fully remote only"]):
        constraints["strict_location"] = True
    if constraints["remote_required"]:
        constraints["strict_location"] = True
    if "no relocation" in normalized:
        constraints["relocation_allowed"] = False
    if any(term in normalized for term in ["international", "global", "worldwide", "any country", "outside us", "outside the us", "outside usa"]):
        constraints["international_allowed"] = True

    for region, aliases in REGION_ALIASES.items():
        if any(contains_term(normalized, alias) for alias in aliases):
            constraints["allowed_regions"].append(region)
            constraints["regions"].append(region)
            if "preferred" not in normalized:
                constraints["strict_location"] = True

    for alias, state in STATE_NAME_TO_CODE.items():
        if (len(alias) <= 2 and alias != "dc") or alias in {"new york", "washington"}:
            continue
        if contains_term(normalized, alias):
            constraints["allowed_states"].append(state)
            constraints["states"].append(state)
            if US_STATE_NAMES.get(state, state) not in display_items:
                display_items.append(US_STATE_NAMES.get(state, state))
            if "preferred" not in normalized:
                constraints["strict_location"] = True

    for country_alias, code in COUNTRY_ALIASES.items():
        if code == "US" and contains_term(normalized, country_alias):
            constraints["allowed_countries"].append("US")
            constraints["country"] = "US"

    # Scan the full text so comma/or/and separated multi-location preferences all survive.
    matched_location_displays = []
    for alias, entry in sorted(vocab["aliases"].items(), key=lambda item: len(item[0]), reverse=True):
        if len(alias) < 5 and alias not in {"nyc", "sf", "la", "dc"}:
            continue
        if not contains_term(normalized, alias):
            continue
        position = normalized.find(alias)
        city = entry.get("city", "")
        state = entry.get("state", "")
        region = entry.get("region", "")
        entry_regions = regions_for_city_state(city, state)
        if region:
            entry_regions.insert(0, region)
        entry_regions = list(dict.fromkeys(entry_regions))
        key = entry.get("key", "")
        if city:
            constraints["cities"].append(display_city_state(city, state))
            constraints["preferred_locations"].append(canonical_location_key(city))
        if state:
            constraints["allowed_states"].append(state)
            constraints["states"].append(state)
        for matched_region in entry_regions:
            constraints["allowed_regions"].append(matched_region)
            constraints["regions"].append(matched_region)
        display = entry.get("display", "")
        if display:
            matched_location_displays.append((position if position >= 0 else 99999, display))
        if key and "preferred" not in normalized:
            constraints["strict_location"] = True

    if any(term in normalized for term in ["any us", "any usa", "united states", "us only", "open to us", "remote within us"]):
        constraints["allowed_countries"].append("US")
        constraints["country"] = "US"
        constraints["strict_location"] = True

    if (constraints["allowed_states"] or constraints["preferred_locations"] or constraints["allowed_regions"]) and not constraints["international_allowed"]:
        constraints["allowed_countries"].append("US")
        constraints["country"] = "US"

    for key in ["allowed_regions", "allowed_states", "allowed_countries", "preferred_locations", "cities", "states", "regions"]:
        constraints[key] = list(dict.fromkeys([x for x in constraints[key] if x]))
    for _, display in sorted(matched_location_displays, key=lambda item: item[0]):
        if display not in display_items:
            display_items.append(display)
    constraints["display"] = ", ".join(dict.fromkeys(display_items)) or str(profile_or_text.get("location_preference", "") if isinstance(profile_or_text, dict) else profile_or_text).strip()
    return constraints


def normalize_job_location(row: pd.Series | Dict) -> Dict:
    cached_normalized = str(row.get("normalized_location_static", "") or "").strip()
    if cached_normalized:
        try:
            region_matches = json.loads(row.get("location_region_matches_json", "[]") or "[]")
        except Exception:
            region_matches = []
        return {
            "raw_location": str(row.get("location", "") or "").strip(),
            "normalized_location": cached_normalized,
            "is_remote": str(row.get("location_is_remote_static", "")).lower() in {"true", "1", "yes"},
            "remote_label": str(row.get("location_remote_label_static", "")).lower() in {"true", "1", "yes"},
            "source_remote": str(row.get("is_remote", "")).lower() in {"true", "1", "yes"},
            "country": str(row.get("location_country_static", "") or ""),
            "state": str(row.get("location_state_static", "") or ""),
            "city": str(row.get("location_city_static", "") or ""),
            "region_matches": region_matches,
            "broad_region": str(row.get("location_broad_region_static", "") or ""),
        }
    raw = str(row.get("location", "") or "").strip()
    title = safe_lower(row.get("title"))
    loc = normalize_location_text(raw)
    location_remote_label = "remote" in loc or loc in {"anywhere", "worldwide", "wfh"}
    title_remote_label = re.search(r"\b(?:remote|wfh|work from home|work-from-home)\b", title) is not None
    source_remote = str(row.get("is_remote", "")).lower() in {"true", "1", "yes"}
    remote_label = bool(location_remote_label or title_remote_label)
    country = ""
    state = ""
    city = ""

    for term, code in NON_US_LOCATION_TERMS.items():
        if contains_term(loc, term):
            country = code
            break
    for alias, code in COUNTRY_ALIASES.items():
        if contains_term(loc, alias):
            country = code
            break

    state_match = re.search(r"(?:^|,\s*|\s)([A-Z]{2})(?:,|\s|$)", raw)
    if state_match and state_match.group(1).upper() in US_STATE_CODES:
        state = state_match.group(1).upper()
        country = "US"
    if not state:
        for alias, code in STATE_NAME_TO_CODE.items():
            if contains_term(loc, alias):
                state = code
                country = "US"
                break

    if not country and any(term in loc for term in ["united states", " usa", " us"]):
        country = "US"
    if loc in {"remote", "anywhere", "worldwide"}:
        country = ""

    parts = [p.strip() for p in re.split(r",|\|", raw) if p.strip()]
    if parts:
        possible_city = re.sub(r"[^A-Za-z .'-]", "", parts[0]).strip()
        if possible_city and safe_lower(possible_city) not in {"remote", "anywhere", "united states", "usa"}:
            city = possible_city

    region_matches = []
    for region, aliases in REGION_ALIASES.items():
        if any(contains_term(loc, alias) for alias in aliases):
            region_matches.append(region)
    broad_region = STATE_BROAD_REGIONS.get(state)
    if broad_region:
        region_matches.append(broad_region)

    has_physical_location = bool(city or state or (country and loc not in {"remote", "anywhere", "worldwide"}))
    is_remote = bool(remote_label or (source_remote and not has_physical_location))

    normalized_parts = []
    if is_remote:
        normalized_parts.append("Remote")
    if city:
        normalized_parts.append(city)
    if state:
        normalized_parts.append(state)
    if country:
        normalized_parts.append("US" if country == "US" else country)

    return {
        "raw_location": raw,
        "normalized_location": ", ".join(dict.fromkeys(normalized_parts)) or raw or "Location not listed",
        "is_remote": is_remote,
        "remote_label": remote_label,
        "source_remote": source_remote,
        "country": country,
        "state": state,
        "city": city,
        "region_matches": sorted(set(region_matches)),
        "broad_region": broad_region or "",
    }


def location_constraint_decision(profile: Dict, row: pd.Series | Dict) -> Dict:
    constraints = parse_location_preferences(profile)
    job_loc = normalize_job_location(row)
    strict = bool(constraints["strict_location"])
    reason = "Flexible or broad location preference"
    score = 0.70
    passed = True

    has_constraints = any(
        constraints[key]
        for key in ["remote_allowed", "remote_required", "preferred_locations", "allowed_regions", "allowed_states", "allowed_countries"]
    )
    if not has_constraints:
        return {**job_loc, "score": score, "reason": reason, "pass": True, "constraints": constraints}

    us_bounded = "US" in constraints["allowed_countries"] and not constraints.get("international_allowed")
    if us_bounded and job_loc["country"] not in {"", "US"}:
        return {**job_loc, "score": 0.0, "reason": "Outside stated location preference: non-US roles were not requested.", "pass": False, "constraints": constraints}

    strict_remote_region = bool(
        constraints["strict_location"]
        and constraints["remote_allowed"]
        and (constraints["allowed_regions"] or constraints["preferred_locations"])
    )
    location_specific_us = bool(
        us_bounded
        and (constraints["allowed_regions"] or constraints["allowed_states"] or constraints["preferred_locations"])
        and not constraints.get("flexible_us")
    )

    if job_loc["is_remote"] and constraints["remote_allowed"] and "US" in constraints["allowed_countries"] and job_loc["country"] in {"", "US"}:
        return {**job_loc, "score": 1.0, "location_tier": 1, "reason": "Location fits: remote role matches remote preference.", "pass": True, "constraints": constraints}

    if job_loc["is_remote"] and constraints["remote_allowed"]:
        if strict_remote_region and not job_loc["remote_label"]:
            return {**job_loc, "score": 0.0, "reason": "Outside stated location preference", "pass": False, "constraints": constraints}
        if "US" in constraints["allowed_countries"] and job_loc["country"] not in {"", "US"}:
            return {**job_loc, "score": 0.0, "reason": "Outside stated location preference", "pass": False, "constraints": constraints}
        return {**job_loc, "score": 1.0, "location_tier": 1, "reason": "Location fits: remote role matches remote preference.", "pass": True, "constraints": constraints}

    if constraints["remote_required"] and not job_loc["is_remote"]:
        passed = False
        score = 0.0
        reason = "Outside stated location preference"
        return {**job_loc, "score": score, "reason": reason, "pass": passed, "constraints": constraints}

    if constraints["preferred_locations"]:
        city_text = safe_lower(job_loc["city"])
        for pref in constraints["preferred_locations"]:
            if pref.replace("_", " ") in city_text:
                city_display = job_loc["city"] or pref.replace("_", " ").title()
                return {**job_loc, "score": 1.0, "location_tier": 1, "reason": f"Location fits: {city_display} matches one of the preferred cities.", "pass": True, "constraints": constraints}

    region_overlap = set(constraints["allowed_regions"]) & set(job_loc["region_matches"])
    metro_overlap = region_overlap & METRO_REGION_KEYS
    if metro_overlap:
        region_name = next(iter(metro_overlap)).replace("_", " ").title()
        city_display = job_loc["city"] or region_name
        return {**job_loc, "score": 0.86, "location_tier": 2, "reason": f"Nearby location: {city_display} is within the {region_name} metro area.", "pass": True, "constraints": constraints}

    if constraints["allowed_states"] and job_loc["state"] in constraints["allowed_states"]:
        city_display = job_loc["city"] or job_loc["state"]
        return {**job_loc, "score": 0.72, "location_tier": 3, "reason": f"Same-state fallback: {city_display} included because few exact-location matches may remain.", "pass": True, "constraints": constraints}

    broad_overlap = region_overlap & BROAD_REGION_KEYS
    if broad_overlap and not strict_remote_region:
        region_name = next(iter(broad_overlap)).replace("_", " ").title()
        return {**job_loc, "score": 0.58, "location_tier": 4, "reason": f"Broader fallback: {region_name} location included because few closer matches may remain.", "pass": True, "constraints": constraints}

    if constraints["allowed_countries"] and job_loc["country"] in constraints["allowed_countries"] and not location_specific_us:
        if job_loc["country"] == "US":
            reason = "Location fits: US-only preference satisfied."
        else:
            reason = "Location fits: country preference satisfied."
        return {**job_loc, "score": 0.92, "location_tier": 1, "reason": reason, "pass": True, "constraints": constraints}

    if not strict and constraints.get("country") in {None, "", "US"} and job_loc["country"] == "US":
        return {**job_loc, "score": 0.42, "location_tier": 5, "reason": "Broader fallback: included because few matches remained after hard filters.", "pass": True, "constraints": constraints}

    if strict and location_specific_us and constraints["remote_allowed"] and job_loc["country"] == "US":
        return {**job_loc, "score": 0.42, "location_tier": 5, "reason": "US fallback: included only after preferred remote or regional matches are prioritized.", "pass": True, "constraints": constraints}

    if not job_loc["raw_location"]:
        return {**job_loc, "score": 0.45 if strict else 0.60, "location_tier": 9, "reason": "Location uncertainty: job location not listed.", "pass": True, "constraints": constraints}

    if strict:
        return {**job_loc, "score": 0.0, "reason": "Outside stated location preference", "pass": False, "constraints": constraints}

    if constraints["remote_allowed"]:
        return {**job_loc, "score": 0.65, "reason": "Location differs from remote preference", "pass": True, "constraints": constraints}
    return {**job_loc, "score": 0.55, "reason": "Location is acceptable but not preferred", "pass": True, "constraints": constraints}


def should_filter_location_for_profile(profile: Dict, row: pd.Series | Dict) -> bool:
    decision = location_constraint_decision(profile, row)
    return bool(decision["constraints"]["strict_location"] and not decision["pass"])


def split_target_phrases(target_role: str) -> List[str]:
    raw = safe_lower(target_role)
    parts = re.split(r"[,;/]|\bor\b", raw)
    phrases = []
    for part in parts:
        part = re.sub(r"\([^)]*\)", " ", part)
        part = re.sub(r"\s+", " ", part).strip()
        if part:
            phrases.append(part)
    return phrases


ANALYTICS_ENGINEERING_TARGET_TERMS = [
    "analytics engineer",
    "analytics engineering",
    "bi engineer",
    "business intelligence engineer",
    "bi engineering",
    "data platform",
    "data platform analyst",
    "data engineer - analytics",
    "data engineer analytics",
    "analytics data engineer",
    "data warehouse engineer",
]

ANALYTICS_ENGINEERING_PREFERRED_TITLES = [
    "analytics engineer",
    "bi engineer",
    "business intelligence engineer",
    "data platform analyst",
    "data engineer - analytics",
    "data engineer analytics",
    "analytics data engineer",
    "data warehouse engineer",
]

ANALYTICS_ENGINEERING_PLATFORM_SIGNALS = [
    "dbt",
    "snowflake",
    "bigquery",
    "data warehouse",
    "warehouse",
    "semantic layer",
    "etl",
    "elt",
    "analytics engineering",
    "bi engineering",
    "business intelligence",
    "looker",
    "tableau",
    "power bi",
    "sql",
    "data platform",
]

ANALYTICS_ENGINEERING_PENALIZED_TITLES = {
    "marketing analyst": ["marketing analyst", "growth marketing analyst", "marketing data analyst"],
    "business analyst": ["business analyst"],
    "ml engineer": ["ml engineer", "machine learning engineer"],
    "data scientist": ["data scientist"],
    "software engineer": ["software engineer", "software development engineer", "generic software engineer"],
    "backend engineer": ["backend engineer", "back end engineer"],
}


def is_analytics_engineering_profile(profile: Dict) -> bool:
    target = safe_lower(profile.get("target_role"))
    return any(term in target for term in ANALYTICS_ENGINEERING_TARGET_TERMS)


def target_explicitly_lists_role(target: str, role_key: str) -> bool:
    target_l = safe_lower(target)
    phrases = split_target_phrases(target_l)
    terms = ANALYTICS_ENGINEERING_PENALIZED_TITLES.get(role_key, [role_key])
    return any(any(term in phrase for term in terms) for phrase in phrases)


def analytics_engineering_preferred_title(title: str) -> bool:
    return any(term in title for term in ANALYTICS_ENGINEERING_PREFERRED_TITLES)


def analytics_engineering_penalized_title(title: str, target: str) -> str:
    for role_key, terms in ANALYTICS_ENGINEERING_PENALIZED_TITLES.items():
        if any(term in title for term in terms) and not target_explicitly_lists_role(target, role_key):
            if role_key == "software engineer" and any(term in title for term in ["analytics engineer", "data engineer"]):
                continue
            return role_key
    return ""


def role_family_hits(text: str, patterns: Iterable[str]) -> int:
    return sum(1 for pattern in patterns if contains_term(text, pattern))


NON_ML_RESEARCH_SIGNALS = [
    "ux research", "user research", "usability studies", "usability testing", "concept studies",
    "concept testing", "surveys", "survey research", "qualitative research", "mixed-method research",
    "mixed method research", "human factors", "hci", "anthropology", "sociology", "market research",
    "research logistics", "research reports", "focus groups",
]

ML_MODELING_SIGNALS = [
    "machine learning", "deep learning", "model training", "pytorch", "tensorflow", "scikit-learn",
    "sklearn", "nlp", "computer vision", "ai research", "statistical modeling", "ml model",
    "predictive modeling", "predictive model", "model deployment", "model serving", "mlops",
    "artificial intelligence", "llm", "large language model", "generative ai", "foundation model",
    "model monitoring", "model inference", "scientific computing",
]

WEAK_NON_ML_ANALYTICS_SIGNALS = [
    "dashboard", "dashboards", "reporting", "reports", "excel", "tableau", "power bi",
    "business intelligence", "data visualization", "visualization", "survey", "surveys",
    "usability", "sql analysis", "marketing analytics", "operations analyst",
]


def has_non_ml_research_signal(text: str) -> bool:
    return any(contains_term(text, term) for term in NON_ML_RESEARCH_SIGNALS)


def has_ml_modeling_signal(text: str) -> bool:
    return any(contains_term(text, term) for term in ML_MODELING_SIGNALS)


def profile_requires_ml_related_roles(profile: Dict) -> bool:
    text = " ".join(
        [
            str(profile.get("target_role", "")),
            str(profile.get("profile_summary", "")),
            str(profile.get("raw_resume_text", "")),
            str(profile.get("location_preference", "")),
        ]
        + [str(x) for x in profile.get("preferred_industries", [])]
        + [str(x) for x in profile.get("dealbreakers", [])]
        + [str(x) for x in profile.get("project_keywords", [])]
    ).lower()
    trigger_terms = [
        "ml engineer", "machine learning engineer", "applied scientist", "research scientist",
        "ai engineer", "machine learning roles only", "all ml-related", "all ml related",
        "ml-focused", "ml focused", "pivot to ml engineering", "machine learning roles",
    ]
    if any(term in text for term in trigger_terms):
        return True
    return "data scientist" in text and any(term in text for term in ["ml-focused", "ml focused", "machine learning", "modeling"])


def profile_targets_technical_ic_roles(profile: Dict) -> bool:
    target = safe_lower(profile.get("target_role"))
    return any(term in target for term in TECHNICAL_IC_TARGET_TERMS)


def profile_explicitly_targets_management(profile: Dict) -> bool:
    target = safe_lower(profile.get("target_role"))
    return any(term in target for term in MANAGEMENT_TARGET_TERMS)


def has_executive_operations_title(row: pd.Series | Dict) -> bool:
    title = safe_lower(row.get("title"))
    return any(re.search(pattern, title) for pattern in EXECUTIVE_OPERATIONS_TITLE_PATTERNS)


def executive_operations_title_mismatch(profile: Dict, row: pd.Series | Dict) -> bool:
    return (
        profile_targets_technical_ic_roles(profile)
        and not profile_explicitly_targets_management(profile)
        and has_executive_operations_title(row)
    )


def data_center_context_without_ml_evidence(row: pd.Series | Dict) -> bool:
    text = safe_lower(f"{row.get('title', '')} {row.get('description', '')} {row.get('clean_job_text', '')}")
    if not any(re.search(pattern, text) for pattern in DATA_CENTER_CONTEXT_PATTERNS):
        return False
    return not has_ml_modeling_signal(text) and not any(
        term in text
        for term in [
            "machine learning engineer", "ml engineer", "data scientist", "applied scientist",
            "model training", "model evaluation", "feature engineering", "mlops", "ml platform",
        ]
    )


def ml_related_decision(profile: Dict, row: pd.Series | Dict) -> Tuple[bool, str]:
    if not profile_requires_ml_related_roles(profile):
        return True, "ML-specific constraint not requested"

    title = safe_lower(row.get("title"))
    text = safe_lower(f"{row.get('title', '')} {row.get('description', '')} {row.get('clean_job_text', '')} {row.get('skills', '')}")
    if executive_operations_title_mismatch(profile, row):
        return False, "Title appears executive/operations-focused, not a technical ML/data IC role"
    if data_center_context_without_ml_evidence(row):
        return False, "Data center context is not evidence of ML/data science role relevance"
    job_families = infer_job_role_families(row, use_cache=False)
    family = max(job_families, key=job_families.get) if job_families else "unclear"
    has_strong_ml = has_ml_modeling_signal(text)
    has_non_ml_research = has_non_ml_research_signal(text) and not has_strong_ml
    weak_only = any(contains_term(text, term) for term in WEAK_NON_ML_ANALYTICS_SIGNALS) and not has_strong_ml

    if has_non_ml_research:
        return False, "Non-ML UX/product/market research signals"
    if family in {"ux_research", "product_research", "market_research", "marketing", "finance_accounting", "bi_analytics"}:
        if not has_strong_ml:
            return False, f"{FRIENDLY_ROLE_FAMILY_LABELS.get(family, family)} without ML/modeling signals"
    if any(term in title for term in ["marketing analyst", "operations analyst", "reporting analyst", "business analyst"]):
        if not has_strong_ml:
            return False, "Analyst title lacks ML/modeling signals"
    if "data scientist" in title or "scientist" in title:
        if has_strong_ml:
            return True, "Data/scientist role has ML/modeling signals"
        return False, "Data/scientist title lacks ML/modeling signals"
    if any(term in title for term in ["researcher", "research scientist", "applied scientist"]):
        if has_strong_ml:
            return True, "Research/scientist role has ML/AI signals"
        return False, "Research title lacks ML/AI/modeling signals"
    if family in {"ml_engineering", "ml_platform_mlops"} and has_strong_ml:
        return True, "ML engineering/platform role with strong ML signals"
    if any(term in title for term in ["machine learning", "ml engineer", "ai engineer", "applied ai"]):
        if has_strong_ml:
            return True, "ML/AI title and modeling signals"
        return False, "ML/AI title lacks supporting modeling evidence"
    if family in {"data_analytics", "software_engineering", "devops", "data_engineering", "analytics_engineering"}:
        if has_strong_ml:
            return True, f"{FRIENDLY_ROLE_FAMILY_LABELS.get(family, family)} role has ML/modeling signals"
        return False, f"{FRIENDLY_ROLE_FAMILY_LABELS.get(family, family)} lacks ML/modeling signals"
    if weak_only:
        return False, "Only weak analytics/reporting signals found"
    if has_strong_ml:
        return True, "Description has strong ML/modeling signals"
    return False, "No strong ML/modeling signal found"


def infer_role_families_from_text(text: str, title_weight: float = 1.0) -> Dict[str, float]:
    title_text = safe_lower(text)
    scores = {}
    for family, config in ROLE_FAMILIES.items():
        title_hits = role_family_hits(title_text, config["title"])
        text_hits = role_family_hits(title_text, config["text"])
        avoid_hits = role_family_hits(title_text, config.get("avoid", []))
        score = title_weight * title_hits + 0.35 * text_hits - 0.60 * avoid_hits
        if score > 0:
            scores[family] = round(float(score), 3)
    return scores


def infer_target_role_families(target_role: str) -> Dict[str, float]:
    target = safe_lower(target_role)
    scores = infer_role_families_from_text(target, title_weight=1.4)
    if any(term in target for term in ["credit risk", "risk data scientist", "risk analytics", "risk modeling", "fintech data scientist"]):
        scores["risk_analytics"] = max(scores.get("risk_analytics", 0), 1.8)
        scores["credit_risk"] = max(scores.get("credit_risk", 0), 1.6)
        scores["data_science"] = max(scores.get("data_science", 0), 1.0)
        scores["financial_analytics"] = max(scores.get("financial_analytics", 0), 0.9)
    if any(term in target for term in ["quantitative analyst", "quant analyst", "quantitative analytics"]):
        scores["quantitative_analytics"] = max(scores.get("quantitative_analytics", 0), 1.8)
        scores["risk_analytics"] = max(scores.get("risk_analytics", 0), 0.9)
    if any(term in target for term in ["fintech", "banking technology", "financial services analytics"]):
        scores["financial_analytics"] = max(scores.get("financial_analytics", 0), 1.2)
        scores["risk_analytics"] = max(scores.get("risk_analytics", 0), 1.0)
    if any(term in target for term in ["ml platform", "mlops", "machine learning platform", "ml infrastructure"]):
        scores["ml_platform_mlops"] = max(scores.get("ml_platform_mlops", 0), 1.7)
        scores["ml_engineering"] = max(scores.get("ml_engineering", 0), 1.1)
        scores.pop("devops", None)
        scores.pop("software_engineering", None)
    if "ml" in target and "engineer" in target:
        scores["ml_engineering"] = max(scores.get("ml_engineering", 0), 1.4)
    if is_analytics_engineering_profile({"target_role": target}):
        scores["analytics_engineering"] = max(scores.get("analytics_engineering", 0), 1.4)
        scores["data_engineering"] = max(scores.get("data_engineering", 0), 0.8)
        scores["data_analytics"] = max(scores.get("data_analytics", 0), 0.7)
    if "business intelligence" in target or "bi " in f"{target} ":
        scores["bi_analytics"] = max(scores.get("bi_analytics", 0), 1.4)
    if "data analyst" in target or "analytics analyst" in target:
        scores["data_analytics"] = max(scores.get("data_analytics", 0), 1.3)
    if "junior data scientist" in target:
        scores["data_science"] = max(scores.get("data_science", 0), 1.0)
    return scores


def infer_job_role_families(row: pd.Series | Dict, use_cache: bool = True) -> Dict[str, float]:
    if use_cache:
        cached_json = str(row.get("role_families_json", "") or "").strip()
        if cached_json:
            try:
                parsed = json.loads(cached_json)
                if isinstance(parsed, dict) and parsed:
                    return {str(k): float(v) for k, v in parsed.items()}
            except Exception:
                pass
        cached_family = str(row.get("role_family_static", "") or "").strip()
        if cached_family:
            return {cached_family: 2.0}
    cache_key = None
    if not use_cache and hasattr(row, "name"):
        try:
            cache_key = int(row.name)
        except Exception:
            cache_key = None
    if cache_key is not None and cache_key in PRECISE_ROLE_FAMILY_CACHE:
        return dict(PRECISE_ROLE_FAMILY_CACHE[cache_key])
    title = safe_lower(row.get("title"))
    text = safe_lower(f"{row.get('title', '')} {row.get('description', '')} {row.get('clean_job_text', '')}")
    scores = infer_role_families_from_text(text, title_weight=0.8)
    if data_center_context_without_ml_evidence(row):
        if any(term in title for term in ["chief", "coo", "ceo", "vp", "director", "head of"]):
            result = {"executive_management": 2.2, "operations": 1.4}
            if cache_key is not None:
                PRECISE_ROLE_FAMILY_CACHE[cache_key] = dict(result)
            return result
        if "operations" in title or "manager" in title:
            result = {"operations": 2.0}
            if cache_key is not None:
                PRECISE_ROLE_FAMILY_CACHE[cache_key] = dict(result)
            return result
    if "marketing analyst" in title or "marketing data analyst" in title:
        result = {"marketing_analytics": 2.0, "marketing": 1.2}
        if cache_key is not None:
            PRECISE_ROLE_FAMILY_CACHE[cache_key] = dict(result)
        return result
    if "credit risk" in title:
        scores["credit_risk"] = max(scores.get("credit_risk", 0), 2.2)
        scores["risk_analytics"] = max(scores.get("risk_analytics", 0), 1.7)
    if "risk data scientist" in title or ("risk" in title and "data scientist" in title):
        scores["risk_analytics"] = max(scores.get("risk_analytics", 0), 2.1)
        scores["data_science"] = max(scores.get("data_science", 0), 1.5)
    if "quantitative analyst" in title or "quant analyst" in title:
        scores["quantitative_analytics"] = max(scores.get("quantitative_analytics", 0), 2.1)
    if any(term in title for term in ["auditor", "audit analyst", "internal audit"]):
        result = {"audit": 2.0}
        if cache_key is not None:
            PRECISE_ROLE_FAMILY_CACHE[cache_key] = dict(result)
        return result
    if any(term in title for term in ["accountant", "accounting analyst"]):
        result = {"accounting": 2.0}
        if cache_key is not None:
            PRECISE_ROLE_FAMILY_CACHE[cache_key] = dict(result)
        return result
    if any(term in title for term in ["account executive", "sales representative", "sales manager"]):
        result = {"sales": 2.0}
        if cache_key is not None:
            PRECISE_ROLE_FAMILY_CACHE[cache_key] = dict(result)
        return result
    if any(term in title for term in ["backend engineer", "back end engineer"]):
        scores["backend_engineering"] = max(scores.get("backend_engineering", 0), 2.0)
    for family, config in ROLE_FAMILIES.items():
        title_hits = role_family_hits(title, config["title"])
        if title_hits:
            scores[family] = max(scores.get(family, 0), 1.2 + title_hits * 0.4)
    if has_non_ml_research_signal(text) and not has_ml_modeling_signal(text):
        if contains_term(text, "market research"):
            scores = {"market_research": max(scores.get("market_research", 0), 2.0)}
        elif any(contains_term(text, term) for term in ["product research", "concept studies", "concept testing"]):
            scores = {"product_research": max(scores.get("product_research", 0), 2.0)}
        else:
            scores = {"ux_research": max(scores.get("ux_research", 0), 2.0)}
    result = scores or {"other": 0.1}
    if cache_key is not None:
        PRECISE_ROLE_FAMILY_CACHE[cache_key] = dict(result)
    return result


def best_role_family_match(target_families: Dict[str, float], job_families: Dict[str, float]) -> Tuple[str, float]:
    if not target_families:
        if job_families:
            family = max(job_families, key=job_families.get)
            return family, 0.70
        return "general", 0.60

    overlap = set(target_families) & set(job_families)
    if overlap:
        family = max(overlap, key=lambda fam: target_families[fam] + job_families[fam])
        target_strength = min(1.0, target_families[family] / 1.4)
        job_strength = min(1.0, job_families[family] / 1.4)
        return family, clamp(0.72 + 0.18 * target_strength + 0.10 * job_strength)

    if not job_families:
        return "unclear", 0.45

    target_best = max(target_families, key=target_families.get)
    job_best = max(job_families, key=job_families.get)
    adjacent = {
        ("data_analytics", "bi_analytics"),
        ("data_analytics", "analytics_engineering"),
        ("data_analytics", "data_science"),
        ("bi_analytics", "analytics_engineering"),
        ("analytics_engineering", "data_engineering"),
        ("data_science", "ml_engineering"),
        ("ml_engineering", "ml_platform_mlops"),
        ("data_engineering", "ml_platform_mlops"),
    }
    if (target_best, job_best) in adjacent or (job_best, target_best) in adjacent:
        return f"{target_best}->{job_best}", 0.58
    return f"{target_best}!={job_best}", 0.22


def parse_excluded_role_families(dealbreakers) -> set:
    if isinstance(dealbreakers, dict) and dealbreakers.get("_excluded_role_families"):
        return set(dealbreakers.get("_excluded_role_families") or [])
    parts = dealbreakers if isinstance(dealbreakers, list) else [str(dealbreakers or "")]
    text = safe_lower(" ".join([str(x) for x in parts]))
    excluded = set()
    mapping = {
        "audit": ["no audit", "avoid audit"],
        "accounting": ["no accounting", "avoid accounting"],
        "sales": ["no pure sales", "no sales", "avoid sales"],
        "ml_engineering": ["no ml engineer", "no machine learning engineer"],
        "machine_learning": ["no ml engineer", "no machine learning engineer"],
        "backend_engineering": ["no backend", "no backend engineer", "avoid backend"],
        "software_engineering": ["no software engineer", "avoid software engineer"],
        "executive_management": ["no manager", "no director", "no executive", "no vp"],
    }
    for family, phrases in mapping.items():
        if any(phrase in text for phrase in phrases):
            excluded.add("ml_engineering" if family == "machine_learning" else family)
    return excluded


def role_has_risk_or_financial_modeling_signal(row: pd.Series | Dict) -> bool:
    text = safe_lower(f"{row.get('title', '')} {row.get('description', '')} {row.get('clean_job_text', '')} {row.get('skills', '')}")
    return any(term in text for term in ["risk", "credit", "fintech", "banking", "financial modeling", "sql", "python", "statistics", "model", "dashboard", "analytics", "data science"])


def compute_role_family_compatibility(profile: Dict, row: pd.Series | Dict) -> Tuple[float, str, str]:
    target_text = build_target_role_text(profile)
    target_families = profile.get("_target_role_families") or infer_target_role_families(target_text)
    job_families = infer_job_role_families(row, use_cache=False)
    family_label, base_score = best_role_family_match(target_families, job_families)
    job_best = max(job_families, key=job_families.get) if job_families else "other"
    excluded = set(profile.get("_excluded_role_families") or parse_excluded_role_families(profile.get("dealbreakers", [])))

    if job_best in excluded:
        target = safe_lower(profile.get("target_role"))
        if job_best == "ml_engineering" and "ml engineer" in target:
            pass
        else:
            return 0.0, f"Excluded role family: {FRIENDLY_ROLE_FAMILY_LABELS.get(job_best, job_best)}", family_label

    target_risk = any(fam in target_families for fam in ["risk_analytics", "credit_risk", "quantitative_analytics"])
    if target_risk:
        if job_best in {"marketing_analytics", "marketing", "audit", "accounting", "sales", "backend_engineering", "software_engineering", "executive_management"}:
            return 0.12, f"{FRIENDLY_ROLE_FAMILY_LABELS.get(job_best, job_best)} is far from risk/credit/quant target", family_label
        if job_best == "financial_analytics" and not role_has_risk_or_financial_modeling_signal(row):
            return 0.28, "Generic financial analyst role lacks risk, analytics, SQL, Python, or modeling signals", family_label
        if job_best == "ml_engineering" and "ml engineer" not in safe_lower(profile.get("target_role")):
            return 0.20, "Generic ML engineering role is not the requested risk/fintech data science target", family_label

    if "analytics_engineering" in target_families and job_best in {"ml_engineering", "backend_engineering", "software_engineering", "executive_management"}:
        if job_best == "ml_engineering" and "ml engineer" in safe_lower(profile.get("target_role")):
            return base_score, "ML engineering explicitly listed in target roles", family_label
        return 0.18, f"{FRIENDLY_ROLE_FAMILY_LABELS.get(job_best, job_best)} is outside analytics engineering target", family_label

    return base_score, f"Role family compatibility: {family_label}", family_label


def apply_role_mismatch_guardrail(profile: Dict, row: pd.Series | Dict, score: float) -> Tuple[float, str]:
    compatibility, note, _ = compute_role_family_compatibility(profile, row)
    if compatibility <= 0.0:
        return 0.05, note
    if compatibility < 0.30:
        return min(score, 0.22), note
    if compatibility < 0.50:
        return min(score, 0.45), note
    return score, note


def should_filter_excluded_role_family(profile: Dict, row: pd.Series | Dict) -> bool:
    excluded = set(profile.get("_excluded_role_families") or parse_excluded_role_families(profile.get("dealbreakers", [])))
    if not excluded:
        return False
    job_families = infer_job_role_families(row, use_cache=False)
    job_best = max(job_families, key=job_families.get) if job_families else "other"
    if job_best == "ml_engineering" and "ml engineer" in safe_lower(profile.get("target_role")):
        return False
    return job_best in excluded


def target_role_fit_score_fn(profile: Dict, row: pd.Series | Dict) -> Tuple[float, str]:
    target = safe_lower(profile.get("target_role"))
    if not target:
        return 0.70, "no specific target role provided"

    title = safe_lower(row.get("title"))
    text = safe_lower(f"{row.get('title', '')} {row.get('description', '')} {row.get('clean_job_text', '')}")
    phrases = split_target_phrases(target)
    target_families = infer_target_role_families(target)
    job_families = infer_job_role_families(row, use_cache=False)
    family_label, family_score = best_role_family_match(target_families, job_families)

    target_has_ml = any(term in target for term in ["machine learning", "ml engineer", "ml-focused", "applied scientist", "ai engineer", "research scientist"])
    target_has_ml_platform = any(term in target for term in ["ml platform", "mlops", "ml infrastructure", "machine learning infrastructure", "ml systems"])
    target_has_data_science = "data scientist" in target or "scientist" in target
    target_has_analytics = any(term in target for term in ["data analyst", "bi analyst", "business intelligence", "analytics engineer", "analytics analyst"])
    target_has_engineering = "analytics engineer" in target or "data engineer" in target
    target_has_analytics_engineering = is_analytics_engineering_profile(profile)
    non_ml_research = has_non_ml_research_signal(text) and not has_ml_modeling_signal(text)
    ml_related, ml_related_reason = ml_related_decision(profile, row)

    if executive_operations_title_mismatch(profile, row):
        return 0.16, "Title appears executive/operations-focused, which does not match the user's technical target roles"
    if profile_targets_technical_ic_roles(profile) and data_center_context_without_ml_evidence(row):
        return 0.18, "Data center context is not enough evidence for ML/data science target fit"

    if profile_requires_ml_related_roles(profile) and not ml_related:
        return 0.12, ml_related_reason

    title_ml = any(term in title for term in ["machine learning", " ml ", "ml engineer", "ai engineer", "applied scientist", "research scientist"])
    title_ml_platform = any(term in title for term in ["ml platform", "mlops", "machine learning ops", "machine learning infrastructure", "ml infrastructure", "ml systems"])
    title_ds = "data scientist" in title or "scientist" in title
    title_analytics = any(term in title for term in ["data analyst", "bi analyst", "business analyst", "business intelligence", "analytics", "analyst"])
    title_engineering = any(term in title for term in ["analytics engineer", "data engineer"])
    title_devops = any(term in title for term in ["devops", "sre", "site reliability", "cloud engineer", "platform engineer"])
    title_marketing = "marketing analyst" in title or "marketing" in title

    strong_ml_text = any(term in text for term in ["machine learning", "deep learning", "model training", "pytorch", "tensorflow", "scikit-learn", "ml model"])
    strong_ml_platform_text = any(
        term in text
        for term in [
            "ml platform", "mlops", "model serving", "model deployment", "distributed training",
            "training infrastructure", "inference infrastructure", "production ml", "feature store",
            "kubeflow", "ray", "vllm", "model monitoring", "machine learning infrastructure",
            "llm deployment", "inference stack",
        ]
    )
    strong_analytics_text = any(term in text for term in ["dashboard", "tableau", "business intelligence", "data analysis", "sql", "reporting", "analytics"])
    limited_production_ml = profile_lacks_production_ml_experience(profile)

    if non_ml_research:
        if target_has_ml:
            return 0.12, "UX/product/market research role lacks ML or modeling signals"
        if target_has_analytics:
            return 0.30, "research role lacks clear analytics or BI target fit"

    if any(phrase and phrase in title for phrase in phrases):
        if any(term in title for term in ["research scientist", "applied scientist", "researcher"]):
            if target_has_ml and not has_ml_modeling_signal(text):
                return 0.28, "research/scientist title lacks ML or modeling signals"
        return 1.00, "target role phrase appears in title"

    if limited_production_ml and target_has_ml and strong_production_ml_requirement(row):
        return 0.34, "role appears to require production ML ownership beyond the stated profile"

    if target_has_analytics_engineering:
        penalized_role = analytics_engineering_penalized_title(title, target)
        if analytics_engineering_preferred_title(title):
            return 1.00, "preferred analytics/BI/data-platform title matches target"
        if "data engineer" in title and any(term in text for term in ANALYTICS_ENGINEERING_PLATFORM_SIGNALS):
            return 0.88, "data engineering role has analytics platform or warehouse signals"
        if any(term in title for term in ["bi analyst", "business intelligence analyst", "data analyst"]):
            if any(term in text for term in ANALYTICS_ENGINEERING_PLATFORM_SIGNALS):
                return 0.74, "analytics/BI analyst role has platform or warehouse signals"
            return 0.56, "analyst title is adjacent but weaker than analytics engineering target"
        if penalized_role:
            return 0.18, f"{penalized_role} title is outside analytics engineering target unless explicitly listed"
        if any(term in text for term in ["analytics engineering", "data warehouse", "semantic layer", "dbt", "snowflake", "bigquery"]):
            return 0.72, "job text has analytics engineering or data warehouse signals"

    if target_has_ml_platform:
        if title_ml_platform:
            return 1.00, "ML platform/MLOps title matches target family"
        if (title_ml or title_devops or title_engineering) and strong_ml_platform_text:
            return 0.86, "job has ML platform or production ML infrastructure signals"
        if (title_ml or title_devops or title_engineering or "engineer" in title) and strong_ml_text:
            return 0.58, "job has ML/AI engineering signals but weaker platform evidence"
        if title_engineering and not strong_ml_platform_text:
            return 0.32, "data engineering role lacks ML platform signals"
        if title_devops and not strong_ml_platform_text:
            return 0.22, "generic DevOps role lacks ML platform signals"
        if title_ds and not strong_ml_platform_text:
            return 0.38, "data science role lacks ML platform signals"

    if target_has_ml and (title_ml or title_ds):
        if ("research" in title or "scientist" in title or "applied scientist" in title) and not has_ml_modeling_signal(text):
            return 0.34, "research/scientist title lacks ML or AI modeling signals"
        if title_ds and not has_ml_modeling_signal(text):
            return 0.26, "data scientist title lacks ML/modeling signals"
        return 0.95, "ML/data science title matches target family"
    if target_has_data_science and title_ds:
        return 0.90, "data scientist title matches target family"
    if target_has_analytics and (title_analytics or title_engineering):
        return 0.95, "analytics title matches target family"
    if target_has_engineering and title_engineering:
        return 0.90, "engineering analytics title matches target family"

    if target_has_ml and strong_ml_text and (title_ds or "engineer" in title or "scientist" in title):
        if "engineer" in title and not (title_ml or title_ds or title_ml_platform):
            return 0.34, "generic engineering title has only weak ML alignment"
        return 0.70, "job text has strong ML signals"
    if target_has_ml and (title_marketing or (title_analytics and not strong_ml_text)):
        return 0.26, "analytics or marketing role lacks ML focus"
    if target_has_ml and not (title_ml or title_ds or title_ml_platform):
        if any(term in title for term in ["database administrator", "automation engineer", "network engineer", "software engineer", "developer", "devops", "data engineer"]):
            return 0.24, "title is outside the ML/data science target family"
    if target_has_analytics and strong_analytics_text and "analyst" in title:
        return 0.75, "job text has analytics signals"
    if target_has_analytics and any(term in title for term in ["software engineer", "software development engineer", "developer", "backend", "frontend"]):
        if not any(term in title for term in ["analytics engineer", "data engineer"]):
            return 0.24, "software engineering title is outside analytics target"

    if any(token in title for token in ["developer", "software engineer", "java", "frontend", "backend", "full stack"]) and not (
        "analytics engineer" in target or "software engineer" in target or "developer" in target
    ):
        return min(0.20, family_score), "title appears outside target role family"

    if family_score >= 0.70:
        return family_score, f"role family match: {family_label}"
    if family_score <= 0.30:
        return family_score, f"role family mismatch: {family_label}"

    return max(0.45, family_score), "target role fit is weak but possible"


def should_filter_role_family_mismatch(profile: Dict, row: pd.Series | Dict) -> bool:
    score, _ = target_role_fit_score_fn(profile, row)
    title = safe_lower(row.get("title"))
    target = safe_lower(profile.get("target_role"))
    text = safe_lower(f"{row.get('title', '')} {row.get('description', '')} {row.get('clean_job_text', '')}")

    if profile_requires_ml_related_roles(profile):
        ml_related, _ = ml_related_decision(profile, row)
        if not ml_related:
            return True

    if has_non_ml_research_signal(text) and not has_ml_modeling_signal(text):
        if any(term in target for term in ["ml engineer", "machine learning", "ml-focused", "applied scientist", "research scientist", "ai engineer"]):
            return True
        if any(term in target for term in ["data analyst", "bi analyst", "analytics engineer"]) and not any(term in text for term in ["analytics", "business intelligence", "sql", "tableau", "dashboard"]):
            return True

    if any(term in target for term in ["ml platform", "mlops", "ml infrastructure", "machine learning infrastructure", "ml systems"]):
        if score < 0.50:
            return True

    if is_analytics_engineering_profile(profile):
        penalized_role = analytics_engineering_penalized_title(title, target)
        if penalized_role and score < 0.35:
            return True

    if any(term in target for term in ["ml engineer", "machine learning", "ml-focused", "applied scientist"]):
        if score < 0.35 and any(term in title for term in ["marketing analyst", "business analyst", "data analyst"]):
            return True
        if score < 0.35 and any(term in title for term in ["database administrator", "automation engineer", "network engineer", "software engineer", "developer", "devops", "data engineer"]):
            return True

    if score >= 0.40:
        return False

    if score < 0.28:
        return True

    # Keep broad searches broad, but remove clearly unrelated software/developer roles.
    if any(term in title for term in ["developer", "software engineer", "java", "frontend", "backend", "full stack"]):
        if not any(term in target for term in ["developer", "software engineer", "analytics engineer", "data engineer"]):
            return True
    if any(term in title for term in ["auditor", "tax", "accountant", "financial analyst"]):
        if not any(term in target for term in ["finance", "financial", "account", "audit", "tax"]):
            return True
    return False


def job_quality_score_fn(row: pd.Series | Dict) -> Tuple[float, str]:
    title = safe_lower(row.get("title"))
    company = safe_lower(row.get("company"))
    description = safe_lower(row.get("description"))
    text = safe_lower(f"{row.get('title', '')} {row.get('company', '')} {row.get('description', '')} {row.get('clean_job_text', '')}")

    score = 0.78
    notes = []

    if title and company and len(description) >= 250:
        score += 0.10
        notes.append("clear job posting")
    elif len(description) < 120:
        score -= 0.25
        notes.append("thin description")

    # Low direct-employer quality: wording markets job-search/training services to candidates
    # rather than describing one concrete role.
    service_patterns = [
        "job seeker",
        "jobseeker",
        "helped jobseekers",
        "we help candidates",
        "our candidates",
        "multiple job offers",
        "candidate marketing",
        "clients are looking",
        "career placement",
        "job placement",
        "placement program",
        "resume marketing",
        "interview preparation",
        "career support",
        "training program",
        "bootcamp",
        "leetcode",
        "get hired",
        "guaranteed interview",
        "career accelerator",
        "job-ready",
        "fresh grads get stuck",
        "ats filters",
    ]
    service_hits = [term for term in service_patterns if term in text]
    if service_hits:
        score -= min(0.70, 0.22 * len(service_hits))
        notes.append("placement/training-style wording")

    # Program postings are not always bad, but talent/trainee programs are weaker for direct job matching.
    program_terms = ["emerging talent program", "talent program", "graduate program", "trainee program", "career accelerator"]
    program_hits = [term for term in program_terms if term in text]
    has_concrete_role_detail = any(term in text for term in ["responsibilities", "qualifications", "requirements", "you will", "what you'll do", "in this role"])
    if program_hits:
        score -= 0.35
        notes.append("program-like posting")
        if not has_concrete_role_detail:
            score -= 0.15
            notes.append("limited role detail")

    # Company/name hints are kept generic; avoid company-specific blocklists.
    placement_company_terms = ["placement", "placements", "staffing", "recruiting", "talent program"]
    if any(term in company for term in placement_company_terms):
        score -= 0.18
        notes.append("placement/job-board style source")

    if any(term in text for term in ["apply now", "responsibilities", "qualifications", "requirements", "what you'll do"]):
        score += 0.04

    if "contract" in text and "full-time" not in text and "full time" not in text:
        score -= 0.08
        notes.append("contract-like wording")

    if any(term in title for term in ["urgent", "immediate hire"]) and len(description) < 250:
        score -= 0.10
        notes.append("thin urgent posting")

    return clamp(score), "; ".join(notes) if notes else "standard job posting quality"


def should_filter_low_quality_job(row: pd.Series | Dict) -> bool:
    score, note = job_quality_score_fn(row)
    if score < 0.55:
        return True
    if score <= 0.72 and "placement/training-style" in note:
        return True

    text = safe_lower(f"{row.get('title', '')} {row.get('company', '')} {row.get('description', '')} {row.get('clean_job_text', '')}")
    direct_service_signals = [
        "job seeker",
        "jobseeker",
        "helped jobseekers",
        "we help candidates",
        "our candidates",
        "multiple job offers",
        "candidate marketing",
        "clients are looking",
        "career placement",
        "job placement",
        "placement program",
        "resume marketing",
        "interview preparation",
        "career support",
        "training program",
        "bootcamp",
        "leetcode",
        "guaranteed interview",
        "career accelerator",
    ]
    has_direct_role_detail = any(term in text for term in ["responsibilities", "qualifications", "requirements", "what you'll do", "in this role", "you will"])
    direct_service_hit_count = sum(1 for term in direct_service_signals if term in text)
    if direct_service_hit_count >= 2:
        return True
    if direct_service_hit_count and not has_direct_role_detail:
        return True

    program_terms = ["emerging talent program", "talent program", "graduate program", "trainee program"]
    has_program = any(term in text for term in program_terms)
    if has_program and not has_direct_role_detail:
        return True

    # Generic placement program postings should not dominate Top-N recommendations.
    company = safe_lower(row.get("company"))
    placement_company_terms = ["placement", "placements", "staffing", "recruiting", "talent program"]
    if any(term in company for term in placement_company_terms):
        return True

    if has_program and any(term in company for term in ["placement", "placements", "jobcertify"]):
        return True

    return False


def should_filter_internship_for_profile(profile: Dict, row: pd.Series | Dict) -> bool:
    target_role = safe_lower(profile.get("target_role"))
    title = safe_lower(row.get("title"))
    description = safe_lower(row.get("description"))
    employment = safe_lower(row.get("employment_type"))

    explicit_internship = (
        "internship" in title
        or re.search(r"\bintern\b", title) is not None
        or re.search(r"\binternship\b", description) is not None
    )

    # Some API rows have noisy employment_type='internship' for non-intern jobs.
    # Do not filter on that field alone unless the text confirms it.
    if employment == "internship" and explicit_internship:
        explicit_internship = True

    if not explicit_internship:
        return False

    if "intern" in target_role or "internship" in target_role:
        return False

    return True


def experience_score_fn(profile: Dict, row: pd.Series) -> Tuple[float, str]:
    seniority = safe_lower(row.get("seniority"))
    title = safe_lower(row.get("title"))
    level = safe_lower(profile.get("candidate_level"))
    years = profile.get("years_experience")
    required_years = extract_required_years(row)

    try:
        years_val = None if years in [None, "", "unknown"] else float(years)
    except Exception:
        years_val = None

    seniority_level = job_seniority_level(row)
    senior_job = seniority_level in {"senior", "executive"}
    junior_job = seniority_level in {"intern", "entry"}

    if required_years is not None and years_val is not None:
        if required_years > years_val + 2:
            return 0.12, f"Requires about {required_years:g}+ years, above candidate experience"
        if required_years > years_val + 1:
            return 0.35, f"Experience requirement may be high ({required_years:g}+ years)"
        if required_years <= years_val + 1:
            return 0.95, "Experience requirement appears compatible"

    if level in {"entry_or_junior", "entry", "junior", "new grad"}:
        if senior_job:
            return 0.08, f"Seniority may be too high ({seniority_level})"
        if seniority_level == "mid":
            return 0.25, "Role may be too advanced for entry-level profile"
        if required_years is None:
            return 0.88, "Required years not listed; unknown, penalized slightly"
        return 1.00, "Seniority is compatible with entry-level profile"
    if level in {"senior_or_experienced", "senior"}:
        if junior_job:
            return 0.20, f"Role may be too junior for experienced candidate ({seniority_level})"
        if required_years is None:
            return 0.92, "Required years not listed; unknown, penalized slightly"
        return 1.00, "Seniority is compatible with experienced profile"
    if years_val is not None and years_val < 2 and senior_job:
        return 0.15, "Low experience estimate versus senior job"
    return 0.65, "Seniority is not explicit; kept as possible match"


def location_score_fn(profile: Dict, row: pd.Series) -> Tuple[float, str]:
    decision = location_constraint_decision(profile, row)
    if not str(row.get("location", "") or "").strip() and str(profile.get("location_preference", "") or "").strip():
        return clamp(min(decision["score"], 0.45)), "Location uncertainty: job location not listed."
    return clamp(decision["score"]), decision["reason"]


def feedback_job_key(row: pd.Series | Dict) -> str:
    return f"{str(row.get('title', '') or '')} | {str(row.get('company', '') or '')}".strip(" |")


def canonical_feedback_reason(reason: str | None) -> str:
    reason_l = safe_lower(reason)
    if not reason_l:
        return ""
    if "company is too small or uncertain" in reason_l:
        return "company too small or uncertain"
    if "visa sponsorship is unclear" in reason_l:
        return "visa sponsorship unclear"
    if "salary or location" in reason_l:
        return "salary or location does not work"
    if "work setup" in reason_l:
        return "work setup not acceptable"
    if "role is not a good fit" in reason_l:
        return "wrong role family"
    if "skills do not match" in reason_l:
        return "skills do not match"
    if "low-quality posting" in reason_l or "low quality posting" in reason_l:
        return "low-quality posting"
    if "company" in reason_l and "unknown" in reason_l:
        return "company too small or uncertain"
    if "small" in reason_l or "startup" in reason_l:
        return "company too small or uncertain"
    if "large employer" in reason_l or "research lab" in reason_l:
        return "company too small or uncertain"
    if "sponsor" in reason_l or "h-1b" in reason_l or "h1b" in reason_l or "visa" in reason_l:
        return "visa sponsorship unclear"
    if "salary" in reason_l:
        return "salary or location does not work"
    if "location" in reason_l:
        return "salary or location does not work"
    if "contract" in reason_l or "temp" in reason_l or "unwanted type" in reason_l:
        return "work setup not acceptable"
    if "too senior" in reason_l:
        return "too senior"
    if "too junior" in reason_l:
        return "too junior"
    if "skill" in reason_l:
        return "skills do not match"
    if "ml" in reason_l or "analytics relevance" in reason_l or "wrong role" in reason_l or "role family" in reason_l:
        return "not enough ML / analytics relevance"
    if "low-quality" in reason_l or "posting quality" in reason_l or "thin posting" in reason_l:
        return "low-quality posting"
    return reason_l


def feedback_learned_signal(reason: str) -> str:
    reason_l = canonical_feedback_reason(reason)
    if reason_l == "company too small or uncertain":
        return "Deprioritize company-size uncertainty; prioritize larger employers/research labs."
    if "visa sponsorship unclear" in reason_l:
        return "Deprioritize sponsor-unclear roles; prioritize known sponsors, research labs, and large employers."
    if reason_l == "salary or location does not work":
        return "Deprioritize salary/location mismatches."
    if reason_l == "work setup not acceptable":
        return "Deprioritize contract/temp or unwanted work setups."
    if reason_l == "low-quality posting":
        return "Deprioritize vague, placement, training, or low-quality postings."
    if reason_l == "wrong role family":
        return "Deprioritize role-family mismatches."
    if reason_l == "skills do not match":
        return "Deprioritize weak skill or ML/analytics relevance matches."
    if reason_l == "too senior":
        return "Deprioritize seniority or required-years mismatches."
    return "Use rejected role traits to improve ranking"


def feedback_final_score_adjustment(row: pd.Series | Dict, feedback: Dict) -> Tuple[float, float, str]:
    if not feedback:
        return 1.0, 0.0, ""

    multiplier = 1.0
    additive = 0.0
    notes: List[str] = []
    company = safe_lower(row.get("company"))
    job_key = safe_lower(feedback_job_key(row))
    rejected_jobs = [safe_lower(x) for x in feedback.get("rejected_jobs", [])]
    rejected_companies = [safe_lower(x) for x in feedback.get("rejected_companies", [])]
    reasons = [canonical_feedback_reason(x) for x in feedback.get("rejection_reasons", [])]

    bucket = safe_lower(row.get("company_size_bucket")) or "unknown"
    confidence = safe_lower(row.get("company_enrichment_confidence", row.get("company_size_confidence", "")))
    employee_count = str(row.get("employee_count_estimate", "") or "").strip()
    meets_100 = company_enrichment.bool_value(row.get("meets_100_employee_threshold", row.get("meets_min_company_size", "")))
    meets_500 = company_enrichment.bool_value(row.get("meets_500_employee_threshold", ""))
    is_large = company_enrichment.bool_value(row.get("is_large_company", ""))
    is_tiny = company_enrichment.bool_value(row.get("is_possible_tiny_startup", ""))
    is_sponsor = company_enrichment.bool_value(row.get("is_known_h1b_sponsor", ""))
    is_research = company_enrichment.bool_value(row.get("is_research_lab", ""))
    sponsor_signal = safe_lower(row.get("company_sponsor_signal", company_sponsor_signal(row)))
    try:
        job_quality_score = float(row.get("job_quality_score", 1.0) or 1.0)
    except Exception:
        job_quality_score = 1.0
    quality_text = safe_lower(f"{row.get('job_quality_reason', '')} {row.get('description', '')} {row.get('clean_job_text', '')}")
    try:
        sponsorship_score = float(row.get("sponsorship_score", 0.5) or 0.5)
    except Exception:
        sponsorship_score = 0.5

    if job_key and job_key in rejected_jobs:
        multiplier *= 0.35
        notes.append("exact rejected job penalty")
    elif company and company in rejected_companies:
        multiplier *= 0.72
        notes.append("same rejected company penalty")

    if any(reason == "company too small or uncertain" for reason in reasons):
        small_or_uncertain = (
            bucket == "unknown"
            or confidence == "low"
            or not employee_count
            or is_tiny
            or bucket in {"1-10", "11-50", "51-99"}
        )
        if small_or_uncertain:
            multiplier *= 0.60 if (is_tiny or bucket in {"1-10", "11-50", "51-99"}) else 0.70
            notes.append("company-size uncertainty/startup risk penalty")
        if not is_large and not is_research and not (meets_100 or meets_500):
            multiplier *= 0.82
            notes.append("lacks large-employer/research-lab signal")
        if meets_100 or meets_500 or is_large or is_research:
            additive += 0.055
            notes.append("larger employer/research-lab boost")

    if any("visa sponsorship unclear" in reason for reason in reasons):
        sponsor_unclear = sponsor_signal == "unknown" or sponsorship_score < 0.55 or not is_sponsor
        if sponsor_unclear and not (is_sponsor or is_research or is_large):
            multiplier *= 0.64
            notes.append("sponsorship uncertainty penalty")
        else:
            additive += 0.05
            notes.append("sponsor-like employer boost")

    if any(reason == "low-quality posting" for reason in reasons):
        quality_risk = (
            job_quality_score < 0.65
            or any(term in quality_text for term in ["placement", "training", "jobseeker", "vague", "thin", "staffing"])
        )
        if quality_risk:
            multiplier *= 0.72
            notes.append("low-quality posting penalty")

    return max(0.10, multiplier), min(0.12, additive), "; ".join(dict.fromkeys(notes))


def feedback_score_fn(row: pd.Series, feedback: Dict) -> Tuple[float, str]:
    if not feedback:
        return 0.50, "No feedback applied yet"

    text = safe_lower(f"{row.get('title')} {row.get('company')} {row.get('location')} {row.get('employment_type')} {row.get('seniority')} {row.get('skills')} {row.get('clean_job_text')}")
    title = safe_lower(row.get("title"))
    location = safe_lower(row.get("location"))
    employment = safe_lower(row.get("employment_type"))
    seniority = safe_lower(row.get("seniority"))
    company = safe_lower(row.get("company"))

    accepted = [safe_lower(x) for x in feedback.get("accepted_terms", [])]
    rejected = [safe_lower(x) for x in feedback.get("rejected_terms", [])]
    saved_companies = [safe_lower(x) for x in feedback.get("saved_companies", [])]
    rejected_companies = [safe_lower(x) for x in feedback.get("rejected_companies", [])]
    rejection_reasons = [canonical_feedback_reason(x) for x in feedback.get("rejection_reasons", [])]

    accepted_hits = sum(1 for x in accepted if x and x in text)
    rejected_hits = sum(1 for x in rejected if x and x in text)

    score = 0.50 + 0.08 * accepted_hits - 0.12 * rejected_hits

    if company and company in saved_companies:
        score += 0.10
    if company and company in rejected_companies:
        score -= 0.12

    if "salary or location does not work" in rejection_reasons:
        score -= 0.10 if location else 0.03
        annual_min, annual_max = annual_salary_range(row)
        if annual_min is None and annual_max is None:
            score -= 0.04
    if "too senior" in rejection_reasons and (seniority in {"senior", "executive"} or any(x in title for x in ["senior", "sr.", "staff", "principal", "lead"])):
        score -= 0.18
    if "too senior" in rejection_reasons:
        required_years = extract_required_years(row)
        if required_years is not None and required_years >= 3:
            score -= 0.12
    if "too junior" in rejection_reasons and (seniority in {"entry_or_junior", "internship"} or any(x in title for x in ["junior", "jr.", "intern"])):
        score -= 0.18
    if "skills do not match" in rejection_reasons:
        score -= 0.04 * rejected_hits
    if "work setup not acceptable" in rejection_reasons and any(x in employment + " " + text for x in ["contract", "temporary", "temp", "part-time"]):
        score -= 0.18
    if "low-quality posting" in rejection_reasons and any(x in text for x in ["placement", "training", "jobseeker", "vague", "thin", "staffing"]):
        score -= 0.10

    multiplier, additive, adjustment_note = feedback_final_score_adjustment(row, feedback)
    score = score * multiplier + additive
    if adjustment_note:
        return clamp(score), f"Feedback adjusted ranking: {adjustment_note}"

    note = "Feedback adjusted ranking"
    return clamp(score), note


def update_feedback_state(feedback: Dict | None, job_row: Dict, action: str, reason: str | None = None) -> Dict:
    feedback = dict(feedback or {})
    action_aliases = {"accept": "save", "reject": "not_interested", "skip": "apply_later"}
    action = action_aliases.get(action, action)
    feedback.setdefault("accepted_terms", [])
    feedback.setdefault("rejected_terms", [])
    feedback.setdefault("saved_companies", [])
    feedback.setdefault("apply_later_companies", [])
    feedback.setdefault("rejected_companies", [])
    feedback.setdefault("rejection_reasons", [])
    feedback.setdefault("saved_jobs", [])
    feedback.setdefault("apply_later_jobs", [])
    feedback.setdefault("rejected_jobs", [])
    feedback.setdefault("feedback_events", [])

    terms = extract_skills(" ".join([str(job_row.get("title", "")), str(job_row.get("skills", "")), str(job_row.get("clean_job_text", ""))]))[:8]
    company = str(job_row.get("company", "") or "")
    job_title = str(job_row.get("title", "") or "")
    job_key = f"{job_title} | {company}".strip(" |")

    if action == "save":
        feedback["accepted_terms"] = sorted(set(feedback["accepted_terms"] + terms))[:40]
        if company:
            feedback["saved_companies"] = sorted(set(feedback["saved_companies"] + [company]))[:25]
        if job_key:
            feedback["saved_jobs"] = sorted(set(feedback["saved_jobs"] + [job_key]))[:50]

    elif action == "apply_later":
        feedback["accepted_terms"] = sorted(set(feedback["accepted_terms"] + terms[:4]))[:40]
        if company:
            feedback["apply_later_companies"] = sorted(set(feedback["apply_later_companies"] + [company]))[:25]
        if job_key:
            feedback["apply_later_jobs"] = sorted(set(feedback["apply_later_jobs"] + [job_key]))[:50]

    elif action == "not_interested":
        canonical_reason = canonical_feedback_reason(reason)
        if canonical_reason in {"skills do not match", "not enough ML / analytics relevance"}:
            feedback["rejected_terms"] = sorted(set(feedback["rejected_terms"] + terms))[:40]
        if company:
            feedback["rejected_companies"] = sorted(set(feedback["rejected_companies"] + [company]))[:25]
        if job_key:
            feedback["rejected_jobs"] = sorted(set(feedback["rejected_jobs"] + [job_key]))[:50]
        if canonical_reason:
            feedback["rejection_reasons"] = sorted(set(feedback["rejection_reasons"] + [canonical_reason]))[:20]
        feedback["feedback_events"].append({
            "rejected_title": job_title,
            "rejected_company": company,
            "rejected_reason": reason or "",
            "learned_signal": feedback_learned_signal(canonical_reason),
            "before_rank": job_row.get("rank", ""),
            "before_score": job_row.get("final_score", ""),
            "after_rank": "",
            "after_score": "",
        })
        feedback["feedback_events"] = feedback["feedback_events"][-30:]

    return feedback


def simulate_feedback_learning(recommender: JobRecommender, profile: Dict, rounds: int = 3, top_n: int = 10) -> pd.DataFrame:
    feedback: Dict = {}
    rows = []
    target_terms = set(extract_skills(" ".join(profile.get("skills", [])) + " " + profile.get("target_role", "")))
    for round_id in range(rounds + 1):
        recs = recommender.recommend_from_profile(profile, top_n=top_n, candidate_k=250, feedback=feedback)
        if recs.empty:
            rows.append({"round": round_id, "avg_final_score": 0.0, "avg_skill_score": 0.0, "accepted_terms": 0, "rejected_terms": 0})
            continue
        rows.append({
            "round": round_id,
            "avg_final_score": round(float(recs["final_score"].mean()), 4),
            "avg_skill_score": round(float(recs["skill_score"].mean()), 4),
            "accepted_terms": len(feedback.get("accepted_terms", [])),
            "rejected_terms": len(feedback.get("rejected_terms", [])),
        })
        if round_id == rounds:
            break
        best = recs.iloc[0].to_dict()
        worst = recs.iloc[-1].to_dict()
        best_terms = set(extract_skills(str(best.get("clean_job_text", ""))))
        if target_terms and len(best_terms & target_terms) >= len(target_terms & set(extract_skills(str(worst.get("clean_job_text", ""))))):
            feedback = update_feedback_state(feedback, best, "accept")
        feedback = update_feedback_state(feedback, worst, "reject")
    return pd.DataFrame(rows)


def build_explanation(
    row: pd.Series,
    matched_skills: List[str],
    role_score: float,
    skill_score: float,
    exp_note: str,
    location_note: str,
    salary_note: str,
    source_score: float,
    feedback_note: str,
    job_quality_note: str = "",
    sponsorship_note: str = "",
    employment_note: str = "",
    dealbreaker_note: str = "",
) -> str:
    parts = []
    if role_score >= 0.70:
        parts.append("strong target-role alignment")
    elif role_score >= 0.45:
        parts.append("moderate target-role alignment")
    else:
        parts.append("semantic match from job description")
    if matched_skills:
        parts.append("matched skills: " + ", ".join(matched_skills[:6]))
    elif skill_score > 0:
        parts.append("job contains relevant technical keywords")
    if exp_note:
        parts.append(exp_note)
    if location_note:
        parts.append(location_note)
    if salary_note:
        parts.append(salary_note)
    if employment_note:
        parts.append(employment_note)
    if dealbreaker_note:
        parts.append(dealbreaker_note)
    if job_quality_note and any(term in job_quality_note.lower() for term in ["thin", "training", "placement", "contract"]):
        parts.append(job_quality_note)
    if sponsorship_note and "No sponsorship requirement" not in sponsorship_note:
        parts.append(sponsorship_note)
    if source_score >= 1:
        parts.append("recent JSearch posting")
    if feedback_note:
        parts.append(feedback_note)
    return "; ".join(parts) + "."


def looks_us_location(location: str) -> bool:
    if not location:
        return True
    if "remote" in location or "united states" in location or "usa" in location or " us" in f" {location}":
        return True
    state_pattern = r"\b(al|ak|az|ar|ca|co|ct|de|fl|ga|hi|ia|id|il|in|ks|ky|la|ma|md|me|mi|mn|mo|ms|mt|nc|nd|ne|nh|nj|nm|nv|ny|oh|ok|or|pa|ri|sc|sd|tn|tx|ut|va|vt|wa|wi|wv|wy|dc)\b"
    return bool(re.search(state_pattern, location.lower()))


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run JobPilot recommendations from a profile JSON file.")
    parser.add_argument("--profile", required=True, help="Path to profile JSON created by resume_parser.py")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=250)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    profile = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    rec = JobRecommender()
    result = rec.recommend_from_profile(profile, top_n=args.top_n, candidate_k=args.candidate_k)
    display_cols = ["rank", "final_score", "title", "company", "location", "seniority", "employment_type", "why_matched"]
    print(result[display_cols].to_string(index=False))
    if args.out:
        result.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"Saved recommendations: {args.out}")


if __name__ == "__main__":
    main()
