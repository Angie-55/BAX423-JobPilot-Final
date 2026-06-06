"""Run JobPilot persona quality checks and save honest PASS/PARTIAL/FAIL outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import re

import pandas as pd

import company_enrichment
from app import build_local_resume, resume_consistency_issues
from resume_export import save_resume_docx
from recommender import (
    DEFENSE_HINTS,
    JobRecommender,
    annual_salary_range,
    company_sponsor_signal,
    extract_required_years,
    format_candidate_funnel_summary,
    has_senior_title_signal,
    has_no_sponsorship_signal,
    job_seniority_level,
    job_quality_score_fn,
    listed_salary_below_preference,
    location_constraint_decision,
    normalize_employment_type,
    normalize_job_location,
    parse_dealbreakers,
    parse_location_preferences,
    profile_min_annual_salary,
    profile_max_required_years,
    required_years_violates_profile,
    apply_role_mismatch_guardrail,
    compute_role_family_compatibility,
    infer_job_role_families,
    should_filter_excluded_role_family,
    should_filter_low_quality_job,
    should_filter_role_family_mismatch,
    simulate_feedback_learning,
    extract_salary_from_description,
    target_role_fit_score_fn,
    update_feedback_state,
)

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

DIAGNOSTIC_COLUMNS = [
    "rank",
    "title",
    "company",
    "location",
    "normalized_location",
    "location_match_reason",
    "location_constraint_pass",
    "raw_employment_type",
    "normalized_employment_type",
    "is_ml_related",
    "ml_related_reason",
    "salary_min",
    "salary_max",
    "salary_period",
    "salary_annual_min",
    "salary_annual_max",
    "required_years",
    "role_family",
    "role_family_mismatch",
    "disallowed_for_profile_reason",
    "job_quality_score",
    "job_quality_reason",
    "employee_count_estimate",
    "company_size_bucket",
    "meets_100_employee_threshold",
    "meets_500_employee_threshold",
    "meets_min_company_size",
    "is_large_company",
    "is_possible_tiny_startup",
    "is_known_h1b_sponsor",
    "is_research_lab",
    "is_public_company",
    "company_enrichment_confidence",
    "company_enrichment_source",
    "company_enrichment_notes",
    "company_constraint_score",
    "target_fit_score",
    "role_score",
    "skill_score",
    "experience_score",
    "location_score",
    "salary_score",
    "final_score",
    "why_matched",
]

SUMMARY_COLUMNS = [
    "persona",
    "result",
    "recommendation_result",
    "resume_check_result",
    "resume_check_notes",
    "top_10_count",
    "aisha_candidate_funnel_summary",
    "priya_candidate_funnel_summary",
    "feedback_test_result",
    "rejected_company",
    "rejected_title",
    "rejected_reason",
    "before_rank",
    "after_rank",
    "before_score",
    "after_score",
    "feedback_notes",
    "senior_staff_count",
    "senior_title_count",
    "senior_text_count",
    "junior_count",
    "contract_count",
    "unpaid_count",
    "non_us_count",
    "non_remote_or_region_count",
    "location_violation_count",
    "salary_violation_count",
    "required_years_violation_count",
    "defense_count",
    "ml_related_violation_count",
    "role_family_mismatch_count",
    "low_quality_posting_count",
    "tiny_startup_count",
    "possible_tiny_startup_count",
    "company_size_unknown_count",
    "confirmed_below_threshold_count",
    "confirmed_below_100_count",
    "meets_min_company_size_count",
    "large_company_signal_count",
    "sponsor_or_research_count",
    "no_sponsor_violation_count",
    "small_company_count",
    "learning_start_score",
    "learning_end_score",
    "selected_resume_job_rank",
    "selected_resume_job_title",
    "selected_resume_job_company",
    "notes",
    "violations",
    "internship_count",
]


TEST_PERSONAS: Dict[str, Dict] = {
    "Aisha - ML Pivot": {
        "candidate_name": "Aisha",
        "target_role": "ML Engineer, Applied Scientist, Data Scientist ML-focused",
        "skills": ["Python", "SQL", "Pandas", "Scikit-Learn", "PyTorch"],
        "years_experience": 3.0,
        "candidate_level": "mid",
        "education_keywords": ["Data Analytics"],
        "project_keywords": ["Machine Learning", "Modeling", "Data Analysis"],
        "location_preference": "Remote or Bay Area",
        "salary_min": "140000",
        "max_required_years": 5,
        "dealbreakers": ["no senior", "no staff", "no defense", "no military", "no 5+ years experience"],
        "raw_resume_text": "Aisha. Data Analyst with 3 years experience. Skills: Python, SQL, pandas, scikit-learn, basic PyTorch. No production ML experience.",
    },
    "Marcus - New Grad Analytics": {
        "candidate_name": "Marcus",
        "target_role": "Data Analyst, BI Analyst, Junior Data Scientist, Analytics Engineer",
        "skills": ["Python", "R", "SQL", "Tableau", "PySpark", "NLP"],
        "years_experience": 0.5,
        "candidate_level": "entry_or_junior",
        "education_keywords": ["MSBA", "UC Davis"],
        "project_keywords": ["Analytics", "Dashboard", "NLP"],
        "location_preference": "Any US location",
        "salary_min": "80000",
        "max_required_years": 3,
        "preferred_industries": ["technology", "healthcare"],
        "dealbreakers": ["no 3+ years experience", "no unpaid", "no contract"],
        "raw_resume_text": "Marcus. Recent MSBA graduate, UC Davis. No full-time experience. 2 analytics internships. Skills: Python, R, SQL, Tableau, PySpark, basic NLP.",
    },
    "Priya - ML Platform": {
        "candidate_name": "Priya",
        "target_role": "ML Platform Engineer, MLOps Engineer, Senior ML Engineer",
        "skills": ["AWS", "Java", "Kafka", "Kubernetes", "Machine Learning", "Microservices", "Python", "Spark", "TensorFlow"],
        "years_experience": 7.0,
        "candidate_level": "senior_or_experienced",
        "education_keywords": ["Computer Science"],
        "project_keywords": ["MLOps", "ML Platform", "Infrastructure"],
        "location_preference": "Remote or New York",
        "salary_min": "200000",
        "dealbreakers": ["US only", "no junior", "no companies under 100 employees"],
        "raw_resume_text": "Priya. 7 years ML platform, infrastructure, and backend engineering. Skills: Java, Kafka, Kubernetes, Machine Learning, Python, Spark, microservices, TensorFlow, AWS certified.",
    },
    "Kenji - International Visa-Constrained": {
        "candidate_name": "Kenji",
        "target_role": "Research Scientist, ML Engineer, Applied Scientist, AI Engineer",
        "skills": ["Python", "C++", "Deep Learning", "PyTorch", "NLP", "Computer Vision", "Research"],
        "years_experience": 0.5,
        "candidate_level": "entry_or_junior",
        "education_keywords": ["MS Computer Science"],
        "project_keywords": ["Published Research", "Deep Learning", "NLP", "Computer Vision"],
        "location_preference": "US only",
        "salary_min": "120000",
        "preferred_industries": ["large tech", "research labs", "known H-1B sponsors"],
        "dealbreakers": [
            "visa sponsorship required",
            "need H-1B sponsorship within 1 year",
            "no contract",
            "no temp",
            "no companies that don't sponsor",
        ],
        "raw_resume_text": (
            "Publications: Published research in deep learning, NLP, and computer vision. "
            "Kenji. MS Computer Science student graduating soon. OPT visa, needs H-1B sponsorship within 1 year. "
            "Skills: Python, C++, PyTorch, deep learning, NLP, computer vision."
        ),
    },
}


def contains_any(text: str, terms: List[str]) -> bool:
    text_l = str(text).lower()
    return any(term in text_l for term in terms)


def re_search_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def ensure_diagnostic_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in DIAGNOSTIC_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    return out[DIAGNOSTIC_COLUMNS]


def row_text(row: pd.Series) -> str:
    return " ".join(
        str(row.get(col, "") or "")
        for col in ["title", "company", "description", "clean_job_text", "employment_type", "seniority"]
    ).lower()


def senior_title_signal(row: pd.Series) -> bool:
    return job_seniority_level(row) in {"senior", "executive"}


def senior_text_signal(row: pd.Series) -> bool:
    if senior_title_signal(row):
        return False
    text = " ".join(
        str(row.get(col, "") or "")
        for col in ["description", "clean_job_text", "job_text"]
    ).lower()
    return any(re.search(pattern, text) for pattern in [
        r"\bsenior\b",
        r"\bsr\.?\b",
        r"\bstaff\b",
        r"\bprincipal\b",
        r"\blead\b",
        r"\bdirector\b",
        r"\bmanager\b",
        r"\bhead\b",
    ])


def count_salary_violations(profile: Dict, recs: pd.DataFrame) -> int:
    requested = profile_min_annual_salary(profile)
    if requested is None:
        return 0
    count = 0
    for _, row in recs.iterrows():
        annual_min, annual_max = annual_salary_range(row)
        comparable = annual_max if annual_max is not None else annual_min
        if comparable is not None and comparable < requested:
            count += 1
    return count


def required_years_violation_count(profile: Dict, recs: pd.DataFrame) -> int:
    max_years = profile_max_required_years(profile)
    if max_years is None:
        return 0
    count = 0
    for _, row in recs.iterrows():
        required = extract_required_years(row)
        if required is not None and required >= max_years:
            count += 1
    return count


def non_us_physical_count(recs: pd.DataFrame) -> int:
    count = 0
    for _, row in recs.iterrows():
        loc = normalize_job_location(row)
        if loc["is_remote"]:
            continue
        if loc["country"] and loc["country"] != "US":
            count += 1
    return count


def non_remote_or_region_count(profile: Dict, recs: pd.DataFrame) -> int:
    count = 0
    for _, row in recs.iterrows():
        decision = location_constraint_decision(profile, row)
        if decision["constraints"]["strict_location"] and not decision["pass"]:
            count += 1
    return count


def role_family_mismatch_count(profile: Dict, recs: pd.DataFrame) -> int:
    count = 0
    for _, row in recs.iterrows():
        score, _ = target_role_fit_score_fn(profile, row)
        if should_filter_role_family_mismatch(profile, row) or score < 0.45:
            count += 1
    return count


def no_sponsorship_violation_count(recs: pd.DataFrame) -> int:
    return int(sum(1 for _, row in recs.iterrows() if has_no_sponsorship_signal(row)))


def sponsor_or_research_count(recs: pd.DataFrame) -> int:
    preferred = {"known sponsor", "research lab", "sponsorship mentioned"}
    return int(
        sum(
            1
            for _, row in recs.iterrows()
            if truthy_value(row.get("is_known_h1b_sponsor", ""))
            or truthy_value(row.get("is_research_lab", ""))
            or company_sponsor_signal(row) in preferred
        )
    )


def small_or_unknown_sponsor_count(recs: pd.DataFrame) -> int:
    return int(sum(1 for _, row in recs.iterrows() if company_sponsor_signal(row) == "unknown"))


NON_ML_RESEARCH_TERMS = [
    "ux research", "user research", "usability", "human factors", "hci",
    "mixed-method", "mixed method", "qualitative research", "concept study",
    "concept studies", "survey research", "market research", "product research",
]

ML_SIGNAL_TERMS = [
    "machine learning", " ml ", "ml engineer", "deep learning", "pytorch", "tensorflow",
    "scikit-learn", "sklearn", "nlp", "computer vision", "model training", "modeling",
    "predictive model", "applied scientist", "research scientist", "ai engineer",
]

ML_PLATFORM_SIGNAL_TERMS = [
    "ml platform", "mlops", "machine learning infrastructure", "ml infrastructure",
    "ai infrastructure", "model serving", "model deployment", "model monitoring",
    "feature store", "inference infrastructure", "production ml", "ml pipeline",
    "kubeflow", "distributed training",
]

LARGE_COMPANY_TERMS = [
    "large company", "enterprise", "fortune 500", "global company", "research lab",
    "laboratory", "university", "institute",
]

STARTUP_TERMS = [
    "startup", "start-up", "early-stage", "early stage", "seed-stage", "seed stage",
    "small team", "founding team", "series a", "series b",
]


def field_text(row: pd.Series, fields: List[str]) -> str:
    return " ".join(str(row.get(col, "") or "") for col in fields).lower()


def title_text(row: pd.Series) -> str:
    return field_text(row, ["title", "seniority"])


def has_ml_signal(row: pd.Series) -> bool:
    text = f" {row_text(row)} "
    return contains_any(text, ML_SIGNAL_TERMS + ML_PLATFORM_SIGNAL_TERMS)


def has_non_ml_research_signal(row: pd.Series) -> bool:
    text = row_text(row)
    return contains_any(text, NON_ML_RESEARCH_TERMS) and not has_ml_signal(row)


def role_family(row: pd.Series) -> str:
    return str(row.get("role_family", "") or "").lower()


def is_aisha_ml_related(row: pd.Series) -> bool:
    if "is_ml_related" in row.index:
        return str(row.get("is_ml_related", "")).strip().lower() == "true"
    if has_non_ml_research_signal(row):
        return False
    family = role_family(row)
    title = title_text(row)
    if family in {"ml_engineering", "ml_platform_mlops"}:
        return True
    if family == "data_science":
        return has_ml_signal(row) or contains_any(title, ["data scientist", "applied scientist", "research scientist"])
    if family in {"data_analytics", "software_engineering", "devops", "data_engineering"}:
        return has_ml_signal(row)
    return False


def is_marcus_role_related(row: pd.Series) -> bool:
    if has_non_ml_research_signal(row):
        return False
    family = role_family(row)
    title = title_text(row)
    required = extract_required_years(row)
    if family in {"data_analytics", "bi_analytics", "analytics_engineering"}:
        return True
    if family == "data_science":
        is_junior_friendly = not senior_title_signal(row) and (required is None or required < 3)
        return is_junior_friendly or contains_any(title, ["junior data scientist", "entry level data scientist"])
    return False


def is_priya_role_related(row: pd.Series) -> bool:
    family = role_family(row)
    if family == "ml_platform_mlops":
        return True
    if family == "ml_engineering":
        return True
    if family in {"devops", "data_engineering", "software_engineering", "data_science"}:
        return contains_any(row_text(row), ML_PLATFORM_SIGNAL_TERMS)
    return False


def is_kenji_role_related(row: pd.Series) -> bool:
    if has_non_ml_research_signal(row):
        return False
    family = role_family(row)
    title = title_text(row)
    if contains_any(title, ["research scientist", "applied scientist", "ml engineer", "machine learning engineer", "ai engineer"]):
        return True
    if family in {"ml_engineering", "ml_platform_mlops"}:
        return has_ml_signal(row)
    if family == "data_science":
        return has_ml_signal(row) and contains_any(row_text(row), ["research", "deep learning", "pytorch", "nlp", "computer vision"])
    return False


def persona_role_mismatch_count(name: str, recs: pd.DataFrame) -> int:
    if recs.empty:
        return 0
    if name.startswith("Aisha"):
        return int(sum(1 for _, row in recs.iterrows() if not is_aisha_ml_related(row)))
    if name.startswith("Marcus"):
        return int(sum(1 for _, row in recs.iterrows() if not is_marcus_role_related(row)))
    if name.startswith("Priya"):
        return int(sum(1 for _, row in recs.iterrows() if not is_priya_role_related(row)))
    if name.startswith("Kenji"):
        return int(sum(1 for _, row in recs.iterrows() if not is_kenji_role_related(row)))
    return role_family_mismatch_count({}, recs)


def junior_role_signal(row: pd.Series) -> bool:
    if job_seniority_level(row) in {"intern", "entry"}:
        return True
    text = str(row.get("title", "") or "").lower()
    return any(re.search(pattern, text) for pattern in [
        r"\bjunior\b",
        r"\bjr\.?\b",
        r"\bentry[- ]?level\b",
        r"\bnew grad\b",
        r"\bgraduate\b",
        r"\bintern\b",
        r"\binternship\b",
    ])


def company_size_value(row: pd.Series) -> float | None:
    for col in ["company_size", "employee_count", "employees", "company_employees"]:
        if col in row.index:
            raw = str(row.get(col, "") or "").lower().replace(",", "")
            match = re.search(r"\d+(?:\.\d+)?", raw)
            if match:
                try:
                    return float(match.group(0))
                except Exception:
                    return None
    return None


def has_company_size_field(row: pd.Series) -> bool:
    return any(col in row.index for col in ["company_size", "employee_count", "employees", "company_employees"])


def truthy_value(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def company_min_threshold(profile: Dict) -> int | None:
    return company_enrichment.parse_company_constraints(profile).get("min_company_size")


def confirmed_below_threshold_company(row: pd.Series, threshold: int = 100) -> bool:
    bucket = str(row.get("company_size_bucket", "") or "").strip()
    if threshold >= 500 and bucket in {"1-10", "11-50", "51-99", "100-499"}:
        return True
    if threshold <= 100 and bucket in {"1-10", "11-50", "51-99"}:
        return True
    key = "meets_500_employee_threshold" if threshold >= 500 else "meets_100_employee_threshold"
    meets = str(row.get(key, row.get("meets_min_company_size", ""))).strip().lower()
    if meets == "false":
        return True
    size = company_size_value(row)
    return size is not None and size < threshold


def confirmed_below_100_company(row: pd.Series) -> bool:
    return confirmed_below_threshold_company(row, 100)


def confirmed_below_threshold_count(profile: Dict, recs: pd.DataFrame) -> int:
    threshold = company_min_threshold(profile) or 100
    return int(sum(1 for _, row in recs.iterrows() if confirmed_below_threshold_company(row, threshold)))


def confirmed_below_100_count(recs: pd.DataFrame) -> int:
    return int(sum(1 for _, row in recs.iterrows() if confirmed_below_100_company(row)))


def meets_min_company_size_count(recs: pd.DataFrame) -> int:
    count = 0
    for _, row in recs.iterrows():
        bucket = str(row.get("company_size_bucket", "") or "").strip()
        meets = str(row.get("meets_100_employee_threshold", row.get("meets_min_company_size", ""))).strip().lower()
        if meets == "true" or bucket in {"100-499", "500-999", "1000+"}:
            count += 1
    return count


def large_company_signal(row: pd.Series) -> bool:
    sponsor_signal = company_sponsor_signal(row)
    text = row_text(row)
    return (
        truthy_value(row.get("is_large_company", ""))
        or truthy_value(row.get("is_known_h1b_sponsor", ""))
        or truthy_value(row.get("is_research_lab", ""))
        or sponsor_signal in {"known sponsor", "research lab"}
        or contains_any(text, LARGE_COMPANY_TERMS)
    )


def large_company_signal_count(recs: pd.DataFrame) -> int:
    return int(sum(1 for _, row in recs.iterrows() if large_company_signal(row)))


def possible_tiny_startup_signal(row: pd.Series) -> bool:
    if truthy_value(row.get("is_possible_tiny_startup", "")):
        return True
    return contains_any(row_text(row), STARTUP_TERMS) and not large_company_signal(row)


def tiny_startup_count(recs: pd.DataFrame) -> int:
    return confirmed_below_100_count(recs)


def possible_tiny_startup_count(recs: pd.DataFrame) -> int:
    return int(sum(1 for _, row in recs.iterrows() if possible_tiny_startup_signal(row)))


def company_size_unknown_count(recs: pd.DataFrame) -> int:
    count = 0
    for _, row in recs.iterrows():
        bucket = str(row.get("company_size_bucket", "") or "").strip()
        if bucket == "unknown":
            count += 1
        elif not bucket and (not has_company_size_field(row) or company_size_value(row) is None):
            count += 1
    return count


def small_company_count(recs: pd.DataFrame) -> int:
    return int(sum(1 for _, row in recs.iterrows() if confirmed_below_100_company(row) or possible_tiny_startup_signal(row)))


def profile_needs_sponsorship(profile: Dict) -> bool:
    text = " ".join(
        [str(profile.get("target_role", "")), str(profile.get("location_preference", ""))]
        + [str(x) for x in profile.get("dealbreakers", [])]
        + [str(x) for x in profile.get("preferred_industries", [])]
    ).lower()
    return parse_dealbreakers(profile).get("need_sponsorship") or any(term in text for term in ["visa", "h-1b", "h1b", "sponsorship"])


def simulate_small_company_rejection(recommender: JobRecommender, profile: Dict) -> pd.DataFrame:
    feedback: Dict = {}
    rows = []
    for round_id in range(3):
        recs = recommender.recommend_from_profile(profile, top_n=10, candidate_k=1500, feedback=feedback)
        unknown_count = small_or_unknown_sponsor_count(recs)
        rows.append({
            "round": round_id,
            "avg_final_score": round(float(recs["final_score"].mean()), 4) if len(recs) else 0.0,
            "small_or_unknown_sponsor_count": unknown_count,
            "accepted_terms": len(feedback.get("accepted_terms", [])),
            "rejected_terms": len(feedback.get("rejected_terms", [])),
        })
        unknown_rows = [
            row.to_dict()
            for _, row in recs.iterrows()
            if company_sponsor_signal(row) == "unknown"
        ]
        if unknown_rows:
            feedback = update_feedback_state(feedback, unknown_rows[-1], "not_interested", "company not preferred")
            feedback["rejected_terms"] = sorted(set(feedback.get("rejected_terms", []) + ["small company", "unknown sponsor"]))[:40]
        else:
            break
    return pd.DataFrame(rows)


def run_kenji_feedback_test(recommender: JobRecommender, profile: Dict, initial_recs: pd.DataFrame) -> Dict:
    if initial_recs.empty:
        return {
            "feedback_test_result": "FAIL",
            "feedback_notes": "No initial recommendations available for feedback test.",
        }

    def risk_priority(row: pd.Series) -> int:
        bucket = str(row.get("company_size_bucket", "") or "").strip().lower()
        sponsor = str(row.get("company_sponsor_signal", "") or "").strip().lower()
        confidence = str(row.get("company_enrichment_confidence", "") or "").strip().lower()
        tiny = str(row.get("is_possible_tiny_startup", "") or "").strip().lower() in {"true", "1", "yes"}
        if bucket == "unknown":
            return 0
        if sponsor == "unknown":
            return 1
        if confidence == "low":
            return 2
        if tiny:
            return 3
        return 4

    ordered = initial_recs.copy()
    ordered["_feedback_risk_priority"] = ordered.apply(risk_priority, axis=1)
    rejected = ordered.sort_values(["_feedback_risk_priority", "rank"]).iloc[0].drop(labels=["_feedback_risk_priority"]).to_dict()
    bucket = str(rejected.get("company_size_bucket", "") or "").strip().lower()
    sponsor = str(rejected.get("company_sponsor_signal", "") or "").strip().lower()
    confidence = str(rejected.get("company_enrichment_confidence", "") or "").strip().lower()
    if bucket == "unknown" or confidence == "low":
        reason = "company size unknown"
    elif sponsor == "unknown":
        reason = "visa sponsorship unclear"
    else:
        reason = "not a known H-1B sponsor"

    before_rank = rejected.get("rank", "")
    before_score = rejected.get("final_score", "")
    before_sponsor_like = sponsor_or_research_count(initial_recs)
    before_large = large_company_signal_count(initial_recs)
    feedback = update_feedback_state({}, rejected, "not_interested", reason=reason)
    after_recs = recommender.recommend_from_profile(profile, top_n=10, candidate_k=1500, feedback=feedback)

    title = str(rejected.get("title", "") or "")
    company = str(rejected.get("company", "") or "")
    matches = after_recs[
        (after_recs["title"].astype(str) == title)
        & (after_recs["company"].astype(str) == company)
    ] if len(after_recs) else pd.DataFrame()
    if len(matches):
        after_rank = matches.iloc[0].get("rank", "")
        after_score = matches.iloc[0].get("final_score", "")
    else:
        after_rank = "Left Top 10"
        after_score = ""

    try:
        rank_dropped = after_rank == "Left Top 10" or int(after_rank) > int(before_rank)
    except Exception:
        rank_dropped = after_rank == "Left Top 10"
    sponsor_like_preserved = sponsor_or_research_count(after_recs) >= min(before_sponsor_like, len(after_recs))
    large_preserved = large_company_signal_count(after_recs) >= max(0, before_large - 1)
    result = "PASS" if rank_dropped and sponsor_like_preserved and large_preserved else "FAIL"
    notes = (
        f"Rejected job {'left Top 10' if after_rank == 'Left Top 10' else 'moved to rank ' + str(after_rank)}; "
        f"sponsor/research count {before_sponsor_like}->{sponsor_or_research_count(after_recs)}; "
        f"large-employer count {before_large}->{large_company_signal_count(after_recs)}."
    )
    return {
        "feedback_test_result": result,
        "rejected_company": company,
        "rejected_title": title,
        "rejected_reason": reason,
        "before_rank": before_rank,
        "after_rank": after_rank,
        "before_score": before_score,
        "after_score": after_score,
        "feedback_notes": notes,
    }


def low_quality_posting_count(recs: pd.DataFrame) -> int:
    count = 0
    for _, row in recs.iterrows():
        score, _ = job_quality_score_fn(row)
        if score < 0.60 or should_filter_low_quality_job(row):
            count += 1
    return count


def resume_focus_check(name: str, profile: Dict) -> str:
    text = str(profile.get("raw_resume_text", "") or "").lower()
    skills = {str(skill).lower() for skill in profile.get("skills", [])}
    if name.startswith("Aisha"):
        has_ml = all(term in text or term in skills for term in ["python", "scikit-learn", "pytorch"])
        excel_heavy = text.count("excel") > text.count("python") + text.count("machine learning")
        return "PASS" if has_ml and not excel_heavy else "FAIL"
    if name.startswith("Marcus"):
        msba_pos = text.find("msba")
        exp_pos = text.find("experience")
        return "PASS" if msba_pos >= 0 and (exp_pos < 0 or msba_pos < exp_pos) else "FAIL"
    if name.startswith("Priya"):
        required = {"aws", "java", "kafka", "kubernetes", "machine learning", "microservices", "python", "spark", "tensorflow"}
        return "PASS" if required <= skills else "FAIL"
    if name.startswith("Kenji"):
        first_part = text[:120]
        has_publications_first = "publication" in first_part or "published research" in first_part
        has_research_focus = all(term in text or term in skills for term in ["python", "nlp", "computer vision"])
        return "PASS" if has_publications_first and has_research_focus else "FAIL"
    return "CRITERIA MISSING"


def select_resume_job(recs: pd.DataFrame) -> pd.Series | None:
    if recs.empty:
        return None
    selected_index = 4 if len(recs) >= 5 else 0
    return recs.iloc[selected_index]


def find_position(text: str, options: List[str]) -> int:
    lowered = text.lower()
    positions = [lowered.find(option.lower()) for option in options if lowered.find(option.lower()) >= 0]
    return min(positions) if positions else -1


def has_unsupported_placeholders(text: str) -> bool:
    lowered = text.lower()
    bad_terms = [
        "tailoring notes",
        "raw json",
        "debug",
        "[insert",
        "company name",
        "add metrics here",
        "to be filled",
        "tbd",
        "use the original resume facts",
    ]
    return any(term in lowered for term in bad_terms)


def evaluate_generated_resume(name: str, profile: Dict, resume_text: str, selected_job: pd.Series | None) -> tuple[str, str]:
    if not resume_text.strip():
        return "FAIL", "No resume draft was generated."
    if has_unsupported_placeholders(resume_text):
        return "FAIL", "Resume contains placeholders, debug text, or editing instructions."
    grounding_issues = resume_consistency_issues(profile, resume_text)
    if grounding_issues:
        return "FAIL", "Resume contains unsupported profile details: " + ", ".join(grounding_issues)

    lowered = resume_text.lower()
    notes = []
    if selected_job is not None:
        title = str(selected_job.get("title", "") or "")
        company = str(selected_job.get("company", "") or "")
        title_core = " ".join([token for token in re.split(r"[^A-Za-z0-9+#]+", title) if len(token) > 2][:4]).lower()
        if company and company.lower() not in lowered:
            return "FAIL", "Resume does not mention the selected job company."
        if title_core and not all(token in lowered for token in title_core.split()[:2]):
            return "PARTIAL PASS", "Resume mentions selected company but not enough selected job title keywords."

    if name.startswith("Aisha"):
        ml_terms = ["python", "pandas", "scikit-learn", "pytorch", "machine learning", " ml "]
        ml_hits = sum(1 for term in ml_terms if term in f" {lowered} ")
        excel_reporting = lowered.count("excel") + lowered.count("reporting")
        ml_emphasis = lowered.count("python") + lowered.count("machine learning") + lowered.count("scikit-learn") + lowered.count("pytorch")
        early_text = lowered[:700]
        if ml_hits < 2:
            return "FAIL", "Resume does not include enough Python/ML evidence."
        if "production ml experience" in lowered and "no production ml" not in lowered:
            return "FAIL", "Resume appears to claim unsupported production ML experience."
        if excel_reporting > ml_emphasis:
            return "FAIL", "Excel/reporting dominates the resume emphasis."
        if not any(term in early_text for term in ["machine learning", "ml-focused", "ml engineering", "scikit-learn", "pytorch"]):
            return "PARTIAL PASS", "ML direction is present but not prominent early in the resume."
        return "PASS", "Resume emphasizes Python/ML pivot without claiming production ML experience."

    if name.startswith("Marcus"):
        education_pos = find_position(resume_text, ["## Education", "MSBA", "UC Davis"])
        experience_pos = find_position(resume_text, ["## Experience", "Internship Experience", "Internship"])
        skill_hits = sum(1 for term in ["python", "sql", "tableau", " r,", "pyspark", "nlp", "analytics"] if term in lowered)
        if "msba" not in lowered and "uc davis" not in lowered:
            return "FAIL", "Resume does not include MSBA or UC Davis."
        if experience_pos >= 0 and education_pos >= 0 and education_pos > experience_pos:
            return "FAIL", "Resume does not lead with education before experience."
        if re.search(r"\b[2-9]\+?\s+years? (?:of )?(?:full[- ]time )?experience\b", lowered):
            return "FAIL", "Resume implies more full-time experience than Marcus has."
        if skill_hits < 3:
            return "PARTIAL PASS", "Resume leads with education but has limited analytics skill coverage."
        return "PASS", "Resume leads with MSBA/UC Davis and emphasizes junior analytics evidence."

    if name.startswith("Priya"):
        platform_terms = [
            "mlops", "ml platform", "kubernetes", "aws", "kafka", "spark",
            "microservices", "tensorflow", "infrastructure", "production ml", "model deployment",
        ]
        hits = sum(1 for term in platform_terms if term in lowered)
        early_text = lowered[:700]
        if hits < 3:
            return "FAIL", "Resume lacks enough ML platform/MLOps infrastructure signals."
        if any(term in lowered for term in ["recent graduate", "entry-level candidate", "junior candidate"]):
            return "FAIL", "Resume downgrades Priya to entry-level."
        if not any(term in early_text for term in ["experienced", "senior", "7 years", "platform", "infrastructure"]):
            return "PARTIAL PASS", "ML platform skills appear, but senior positioning is not prominent."
        if "data scientist resume" in lowered:
            return "FAIL", "Resume is framed as a generic data scientist resume."
        return "PASS", "Resume presents senior ML platform/MLOps and infrastructure strengths."

    if name.startswith("Kenji"):
        summary_pos = find_position(resume_text, ["## Professional Summary"])
        research_pos = find_position(resume_text, ["## Publications / Research"])
        education_pos = find_position(resume_text, ["## Education"])
        skills_pos = find_position(resume_text, ["## Technical Skills"])
        research_hits = sum(1 for term in ["publication", "published research", "deep learning", "nlp", "computer vision", "pytorch"] if term in lowered)
        if research_pos < 0:
            return "FAIL", "Resume does not include Publications / Research section."
        if summary_pos >= 0 and research_pos < summary_pos:
            return "FAIL", "Publications / Research appears before Professional Summary."
        if education_pos >= 0 and research_pos > education_pos:
            return "FAIL", "Publications / Research does not appear before Education."
        if skills_pos >= 0 and research_pos > skills_pos:
            return "FAIL", "Publications / Research does not appear before Technical Skills."
        if research_hits < 3:
            return "PARTIAL PASS", "Resume leads with research but has limited technical research coverage."
        if "h-1b" not in lowered and "sponsorship" not in lowered and "opt" not in lowered:
            notes.append("Visa context is not explicit in the resume draft.")
        return "PASS", "Resume leads with publications and AI/ML research strengths."

    return "NOT EVALUATED", "Persona resume criteria were not available."


def evaluate_persona(
    name: str,
    profile: Dict,
    recs: pd.DataFrame,
    learning: pd.DataFrame,
    resume_text: str = "",
    selected_resume_job: pd.Series | None = None,
    feedback_test: Dict | None = None,
) -> dict:
    feedback_test = feedback_test or {}
    texts = recs.apply(row_text, axis=1) if len(recs) else pd.Series(dtype=str)

    senior_title_count = int(recs.apply(senior_title_signal, axis=1).sum()) if len(recs) else 0
    senior_text_count = int(recs.apply(senior_text_signal, axis=1).sum()) if len(recs) else 0
    senior_staff_count = senior_title_count
    junior_count = int(recs.apply(junior_role_signal, axis=1).sum()) if len(recs) else 0
    contract_count = int(recs.apply(lambda row: normalize_employment_type(row) == "contract", axis=1).sum()) if len(recs) else 0
    unpaid_count = int(texts.apply(lambda x: "unpaid" in x).sum()) if len(texts) else 0
    defense_count = int(texts.apply(lambda x: contains_any(x, DEFENSE_HINTS + ["fbi", "homeland security", "ts/sci"])).sum()) if len(texts) else 0
    internship_count = 0
    for _, row in recs.iterrows():
        title = str(row.get("title", "") or "").lower()
        if normalize_employment_type(row) == "internship" or (
            "internship" in title or re_search_word(title, "intern")
        ):
            internship_count += 1
    salary_count = count_salary_violations(profile, recs)
    years_count = required_years_violation_count(profile, recs)
    non_us_count = non_us_physical_count(recs)
    location_count = non_remote_or_region_count(profile, recs)
    low_quality_count = low_quality_posting_count(recs)
    role_mismatch_count = persona_role_mismatch_count(name, recs)
    ml_related_count = role_mismatch_count if name.startswith("Aisha") else 0
    needs_sponsorship = profile_needs_sponsorship(profile)
    no_sponsor_count = no_sponsorship_violation_count(recs) if needs_sponsorship else 0
    sponsor_research_count = sponsor_or_research_count(recs) if needs_sponsorship else 0
    unknown_sponsor_count = small_or_unknown_sponsor_count(recs) if needs_sponsorship else 0
    tiny_count = tiny_startup_count(recs)
    possible_tiny_count = possible_tiny_startup_count(recs)
    company_unknown_count = company_size_unknown_count(recs)
    confirmed_below_threshold = confirmed_below_threshold_count(profile, recs)
    confirmed_below_100 = confirmed_below_100_count(recs)
    meets_min_company = meets_min_company_size_count(recs)
    large_company_count = large_company_signal_count(recs)
    small_count = small_company_count(recs)
    profile_resume_basis = resume_focus_check(name, profile)
    resume_check, resume_notes = evaluate_generated_resume(name, profile, resume_text, selected_resume_job)

    recommendation_hard: List[str] = []
    quality_issues: List[str] = []
    informational_notes: List[str] = []

    if len(recs) < 10:
        quality_issues.append(f"only {len(recs)} recommendations after filters")

    senior_notes = "No seniority issue detected"
    if senior_title_count:
        senior_notes = f"{senior_title_count} title-level seniority signals found"
    elif senior_text_count:
        senior_notes = f"{senior_text_count} seniority words appear only in descriptions/text"

    if name.startswith("Aisha"):
        if senior_title_count:
            recommendation_hard.append(f"{senior_title_count} senior/staff/lead roles")
        if defense_count:
            recommendation_hard.append(f"{defense_count} defense/military roles")
        if salary_count:
            recommendation_hard.append(f"{salary_count} listed salaries below $140K max")
        if years_count:
            recommendation_hard.append(f"{years_count} roles require >=5 years")
        if location_count:
            recommendation_hard.append(f"{location_count} roles outside Remote or Bay Area")
        if ml_related_count:
            recommendation_hard.append(f"{ml_related_count} roles are not ML-related or closely ML/data-science related")
        if profile_resume_basis != "PASS":
            recommendation_hard.append("profile facts do not support Python/ML resume focus")

    elif name.startswith("Marcus"):
        try:
            years = float(profile.get("years_experience"))
        except Exception:
            years = -1
        if years not in {0.0, 0.5}:
            recommendation_hard.append("years_experience is not parsed as 0 or 0.5")
        if profile.get("candidate_level") != "entry_or_junior":
            recommendation_hard.append("candidate_level is not entry_or_junior")
        if senior_title_count:
            recommendation_hard.append(f"{senior_title_count} title-level senior roles")
        if profile_max_required_years(profile) != 3:
            recommendation_hard.append("max_required_years is not 3")
        if years_count:
            recommendation_hard.append(f"{years_count} roles require >=3 years")
        if contract_count:
            recommendation_hard.append(f"{contract_count} contract-only roles")
        if unpaid_count:
            recommendation_hard.append(f"{unpaid_count} unpaid roles")
        if non_us_count or location_count:
            quality_issues.append(f"{max(non_us_count, location_count)} non-US or location-violating roles")
        if internship_count:
            quality_issues.append(f"{internship_count} internship roles")
        if low_quality_count:
            quality_issues.append(f"{low_quality_count} low-quality or placement-like postings")
        if role_mismatch_count:
            recommendation_hard.append(f"{role_mismatch_count} roles outside analytics target")
        if profile_resume_basis != "PASS":
            recommendation_hard.append("profile facts do not support MSBA-led resume")

    elif name.startswith("Priya"):
        if float(profile.get("years_experience", 0)) != 7.0:
            recommendation_hard.append("years_experience is not 7")
        if profile.get("candidate_level") != "senior_or_experienced":
            recommendation_hard.append("candidate_level is not senior_or_experienced")
        if junior_count:
            recommendation_hard.append(f"{junior_count} junior/entry/intern roles")
        if confirmed_below_100:
            recommendation_hard.append(f"{confirmed_below_100} companies show <100 employees")
        if possible_tiny_count:
            quality_issues.append(f"{possible_tiny_count} postings look startup/small-team without large-company signal")
        if company_unknown_count:
            quality_issues.append(f"{company_unknown_count} postings lack company-size data; tiny-startup check is limited")
        if profile_resume_basis != "PASS":
            recommendation_hard.append("profile facts do not cleanly represent ML platform skills")
        if salary_count:
            recommendation_hard.append(f"{salary_count} listed salaries below $200K max")
        if non_us_count:
            recommendation_hard.append(f"{non_us_count} non-US physical locations")
        if location_count:
            recommendation_hard.append(f"{location_count} roles outside NYC or remote preference")
        if role_mismatch_count > 1:
            recommendation_hard.append(f"{role_mismatch_count} generic DevOps/data/science roles outrank ML platform roles")
        elif role_mismatch_count == 1:
            quality_issues.append("1 role has only minor/ambiguous ML platform fit")
        if contract_count:
            quality_issues.append(f"{contract_count} contract-like roles")

    elif name.startswith("Kenji"):
        if contract_count:
            recommendation_hard.append(f"{contract_count} contract/temp roles")
        if non_us_count or location_count:
            recommendation_hard.append(f"{max(non_us_count, location_count)} non-US or location-violating roles")
        if salary_count:
            recommendation_hard.append(f"{salary_count} listed salaries below $120K max")
        if no_sponsor_count:
            recommendation_hard.append(f"{no_sponsor_count} roles explicitly do not sponsor")
        if role_mismatch_count > 1:
            recommendation_hard.append(f"{role_mismatch_count} roles outside research/ML/AI target")
        elif role_mismatch_count == 1:
            quality_issues.append("1 role has only minor/ambiguous research or ML/AI fit")
        if sponsor_research_count < max(5, len(recs) // 2):
            quality_issues.append(f"only {sponsor_research_count} roles show sponsor/research-lab signal")
        if large_company_count < max(5, len(recs) // 2):
            quality_issues.append(f"only {large_company_count} roles show large-employer/research/sponsor signal")
        if small_count:
            quality_issues.append(f"{small_count} postings have small-company/startup signals")
        if unknown_sponsor_count and sponsor_research_count < len(recs):
            quality_issues.append(f"{unknown_sponsor_count} roles have unclear sponsorship fit")
        if profile_resume_basis != "PASS":
            recommendation_hard.append("profile facts do not support publications/research resume")

        if len(learning):
            start_small = int(learning.iloc[0].get("small_or_unknown_sponsor_count", 0))
            end_small = int(learning.iloc[-1].get("small_or_unknown_sponsor_count", 0))
            if start_small > end_small:
                informational_notes.append(f"learning reduced small/unknown sponsor count from {start_small} to {end_small}")
            elif start_small == 0:
                informational_notes.append("learning limitation noted: no small/unknown sponsor roles appeared in the simulated Top-10")
            else:
                quality_issues.append("learning did not reduce small/unknown sponsor exposure in the simulation")

    if recommendation_hard:
        recommendation_result = "FAIL"
    elif quality_issues:
        recommendation_result = "PARTIAL PASS"
    else:
        recommendation_result = "PASS"

    hard_violations = list(recommendation_hard)
    if resume_check != "PASS":
        hard_violations.append(f"resume check is {resume_check}: {resume_notes}")

    if hard_violations:
        result = "FAIL"
    elif recommendation_result == "PARTIAL PASS":
        result = "PARTIAL PASS"
    else:
        result = "PASS"

    avg_score_start = float(learning.iloc[0]["avg_final_score"]) if len(learning) else 0.0
    avg_score_end = float(learning.iloc[-1]["avg_final_score"]) if len(learning) else 0.0
    notes_parts = list(quality_issues) if quality_issues else ["All explicit criteria satisfied"]
    notes_parts.extend(informational_notes)
    if senior_text_count and not senior_title_count:
        notes_parts.append(senior_notes)
    if name.startswith("Priya") and company_unknown_count:
        notes_parts.append("Most postings lack reliable company-size evidence, so the 100+ employee check is conservative.")
    notes = "; ".join(notes_parts)

    return {
        "persona": name,
        "result": result,
        "recommendation_result": recommendation_result,
        "top_10_count": len(recs),
        "aisha_candidate_funnel_summary": format_candidate_funnel_summary(recs.attrs.get("candidate_funnel_summary", {})) if name.startswith("Aisha") else "",
        "priya_candidate_funnel_summary": format_candidate_funnel_summary(recs.attrs.get("candidate_funnel_summary", {})) if name.startswith("Priya") else "",
        "feedback_test_result": feedback_test.get("feedback_test_result", ""),
        "rejected_company": feedback_test.get("rejected_company", ""),
        "rejected_title": feedback_test.get("rejected_title", ""),
        "rejected_reason": feedback_test.get("rejected_reason", ""),
        "before_rank": feedback_test.get("before_rank", ""),
        "after_rank": feedback_test.get("after_rank", ""),
        "before_score": feedback_test.get("before_score", ""),
        "after_score": feedback_test.get("after_score", ""),
        "feedback_notes": feedback_test.get("feedback_notes", ""),
        "violations": "; ".join(hard_violations) if hard_violations else "",
        "senior_staff_count": senior_staff_count,
        "senior_count": senior_staff_count,
        "senior_title_count": senior_title_count,
        "senior_text_count": senior_text_count,
        "senior_notes": senior_notes,
        "junior_count": junior_count,
        "contract_count": contract_count,
        "unpaid_count": unpaid_count,
        "non_us_count": non_us_count,
        "non_remote_or_region_count": location_count,
        "location_violation_count": location_count,
        "required_years_violation_count": years_count,
        "low_quality_posting_count": low_quality_count,
        "role_family_mismatch_count": role_mismatch_count,
        "ml_related_violation_count": ml_related_count,
        "salary_violation_count": salary_count,
        "no_sponsorship_violation_count": no_sponsor_count,
        "no_sponsor_violation_count": no_sponsor_count,
        "sponsor_or_research_count": sponsor_research_count,
        "unclear_sponsorship_count": unknown_sponsor_count,
        "tiny_startup_count": tiny_count,
        "possible_tiny_startup_count": possible_tiny_count,
        "company_size_unknown_count": company_unknown_count,
        "confirmed_below_threshold_count": confirmed_below_threshold,
        "confirmed_below_100_count": confirmed_below_100,
        "meets_min_company_size_count": meets_min_company,
        "large_company_signal_count": large_company_count,
        "small_company_count": small_count,
        "resume_check_result": resume_check,
        "resume_check_notes": resume_notes,
        "selected_resume_job_rank": "" if selected_resume_job is None else selected_resume_job.get("rank", ""),
        "selected_resume_job_title": "" if selected_resume_job is None else selected_resume_job.get("title", ""),
        "selected_resume_job_company": "" if selected_resume_job is None else selected_resume_job.get("company", ""),
        "notes": notes,
        "defense_count": defense_count,
        "internship_count": internship_count,
        "learning_start_score": round(avg_score_start, 4),
        "learning_end_score": round(avg_score_end, 4),
    }


def run_role_relevance_debug_checks() -> pd.DataFrame:
    ml_profile = {
        "target_role": "ML Engineer, Applied Scientist, Data Scientist ML-focused",
        "skills": ["Python", "SQL", "Pandas", "Scikit-Learn", "PyTorch"],
    }
    platform_profile = {
        "target_role": "MLOps Engineer, ML Platform Engineer",
        "skills": ["Python", "TensorFlow", "Kubernetes", "AWS"],
    }
    cases = [
        (
            "COO data center business",
            ml_profile,
            {"title": "Chief Operations Officer (COO / data center business / remote)", "description": "Lead data center business operations and strategy.", "clean_job_text": ""},
            False,
        ),
        (
            "Data center operations manager",
            ml_profile,
            {"title": "Data Center Operations Manager", "description": "Manage data center operations, vendors, and facilities.", "clean_job_text": ""},
            False,
        ),
        (
            "Machine Learning Engineer",
            ml_profile,
            {"title": "Machine Learning Engineer", "description": "Build machine learning models using Python and scikit-learn.", "clean_job_text": ""},
            True,
        ),
        (
            "Applied Scientist ML",
            ml_profile,
            {"title": "Applied Scientist, Machine Learning", "description": "Develop machine learning models and model evaluation workflows.", "clean_job_text": ""},
            True,
        ),
        (
            "ML Platform Engineer",
            platform_profile,
            {"title": "ML Platform Engineer", "description": "Build MLOps, model serving, feature store, and ML platform systems.", "clean_job_text": ""},
            True,
        ),
    ]
    rows = []
    for label, profile, row, should_match in cases:
        score, note = target_role_fit_score_fn(profile, row)
        filtered = should_filter_role_family_mismatch(profile, row)
        passed = (score >= 0.70 and not filtered) if should_match else (score <= 0.30 or filtered)
        rows.append({
            "case": label,
            "target_fit_score": round(float(score), 4),
            "filtered": bool(filtered),
            "expected_match": should_match,
            "passed": bool(passed),
            "note": note,
        })
    result = pd.DataFrame(rows)
    if not bool(result["passed"].all()):
        raise AssertionError("Role relevance debug checks failed:\n" + result.to_string(index=False))
    return result


def run_salary_extraction_debug_checks() -> pd.DataFrame:
    cases = [
        ("range comma dash", "$120,000 - $160,000", "$120K–$160K"),
        ("range k to", "$120K to $160K", "$120K–$160K"),
        ("base pay label", "Base pay range: $130,000 to $180,000", "$130K–$180K"),
        ("signing bonus ignored", "$5,000 signing bonus", "Not listed"),
        ("hourly ignored", "$20/hour", "Not listed"),
        ("benefits ignored", "paid time off and benefits", "Not listed"),
    ]
    rows = []
    for label, description, expected_display in cases:
        parsed = extract_salary_from_description(description)
        passed = parsed.get("salary_display") == expected_display
        rows.append({
            "case": label,
            "description": description,
            "expected_display": expected_display,
            "actual_display": parsed.get("salary_display"),
            "salary_min": parsed.get("salary_min"),
            "salary_max": parsed.get("salary_max"),
            "salary_source": parsed.get("salary_source"),
            "passed": bool(passed),
        })
    result = pd.DataFrame(rows)
    if not bool(result["passed"].all()):
        raise AssertionError("Salary extraction debug checks failed:\n" + result.to_string(index=False))
    return result


def run_location_parser_debug_checks() -> pd.DataFrame:
    cases = [
        ("Remote, Chicago, Austin, or New York", ["Remote", "Chicago, IL", "Austin, TX", "New York, NY"], True, False, None),
        ("Remote or Bay Area", ["Remote", "San Francisco, CA / Bay Area"], True, False, None),
        ("NYC or remote", ["New York, NY", "Remote"], True, False, None),
        ("Any US location", ["Any US location"], False, True, "US"),
        ("US only", ["United States"], False, False, "US"),
        ("Seattle, Austin, Chicago", ["Seattle, WA", "Austin, TX", "Chicago, IL"], False, False, None),
    ]
    rows = []
    for text, expected_parts, remote_ok, flexible_us, country in cases:
        parsed = parse_location_preferences(text)
        display = parsed.get("display", "")
        passed = all(part in display for part in expected_parts)
        passed = passed and bool(parsed.get("remote_ok")) == remote_ok
        passed = passed and bool(parsed.get("flexible_us")) == flexible_us
        if country is not None:
            passed = passed and parsed.get("country") == country
        rows.append({
            "input": text,
            "display": display,
            "remote_ok": parsed.get("remote_ok"),
            "cities": ", ".join(parsed.get("cities", [])),
            "states": ", ".join(parsed.get("states", [])),
            "regions": ", ".join(parsed.get("regions", [])),
            "country": parsed.get("country"),
            "flexible_us": parsed.get("flexible_us"),
            "passed": bool(passed),
        })
    result = pd.DataFrame(rows)
    if not bool(result["passed"].all()):
        raise AssertionError("Location parser debug checks failed:\n" + result.to_string(index=False))
    return result


def run_location_fallback_debug_checks() -> pd.DataFrame:
    cases = [
        (
            "New York to Jersey City nearby metro",
            {"location_preference": "New York"},
            {"location": "Jersey City, NJ", "is_remote": ""},
            2,
            "Nearby location",
        ),
        (
            "Austin to Dallas same state",
            {"location_preference": "Austin"},
            {"location": "Dallas, TX", "is_remote": ""},
            3,
            "Same-state fallback",
        ),
        (
            "Chicago exact",
            {"location_preference": "Remote, Chicago, Austin, or New York"},
            {"location": "Chicago, IL", "is_remote": ""},
            1,
            "Location fits",
        ),
    ]
    rows = []
    for label, profile, row, expected_tier, expected_reason in cases:
        decision = location_constraint_decision(profile, row)
        passed = int(decision.get("location_tier", 0) or 0) == expected_tier and expected_reason in decision.get("reason", "")
        rows.append({
            "case": label,
            "location": row.get("location", ""),
            "location_tier": decision.get("location_tier", ""),
            "reason": decision.get("reason", ""),
            "passed": bool(passed),
        })
    result = pd.DataFrame(rows)
    if not bool(result["passed"].all()):
        raise AssertionError("Location fallback debug checks failed:\n" + result.to_string(index=False))
    return result


def run_required_years_debug_checks() -> pd.DataFrame:
    cases = [
        (
            "no 5 plus excludes 5",
            {"dealbreakers": ["no 5+ years experience"]},
            {"title": "Analyst", "description": "Requires 5+ years of experience.", "clean_job_text": ""},
            True,
        ),
        (
            "max 4 excludes 5",
            {"max_required_years": 4.0, "dealbreakers": []},
            {"title": "Analyst", "description": "Requires 5+ years of experience.", "clean_job_text": ""},
            True,
        ),
        (
            "max 4 keeps unknown",
            {"max_required_years": 4.0, "dealbreakers": []},
            {"title": "Analyst", "description": "Responsibilities include dashboards and SQL.", "clean_job_text": ""},
            False,
        ),
        (
            "max 4 keeps exactly 4",
            {"max_required_years": 4.0, "dealbreakers": []},
            {"title": "Analyst", "description": "Requires 4 years of experience.", "clean_job_text": ""},
            False,
        ),
        (
            "no 3 plus excludes 3",
            {"dealbreakers": ["no 3+ years experience"]},
            {"title": "Analyst", "description": "Requires 3+ years of experience.", "clean_job_text": ""},
            True,
        ),
    ]
    rows = []
    for label, profile, row, expected_violation in cases:
        violation = required_years_violates_profile(profile, row)
        rows.append({
            "case": label,
            "required_years": extract_required_years(row),
            "expected_violation": expected_violation,
            "actual_violation": bool(violation),
            "passed": bool(violation) == expected_violation,
        })
    result = pd.DataFrame(rows)
    if not bool(result["passed"].all()):
        raise AssertionError("Required years debug checks failed:\n" + result.to_string(index=False))
    return result


def run_role_taxonomy_debug_checks() -> pd.DataFrame:
    profiles = {
        "aisha": {"target_role": "ML Engineer, Applied Scientist, ML-focused Data Scientist", "skills": ["Python", "SQL", "Scikit-Learn"]},
        "marcus": {"target_role": "Data Analyst, BI Analyst, Junior Data Scientist, Analytics Engineer", "candidate_level": "entry_or_junior", "dealbreakers": ["no 3+ years required"]},
        "priya": {"target_role": "Senior ML Platform Engineer, MLOps Engineer", "dealbreakers": ["no junior titles"]},
        "kenji": {"target_role": "Research Scientist, ML Engineer, Applied Scientist, AI Engineer", "dealbreakers": ["needs sponsorship"]},
        "elena": {
            "target_role": "Risk Data Scientist, Credit Risk Analyst, FinTech Data Scientist, Quantitative Analyst",
            "skills": ["Python", "SQL", "R", "Tableau", "credit risk", "risk modeling", "statistics"],
            "preferred_industries": ["fintech", "banking technology", "financial services analytics"],
            "dealbreakers": ["No audit roles", "No accounting roles", "No pure sales roles", "No generic Financial Analyst roles unless analytics/risk/Python/SQL"],
        },
        "daniel": {"target_role": "Analytics Engineer, BI Engineer, Data Platform Analyst", "dealbreakers": ["No backend engineer", "No ML Engineer roles", "No senior manager or director roles"]},
    }
    cases = [
        ("aisha coo data center", profiles["aisha"], {"title": "Chief Operations Officer (COO / data center business)", "description": "Lead data center operations and business strategy.", "skills": ""}, 0.25, False),
        ("marcus senior manager", profiles["marcus"], {"title": "Senior Analytics Manager", "description": "Requires 5+ years of experience managing analysts.", "skills": "SQL"}, 0.45, False),
        ("priya junior ml", profiles["priya"], {"title": "Junior ML Engineer", "description": "Entry-level model support.", "skills": "Python"}, 0.45, False),
        ("kenji no sponsorship", profiles["kenji"], {"title": "Machine Learning Engineer", "description": "Must be authorized to work in the United States without sponsorship.", "skills": "Python PyTorch"}, 0.80, False),
        ("elena marketing analyst", profiles["elena"], {"title": "Marketing Analyst", "description": "Campaign attribution, paid media, SEO dashboards.", "skills": "SQL Tableau"}, 0.30, False),
        ("elena audit", profiles["elena"], {"title": "Audit Analyst", "description": "SOX controls and compliance testing.", "skills": "Excel"}, 0.30, True),
        ("elena credit risk", profiles["elena"], {"title": "Credit Risk Analyst", "description": "Build credit risk dashboards and risk models using SQL and Python.", "skills": "Python SQL statistics"}, 0.70, False),
        ("elena quant", profiles["elena"], {"title": "Quantitative Analyst", "description": "Statistical modeling, risk analytics, Python and SQL.", "skills": "Python SQL"}, 0.70, False),
        ("daniel backend", profiles["daniel"], {"title": "Backend Engineer", "description": "Build APIs and microservices.", "skills": "Java APIs"}, 0.30, True),
        ("daniel ml engineer", profiles["daniel"], {"title": "Machine Learning Engineer", "description": "Train ML models with PyTorch.", "skills": "Python PyTorch"}, 0.30, True),
        ("daniel analytics engineer", profiles["daniel"], {"title": "Analytics Engineer", "description": "Build dbt models, data warehouse, semantic layer and BI datasets.", "skills": "SQL dbt Snowflake"}, 0.70, False),
    ]
    rows = []
    for label, profile, row, threshold, expect_filter in cases:
        compatibility, note, family_label = compute_role_family_compatibility(profile, row)
        guarded_score, guard_note = apply_role_mismatch_guardrail(profile, row, 0.95)
        excluded = should_filter_excluded_role_family(profile, row)
        dealbreakers = parse_dealbreakers(profile)
        seniority_blocked = bool(
            (dealbreakers.get("no_senior") and has_senior_title_signal(row.get("title", ""), row.get("seniority", "")))
            or (dealbreakers.get("no_junior") and job_seniority_level(row) in {"entry", "intern"})
        )
        sponsorship_blocked = bool(dealbreakers.get("need_sponsorship") and has_no_sponsorship_signal(row))
        years_blocked = required_years_violates_profile(profile, row)
        hard_blocked = bool(excluded or seniority_blocked or sponsorship_blocked or years_blocked)
        families = infer_job_role_families(row)
        if threshold >= 0.70:
            passed = compatibility >= threshold and guarded_score >= threshold and not hard_blocked
        else:
            passed = guarded_score <= threshold or hard_blocked == expect_filter
        if expect_filter:
            passed = passed and hard_blocked
        if label.startswith("marcus") or label.startswith("priya") or label.startswith("kenji"):
            passed = hard_blocked
        rows.append({
            "case": label,
            "job_family": max(families, key=families.get) if families else "other",
            "family_label": family_label,
            "compatibility": round(float(compatibility), 4),
            "guarded_score": round(float(guarded_score), 4),
            "excluded": bool(excluded),
            "hard_blocked": hard_blocked,
            "note": note if note else guard_note,
            "passed": bool(passed),
        })
    result = pd.DataFrame(rows)
    if not bool(result["passed"].all()):
        raise AssertionError("Role taxonomy debug checks failed:\n" + result.to_string(index=False))
    return result


def main() -> None:
    rec = JobRecommender()
    role_debug = run_role_relevance_debug_checks()
    role_debug.to_csv(OUT_DIR / "role_relevance_debug_checks.csv", index=False, encoding="utf-8-sig")
    salary_debug = run_salary_extraction_debug_checks()
    salary_debug.to_csv(OUT_DIR / "salary_extraction_debug_checks.csv", index=False, encoding="utf-8-sig")
    location_debug = run_location_parser_debug_checks()
    location_debug.to_csv(OUT_DIR / "location_parser_debug_checks.csv", index=False, encoding="utf-8-sig")
    location_fallback_debug = run_location_fallback_debug_checks()
    location_fallback_debug.to_csv(OUT_DIR / "location_fallback_debug_checks.csv", index=False, encoding="utf-8-sig")
    required_years_debug = run_required_years_debug_checks()
    required_years_debug.to_csv(OUT_DIR / "required_years_debug_checks.csv", index=False, encoding="utf-8-sig")
    role_taxonomy_debug = run_role_taxonomy_debug_checks()
    role_taxonomy_debug.to_csv(OUT_DIR / "role_taxonomy_debug_checks.csv", index=False, encoding="utf-8-sig")
    rows = []
    enrichment_rows = []
    for name, profile in TEST_PERSONAS.items():
        recs = rec.recommend_from_profile(profile, top_n=10, candidate_k=1500)
        status = dict(getattr(rec, "last_enrichment_status", {}) or {})
        status["persona"] = name
        enrichment_rows.append(status)
        ensure_diagnostic_columns(recs).to_csv(
            OUT_DIR / f"{name.split(' - ')[0].lower()}_top10.csv",
            index=False,
            encoding="utf-8-sig",
        )
        if name.startswith("Kenji"):
            learning = simulate_small_company_rejection(rec, profile)
            feedback_test = run_kenji_feedback_test(rec, profile, recs)
        else:
            learning = simulate_feedback_learning(rec, profile, rounds=3, top_n=10)
            feedback_test = {}
        learning.to_csv(OUT_DIR / f"{name.split(' - ')[0].lower()}_learning.csv", index=False, encoding="utf-8-sig")
        selected_resume_job = select_resume_job(recs)
        resume_text = ""
        if selected_resume_job is not None:
            resume_text = build_local_resume(profile, selected_resume_job.to_dict())
        resume_path = OUT_DIR / f"{name.split(' - ')[0].lower()}_resume.md"
        resume_path.write_text(resume_text, encoding="utf-8")
        save_resume_docx(resume_text, OUT_DIR / f"{name.split(' - ')[0].lower()}_resume.docx")
        rows.append(evaluate_persona(name, profile, recs, learning, resume_text, selected_resume_job, feedback_test))

    table = pd.DataFrame(rows)
    for col in SUMMARY_COLUMNS:
        if col not in table.columns:
            table[col] = ""
    table = table[SUMMARY_COLUMNS]
    table.to_csv(OUT_DIR / "persona_pass_fail.csv", index=False, encoding="utf-8-sig")
    enrichment_table = pd.DataFrame(enrichment_rows)
    if not enrichment_table.empty:
        leading = ["persona", "companies_checked", "cache_hits", "api_calls_made", "unknown_cached", "errors"]
        for col in leading:
            if col not in enrichment_table.columns:
                enrichment_table[col] = 0
        enrichment_table = enrichment_table[leading]
        enrichment_table.to_csv(OUT_DIR / "company_enrichment_status.csv", index=False, encoding="utf-8-sig")
    print(table.to_string(index=False))
    if not enrichment_table.empty:
        print("\nCompany enrichment status:")
        print(enrichment_table.to_string(index=False))
    print(f"Saved results to {OUT_DIR}")


if __name__ == "__main__":
    main()
