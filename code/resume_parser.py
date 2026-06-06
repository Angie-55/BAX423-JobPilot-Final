"""
Improved structured resume parser for JobPilot.

Key design choices:
- Extracts PDF / DOCX / TXT resume text.
- Produces structured fields for ranking instead of using the whole resume directly.
- Does NOT infer work experience from education dates.
- Infers candidate level conservatively from work/internship sections.

Run:
    python resume_parser.py --resume "Resume 1.docx" --target-role "Data Analyst" --location "California" --dealbreakers "no senior,no contract" --out profile1.json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

SKILL_KEYWORDS = [
    "python", "sql", "r", "excel", "tableau", "power bi", "machine learning", "deep learning",
    "aws", "azure", "gcp", "spark", "pandas", "numpy", "scikit-learn", "sklearn", "tensorflow", "pytorch",
    "nlp", "statistics", "a/b testing", "ab testing", "java", "javascript", "typescript", "react", "node",
    "docker", "kubernetes", "git", "linux", "airflow", "dbt", "snowflake", "looker", "salesforce",
    "financial modeling", "forecasting", "accounting", "risk", "cybersecurity", "cloud", "etl",
    "data visualization", "communication", "stakeholder", "product management", "agile", "scrum",
    "figma", "ux", "ui", "marketing", "seo", "consulting", "powerpoint", "gaap", "audit",
    "power query", "analytics", "data analysis", "dashboard", "visualization", "regression", "classification",
    "data mining", "business intelligence", "data cleaning", "data preprocessing", "experimental design",
    "customer analytics", "financial analysis", "financial reporting", "market research", "statistical modeling",
    "sas", "stata", "vba", "jira", "confluence", "bigquery", "redshift", "data warehousing",
    "api", "rest api", "crm", "experimentation", "causal inference", "time series", "mlops",
    "model deployment", "data pipeline", "data pipelines", "excel pivot tables",
]

EDUCATION_KEYWORDS = [
    "master", "msba", "mba", "m.s.", "msc", "bachelor", "b.s.", "ba", "phd", "university", "college",
    "degree", "business analytics", "data science", "computer science", "accounting", "finance", "statistics",
    "economics", "information systems", "management", "marketing",
]

EDUCATION_KEYWORD_PATTERNS = [
    ("MSBA", r"(?<![A-Za-z0-9])msba(?![A-Za-z0-9])"),
    ("MBA", r"(?<![A-Za-z0-9])mba(?![A-Za-z0-9])"),
    ("MS", r"(?<![A-Za-z0-9])m\.?s\.?(?![A-Za-z0-9])"),
    ("MSc", r"(?<![A-Za-z0-9])msc(?![A-Za-z0-9])"),
    ("Master", r"\bmaster(?:'s|s)?\b"),
    ("BA", r"(?<![A-Za-z0-9])b\.?a\.?(?![A-Za-z0-9])"),
    ("BS", r"(?<![A-Za-z0-9])b\.?s\.?(?![A-Za-z0-9])"),
    ("Bachelor", r"\bbachelor(?:'s|s)?\b"),
    ("PhD", r"(?<![A-Za-z0-9])ph\.?d\.?(?![A-Za-z0-9])"),
    ("UC Davis", r"\buc\s+davis\b|\buniversity of california,\s*davis\b"),
    ("University", r"\buniversity\b"),
    ("College", r"\bcollege\b"),
    ("Degree", r"\bdegree\b"),
    ("Business Analytics", r"\bbusiness analytics\b"),
    ("Data Science", r"\bdata science\b"),
    ("Computer Science", r"\bcomputer science\b"),
    ("Accounting", r"\baccounting\b"),
    ("Finance", r"\bfinance\b"),
    ("Statistics", r"\bstatistics\b"),
    ("Economics", r"\beconomics\b"),
    ("Information Systems", r"\binformation systems\b"),
    ("Management", r"\bmanagement\b"),
    ("Marketing", r"\bmarketing\b"),
]

TITLE_PATTERNS = [
    "data analyst", "business analyst", "financial analyst", "marketing analyst", "product analyst", "data scientist",
    "machine learning engineer", "software engineer", "consultant", "associate", "intern", "auditor",
    "tax consultant", "accountant", "project manager", "product manager", "research assistant", "teaching assistant",
    "graduate assistant", "analyst intern", "data intern", "finance intern", "marketing intern",
    "bi analyst", "analytics engineer", "data engineer", "ml engineer", "mlops engineer",
    "applied scientist", "decision scientist", "growth analyst", "reporting analyst",
]

SECTION_HEADERS = {
    "education": ["education", "academic background"],
    "experience": ["experience", "work experience", "professional experience", "employment", "internship", "internships"],
    "projects": ["projects", "academic projects", "selected projects", "project experience", "research projects", "course projects"],
    "skills": ["skills", "technical skills", "tools", "core skills"],
}

@dataclass
class CandidateProfile:
    target_role: str
    skills: List[str]
    years_experience: Optional[float]
    candidate_level: str
    education_keywords: List[str]
    past_titles: List[str]
    project_keywords: List[str]
    location_preference: str
    dealbreakers: List[str]
    raw_text_length: int
    profile_summary: str


def clean_resume_text(text: str) -> str:
    if not text:
        return ""
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-", "\u2022": " ",
        "â€™": "'", "â€œ": '"', "â€\x9d": '"', "â€“": "-", "â€¢": " ", "鈥?": " ", "鈥檚": "'s", "鈥檛": "'t",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    text = re.sub(r"[\t\r]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


def extract_text_from_pdf(path: Path) -> str:
    try:
        import pdfplumber
    except ImportError as exc:
        raise ImportError("Please install pdfplumber: pip install pdfplumber") from exc
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)


def extract_text_from_docx(path: Path) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        raise ImportError("Please install python-docx: pip install python-docx") from exc
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def load_resume_text(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Resume file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = extract_text_from_pdf(path)
    elif suffix == ".docx":
        text = extract_text_from_docx(path)
    elif suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
    else:
        raise ValueError("Unsupported resume format. Use PDF, DOCX, TXT, or MD.")
    return clean_resume_text(text)


def section_text(text: str, section: str) -> str:
    """Best-effort extraction of a resume section. Returns empty string if not found."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    header_to_section = {}
    for sec, headers in SECTION_HEADERS.items():
        for h in headers:
            header_to_section[h] = sec

    current = None
    buckets = {k: [] for k in SECTION_HEADERS}
    for line in lines:
        normalized = re.sub(r"[^a-z ]", "", line.lower()).strip()
        matched_section = None
        for h, sec in header_to_section.items():
            if normalized == h or normalized.startswith(h + " ") and len(normalized) <= len(h) + 15:
                matched_section = sec
                break
        if matched_section:
            current = matched_section
            continue
        if current:
            buckets[current].append(line)
    return "\n".join(buckets.get(section, []))


def normalize_skill_name(skill: str) -> str:
    mapping = {
        "sklearn": "Scikit-Learn", "scikit-learn": "Scikit-Learn", "ab testing": "A/B Testing", "a/b testing": "A/B Testing",
        "sql": "SQL", "r": "R", "aws": "AWS", "gcp": "GCP", "nlp": "NLP", "etl": "ETL", "gaap": "GAAP",
        "ui": "UI", "ux": "UX", "msba": "MSBA", "api": "API", "rest api": "REST API",
        "gpa": "GPA", "uc davis": "UC Davis",
        "mlops": "MLOps", "vba": "VBA", "sas": "SAS", "crm": "CRM",
        "data pipelines": "Data Pipeline",
        "pytorch": "PyTorch", "tensorflow": "TensorFlow",
    }
    return mapping.get(skill.lower(), skill.title())


def skill_mention_is_negated(text: str, start: int, end: int) -> bool:
    base = max(0, start - 55)
    window = text[base: min(len(text), end + 45)].lower()
    patterns = [
        r"\b(no|not|without|lack(?:ing)?|limited)\b.{0,55}\b(experience|background|ownership)\b",
        r"\bno\s+production\s+(?:ml|machine learning)\s+experience\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, window):
            if base + match.start() <= start:
                return True
    return False


def extract_skills(text: str) -> List[str]:
    text_l = text.lower()
    found = []
    for skill in SKILL_KEYWORDS:
        if len(skill) <= 3:
            pattern = r"(?<![a-zA-Z0-9])" + re.escape(skill.lower()) + r"(?![a-zA-Z0-9])"
            matched = re.search(pattern, text_l)
        else:
            matched = re.search(re.escape(skill.lower()), text_l)
        if matched and not skill_mention_is_negated(text_l, matched.start(), matched.end()):
            found.append(normalize_skill_name(skill))
    return sorted(set(found))


def extract_years_experience(text: str) -> Optional[float]:
    """Conservative years extraction. Do not use education dates as work experience."""
    text_l = text.lower()
    explicit_patterns = [
        r"(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)\s+of\s+(?:professional\s+)?experience",
        r"(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)\s+(?:professional\s+)?experience",
        r"experience\s*[:\-]?\s*(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)",
    ]
    vals = []
    for pattern in explicit_patterns:
        for match in re.finditer(pattern, text_l):
            start = match.start()
            context = text_l[max(0, start - 24):start]
            if re.search(r"\b(no|not|without|less than|under|avoid)\b", context):
                continue
            vals.append(float(match.group(1)))
    if vals:
        return max(vals)

    exp_text = section_text(text, "experience")
    if not exp_text:
        # If the resume contains intern roles but no work section, treat as entry-level rather than inferring from school dates.
        if re.search(r"\bintern(ship)?\b", text_l):
            return 0.5
        return None

    # Count month-year ranges in the experience section only.
    month_year = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+((?:19|20)\d{2})"
    years = [int(y) for y in re.findall(month_year, exp_text.lower())]
    # Also capture numeric year ranges in experience section, but not if dominated by education-only text.
    years += [int(y) for y in re.findall(r"\b((?:19|20)\d{2})\b", exp_text)]
    years = sorted({y for y in years if 1990 <= y <= 2030})
    if len(years) >= 2:
        span = max(years) - min(years)
        # Conservative cap: internships/projects should not accidentally become 4-5 years.
        if "intern" in exp_text.lower():
            return min(float(max(span, 0.5)), 1.0)
        if 0 < span <= 12:
            return float(span)
    if re.search(r"\bintern(ship)?\b", exp_text.lower()):
        return 0.5
    return None


def infer_candidate_level(years: Optional[float], past_titles: List[str], dealbreakers: List[str]) -> str:
    joined = " ".join(past_titles + dealbreakers).lower()
    if any(x in joined for x in ["intern", "new grad", "entry", "junior", "no senior"]):
        return "entry_or_junior"
    if years is None:
        return "unknown"
    if years < 2:
        return "entry_or_junior"
    if years < 5:
        return "mid"
    return "senior_or_experienced"


def extract_education_keywords(text: str) -> List[str]:
    edu_text = section_text(text, "education") or text
    text_l = edu_text.lower()
    found = []
    for label, pattern in EDUCATION_KEYWORD_PATTERNS:
        if re.search(pattern, text_l, flags=re.I):
            found.append(label)
    return sorted(set(found))


def extract_past_titles(text: str) -> List[str]:
    exp_text = section_text(text, "experience") or text
    text_l = exp_text.lower()
    found = [title.title() for title in TITLE_PATTERNS if title in text_l]
    return sorted(set(found))


def extract_project_keywords(text: str) -> List[str]:
    proj_text = section_text(text, "projects") or text
    text_l = proj_text.lower()
    project_terms = [
        "dashboard", "prediction", "classification", "regression", "recommendation", "forecasting", "machine learning",
        "data visualization", "etl", "database", "customer analytics", "pricing", "optimization", "nlp", "sentiment",
        "data analysis", "business intelligence", "statistical modeling",
    ]
    found = []
    for term in project_terms:
        match = re.search(re.escape(term), text_l)
        if match and not skill_mention_is_negated(text_l, match.start(), match.end()):
            found.append(term.title())
    if section_text(text, "projects") and "Projects" not in found:
        found.append("Projects")
    return sorted(set(found))


def parse_dealbreakers(value: str | List[str] | None) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = re.split(r"[,;]", value)
    return [x.strip().lower() for x in items if x and x.strip()]


def build_profile_summary(profile: CandidateProfile) -> str:
    years_text = "unknown" if profile.years_experience is None else f"{profile.years_experience:g} years"
    return (
        f"Target role: {profile.target_role or 'Not specified'}. "
        f"Candidate level: {profile.candidate_level}. "
        f"Skills: {', '.join(profile.skills) if profile.skills else 'Not detected'}. "
        f"Work experience: {years_text}. "
        f"Past titles: {', '.join(profile.past_titles) if profile.past_titles else 'Not detected'}. "
        f"Education: {', '.join(profile.education_keywords) if profile.education_keywords else 'Not detected'}. "
        f"Projects: {', '.join(profile.project_keywords) if profile.project_keywords else 'Not detected'}. "
        f"Location preference: {profile.location_preference or 'Flexible'}. "
        f"Dealbreakers: {', '.join(profile.dealbreakers) if profile.dealbreakers else 'None'}."
    )


def parse_resume(resume_text: str, target_role: str = "", location_preference: str = "", dealbreakers: str | List[str] | None = None) -> CandidateProfile:
    text = clean_resume_text(resume_text)
    dealbreaker_list = parse_dealbreakers(dealbreakers)
    skills = extract_skills(text)
    education = extract_education_keywords(text)
    titles = extract_past_titles(text)
    projects = extract_project_keywords(text)
    years = extract_years_experience(text)
    level = infer_candidate_level(years, titles, dealbreaker_list)
    profile = CandidateProfile(
        target_role=target_role,
        skills=skills,
        years_experience=years,
        candidate_level=level,
        education_keywords=education,
        past_titles=titles,
        project_keywords=projects,
        location_preference=location_preference,
        dealbreakers=dealbreaker_list,
        raw_text_length=len(text),
        profile_summary="",
    )
    profile.profile_summary = build_profile_summary(profile)
    return profile


def parse_resume_file(resume_path: str | Path, target_role: str = "", location_preference: str = "", dealbreakers: str | List[str] | None = None) -> CandidateProfile:
    text = load_resume_text(resume_path)
    return parse_resume(text, target_role, location_preference, dealbreakers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a resume into structured JobPilot profile fields.")
    parser.add_argument("--resume", required=True, help="Path to resume PDF/DOCX/TXT. Use quotes if path has spaces.")
    parser.add_argument("--target-role", default="", help="User target role, e.g., Data Analyst")
    parser.add_argument("--location", default="", help="Location preference, e.g., California or Remote")
    parser.add_argument("--dealbreakers", default="", help="Comma-separated dealbreakers, e.g., no senior,no contract")
    parser.add_argument("--out", default="", help="Optional output JSON path")
    args = parser.parse_args()

    profile = parse_resume_file(args.resume, args.target_role, args.location, args.dealbreakers)
    payload = asdict(profile)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved profile JSON: {args.out}")


if __name__ == "__main__":
    main()
