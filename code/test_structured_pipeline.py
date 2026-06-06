import json
from pathlib import Path
from recommender import JobRecommender

profile_path = Path("profile1.json")
if not profile_path.exists():
    raise FileNotFoundError("Create profile1.json first with resume_parser.py")

profile = json.loads(profile_path.read_text(encoding="utf-8"))
rec = JobRecommender()
result = rec.recommend_from_profile(profile, top_n=10, candidate_k=150)
print(result[["final_score", "title", "company", "seniority", "employment_type", "why_matched"]].to_string(index=False))
