"""
Build dense embeddings and a FAISS index for the final JobPilot dataset.

Run:
    python code/build_embeddings.py

Inputs:
    data/processed/jobs_final.csv

Outputs:
    data/processed/job_embeddings.npy
    data/processed/faiss.index
    data/processed/job_index_map.csv
"""

from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data/processed")
JOBS_CSV = DATA_DIR / "jobs_final.csv"
EMBEDDINGS_PATH = DATA_DIR / "job_embeddings.npy"
FAISS_PATH = DATA_DIR / "faiss.index"
INDEX_MAP_PATH = DATA_DIR / "job_index_map.csv"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_text(row: pd.Series) -> str:
    """Create compact job text for semantic matching."""
    base_text = str(row.get("clean_job_text", "") or row.get("job_text", "") or "")
    title = str(row.get("title", "") or "")
    company = str(row.get("company", "") or "")
    location = str(row.get("location", "") or "")
    skills = str(row.get("skills", "") or "")
    seniority = str(row.get("seniority", "") or "")
    employment_type = str(row.get("employment_type", "") or "")
    source = str(row.get("source", "") or "")

    compact = f"""
Title: {title}
Company: {company}
Location: {location}
Seniority: {seniority}
Employment Type: {employment_type}
Skills: {skills}
Source: {source}
Description: {base_text[:3000]}
"""
    return " ".join(compact.split())


def main() -> None:
    if not JOBS_CSV.exists():
        raise FileNotFoundError(f"Missing {JOBS_CSV}. Run merge_job_datasets.py first.")

    df = pd.read_csv(JOBS_CSV).fillna("")
    if df.empty:
        raise RuntimeError("jobs_final.csv is empty.")

    texts = df.apply(get_embedding_text, axis=1).tolist()

    print(f"Loaded {len(df):,} jobs")
    print(f"Embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    np.save(EMBEDDINGS_PATH, embeddings)
    faiss.write_index(index, str(FAISS_PATH))

    index_map = pd.DataFrame({
        "faiss_id": range(len(df)),
        "job_id": df["job_id"].astype(str),
        "title": df["title"],
        "company": df["company"],
        "source": df["source"],
    })
    index_map.to_csv(INDEX_MAP_PATH, index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"Saved embeddings: {EMBEDDINGS_PATH}")
    print(f"Saved FAISS index: {FAISS_PATH}")
    print(f"Saved index map: {INDEX_MAP_PATH}")


if __name__ == "__main__":
    main()
