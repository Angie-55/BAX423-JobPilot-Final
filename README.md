# JobPilot - Smart Job Matcher and Resume Builder

JobPilot is a BAX-423 final project application. It ingests job postings, keeps an offline dataset snapshot, builds sentence-transformer embeddings, retrieves candidates with FAISS, applies hard filters, re-ranks jobs with structured scores, learns from feedback, and generates a tailored resume draft for a selected role.

## Project Structure

```text
Wang_Hanzhi_BAX423_Final/
  code/
    app.py
    recommender.py
    resume_parser.py
    build_embeddings.py
    collect_jsearch_jobs.py
    clean_jobs.py
    merge_job_datasets.py
    sample_techmap_jobs_fixed.py
    evaluate_personas.py
    requirements.txt
  data/
    processed/
      jobs_final.csv
      faiss.index
      job_embeddings.npy
      job_index_map.csv
  brief.pdf
  prompts.md
  README.md
```

## Data

The final offline dataset is `data/processed/jobs_final.csv`. It combines a Kaggle Techmap sample with JSearch API rows and is deduplicated by title, company, and location. The processed folder also includes FAISS and embedding artifacts when available, so graders can run without live API calls.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r code/requirements.txt
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
```

## Environment

Copy `.env.example` to `.env` for local use:

```bash
copy .env.example .env
```

Set `OPENROUTER_API_KEY` only on your local machine or deployment environment. Do not submit a real `.env` file with API keys.

## Build Embeddings

If the FAISS artifacts are missing, rebuild them from the processed CSV:

```bash
python code/build_embeddings.py
```

This creates:

```text
data/processed/job_embeddings.npy
data/processed/faiss.index
data/processed/job_index_map.csv
```

## Run the App

```bash
streamlit run code/app.py --server.fileWatcherType none
```

The app supports:

- Resume file upload or pasted profile text
- Flexible profile parsing and editable fields
- Preference and dealbreaker confirmation
- FAISS semantic retrieval plus hard filters
- Multi-stage ranking with diagnostic scores
- Why-ranked explanations
- Save, apply-later, and not-interested feedback
- OpenRouter resume generation with local fallback
- CSV export with salary, required-years, quality, target-fit, and score fields
- Batch market analytics from the full offline dataset

## Run Persona Tests

After embeddings exist:

```bash
python code/evaluate_personas.py
```

Outputs are saved in `outputs/`, including `persona_pass_fail.csv`.

## Rebuild the Dataset

The final processed snapshot is already included. To rebuild from raw/API sources:

```bash
python code/collect_jsearch_jobs.py --target 1000 --pages-per-query 1 --sleep 8
python code/clean_jobs.py
python code/merge_job_datasets.py
python code/build_embeddings.py
```

If rebuilding the Kaggle sample, place the raw Techmap file under `data/raw/` and run:

```bash
python code/sample_techmap_jobs_fixed.py --input data/raw/jobs.jsonl.gz --output data/raw/kaggle_jobs_30000.csv --target 30000
```

## Deployment

For Streamlit Cloud or Google Cloud, upload the project, install `code/requirements.txt`, set environment variables in the hosting platform, include `data/processed/jobs_final.csv`, and include or rebuild the embedding artifacts. Launch with:

```bash
streamlit run code/app.py --server.port 8080 --server.address 0.0.0.0 --server.fileWatcherType none
```

## Submission

Submit a ZIP named `Wang_Hanzhi_BAX423_Final.zip` containing `code/`, `data/`, `brief.pdf`, `prompts.md`, and this `README.md`. Do not include real API keys or a populated `.env`.
