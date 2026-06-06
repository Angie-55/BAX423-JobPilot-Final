# Key AI Prompts Used During Development

This file summarizes the main AI-assisted development prompts used to build JobPilot. The prompts are written at the feature and architecture level rather than listing every small UI or bug-fix change.

## Prompt 1 - End-to-end job data pipeline

Prompt: Help me design an offline job data pipeline that ingests job postings from multiple sources, normalizes schemas, removes noisy HTML, extracts searchable text, deduplicates jobs, and saves a final processed dataset for a Streamlit recommendation app.

Purpose: I used this to structure the ingestion, cleaning, merging, and processed-output workflow. The final app uses an offline snapshot so recommendations can run without live job API calls.

## Prompt 2 - Job schema normalization and feature extraction

Prompt: Create reusable job normalization logic for title, company, location, employment type, seniority, required years, salary, skills, role family, and clean job description text.

Purpose: I used this to make raw job records comparable across sources. I adjusted the implementation so normalized fields represent the app's final interpretation instead of exposing raw source fields directly to users.

## Prompt 3 - Semantic retrieval with embeddings and FAISS

Prompt: Build a sentence-transformers embedding pipeline for job postings, save job embeddings and a FAISS index, and retrieve candidate jobs using the user's profile and target role text.

Purpose: I used this to implement fast semantic candidate retrieval before detailed scoring. The FAISS artifacts are saved under `data/processed/` so the app can load them during demo and deployment.

## Prompt 4 - Static job feature indexing and caching

Prompt: Optimize the recommender so static job-level features are precomputed or cached once instead of recomputed for the full dataset on every search. Include role profile text, normalized title/company/location, role family, duplicate grouping keys, and reusable location vocabulary.

Purpose: I used this to add the role/static feature index and location vocabulary cache while keeping correctness-sensitive scoring logic precise at runtime. This improved demo speed without relying on approximate cached values for hard filters.

## Prompt 5 - Multi-stage recommendation ranking

Prompt: Design a two-stage recommender that first retrieves a candidate pool, then applies hard filters and detailed scoring for role fit, skills, experience, salary, location, employment type, company constraints, posting quality, sponsorship fit, and feedback.

Purpose: I used this to build JobPilot's recommendation pipeline. The ranking combines semantic similarity with structured business rules so each persona's dealbreakers and preferences are reflected in the Top 10.

## Prompt 6 - Company enrichment and cache-first workflow

Prompt: Add company enrichment fields such as size bucket, employee estimate, large-company signal, research-lab signal, and H-1B sponsor signal using a cache-first design that avoids repeated API calls.

Purpose: I used this to support company-size and sponsorship-sensitive personas such as Priya and Kenji. The app reads cached enrichment first and can run offline for grading.

## Prompt 7 - Resume/profile parsing

Prompt: Build a conservative resume and persona parser that extracts editable profile fields such as name, skills, years of experience, target roles, education keywords, preferences, dealbreakers, salary requirements, and location preferences.

Purpose: I used this to support both uploaded resumes and pasted persona/profile text. The UI lets users review and edit extracted fields before matching.

## Prompt 8 - Location and salary intelligence

Prompt: Improve matching by parsing multi-location preferences, building location aliases from the job dataset, supporting remote and US-only preferences, and extracting annual salary ranges from job descriptions when structured salary fields are missing.

Purpose: I used this to make location and salary matching more accurate while keeping uncertain values transparent in the UI and CSV export.

## Prompt 9 - Adaptive feedback learning

Prompt: Add user feedback actions for Save, Apply later, and Not interested, with simple visible reasons mapped to richer backend learning signals that can adjust future ranking.

Purpose: I used this to show how JobPilot can adapt recommendations after feedback while keeping the customer-facing UI simple.

## Prompt 10 - Tailored resume and cover letter generation

Prompt: Generate grounded resume and cover letter drafts using only the current user profile, the selected job, matched skills, and supported evidence. Include OpenRouter generation with a local fallback, and avoid unsupported facts or persona contamination.

Purpose: I used this to add tailored resume and cover letter outputs for the selected recommendation. The generation workflow is separate from ranking and uses the user's chosen job rather than always using the top result.

## Prompt 11 - Customer-facing Streamlit interface

Prompt: Build a step-by-step Streamlit interface for resume upload or profile paste, editable profile review, preference confirmation, ranked job cards, clean explanations, feedback actions, job-specific resume and cover letter generation, and a user-facing CSV download.

Purpose: I used this to create the main application workflow. I refined the UI so debug details are hidden, similar postings are grouped, and the exported CSV matches the visible recommendation cards.

## Prompt 12 - Job market analytics dashboard

Prompt: Add a lightweight dashboard that summarizes the offline job market dataset with top skills, salary distribution, demand by location, and demand by role family.

Purpose: I used this to add a batch analytics feature that complements individual recommendations and supports the project requirement for market-level insight.

## Prompt 13 - Persona evaluation and validation

Prompt: Create repeatable persona evaluation tests for Aisha, Marcus, Priya, and Kenji, including recommendation counts, hard-filter checks, company-size checks, sponsorship checks, feedback behavior, and resume consistency checks.

Purpose: I used this to validate the recommender against required personas and to produce CSV outputs for the final evaluation.

## Prompt 14 - Cloud deployment preparation

Prompt: Prepare the Streamlit app for cloud deployment with a Dockerfile, environment-variable configuration, offline artifacts, model download during image build, and a Cloud Run-compatible startup command.

Purpose: I used this to make the app deployable on Google Cloud while keeping API keys out of code and out of the Docker image.
