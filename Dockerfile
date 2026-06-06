FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY code/requirements.txt /app/code/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /app/code/requirements.txt

# Download the embedding model during image build so runtime can stay offline.
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY . /app

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    ENABLE_ONLINE_COMPANY_ENRICHMENT=false \
    FORCE_REFRESH_COMPANY_ENRICHMENT=false

EXPOSE 8080

CMD ["sh", "-c", "streamlit run code/app.py --server.address=0.0.0.0 --server.port=${PORT:-8080} --server.fileWatcherType=none"]
