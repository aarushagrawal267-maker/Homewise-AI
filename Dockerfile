# HomeWise AI — production container
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=8501
# libgomp1 is required by LightGBM / XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user (also required by Hugging Face Spaces)
RUN useradd -m -u 1000 app
WORKDIR /home/app/homewise
COPY --chown=app:app requirements.txt .
RUN pip install -r requirements.txt
COPY --chown=app:app . .
USER app

# Bake the checksum-verified dataset into the image (the app also fetches it on first run if missing)
RUN python scripts/download_data.py || echo "Dataset will be downloaded on first run"

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD curl -fs http://localhost:${PORT}/_stcore/health || exit 1
CMD streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true
