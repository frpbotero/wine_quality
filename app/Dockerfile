FROM python:3.12-slim

WORKDIR /app

# System deps (curl for healthcheck)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps (API + Streamlit UI)
COPY app/requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

# Application code
COPY app/database.py   ./database.py
COPY app/models.py     ./models.py
COPY app/main.py       ./main.py
COPY app/streamlit_ui.py ./streamlit_ui.py

# FastAPI (8000) + Streamlit (8501)
EXPOSE 8000 8501

# Uvicorn starts FastAPI; FastAPI lifespan spawns Streamlit as subprocess
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]