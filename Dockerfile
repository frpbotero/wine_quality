FROM python:3.12-slim

WORKDIR /app

# System deps (curl for healthcheck)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps (API + Streamlit UI)
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

COPY database.py   ./database.py
COPY models.py     ./models.py
COPY main.py       ./main.py
COPY streamlit_ui.py ./streamlit_ui.py
COPY supabase_logger.py ./supabase_logger.py

# FastAPI (8000) + Streamlit (8501)
EXPOSE 8000 8501

# Uvicorn starts FastAPI; FastAPI lifespan spawns Streamlit as subprocess
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]