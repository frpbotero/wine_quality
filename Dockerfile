FROM python:3.12-slim

WORKDIR /app

# System deps (curl for healthcheck)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps (API + Streamlit UI)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy wine_project files to app directory
COPY wine_project/database.py   ./database.py
COPY wine_project/models.py     ./models.py
COPY wine_project/main.py       ./main.py
COPY wine_project/streamlit_ui.py ./streamlit_ui.py
COPY wine_project/supabase_logger.py ./supabase_logger.py

# Copy trained models
COPY wine_project/models/ ./models/

# FastAPI (8000) + Streamlit (8501)
EXPOSE 8000 8501

# Start Streamlit with FastAPI
CMD ["streamlit", "run", "streamlit_ui.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
