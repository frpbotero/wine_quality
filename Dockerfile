FROM python:3.12-slim

WORKDIR /app

# System deps (curl for healthcheck)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps (API + Streamlit UI)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files to app directory
COPY database.py   ./database.py
COPY models.py     ./models.py
COPY streamlit_ui.py ./streamlit_ui.py
COPY supabase_logger.py ./supabase_logger.py

# Copy trained models and evaluation reports
COPY models/ ./models/
COPY reports/ ./reports/

# Streamlit (8501)
EXPOSE 8501

# Start Streamlit with FastAPI
CMD ["streamlit", "run", "streamlit_ui.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
