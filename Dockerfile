FROM python:3.12-slim

WORKDIR /app

# System deps (curl for healthcheck + git for dvc)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl git \
    && rm -rf /var/lib/apt/lists/*

# Python deps (API + Streamlit UI)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files to app directory
COPY database.py   ./database.py
COPY models.py     ./models.py
COPY streamlit_ui.py ./streamlit_ui.py
COPY supabase_logger.py ./supabase_logger.py

# Copy DVC config and pointers
COPY .dvc/ ./.dvc/
COPY dvc.yaml dvc.lock ./
COPY models/ ./models/
COPY reports/ ./reports/

# Build args para autenticar no DagsHub e baixar o modelo treinado
ARG DAGSHUB_USERNAME
ARG DAGSHUB_TOKEN

RUN dvc remote modify myremote --local auth basic && \
    dvc remote modify myremote --local user ${DAGSHUB_USERNAME} && \
    dvc remote modify myremote --local password ${DAGSHUB_TOKEN} && \
    dvc pull models/ --no-run-cache

# Streamlit (8501)
EXPOSE 8501

CMD ["streamlit", "run", "streamlit_ui.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
