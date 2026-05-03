FROM python:3.12-slim

WORKDIR /app

# Install minimal Python dependencies required by the API
COPY requirements-api.txt ./

# Ensure `curl` is available for the Docker healthcheck
RUN apt-get update \
	&& apt-get install -y --no-install-recommends curl \
	&& rm -rf /var/lib/apt/lists/* \
	&& pip install --no-cache-dir -r requirements-api.txt

# Copy application code (main, database, models and model artifact is mounted via compose)
COPY . .

EXPOSE 8000

# Run Uvicorn serving the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]