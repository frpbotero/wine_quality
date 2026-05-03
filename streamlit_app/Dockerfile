FROM python:3.12-slim

WORKDIR /app

COPY streamlit_app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY streamlit_app/app.py ./app.py
COPY app/best_model.pkl /app/best_model.pkl

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
