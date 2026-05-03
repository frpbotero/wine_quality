from __future__ import annotations

import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv


st.set_page_config(page_title="Wine Quality (Binary)", page_icon="🍷", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
PROCESSED_PATH = ROOT / "data" / "processed" / "wine_processed.parquet"
MODEL_FALLBACK = ROOT / "app" / "best_model.pkl"
SKEWED_COLUMNS = ["sulphates", "chlorides", "residual_sugar"]

load_dotenv(ENV_PATH)


@st.cache_resource(show_spinner=True)
def load_model():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    username = os.getenv("DAGSHUB_USERNAME", "")
    token = os.getenv("DAGSHUB_TOKEN", "")
    model_name = os.getenv("MLFLOW_MODEL_NAME", "wine-quality-binary")

    if tracking_uri and username and token:
        try:
            os.environ["MLFLOW_TRACKING_USERNAME"] = username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = token
            mlflow.set_tracking_uri(tracking_uri)
            model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
            return model, "MLflow Registry"
        except Exception:
            pass

    if MODEL_FALLBACK.exists():
        return joblib.load(MODEL_FALLBACK), f"Fallback local ({MODEL_FALLBACK.name})"
    raise RuntimeError("Modelo não encontrado no MLflow e sem fallback local.")


@st.cache_data(show_spinner=False)
def load_eda_df() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_PATH)


def _prepare_features(payload: dict[str, float]) -> pd.DataFrame:
    df = pd.DataFrame([payload])
    df["sulphates_log"] = np.log1p(df["sulphates"])
    df["chlorides_log"] = np.log1p(df["chlorides"])
    df["residual_sugar_log"] = np.log1p(df["residual_sugar"])
    return df[
        [
            "alcohol",
            "volatile_acidity",
            "citric_acid",
            "density",
            "sulphates_log",
            "chlorides_log",
            "residual_sugar_log",
        ]
    ]


def predict(payload: dict[str, float]) -> tuple[int, float]:
    model, _ = load_model()
    features = _prepare_features(payload)
    pred = int(model.predict(features)[0])
    proba = float(model.predict_proba(features)[0][1])
    return pred, proba


st.title("Wine Quality Classifier")
model, source = load_model()
st.caption(f"Modelo carregado via: {source}")

tab_predict, tab_eda = st.tabs(["Predição", "EDA"])

with tab_predict:
    c1, c2 = st.columns(2)
    with c1:
        alcohol = st.slider("alcohol", 8.0, 15.0, 10.0, 0.1)
        volatile_acidity = st.slider("volatile acidity", 0.08, 1.60, 0.50, 0.01)
        citric_acid = st.slider("citric acid", 0.0, 1.7, 0.30, 0.01)
        density = st.slider("density", 0.9870, 1.0400, 0.9960, 0.0001, format="%.4f")
    with c2:
        sulphates = st.slider("sulphates", 0.20, 2.10, 0.60, 0.01)
        chlorides = st.slider("chlorides", 0.009, 0.65, 0.08, 0.001)
        residual_sugar = st.slider("residual sugar", 0.5, 16.0, 2.0, 0.1)

    if "history" not in st.session_state:
        st.session_state.history = []

    if st.button("Classificar", use_container_width=True):
        payload = {
            "alcohol": alcohol,
            "volatile_acidity": volatile_acidity,
            "citric_acid": citric_acid,
            "density": density,
            "sulphates": sulphates,
            "chlorides": chlorides,
            "residual_sugar": residual_sugar,
        }
        pred, prob = predict(payload)
        label = "🍷 Boa Qualidade" if pred == 1 else "❌ Não-Boa"
        st.subheader(label)
        st.progress(int(round(prob * 100)))
        st.write(f"Probabilidade de boa qualidade: **{prob:.2%}**")
        st.session_state.history.append({"prediction": pred, "probability": prob, **payload})

    if st.session_state.history:
        st.markdown("### Histórico")
        st.dataframe(pd.DataFrame(st.session_state.history).tail(20), use_container_width=True)

with tab_eda:
    if not PROCESSED_PATH.exists():
        st.info("Arquivo processado não encontrado. Rode preprocessing e train antes.")
    else:
        eda_df = load_eda_df()
        st.markdown("### Distribuições")
        for col in ["alcohol", "volatile_acidity", "citric_acid", "density"]:
            st.bar_chart(eda_df[col], use_container_width=True)
        st.markdown("### Correlação")
        st.dataframe(eda_df.corr(numeric_only=True), use_container_width=True)
