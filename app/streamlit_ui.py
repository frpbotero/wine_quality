"""
Streamlit UI — Wine Quality Classifier (3 classes)
Iniciado como subprocesso pelo FastAPI lifespan em main.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT.parent / ".env")

PROCESSED_PATH = ROOT.parent / "data" / "processed" / "wine_processed.parquet"
MODEL_FALLBACK = ROOT / "best_model.pkl"
API_URL = os.getenv("API_URL", "http://localhost:8000")

CLASS_LABELS = {0: "🔴 Ruim", 1: "🟡 Médio", 2: "🟢 Bom"}

st.set_page_config(page_title="Wine Quality Classifier", page_icon="🍷", layout="wide")


# ── Model loader ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Carregando modelo…")
def load_model():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    username = os.getenv("DAGSHUB_USERNAME", "")
    token = os.getenv("DAGSHUB_TOKEN", "")
    model_name = os.getenv("MLFLOW_MODEL_NAME", "wine-quality")

    if tracking_uri and username and token:
        try:
            os.environ["MLFLOW_TRACKING_USERNAME"] = username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = token
            mlflow.set_tracking_uri(tracking_uri)
            return mlflow.sklearn.load_model(
                f"models:/{model_name}/Production"
            ), "MLflow Registry"
        except Exception:
            pass

    if MODEL_FALLBACK.exists():
        return joblib.load(MODEL_FALLBACK), f"Fallback local ({MODEL_FALLBACK.name})"

    raise RuntimeError("Modelo não encontrado.")


@st.cache_data(show_spinner=False)
def load_eda_df() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_PATH)


FEATURE_ORDER = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "ph",
    "sulphates",
    "alcohol",
]


def _predict_local(payload: dict) -> tuple[int, dict]:
    model, _ = load_model()
    df = pd.DataFrame([payload])[FEATURE_ORDER]
    quality = int(model.predict(df)[0])
    proba = model.predict_proba(df)[0]
    classes = [int(c) for c in model.classes_]
    probs = {
        CLASS_LABELS.get(c, str(c)): round(float(p), 4) for c, p in zip(classes, proba)
    }
    return quality, probs


def _predict_api(payload: dict) -> tuple[int, dict]:
    resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data["quality"], data["probabilities"]


# ── UI ───────────────────────────────────────────────────────────────────────
st.title("🍷 Wine Quality Classifier")

try:
    _, source = load_model()
    st.caption(f"Modelo: {source}")
except Exception as e:
    st.error(f"Modelo não disponível: {e}")

tab_predict, tab_history, tab_eda = st.tabs(["Predição", "Histórico", "EDA"])

# ── Tab: Predição ─────────────────────────────────────────────────────────
with tab_predict:
    st.markdown("Ajuste os parâmetros físico-químicos e clique em **Classificar**.")

    c1, c2, c3 = st.columns(3)
    with c1:
        fixed_acidity = st.slider("fixed acidity", 4.0, 16.0, 7.4, 0.1)
        volatile_acidity = st.slider("volatile acidity", 0.08, 1.60, 0.50, 0.01)
        citric_acid = st.slider("citric acid", 0.0, 1.70, 0.30, 0.01)
        residual_sugar = st.slider("residual sugar", 0.5, 16.0, 2.0, 0.1)
    with c2:
        chlorides = st.slider("chlorides", 0.009, 0.65, 0.08, 0.001)
        free_sulfur_dioxide = st.slider("free sulfur dioxide", 1.0, 72.0, 15.0, 1.0)
        total_sulfur_dioxide = st.slider("total sulfur dioxide", 6.0, 289.0, 46.0, 1.0)
    with c3:
        density = st.slider("density", 0.9870, 1.0040, 0.9960, 0.0001, format="%.4f")
        ph = st.slider("pH", 2.70, 4.00, 3.30, 0.01)
        sulphates = st.slider("sulphates", 0.20, 2.00, 0.60, 0.01)
        alcohol = st.slider("alcohol", 8.0, 15.0, 10.0, 0.1)

    use_api = st.toggle("Usar API REST (/predict)", value=False)

    if "history" not in st.session_state:
        st.session_state.history = []

    if st.button("🔍 Classificar", use_container_width=True):
        payload = {
            "fixed_acidity": fixed_acidity,
            "volatile_acidity": volatile_acidity,
            "citric_acid": citric_acid,
            "residual_sugar": residual_sugar,
            "chlorides": chlorides,
            "free_sulfur_dioxide": free_sulfur_dioxide,
            "total_sulfur_dioxide": total_sulfur_dioxide,
            "density": density,
            "ph": ph,
            "sulphates": sulphates,
            "alcohol": alcohol,
        }

        try:
            if use_api:
                quality, probs = _predict_api(payload)
            else:
                quality, probs = _predict_local(payload)

            label = CLASS_LABELS.get(quality, str(quality))
            st.subheader(f"Qualidade prevista: {label}")

            prob_df = pd.DataFrame(
                {"Classe": list(probs.keys()), "Probabilidade": list(probs.values())}
            ).set_index("Classe")
            st.bar_chart(prob_df)

            st.session_state.history.append(
                {"qualidade": label, "quality_int": quality, **probs, **payload}
            )
        except Exception as exc:
            st.error(f"Erro na predição: {exc}")

# ── Tab: Histórico ────────────────────────────────────────────────────────
with tab_history:
    if st.session_state.get("history"):
        st.dataframe(
            pd.DataFrame(st.session_state.history).tail(20),
            use_container_width=True,
        )
    else:
        # Tentar buscar histórico da API
        try:
            resp = requests.get(f"{API_URL}/simulations?limit=20", timeout=5)
            resp.raise_for_status()
            rows = resp.json()
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.info("Nenhuma simulação registrada ainda.")
        except Exception:
            st.info("Faça uma predição para ver o histórico aqui.")

# ── Tab: EDA ──────────────────────────────────────────────────────────────
with tab_eda:
    if not PROCESSED_PATH.exists():
        st.info("Arquivo processado não encontrado. Rode preprocessing e train antes.")
    else:
        eda_df = load_eda_df()

        st.markdown("### Distribuição das Classes")
        class_counts = (
            eda_df["quality"]
            .map({0: "Ruim(0)", 1: "Médio(1)", 2: "Bom(2)"})
            .value_counts()
        )
        st.bar_chart(class_counts)

        st.markdown("### Distribuições das Features")
        feat_cols = [c for c in eda_df.columns if c != "quality"]
        selected = st.multiselect("Features", feat_cols, default=feat_cols[:4])
        for col in selected:
            st.markdown(f"**{col}**")
            st.bar_chart(eda_df[col], use_container_width=True)

        st.markdown("### Correlação")
        st.dataframe(
            eda_df.corr(numeric_only=True).style.background_gradient(cmap="RdYlGn"),
            use_container_width=True,
        )
