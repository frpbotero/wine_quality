"""
Streamlit UI — Wine Quality Classifier (3 classes: Ruim / Médio / Bom)
Iniciado como subprocesso pelo FastAPI lifespan em main.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

import supabase_logger

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT.parent / ".env")

MODEL_FALLBACK = ROOT / "models" / "best_model.pkl"
API_URL = os.getenv("API_URL", "http://localhost:8000")
EVALUATION_REPORT_PATH = ROOT / "reports" / "evaluation_report.json"
TRAINING_REPORT_PATH = ROOT / "reports" / "training_report.json"

CLASS_LABELS = {0: "🔴 Ruim", 1: "🟡 Médio", 2: "🟢 Bom"}

WINE_TYPES = {"🍷 Tinto": "red", "🥂 Branco": "white"}

st.set_page_config(page_title="Wine Quality Classifier", page_icon="🍷", layout="wide")


# ── Model loader ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Carregando modelo…")
def load_model():
    """Carrega o modelo do fallback local (joblib)."""

    if MODEL_FALLBACK.exists():
        try:
            model = joblib.load(MODEL_FALLBACK)
            return model, f"✅ Carregado: {MODEL_FALLBACK.name}"
        except Exception as e:
            st.error(f"Erro ao carregar modelo: {e}")
            return None, f"❌ Erro ao carregar {MODEL_FALLBACK.name}"

    st.error("Modelo não encontrado no MLflow nem como fallback local.")
    return None, "❌ Modelo não disponível"


@st.cache_data(show_spinner="Carregando relatório de avaliação…")
def load_evaluation_report() -> dict:
    """Carrega relatório de avaliação gerado por evaluate.py."""
    if EVALUATION_REPORT_PATH.exists():
        with open(EVALUATION_REPORT_PATH) as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner="Carregando relatório de treinamento…")
def load_training_report() -> dict:
    """Carrega relatório de treinamento gerado por train.py."""
    if TRAINING_REPORT_PATH.exists():
        with open(TRAINING_REPORT_PATH) as f:
            return json.load(f)
    return {}


FEATURE_ORDER_RAW = [
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
    "type_red",
    "type_white",
]


def _get_expected_features(model) -> list[str]:
    """Detecta as features esperadas pelo modelo (pipeline ou estimador direto)."""
    for obj in [
        model,
        getattr(model, "steps", [[None, None]])[0][1]
        if hasattr(model, "steps")
        else None,
    ]:
        if obj is not None and hasattr(obj, "feature_names_in_"):
            return list(obj.feature_names_in_)
    # Fallback: tenta o transformer dentro do pipeline
    try:
        return list(model.named_steps["preprocessor"].transformers_[0][2])
    except Exception:
        return FEATURE_ORDER_RAW


def _prepare_features(payload: dict, model=None) -> pd.DataFrame:
    """Prepara features para o modelo com transformações (11 features químicas + type one-hot encoded + log transforms)."""
    df = pd.DataFrame([payload])
    
    # One-hot encoding para 'type' (red/white)
    if "type" in df.columns:
        df = pd.get_dummies(df, columns=["type"], prefix="type", drop_first=False)
    
    # Aplicar as mesmas transformações que em main.py
    df["sulphates_log"] = np.log1p(df["sulphates"])
    df["chlorides_log"] = np.log1p(df["chlorides"])
    df["residual_sugar_log"] = np.log1p(df["residual_sugar"])
    
    if model is not None:
        expected = _get_expected_features(model)
        return df[expected]
    
    # Fallback: retorna as 11 features químicas + type one-hot encoded
    return df[
        [
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
            "type_red",
            "type_white",
        ]
    ]


def _predict_local(payload: dict) -> tuple[int, list[float], list[int]]:
    model, source = load_model()
    if model is None:
        raise RuntimeError(f"Modelo não disponível: {source}")
    features = _prepare_features(payload, model)
    quality = int(model.predict(features)[0])
    proba = list(map(float, model.predict_proba(features)[0]))
    classes = [int(c) for c in model.classes_]
    return quality, proba, classes


RAW_FEATURES = [
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
    "type",
]


def _predict_api(payload: dict) -> tuple[int, list[float], list[int]]:
    api_payload = {k: payload[k] for k in RAW_FEATURES}
    resp = requests.post(f"{API_URL}/predict", json=api_payload, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    quality = data["quality"]
    probs_dict: dict = data["probabilities"]
    # API labels (no emoji) → int class index via CLASS_LABELS_API
    # CLASS_LABELS values may have emojis; strip to match API output
    api_label_to_int = {v.split(" ", 1)[-1].strip(): k for k, v in CLASS_LABELS.items()}
    # Also try exact match first
    api_label_to_int.update({v: k for k, v in CLASS_LABELS.items()})
    classes = sorted(
        api_label_to_int[lbl] for lbl in probs_dict if lbl in api_label_to_int
    )
    if not classes:
        # Fallback: use positional order returned by the API
        classes = list(range(len(probs_dict)))
    proba = [list(probs_dict.values())[i] for i in range(len(classes))]
    return quality, proba, classes


# ── UI ───────────────────────────────────────────────────────────────────────
st.title("🍷 Wine Quality Classifier")

wine_type_label = st.radio("Tipo de vinho", list(WINE_TYPES.keys()), horizontal=True)
wine_type = WINE_TYPES[wine_type_label]

try:
    _, source = load_model()
    st.sidebar.success(f"✅ Modelo: {source}")
except Exception as e:
    st.sidebar.error(f"Modelo não disponível: {e}")

# ── Sidebar com informações ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 Modelo em Uso")
    
    training_report = load_training_report()
    evaluation_report = load_evaluation_report()
    
    if training_report:
        # Encontrar melhor modelo
        best_model_name = None
        best_f1 = 0
        for name, metrics in training_report.items():
            if metrics.get("val_f1", 0) > best_f1:
                best_f1 = metrics.get("val_f1", 0)
                best_model_name = name
        
        if best_model_name:
            # Extrair nome e estratégia
            parts = best_model_name.split("_", 1)
            strategy = parts[0] if parts else "Unknown"
            model_type = "_".join(parts[1:]) if len(parts) > 1 else "Unknown"
            
            st.markdown(f"**📛 Nome:** `{best_model_name}`")
            st.markdown(f"**⚙️ Estratégia:** `{strategy}`")
            st.markdown(f"**🧠 Algoritmo:** `{model_type}`")
            
            st.markdown("---")
            st.markdown("#### 📊 Métricas de Validação")
            
            train_metrics = training_report[best_model_name]
            col1, col2 = st.columns(2)
            with col1:
                st.metric("✅ Acurácia", f"{train_metrics.get('val_accuracy', 0):.4f}")
                st.metric("📈 Recall", f"{train_metrics.get('val_recall', 0):.4f}")
            with col2:
                st.metric("🎯 F1-Score", f"{train_metrics.get('val_f1', 0):.4f}")
                st.metric("🔍 Precisão", f"{train_metrics.get('val_precision', 0):.4f}")
            
            if best_model_name in evaluation_report:
                st.markdown("---")
                st.markdown("#### 🏆 Métricas de Teste")
                test_metrics = evaluation_report[best_model_name]
                col3, col4 = st.columns(2)
                with col3:
                    st.metric("✅ Acurácia", f"{test_metrics.get('test_accuracy', 0):.4f}")
                    st.metric("📈 Recall", f"{test_metrics.get('test_recall', 0):.4f}")
                with col4:
                    st.metric("🎯 F1-Score", f"{test_metrics.get('test_f1', 0):.4f}")
                    st.metric("🔍 Precisão", f"{test_metrics.get('test_precision', 0):.4f}")
    else:
        st.info("📊 Execute `python src/train.py` para gerar relatório.")

tab_predict, tab_history = st.tabs(
    ["🔮 Predição", "📜 Histórico"]
)

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
                quality, proba, classes = _predict_api(payload)
            else:
                quality, proba, classes = _predict_local(payload)

            label = CLASS_LABELS.get(quality, str(quality))
            q_idx = classes.index(quality) if quality in classes else 0
            confidence = proba[q_idx]

            st.subheader(f"Qualidade prevista: {label}")
            st.metric("Confiança da predição", f"{confidence:.1%}")

            prob_df = pd.DataFrame(
                {"Probabilidade": proba},
                index=[CLASS_LABELS.get(c, str(c)) for c in classes],
            )
            st.bar_chart(prob_df)

            st.session_state.history.append(
                {
                    "tipo": wine_type_label,
                    "qualidade": label,
                    "confiança": f"{confidence:.1%}",
                    **{
                        CLASS_LABELS.get(classes[i], str(classes[i])): f"{proba[i]:.1%}"
                        for i in range(len(proba))
                    },
                    **payload,
                }
            )
            
            # Salvar no Supabase
            try:
                proba_dict = {
                    CLASS_LABELS.get(classes[i], str(classes[i])): proba[i]
                    for i in range(len(proba))
                }
                supabase_logger.log_prediction(
                    features=payload,
                    predicted_quality=quality,
                    quality_label=label.replace("🔴 ", "").replace("🟡 ", "").replace("🟢 ", ""),
                    probabilities=proba_dict,
                    elapsed_ms=0
                )
                st.success("✅ Predição salva no Supabase!")
            except Exception as e:
                st.warning(f"⚠️ Não foi possível salvar no Supabase: {e}")
                
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
