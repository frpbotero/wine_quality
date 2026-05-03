"""
Streamlit UI — Wine Quality Predictor
Carrega o modelo diretamente do MLflow Model Registry (stage: Production).
"""

import os, time
import numpy as np
import pandas as pd
import requests
import streamlit as st
import mlflow
import mlflow.sklearn

st.set_page_config(page_title="🍷 Wine Quality Predictor", page_icon="🍷", layout="wide")

API_URL          = os.getenv("API_URL",              "http://api:8000")
MLFLOW_URI       = os.getenv("MLFLOW_TRACKING_URI",  "")
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME",     "")
DAGSHUB_TOKEN    = os.getenv("DAGSHUB_TOKEN",        "")
REGISTERED_MODEL = os.getenv("MLFLOW_MODEL_NAME",    "wine-quality")


# ── Carregamento via MLflow Registry ─────────────────────────────────────────

@st.cache_resource(show_spinner="Carregando modelo do MLflow Registry...")
def load_model_from_registry():
    if MLFLOW_URI:
        try:
            mlflow.set_tracking_uri(MLFLOW_URI)
            os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
            os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

            model_uri = f"models:/{REGISTERED_MODEL}/Production"
            model = mlflow.sklearn.load_model(model_uri)

            client = mlflow.MlflowClient()
            versions = client.get_latest_versions(REGISTERED_MODEL, stages=["Production"])
            version_info = versions[0] if versions else None
            return model, model_uri, version_info
        except Exception as e:
            st.warning(f"Nao foi possivel carregar do MLflow: {e}\n\nUsando fallback local.")

    # Fallback local
    import pickle
    pkl_path = os.getenv("MODEL_PATH", "mlp_wine.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            return pickle.load(f), f"local://{pkl_path}", None

    raise RuntimeError("Modelo nao encontrado. Configure MLFLOW_TRACKING_URI ou mlp_wine.pkl.")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["alcohol_sulphates"] = df["alcohol"] * df["sulphates"]
    df["acid_ratio"]        = df["fixed acidity"] / (df["volatile acidity"] + 1e-6)
    df["so2_ratio"]         = df["free sulfur dioxide"] / (df["total sulfur dioxide"] + 1e-6)
    for col in ["residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide"]:
        df[f"log_{col.replace(' ', '_')}"] = np.log1p(df[col])
    return df


def build_input_df(w: dict) -> pd.DataFrame:
    raw = pd.DataFrame([{
        "fixed acidity": w["fixed_acidity"], "volatile acidity": w["volatile_acidity"],
        "citric acid": w["citric_acid"],     "residual sugar": w["residual_sugar"],
        "chlorides": w["chlorides"],          "free sulfur dioxide": w["free_sulfur_dioxide"],
        "total sulfur dioxide": w["total_sulfur_dioxide"], "density": w["density"],
        "pH": w["pH"], "sulphates": w["sulphates"], "alcohol": w["alcohol"], "type": w["type"],
    }])
    return add_features(raw)


# ── Header ───────────────────────────────────────────────────────────────────

st.title("🍷 Wine Quality Predictor")

try:
    model, model_uri, version_info = load_model_from_registry()
    if version_info:
        st.caption(
            f"Modelo `{REGISTERED_MODEL}` v{version_info.version} | "
            f"Stage: **{version_info.current_stage}** | "
            f"Run: `{version_info.run_id[:8]}...`"
        )
    else:
        st.caption(f"Modelo carregado de: `{model_uri}`")
    model_loaded = True
except Exception as err:
    st.error(f"Erro ao carregar modelo: {err}")
    model_loaded = False

st.divider()

# ── Sidebar — inputs ─────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Atributos do Vinho")
    wine_type        = st.selectbox("Tipo", ["red", "white"])
    fixed_acidity    = st.slider("Fixed Acidity",    3.8,  15.9,  7.4,   0.1)
    volatile_acidity = st.slider("Volatile Acidity", 0.08,  1.58,  0.70,  0.01)
    citric_acid      = st.slider("Citric Acid",      0.0,   1.66,  0.0,   0.01)
    residual_sugar   = st.slider("Residual Sugar",   0.6,  65.8,   1.9,   0.1)
    chlorides        = st.slider("Chlorides",        0.009, 0.611, 0.076, 0.001)
    free_so2         = st.slider("Free SO2",         1.0,  289.0, 11.0,   1.0)
    total_so2        = st.slider("Total SO2",        6.0,  440.0, 34.0,   1.0)
    density          = st.slider("Density",          0.9871, 1.039, 0.9978, 0.0001, format="%.4f")
    ph               = st.slider("pH",               2.72,  4.01,  3.51,  0.01)
    sulphates        = st.slider("Sulphates",        0.22,  2.0,   0.56,  0.01)
    alcohol          = st.slider("Alcohol (%)",      8.0,  14.9,   9.4,   0.1)
    st.divider()
    predict_btn  = st.button("Prever Qualidade", use_container_width=True, disabled=not model_loaded)
    use_api      = st.checkbox("Registrar via API (salva no banco)", value=False)

# ── Predicao ─────────────────────────────────────────────────────────────────

col1, col2 = st.columns(2)

if predict_btn:
    wine_input = dict(
        fixed_acidity=fixed_acidity, volatile_acidity=volatile_acidity,
        citric_acid=citric_acid,     residual_sugar=residual_sugar,
        chlorides=chlorides,          free_sulfur_dioxide=free_so2,
        total_sulfur_dioxide=total_so2, density=density,
        pH=ph, sulphates=sulphates, alcohol=alcohol, type=wine_type,
    )

    quality = probabilities = source = None

    if use_api:
        try:
            resp = requests.post(f"{API_URL}/predict", json=wine_input, timeout=10)
            resp.raise_for_status()
            d = resp.json()
            quality, probabilities = d["quality"], d["probabilities"]
            source = f"API ({d['elapsed_ms']} ms, salvo no banco)"
        except requests.exceptions.ConnectionError:
            st.warning("API indisponivel — usando predicao local.")

    if quality is None:
        t0 = time.perf_counter()
        df_in = build_input_df(wine_input)
        quality = int(model.predict(df_in)[0])
        proba   = model.predict_proba(df_in)[0]
        probabilities = {str(c): round(float(p), 4) for c, p in zip(model.classes_, proba)}
        source = f"local ({round((time.perf_counter()-t0)*1000, 2)} ms, MLflow Registry)"

    with col1:
        st.metric("Qualidade Prevista", quality, help="Escala de 3 a 9")
        st.caption("⭐" * max(1, quality - 3))
        if quality <= 4:
            st.error("Qualidade baixa")
        elif quality <= 6:
            st.warning("Qualidade media")
        else:
            st.success("Qualidade alta")
        st.caption(f"Fonte: {source}")

    with col2:
        proba_df = pd.DataFrame.from_dict(probabilities, orient="index", columns=["Prob"]).sort_index()
        st.bar_chart(proba_df, color="#4e8cff")

# ── Historico ────────────────────────────────────────────────────────────────

st.divider()
with st.expander("Historico de Simulacoes (PostgreSQL via API)", expanded=False):
    try:
        r = requests.get(f"{API_URL}/simulations?limit=20", timeout=5)
        r.raise_for_status()
        rows = r.json()
        if rows:
            df_hist = pd.DataFrame(rows)[["id", "created_at", "predicted_quality", "elapsed_ms"]]
            df_hist.columns = ["ID", "Data/Hora", "Qualidade", "ms"]
            st.dataframe(df_hist, use_container_width=True)
        else:
            st.info("Nenhuma simulacao registrada ainda.")
    except Exception as e:
        st.warning(f"API indisponivel: {e}")

# ── Info ─────────────────────────────────────────────────────────────────────

with st.expander("Informacoes do Modelo", expanded=False):
    st.markdown(f"""
| Campo | Valor |
|---|---|
| Registry | `{REGISTERED_MODEL}` |
| Tracking URI | `{MLFLOW_URI or 'nao configurado'}` |
| Model URI | `{model_uri if model_loaded else 'N/A'}` |
| Features | 19 (11 originais + 7 derivadas) |
| Balanceamento | SMOTENC (classes 3, 4, 8, 9) |
""")
