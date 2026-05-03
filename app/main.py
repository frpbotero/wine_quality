"""
Wine Quality Prediction API + Streamlit UI (single container)

FastAPI roda na porta 8000.
Streamlit é iniciado como subprocesso na porta 8501 no startup.
"""

import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

import database
from models import Base, SimulationLog
import supabase_logger

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

MODEL_PATH = os.getenv(
    "MODEL_PATH", str(Path(__file__).resolve().parent / "best_model.pkl")
)
STREAMLIT_UI = Path(__file__).resolve().parent / "streamlit_ui.py"

# Classes: 0=Ruim (quality<=5), 1=Médio (quality=6), 2=Bom (quality>=7)
CLASS_LABELS = {0: "Ruim", 1: "Médio", 2: "Bom"}

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_model():
    """Carrega o modelo via MLflow Registry ou fallback local (joblib)."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    username = os.getenv("DAGSHUB_USERNAME", "")
    token = os.getenv("DAGSHUB_TOKEN", "")
    model_name = os.getenv("MLFLOW_MODEL_NAME", "wine-quality")

    if tracking_uri and username and token:
        try:
            os.environ["MLFLOW_TRACKING_USERNAME"] = username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = token
            mlflow.set_tracking_uri(tracking_uri)
            return mlflow.sklearn.load_model(f"models:/{model_name}/Production")
        except Exception as exc:
            print(f"[main] MLflow falhou ({exc}), usando fallback local.")

    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    raise RuntimeError(f"Modelo não encontrado em {MODEL_PATH} nem no MLflow Registry.")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

_streamlit_proc: subprocess.Popen | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _streamlit_proc

    # Banco
    database.wait_for_db()
    Base.metadata.create_all(bind=database.engine)

    # Modelo
    app.state.model = _load_model()

    # Streamlit como subprocesso
    if STREAMLIT_UI.exists():
        _streamlit_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(STREAMLIT_UI),
                "--server.port=8501",
                "--server.address=0.0.0.0",
                "--server.headless=true",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"[main] Streamlit iniciado (PID={_streamlit_proc.pid}) na porta 8501")
    else:
        print(
            f"[main] streamlit_ui.py não encontrado em {STREAMLIT_UI}, UI desabilitada."
        )

    yield

    if _streamlit_proc is not None:
        _streamlit_proc.terminate()
        _streamlit_proc.wait()
        print("[main] Streamlit encerrado.")


app = FastAPI(
    title="Wine Quality API",
    version="2.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class WineFeatures(BaseModel):
    fixed_acidity: float = Field(..., examples=[7.4])
    volatile_acidity: float = Field(..., examples=[0.70])
    citric_acid: float = Field(..., examples=[0.00])
    residual_sugar: float = Field(..., examples=[1.9])
    chlorides: float = Field(..., examples=[0.076])
    free_sulfur_dioxide: float = Field(..., examples=[11.0])
    total_sulfur_dioxide: float = Field(..., examples=[34.0])
    density: float = Field(..., examples=[0.9978])
    ph: float = Field(..., examples=[3.51])
    sulphates: float = Field(..., examples=[0.56])
    alcohol: float = Field(..., examples=[9.4])


class PredictionResponse(BaseModel):
    quality: int
    quality_label: str
    probabilities: dict[str, float]
    elapsed_ms: float


# ---------------------------------------------------------------------------
# DB dependency
# ---------------------------------------------------------------------------


def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Feature engineering — espelha preprocessing.py (11 features brutas)
# ---------------------------------------------------------------------------

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


def _build_dataframe(wine: WineFeatures) -> pd.DataFrame:
    return pd.DataFrame([wine.model_dump()])[FEATURE_ORDER]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["infra"])
def health_check():
    """Liveness probe."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(wine: WineFeatures, db: Session = Depends(get_db)):
    """Prediz a qualidade do vinho (0=Ruim, 1=Médio, 2=Bom)."""
    model = app.state.model
    t0 = time.perf_counter()

    df = _build_dataframe(wine)

    try:
        quality = int(model.predict(df)[0])
        proba = model.predict_proba(df)[0]
        classes = [str(c) for c in model.classes_]
        probabilities = {
            CLASS_LABELS.get(int(c), c): round(float(p), 4)
            for c, p in zip(classes, proba)
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {exc}")

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    log = SimulationLog(
        input_data=wine.model_dump(),
        predicted_quality=quality,
        probabilities=probabilities,
        elapsed_ms=elapsed_ms,
    )
    db.add(log)
    db.commit()

    # Registra também no Supabase (falha silenciosa)
    supabase_logger.log_prediction(
        features=wine.model_dump(),
        predicted_quality=quality,
        quality_label=CLASS_LABELS.get(quality, str(quality)),
        probabilities=probabilities,
        elapsed_ms=elapsed_ms,
    )

    return PredictionResponse(
        quality=quality,
        quality_label=CLASS_LABELS.get(quality, str(quality)),
        probabilities=probabilities,
        elapsed_ms=elapsed_ms,
    )


@app.get("/simulations", tags=["history"])
def list_simulations(limit: int = 50, db: Session = Depends(get_db)):
    """Retorna as últimas simulações registradas."""
    rows = (
        db.query(SimulationLog)
        .order_by(SimulationLog.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": r.id,
            "created_at": r.created_at.isoformat(),
            "predicted_quality": r.predicted_quality,
            "quality_label": CLASS_LABELS.get(r.predicted_quality, ""),
            "elapsed_ms": r.elapsed_ms,
            "input_data": r.input_data,
        }
        for r in rows
    ]
