"""
Wine Quality Prediction API
"""

import pickle
import time
import os
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

import database
from models import Base, SimulationLog

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    database.wait_for_db()            # aguarda Postgres estar pronto
    Base.metadata.create_all(bind=database.engine)
    app.state.model = _load_model()
    yield

app = FastAPI(
    title="Wine Quality API",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

MODEL_PATH = os.getenv("MODEL_PATH", "mlp_wine.pkl")

def _load_model():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Modelo não encontrado em {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class WineFeatures(BaseModel):
    fixed_acidity: float        = Field(..., example=7.4)
    volatile_acidity: float     = Field(..., example=0.70)
    citric_acid: float          = Field(..., example=0.00)
    residual_sugar: float       = Field(..., example=1.9)
    chlorides: float            = Field(..., example=0.076)
    free_sulfur_dioxide: float  = Field(..., example=11.0)
    total_sulfur_dioxide: float = Field(..., example=34.0)
    density: float              = Field(..., example=0.9978)
    pH: float                   = Field(..., example=3.51)
    sulphates: float            = Field(..., example=0.56)
    alcohol: float              = Field(..., example=9.4)
    type: str                   = Field(..., example="red")  # "red" | "white"


class PredictionResponse(BaseModel):
    quality: int
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
# Feature engineering (deve espelhar o notebook)
# ---------------------------------------------------------------------------

def _build_dataframe(wine: WineFeatures) -> pd.DataFrame:
    row = {
        "fixed acidity":        wine.fixed_acidity,
        "volatile acidity":     wine.volatile_acidity,
        "citric acid":          wine.citric_acid,
        "residual sugar":       wine.residual_sugar,
        "chlorides":            wine.chlorides,
        "free sulfur dioxide":  wine.free_sulfur_dioxide,
        "total sulfur dioxide": wine.total_sulfur_dioxide,
        "density":              wine.density,
        "pH":                   wine.pH,
        "sulphates":            wine.sulphates,
        "alcohol":              wine.alcohol,
        "type":                 wine.type,
    }
    df = pd.DataFrame([row])

    # Features derivadas — espelho do notebook
    df["alcohol_sulphates"]  = df["alcohol"] * df["sulphates"]
    df["acid_ratio"]         = df["fixed acidity"] / (df["volatile acidity"] + 1e-6)
    df["so2_ratio"]          = df["free sulfur dioxide"] / (df["total sulfur dioxide"] + 1e-6)
    for col in ["residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide"]:
        df[f"log_{col.replace(' ', '_')}"] = np.log1p(df[col])
    return df

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["infra"])
def health_check():
    """Liveness probe — usada pelo Docker e load balancer."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(wine: WineFeatures, db: Session = Depends(get_db)):
    """Prediz a qualidade do vinho e registra a simulação no banco."""
    model = app.state.model
    t0 = time.perf_counter()

    df = _build_dataframe(wine)

    try:
        quality = int(model.predict(df)[0])
        proba   = model.predict_proba(df)[0]
        classes = [str(c) for c in model.classes_]
        probabilities = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {exc}")

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    # Persiste no banco
    log = SimulationLog(
        input_data=wine.model_dump(),
        predicted_quality=quality,
        probabilities=probabilities,
        elapsed_ms=elapsed_ms,
    )
    db.add(log)
    db.commit()

    return PredictionResponse(
        quality=quality,
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
            "elapsed_ms": r.elapsed_ms,
            "input_data": r.input_data,
        }
        for r in rows
    ]
