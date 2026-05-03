"""
Registra cada predição na tabela `wine_predictions` do Supabase.

DDL para criar a tabela no Supabase (SQL Editor):
----------------------------------------------------
CREATE TABLE IF NOT EXISTS wine_predictions (
    id                   bigserial PRIMARY KEY,
    created_at           timestamptz NOT NULL DEFAULT now(),
    -- features de entrada
    fixed_acidity        numeric NOT NULL,
    volatile_acidity     numeric NOT NULL,
    citric_acid          numeric NOT NULL,
    residual_sugar       numeric NOT NULL,
    chlorides            numeric NOT NULL,
    free_sulfur_dioxide  numeric NOT NULL,
    total_sulfur_dioxide numeric NOT NULL,
    density              numeric NOT NULL,
    ph                   numeric NOT NULL,
    sulphates            numeric NOT NULL,
    alcohol              numeric NOT NULL,
    -- resultado
    predicted_quality    smallint NOT NULL,  -- 0=Ruim | 1=Médio | 2=Bom
    quality_label        text     NOT NULL,
    prob_ruim            numeric  NOT NULL,
    prob_medio           numeric  NOT NULL,
    prob_bom             numeric  NOT NULL,
    elapsed_ms           numeric  NOT NULL
);
----------------------------------------------------
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_client = None  # inicializado na primeira chamada (lazy)


def _get_client():
    global _client
    if _client is None:
        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_KEY", "")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL e SUPABASE_KEY não configurados.")
        from supabase import create_client

        _client = create_client(url, key)
    return _client


SUPABASE_TABLE = os.getenv("SUPABASE_PREDICTIONS_TABLE", "wine_predictions")


def log_prediction(
    features: dict,
    predicted_quality: int,
    quality_label: str,
    probabilities: dict[str, float],  # {"Ruim": .., "Médio": .., "Bom": ..}
    elapsed_ms: float,
) -> None:
    """
    Insere uma linha em `wine_predictions` no Supabase.
    Erros são apenas logados — nunca propagados para o endpoint.
    """
    try:
        row = {
            **{k: float(v) for k, v in features.items()},
            "predicted_quality": int(predicted_quality),
            "quality_label": quality_label,
            "prob_ruim": float(probabilities.get("Ruim", 0.0)),
            "prob_medio": float(probabilities.get("Médio", 0.0)),
            "prob_bom": float(probabilities.get("Bom", 0.0)),
            "elapsed_ms": float(elapsed_ms),
        }
        _get_client().table(SUPABASE_TABLE).insert(row).execute()
        logger.info("[supabase] Predição registrada: quality=%d", predicted_quality)
    except Exception as exc:
        logger.warning("[supabase] Falha ao registrar predição: %s", exc)
