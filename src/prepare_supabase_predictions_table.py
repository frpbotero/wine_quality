"""
Cria/garante a existência da tabela de predições no Supabase.

Schema baseado no cabeçalho do dataset winequalityN.csv (Kaggle):
  fixed acidity | volatile acidity | citric acid | residual sugar | chlorides |
  free sulfur dioxide | total sulfur dioxide | density | pH | sulphates | alcohol | type | quality

Colunas extras de predição: predicted_quality, quality_label, prob_not_good,
prob_good, elapsed_ms, wine_type.

Requer SUPABASE_DATABASE_URL (ou SUPABASE_DB_URL / DATABASE_URL) apontando para
o Postgres do Supabase com permissão DDL.
"""

from __future__ import annotations

import sys
from datetime import datetime, UTC
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import supabase_logger

MARKER_PATH = ROOT / "reports" / "supabase_predictions_table_ready.txt"


def run() -> None:
    env_path = ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)

    MARKER_PATH.parent.mkdir(parents=True, exist_ok=True)

    ok = supabase_logger.ensure_predictions_table()
    status = "ok" if ok else "failed"
    content = (
        f"status={status}\n"
        f"table={supabase_logger.SUPABASE_TABLE}\n"
        f"timestamp={datetime.now(UTC).isoformat()}\n"
        f"hint=configure SUPABASE_DB_URL/SUPABASE_DATABASE_URL com URL postgres válida\n"
    )
    MARKER_PATH.write_text(content, encoding="utf-8")

    if not ok:
        # Não bloqueia o pipeline: apenas avisa e grava o marcador com status=failed
        print(
            "[prepare_supabase_predictions_table] ⚠️  Não foi possível criar a "
            "tabela de predições no Supabase. "
            "Configure SUPABASE_DB_URL/SUPABASE_DATABASE_URL com URL Postgres válida."
        )
        return

    print(
        f"[prepare_supabase_predictions_table] ✅ Tabela garantida: "
        f"{supabase_logger.SUPABASE_TABLE}"
    )
    print(f"[prepare_supabase_predictions_table] Marker: {MARKER_PATH}")


if __name__ == "__main__":
    run()
