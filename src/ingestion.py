from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(str(ROOT / ".env"), override=True)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TABLE_NAME = os.getenv("SUPABASE_TABLE", "wine_quality")

DEFAULT_SOURCE = ROOT / "data" / "winequality-red.csv"
DEFAULT_RAW_OUT = ROOT / "data" / "raw" / "winequality-red.csv"

BATCH_SIZE = 500


def _get_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)


RENAME_MAP = {
    "ph": "ph",
    "fixed_acidity": "fixed_acidity",
    # adicione aqui qualquer coluna com nome diferente
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda c: c.strip().replace(" ", "_").lower())
    return df.rename(columns=RENAME_MAP)


def _load_csv(path: Path) -> pd.DataFrame:
    """Lê CSV com separador ';' (padrão UCI) ou ',' como fallback."""
    df = pd.read_csv(path, sep=";")
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=",")
    return _normalize_columns(df)


def fetch_wine_data() -> pd.DataFrame:
    client = _get_client()

    # Debug: contar registros via RPC
    count_resp = client.table(TABLE_NAME).select("*", count="exact").limit(1).execute()
    print(f"[debug] Count exato: {count_resp.count}")
    print(f"[debug] Primeiros dados: {count_resp.data}")

    all_rows = []
    page = 0
    page_size = 1000

    while True:
        response = (
            client.table(TABLE_NAME)
            .select("*")
            .range(page * page_size, (page + 1) * page_size - 1)
            .execute()
        )
        batch = response.data
        if not batch:
            break
        all_rows.extend(batch)
        if len(batch) < page_size:
            break
        page += 1

    print(f"[ingestion] Total buscado: {len(all_rows)} linhas")
    return pd.DataFrame(all_rows)
    return pd.DataFrame(all_rows)


def _seed(source_csv: Path) -> None:
    print("[ingestion] Iniciando seed (upsert)…")
    df = _load_csv(source_csv)
    records = df.to_dict(orient="records")
    client = _get_client()

    for i in range(0, len(records), BATCH_SIZE):
        client.table(TABLE_NAME).upsert(
            records[i : i + BATCH_SIZE],
            on_conflict="id",  # ← chave de conflito explícita
        ).execute()
        print(
            f"[ingestion] Seed: {min(i + BATCH_SIZE, len(records))}/{len(records)} linhas"
        )

    print(f"[ingestion] ✅ Seed completo — {len(records)} linhas em '{TABLE_NAME}'")


def ingest(source_csv: Path = DEFAULT_SOURCE, raw_out: Path = DEFAULT_RAW_OUT) -> None:
    raw_out.parent.mkdir(parents=True, exist_ok=True)

    try:
        print("[ingestion] Conectando via API REST (HTTPS)…")
        df = fetch_wine_data()

        # Tabela vazia → popular primeiro, depois buscar
        if df.empty:
            _seed(source_csv)
            df = fetch_wine_data()

        if df.empty:
            raise RuntimeError(
                "Seed executado mas tabela ainda vazia. Verifique o Supabase."
            )

        df = _normalize_columns(df)
        df.to_csv(raw_out, index=False)
        print(f"[ingestion] Shape: {df.shape}")
        print(f"[ingestion] ✅ Dados salvos em: {raw_out}")

    except Exception as e:
        print(f"[ingestion] ⚠️  Fallback local: {e}")
        df = _load_csv(source_csv)
        df.to_csv(raw_out, index=False)
        print(f"[ingestion] Dados salvos localmente em: {raw_out}")


if __name__ == "__main__":
    ingest()
