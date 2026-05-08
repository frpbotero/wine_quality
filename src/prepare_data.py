from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = ROOT / "data" / "processed" / "wine_processed.parquet"
SPLITS_DIR = ROOT / "data" / "processed" / "splits"

TARGET = "quality"


def prepare(
    processed_path: Path = PROCESSED_PATH,
    splits_dir: Path = SPLITS_DIR,
) -> None:
    """
    Gera os splits train/val/test a partir do parquet processado.

    Responsabilidades:
    - One-hot encoding de 'type' (red/white) → type_red, type_white
    - Split estratificado 60/20/20
    - Salvar splits BRUTOS (sem StandardScaler)

    ⚠️  O StandardScaler NÃO é aplicado aqui.
        Cada Pipeline em train.py faz seu próprio fit apenas no treino,
        evitando duplicidade e data leakage.
    """
    df = pd.read_parquet(processed_path)
    print(f"[prepare] Dados carregados: {df.shape}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    # ── One-hot encoding para 'type' (red/white) ───────────────────────────────
    if "type" in X.columns:
        X = pd.get_dummies(X, columns=["type"], prefix="type", drop_first=False, dtype=float)
        type_cols = [c for c in X.columns if c.startswith("type_")]
        print(f"[prepare] One-hot encoding aplicado para 'type': {type_cols}")

    feature_cols = X.columns.tolist()
    print(f"[prepare] Features ({len(feature_cols)}): {feature_cols}")
    print(f"[prepare] Distribuição de classes:\n{y.value_counts().sort_index()}\n")

    # ── Split 60 / 20 / 20 (estratificado) ────────────────────────────────────
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    print("Split 60/20/20:")
    print(f"  Treino:    {len(X_train)} ({len(X_train) / len(X) * 100:.1f}%)")
    print(f"  Validação: {len(X_val)} ({len(X_val) / len(X) * 100:.1f}%)")
    print(f"  Teste:     {len(X_test)} ({len(X_test) / len(X) * 100:.1f}%)\n")

    # ── Salvar splits brutos (features + target) ───────────────────────────────
    splits_dir.mkdir(parents=True, exist_ok=True)

    for split_name, X_split, y_split in [
        ("train", X_train, y_train),
        ("val",   X_val,   y_val),
        ("test",  X_test,  y_test),
    ]:
        df_split = X_split.copy()
        df_split[TARGET] = y_split.values
        split_path = splits_dir / f"{split_name}.parquet"
        df_split.to_parquet(split_path, index=False)
        print(f"[prepare] {split_name:5} → {split_path.name}  ({len(df_split)} amostras, {len(feature_cols)} features)")

    print("\n[prepare] ✅ Splits salvos com sucesso!")


if __name__ == "__main__":
    prepare()
