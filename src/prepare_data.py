from __future__ import annotations

import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = ROOT / "data" / "processed" / "wine_processed.parquet"
SPLITS_DIR = ROOT / "data" / "processed" / "splits"
PREPROCESSOR_PATH = ROOT / "models" / "preprocessors" / "preprocessor.joblib"

TARGET = "quality"


def prepare(
    processed_path: Path = PROCESSED_PATH,
    splits_dir: Path = SPLITS_DIR,
    preprocessor_path: Path = PREPROCESSOR_PATH,
) -> None:
    """
    Prepare splits and preprocessor:
    1. Load processed data
    2. Split 60/20/20 (train/val/test) stratified
    3. Build preprocessor (StandardScaler on numeric columns)
    4. Fit preprocessor on train, transform all splits
    5. Save train/val/test.parquet + preprocessor.joblib

    ⚠️  CRITICAL: No data leakage — preprocessor fitted ONLY on train
    """
    # ── Load data ──────────────────────────────────────────────────────────────
    df = pd.read_parquet(processed_path)
    print(f"[prepare] Dados carregados: {df.shape}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    print(f"[prepare] Colunas numéricas: {len(num_cols)}")
    print(f"[prepare] Distribuição de classes:\n{y.value_counts().sort_index()}\n")

    # ── Split 60 / 20 / 20 ─────────────────────────────────────────────────────
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    print("Split 60/20/20:")
    print(f"  Treino:     {len(X_train)} ({len(X_train) / len(X) * 100:.1f}%)")
    print(f"  Validação:  {len(X_val)} ({len(X_val) / len(X) * 100:.1f}%)")
    print(f"  Teste:      {len(X_test)} ({len(X_test) / len(X) * 100:.1f}%)\n")

    # ── Build preprocessor ─────────────────────────────────────────────────────
    preprocessor = ColumnTransformer([("num", StandardScaler(), num_cols)])

    # ── Fit on TRAIN ONLY (critical for no data leakage) ──────────────────────
    print("[prepare] Fitting preprocessor on train split…")
    X_train_p = preprocessor.fit_transform(X_train)
    X_val_p = preprocessor.transform(X_val)
    X_test_p = preprocessor.transform(X_test)
    print("[prepare] ✅ Preprocessor fitted and applied\n")

    # ── Save splits to parquet ─────────────────────────────────────────────────
    splits_dir.mkdir(parents=True, exist_ok=True)

    for split_name, X_p, y_s in [
        ("train", X_train_p, y_train),
        ("val", X_val_p, y_val),
        ("test", X_test_p, y_test),
    ]:
        df_split = pd.DataFrame(X_p, columns=num_cols)
        df_split[TARGET] = y_s.values if hasattr(y_s, "values") else y_s

        split_path = splits_dir / f"{split_name}.parquet"
        df_split.to_parquet(split_path, index=False)
        print(
            f"[prepare] {split_name:8} → {split_path.name:15}  ({len(df_split)} samples)"
        )

    # ── Save preprocessor ──────────────────────────────────────────────────────
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, preprocessor_path)
    print(f"\n[prepare] 🔧 Preprocessor salvo em: {preprocessor_path}")
    print("[prepare] ✅ Preparação concluída com sucesso!")


if __name__ == "__main__":
    prepare()
