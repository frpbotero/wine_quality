from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

try:
    import dagshub
except Exception:  # pragma: no cover
    dagshub = None


ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = ROOT / "data" / "processed" / "splits"
MODELS_DIR = ROOT / "models" / "trained"
APP_MODEL_PATH = ROOT / "models" / "best_model.pkl"
REPORT_PATH = ROOT / "reports" / "training_report.json"

TARGET = "quality"

# Classes: 0=Ruim (<7), 1=Bom (>=7) — binário
# Features: 11 químicas + type (red/white) one-hot encoded = 13 features totais
CLASS_NAMES = ["Ruim(0)", "Bom(1)"]


# ── DuckDB helpers ─────────────────────────────────────────────────────────────

def _load_splits_via_duckdb() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carrega os splits de treino e validação usando DuckDB.

    Estratégia (em ordem de prioridade):
    1. Postgres do Supabase via DuckDB postgres extension (DATABASE_URL).
       Executa o mesmo SQL de preprocessing inline, sem depender do parquet.
    2. Fallback: lê os parquets locais via DuckDB (equivalente a pd.read_parquet).
    """
    load_dotenv(ROOT / ".env")
    db_url = (os.getenv("DATABASE_URL") or "").strip().strip('"').strip("'")
    table = os.getenv("SUPABASE_TABLE", "wine_quality")

    if db_url and (db_url.startswith("postgresql://") or db_url.startswith("postgres://")):
        print(f"[train] 🦆 DuckDB: conectando ao Supabase Postgres ({table})…")
        try:
            con = duckdb.connect()
            con.execute("INSTALL postgres; LOAD postgres;")
            con.execute(f"ATTACH '{db_url}' AS supabase (TYPE POSTGRES, READ_ONLY);")

            # Preprocessamento inline no DuckDB (mesmo critério do preprocessing.py)
            query = f"""
            WITH src AS (
                SELECT
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
                    CASE WHEN CAST("quality" AS INTEGER) < 7 THEN 0 ELSE 1 END AS quality
                FROM supabase.public."{table}"
                WHERE quality IS NOT NULL
            )
            SELECT * FROM src
            """
            df = con.execute(query).df()
            con.close()

            print(f"[train] 🦆 DuckDB: {len(df)} linhas carregadas do Supabase")

            # Remover linhas com NaN nas features numéricas
            before = len(df)
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            df = df.dropna(subset=[c for c in num_cols if c != TARGET])
            dropped = before - len(df)
            if dropped:
                print(f"[train] ⚠️  {dropped} linhas removidas por NaN nas features")

            # One-hot para 'type'
            if "type" in df.columns:
                df = pd.get_dummies(df, columns=["type"], prefix="type", drop_first=False)

            X = df.drop(columns=[TARGET])
            y = df[TARGET].astype(int)

            from sklearn.model_selection import train_test_split
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=0.20, random_state=42, stratify=y
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
            )

            train_df = X_train.copy(); train_df[TARGET] = y_train.values
            val_df = X_val.copy(); val_df[TARGET] = y_val.values

            print(f"[train] DuckDB split: treino={len(train_df)} | val={len(val_df)}")
            return train_df, val_df

        except Exception as exc:
            print(f"[train] ⚠️  DuckDB/Supabase falhou ({exc}). Usando parquets locais…")

    # Fallback: parquets locais via DuckDB
    train_path = str(SPLITS_DIR / "train.parquet")
    val_path = str(SPLITS_DIR / "val.parquet")

    if not Path(train_path).exists() or not Path(val_path).exists():
        raise FileNotFoundError(
            "Splits não encontrados. Execute 'python src/prepare_data.py' primeiro.\n"
            f"  Train: {train_path}\n"
            f"  Val:   {val_path}"
        )

    print("[train] 🦆 DuckDB: lendo parquets locais…")
    con = duckdb.connect()
    train_df = con.execute(f"SELECT * FROM read_parquet('{train_path}')").df()
    val_df = con.execute(f"SELECT * FROM read_parquet('{val_path}')").df()
    con.close()
    print(f"[train] DuckDB parquet: treino={len(train_df)} | val={len(val_df)}")
    return train_df, val_df


def _setup_mlflow() -> None:
    load_dotenv(ROOT / ".env")
    user = os.getenv("DAGSHUB_USERNAME", "")
    token = os.getenv("DAGSHUB_TOKEN", "")
    repo_name = os.getenv("DAGSHUB_REPO_NAME", "")
    experiment = os.getenv("MLFLOW_EXPERIMENT", "wine-predict")

    if user and token and repo_name:
        # Configure DagsHub remote tracking
        dagshub_uri = f"https://dagshub.com/{user}/{repo_name}.mlflow"
        mlflow.set_tracking_uri(dagshub_uri)
        os.environ["MLFLOW_TRACKING_USERNAME"] = user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        print(f"[train] ✅ MLflow configurado para DagsHub: {dagshub_uri}")
    else:
        # Fallback to local MLflow
        print("[train] ⚠️  Credenciais DagsHub não encontradas. Usando MLflow local em ./mlruns")
        mlflow.set_tracking_uri("./mlruns")

    # Restore deleted experiment or create a new one
    client = mlflow.tracking.MlflowClient()
    existing = client.search_experiments(
        view_type=mlflow.entities.ViewType.DELETED_ONLY
    )
    deleted = [e for e in existing if e.name == experiment]
    if deleted:
        client.restore_experiment(deleted[0].experiment_id)
        print(f"[train] ♻️  Experimento deletado '{experiment}' restaurado.")

    mlflow.set_experiment(experiment)


def _make_preprocessor(num_cols: list[str]) -> ColumnTransformer:
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline as SkPipeline

    num_pipeline = SkPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([("num", num_pipeline, num_cols)])


def _get_classifiers() -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "knn": KNeighborsClassifier(),
        "svm": SVC(random_state=42),
        "xgboost": XGBClassifier(eval_metric="mlogloss", random_state=42),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(200, 100), max_iter=2000, random_state=42
        ),
    }


def _compute_metrics(y_true, y_pred, prefix: str = "") -> dict[str, float]:
    return {
        f"{prefix}accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        f"{prefix}precision": round(
            float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            4,
        ),
        f"{prefix}recall": round(
            float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        ),
        f"{prefix}f1": round(
            float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        ),
    }


def _promote_best(registered_name: str, best_run_id: str) -> None:
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{registered_name}'")
    for v in versions:
        if v.run_id == best_run_id:
            try:
                # MLflow >=2.9: use aliases instead of deprecated stages
                client.set_registered_model_alias(
                    name=registered_name,
                    alias="champion",
                    version=v.version,
                )
                print(
                    f"[train] Alias 'champion' definido para {registered_name} v{v.version}"
                )
            except Exception as exc:
                print(f"[train] Aviso ao promover modelo: {exc}")
            break


def train() -> None:
    _setup_mlflow()

    # ── Load preprocessed splits via DuckDB ────────────────────────────────────
    train_df, val_df = _load_splits_via_duckdb()

    col_names = train_df.drop(columns=[TARGET]).columns.tolist()

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET].values.astype(int)

    X_val = val_df.drop(columns=[TARGET])
    y_val = val_df[TARGET].values.astype(int)

    # ── Garantir sem NaN (imputação pela mediana — sem data leakage) ───────────
    from sklearn.impute import SimpleImputer
    _imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(_imputer.fit_transform(X_train), columns=col_names)
    X_val = pd.DataFrame(_imputer.transform(X_val), columns=col_names)

    nan_count = int(pd.DataFrame(X_train).isna().sum().sum())
    if nan_count == 0:
        print("[train] ✅ Sem NaN após imputação")
    else:
        print(f"[train] ⚠️  Ainda há {nan_count} NaN após imputação — verifique os dados")

    print("[train] Splits carregados:")
    print(f"  Treino: {X_train.shape}")
    print(f"  Validação: {X_val.shape}")
    print(
        f"[train] Distribuição de classes (treino):\n{pd.Series(y_train).value_counts().sort_index()}\n"
    )

    best_score: float = -np.inf
    best_model = None
    best_run_id: str | None = None
    best_registered: str = ""
    metrics_report: dict[str, dict] = {}

    # ══════════════════════════════════════════════════════════════════════════════
    # Estratégia 1 — SMOTE (Over-sampling)
    # ══════════════════════════════════════════════════════════════════════════════
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    # Convert back to DataFrame to preserve column names for ColumnTransformer
    X_train_smote = pd.DataFrame(X_train_smote, columns=col_names)

    print(f"[SMOTE] Amostras após balanceamento: {len(X_train_smote)}")
    print(pd.Series(y_train_smote).value_counts().sort_index().to_string(), "\n")

    STRATEGY = "SMOTE"
    for clf_name, clf in _get_classifiers().items():
        run_name = f"{STRATEGY}_{clf_name}"
        print(f"{'=' * 60}\n  {STRATEGY}  |  {clf_name}")

        pipeline = Pipeline(
            [
                ("preprocessor", _make_preprocessor(col_names)),
                ("classifier", clf),
            ]
        )
        pipeline.fit(X_train_smote, y_train_smote)

        y_val_pred = pipeline.predict(X_val)
        val_metrics = _compute_metrics(y_val, y_val_pred, prefix="val_")

        print(f"  Val  → Acc:{val_metrics['val_accuracy']}  F1:{val_metrics['val_f1']}")
        print(classification_report(y_val, y_val_pred, target_names=CLASS_NAMES))

        metrics_report[run_name] = val_metrics

        # ── Save model ─────────────────────────────────────────────────────────
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"{run_name}.joblib"
        joblib.dump(pipeline, model_path)

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param("strategy", STRATEGY)
            mlflow.log_param("model", clf_name)
            mlflow.log_param("train_samples", int(len(X_train_smote)))
            mlflow.log_param("val_samples", int(len(X_val)))
            mlflow.log_metrics(val_metrics)
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                registered_model_name=f"wine_{STRATEGY}_{clf_name}",
            )
            if val_metrics["val_f1"] > best_score:
                best_score = val_metrics["val_f1"]
                best_model = pipeline
                best_run_id = run.info.run_id
                best_registered = f"wine_{STRATEGY}_{clf_name}"

        print(f"  ✅ Registrado → wine_{STRATEGY}_{clf_name}")
        print(f"  💾 Salvo em: {model_path}\n")

    print("🎉 SMOTE — todos os modelos treinados e salvos!\n")

    # ══════════════════════════════════════════════════════════════════════════════
    # Estratégia 2 — Tomek Links (Under-sampling)
    # ══════════════════════════════════════════════════════════════════════════════
    preprocessor_tomek = _make_preprocessor(col_names)
    X_train_trans = preprocessor_tomek.fit_transform(X_train)
    X_val_trans = preprocessor_tomek.transform(X_val)

    y_train_int = y_train.astype(int)
    y_val_int = y_val.astype(int)

    tomek = TomekLinks()
    X_train_tomek, y_train_tomek = tomek.fit_resample(X_train_trans, y_train_int)

    removed = len(X_train_trans) - len(X_train_tomek)
    print(
        f"[Tomek] Amostras após remoção: {len(X_train_tomek)}  (removidas: {removed})"
    )
    print(pd.Series(y_train_tomek).value_counts().sort_index().to_string(), "\n")

    STRATEGY = "Tomek"
    for clf_name, clf in _get_classifiers().items():
        run_name = f"{STRATEGY}_{clf_name}"
        print(f"{'=' * 60}\n  {STRATEGY}  |  {clf_name}")

        clf.fit(X_train_tomek, y_train_tomek)

        y_val_pred = clf.predict(X_val_trans)
        val_metrics = _compute_metrics(y_val_int, y_val_pred, prefix="val_")

        print(f"  Val  → Acc:{val_metrics['val_accuracy']}  F1:{val_metrics['val_f1']}")
        print(classification_report(y_val_int, y_val_pred, target_names=CLASS_NAMES))

        metrics_report[run_name] = val_metrics

        # Embrulha preprocessor + clf num Pipeline para predição end-to-end.
        # O preprocessor já foi fitado em X_train; combinamos aqui para que
        # predict() receba features brutas (sem pré-escalonamento manual).
        tomek_pipeline = Pipeline([
            ("preprocessor", preprocessor_tomek),
            ("classifier", clf),
        ])

        # ── Save model ─────────────────────────────────────────────────────────
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"{run_name}.joblib"
        joblib.dump(tomek_pipeline, model_path)

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param("strategy", STRATEGY)
            mlflow.log_param("model", clf_name)
            mlflow.log_param("train_samples", int(len(X_train_tomek)))
            mlflow.log_param("val_samples", int(len(X_val_trans)))
            mlflow.log_metrics(val_metrics)
            mlflow.sklearn.log_model(
                tomek_pipeline,
                artifact_path="model",
                registered_model_name=f"wine_{STRATEGY}_{clf_name}",
            )
            if val_metrics["val_f1"] > best_score:
                best_score = val_metrics["val_f1"]
                best_model = tomek_pipeline
                best_run_id = run.info.run_id
                best_registered = f"wine_{STRATEGY}_{clf_name}"

        print(f"  ✅ Registrado → wine_{STRATEGY}_{clf_name}")
        print(f"  💾 Salvo em: {model_path}\n")

    print("🎉 Tomek — todos os modelos treinados e salvos!\n")

    # ══════════════════════════════════════════════════════════════════════════════
    # Salvar melhor modelo + promover no MLflow
    # ══════════════════════════════════════════════════════════════════════════════
    if best_model is None or best_run_id is None:
        raise RuntimeError("Treinamento falhou: nenhum modelo foi salvo.")

    APP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, APP_MODEL_PATH)
    print(
        f"[train] 🏆 Melhor modelo ({best_registered}, F1={best_score:.4f}) salvo em: {APP_MODEL_PATH}"
    )

    _promote_best(best_registered, best_run_id)

    # ── Resumo tabular ─────────────────────────────────────────────────────────
    rows = [
        {
            "run": run_name,
            "val_f1": m.get("val_f1", "-"),
            "val_accuracy": m.get("val_accuracy", "-"),
        }
        for run_name, m in metrics_report.items()
    ]
    print("\n" + pd.DataFrame(rows).set_index("run").to_string())

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(metrics_report, indent=2), encoding="utf-8")
    print(f"\n[train] Relatório salvo em: {REPORT_PATH}")


if __name__ == "__main__":
    train()
