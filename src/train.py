from __future__ import annotations

import json
import os
from pathlib import Path

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
from sklearn.model_selection import train_test_split
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
DATA_PATH = ROOT / "data" / "processed" / "wine_processed.parquet"
APP_MODEL_PATH = ROOT / "app" / "best_model.pkl"
REPORT_PATH = ROOT / "reports" / "model_comparison.json"

TARGET = "quality"

# Classes: 0=Ruim (quality<=5), 1=Médio (quality=6), 2=Bom (quality>=7)
CLASS_NAMES = ["Ruim(0)", "Médio(1)", "Bom(2)"]


def _setup_mlflow() -> str:
    load_dotenv(ROOT / ".env")
    user = os.getenv("DAGSHUB_USERNAME", "")
    token = os.getenv("DAGSHUB_TOKEN", "")
    repo_name = os.getenv("DAGSHUB_REPO_NAME", "")
    experiment = os.getenv("MLFLOW_EXPERIMENT", "wine-quality")

    if user and token and repo_name and dagshub is not None:
        dagshub.auth.add_app_token(token)
        dagshub.init(repo_name=repo_name, repo_owner=user, mlflow=True)
        os.environ["MLFLOW_TRACKING_USERNAME"] = user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    mlflow.set_experiment(experiment)
    return os.getenv("MLFLOW_MODEL_NAME", "wine-quality")


def _make_preprocessor(num_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer([("num", StandardScaler(), num_cols)])


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
                client.transition_model_version_stage(
                    name=registered_name,
                    version=v.version,
                    stage="Production",
                    archive_existing_versions=True,
                )
            except Exception as exc:
                print(f"[train] Aviso ao promover modelo: {exc}")
            break


def train() -> None:
    _setup_mlflow()
    df = pd.read_parquet(DATA_PATH)

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    print(f"[train] Shape: {X.shape}  |  Classes: {sorted(y.unique())}")
    print(f"[train] Distribuição:\n{y.value_counts().sort_index()}\n")

    # ── Split 60 / 20 / 20 ─────────────────────────────────────────────────────
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    print(
        f"[train] Treino={len(X_train)}  Validação={len(X_val)}  Teste={len(X_test)}\n"
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
    X_train_smote = pd.DataFrame(X_train_smote, columns=X_train.columns)
    y_train_smote = np.array(y_train_smote, dtype=int)

    print(f"[SMOTE] Amostras após balanceamento: {len(X_train_smote)}")
    print(pd.Series(y_train_smote).value_counts().sort_index().to_string(), "\n")

    STRATEGY = "SMOTE"
    for clf_name, clf in _get_classifiers().items():
        run_name = f"{STRATEGY}_{clf_name}"
        print(f"{'=' * 60}\n  {STRATEGY}  |  {clf_name}")

        pipeline = Pipeline(
            [
                ("preprocessor", _make_preprocessor(num_cols)),
                ("classifier", clf),
            ]
        )
        pipeline.fit(X_train_smote, y_train_smote)

        y_val_pred = pipeline.predict(X_val)
        y_test_pred = pipeline.predict(X_test)
        val_metrics = _compute_metrics(y_val, y_val_pred, prefix="val_")
        test_metrics = _compute_metrics(y_test, y_test_pred, prefix="test_")
        all_metrics = {**val_metrics, **test_metrics}

        print(f"  Val  → Acc:{val_metrics['val_accuracy']}  F1:{val_metrics['val_f1']}")
        print(
            f"  Test → Acc:{test_metrics['test_accuracy']}  F1:{test_metrics['test_f1']}"
        )
        print(classification_report(y_test, y_test_pred, target_names=CLASS_NAMES))

        metrics_report[run_name] = all_metrics

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param("strategy", STRATEGY)
            mlflow.log_param("model", clf_name)
            mlflow.log_param("train_samples", int(len(X_train_smote)))
            mlflow.log_param("val_samples", int(len(X_val)))
            mlflow.log_param("test_samples", int(len(X_test)))
            mlflow.log_metrics(all_metrics)
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                registered_model_name=f"wine_{STRATEGY}_{clf_name}",
            )
            if test_metrics["test_f1"] > best_score:
                best_score = test_metrics["test_f1"]
                best_model = pipeline
                best_run_id = run.info.run_id
                best_registered = f"wine_{STRATEGY}_{clf_name}"

        print(f"  ✅ Registrado → wine_{STRATEGY}_{clf_name}\n")

    print("🎉 SMOTE — todos os modelos registrados!\n")

    # ══════════════════════════════════════════════════════════════════════════════
    # Estratégia 2 — Tomek Links (Under-sampling)
    # ══════════════════════════════════════════════════════════════════════════════
    preprocessor_tomek = _make_preprocessor(num_cols)
    X_train_trans = preprocessor_tomek.fit_transform(X_train)
    X_val_trans = preprocessor_tomek.transform(X_val)
    X_test_trans = preprocessor_tomek.transform(X_test)

    y_train_int = y_train.values.astype(int)
    y_val_int = y_val.values.astype(int)
    y_test_int = y_test.values.astype(int)

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
        y_test_pred = clf.predict(X_test_trans)
        val_metrics = _compute_metrics(y_val_int, y_val_pred, prefix="val_")
        test_metrics = _compute_metrics(y_test_int, y_test_pred, prefix="test_")
        all_metrics = {**val_metrics, **test_metrics}

        print(f"  Val  → Acc:{val_metrics['val_accuracy']}  F1:{val_metrics['val_f1']}")
        print(
            f"  Test → Acc:{test_metrics['test_accuracy']}  F1:{test_metrics['test_f1']}"
        )
        print(classification_report(y_test_int, y_test_pred, target_names=CLASS_NAMES))

        metrics_report[run_name] = all_metrics

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param("strategy", STRATEGY)
            mlflow.log_param("model", clf_name)
            mlflow.log_param("train_samples", int(len(X_train_tomek)))
            mlflow.log_param("val_samples", int(len(X_val_trans)))
            mlflow.log_param("test_samples", int(len(X_test_trans)))
            mlflow.log_metrics(all_metrics)
            mlflow.sklearn.log_model(
                clf,
                artifact_path="model",
                registered_model_name=f"wine_{STRATEGY}_{clf_name}",
            )
            mlflow.sklearn.log_model(
                preprocessor_tomek,
                artifact_path="preprocessor",
            )
            if test_metrics["test_f1"] > best_score:
                best_score = test_metrics["test_f1"]
                best_model = clf  # type: ignore[assignment]
                best_run_id = run.info.run_id
                best_registered = f"wine_{STRATEGY}_{clf_name}"

        print(f"  ✅ Registrado → wine_{STRATEGY}_{clf_name}\n")

    print("🎉 Tomek — todos os modelos registrados!\n")

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
            "test_f1": m.get("test_f1", "-"),
            "test_accuracy": m.get("test_accuracy", "-"),
        }
        for run_name, m in metrics_report.items()
    ]
    print("\n" + pd.DataFrame(rows).set_index("run").to_string())

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(metrics_report, indent=2), encoding="utf-8")
    print(f"\n[train] Relatório salvo em: {REPORT_PATH}")


if __name__ == "__main__":
    train()
