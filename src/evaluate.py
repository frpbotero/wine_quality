from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

try:
    import dagshub
except Exception:  # pragma: no cover
    dagshub = None


ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = ROOT / "data" / "processed" / "splits"
MODELS_DIR = ROOT / "models" / "trained"
REPORT_PATH = ROOT / "reports" / "evaluation_report.json"

TARGET = "quality"
CLASS_NAMES = ["Ruim(0)", "Bom(1)"]


def _setup_mlflow() -> None:
    """Setup MLflow connection - using local tracking."""
    import os

    load_dotenv(ROOT / ".env")
    experiment = os.getenv("MLFLOW_EXPERIMENT", "wine-predict")

    # Use local MLflow for evaluation
    print("[evaluate] ℹ️ Using local MLflow tracking at ./mlruns")
    print("[evaluate] 💡 To push to DagsHub: dvc push && dvc dag")
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment(experiment)


def _compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, prefix: str = ""
) -> dict[str, float]:
    """Compute evaluation metrics."""
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


def _save_confusion_matrix_plot(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str
) -> Path:
    """Generate and save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        yticklabels=CLASS_NAMES,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    fig.tight_layout()

    plot_path = ROOT / "reports" / f"confusion_matrix_{model_name}.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def _save_roc_plot(
    y_true: np.ndarray, y_pred_proba: np.ndarray, model_name: str
) -> Path:
    """Generate and save ROC curve plot (one-vs-rest for multiclass)."""
    from sklearn.preprocessing import label_binarize

    y_bin = label_binarize(y_true, classes=list(range(len(CLASS_NAMES))))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(len(CLASS_NAMES)):
        try:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
            auc = roc_auc_score(y_bin[:, i], y_pred_proba[:, i])
            ax.plot(fpr, tpr, label=f"{CLASS_NAMES[i]} (AUC={auc:.3f})")
        except Exception:
            pass

    ax.plot([0, 1], [0, 1], "k--", lw=2, label="Random")
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"ROC Curve - {model_name}",
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    plot_path = ROOT / "reports" / f"roc_curve_{model_name}.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def _get_model_features(model) -> list[str] | None:
    """
    Detecta a lista de features esperadas pelo modelo.
    Funciona para Pipeline (SMOTE) e estimadores diretos (Tomek).
    """
    # Pipeline: verifica o primeiro step (preprocessor ColumnTransformer)
    if hasattr(model, "named_steps"):
        preprocessor = model.named_steps.get("preprocessor")
        if preprocessor is not None and hasattr(preprocessor, "feature_names_in_"):
            return list(preprocessor.feature_names_in_)
        # ColumnTransformer: pega colunas do transformador numérico
        if preprocessor is not None and hasattr(preprocessor, "transformers_"):
            try:
                return list(preprocessor.transformers_[0][2])
            except Exception:
                pass
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
    # Estimador direto: n_features_in_ sem nomes → não conseguimos inferir
    return None


def _align_features(X: pd.DataFrame, model) -> pd.DataFrame:
    """
    Alinha o DataFrame de features com o que o modelo espera:
    - Adiciona colunas ausentes com valor 0
    - Remove colunas extras
    - Reordena para a ordem correta
    Se não for possível detectar as features esperadas, retorna X inalterado.
    """
    expected = _get_model_features(model)
    if expected is None:
        return X

    missing = set(expected) - set(X.columns)
    if missing:
        print(f"  ℹ️  Adicionando colunas ausentes com 0: {sorted(missing)}")
        for col in missing:
            X = X.copy()
            X[col] = 0

    # Seleciona e reordena apenas as colunas esperadas
    return X[expected]


def evaluate() -> None:
    """
    Evaluate all trained models on test set.

    1. Load test split
    2. Load all trained models
    3. Evaluate each model
    4. Log metrics and plots to MLflow
    5. Generate evaluation report
    """
    _setup_mlflow()

    # ── Load test set ──────────────────────────────────────────────────────────
    test_path = SPLITS_DIR / "test.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"Test split not found: {test_path}")

    test_df = pd.read_parquet(test_path)
    X_test = test_df.drop(columns=[TARGET])  # Keep as DataFrame for pipelines
    y_test = test_df[TARGET].values.astype(int)

    print(f"[evaluate] Test set carregado: {X_test.shape}")
    print(
        f"[evaluate] Distribuição:\n{pd.Series(y_test).value_counts().sort_index()}\n"
    )

    # ── Find all trained models ────────────────────────────────────────────────
    if not MODELS_DIR.exists():
        raise FileNotFoundError(f"Models directory not found: {MODELS_DIR}")

    model_files = sorted(MODELS_DIR.glob("*.joblib"))
    if not model_files:
        raise FileNotFoundError(f"No trained models found in {MODELS_DIR}")

    print(f"[evaluate] Encontrados {len(model_files)} modelos treinados")
    print("Models found:")
    for mf in model_files:
        print(f"  - {mf.name}")
    print()

    # ── Evaluate each model ────────────────────────────────────────────────────
    evaluation_results: dict[str, dict] = {}

    for model_path in model_files:
        model_name = model_path.stem  # Remove .joblib extension

        try:
            model = joblib.load(model_path)
            print(f"{'=' * 70}")
            print(f"  {model_name}")
            print(f"{'=' * 70}")

            # Alinhar features do test com as esperadas pelo modelo
            X_test_aligned = _align_features(X_test, model)

            # Predictions
            y_pred = model.predict(X_test_aligned)

            # Metrics
            metrics = _compute_metrics(y_test, y_pred, prefix="test_")
            print(f"  Accuracy: {metrics['test_accuracy']:.4f}")
            print(f"  F1:       {metrics['test_f1']:.4f}")
            print(f"  Precision: {metrics['test_precision']:.4f}")
            print(f"  Recall:   {metrics['test_recall']:.4f}\n")

            print("  Classification Report:")
            print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

            evaluation_results[model_name] = metrics

            # ── Log to MLflow ──────────────────────────────────────────────────
            with mlflow.start_run(run_name=f"{model_name}_test_eval") as run:
                mlflow.set_tag("stage", "test_evaluation")
                mlflow.log_params({"model_name": model_name})
                mlflow.log_metrics(metrics)

                # Log plots
                try:
                    cm_plot = _save_confusion_matrix_plot(y_test, y_pred, model_name)
                    mlflow.log_artifact(str(cm_plot), artifact_path="plots")
                    print(f"  📊 Confusion matrix saved: {cm_plot.name}")
                except Exception as e:
                    print(f"  ⚠️  Confusion matrix failed: {e}")

                try:
                    if hasattr(model, "predict_proba"):
                        y_proba = model.predict_proba(X_test_aligned)
                        roc_plot = _save_roc_plot(y_test, y_proba, model_name)
                        mlflow.log_artifact(str(roc_plot), artifact_path="plots")
                        print(f"  📊 ROC curve saved: {roc_plot.name}")
                except Exception as e:
                    print(f"  ⚠️  ROC curve failed: {e}")

                print(f"  ✅ Logged to MLflow (run_id: {run.info.run_id})\n")

        except Exception as e:
            print(f"  ❌ Error evaluating {model_name}: {e}\n")

    # ── Save evaluation report ─────────────────────────────────────────────────
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(evaluation_results, indent=2), encoding="utf-8")
    print(f"\n[evaluate] 📄 Relatório salvo em: {REPORT_PATH}\n")

    # ── Garantir placeholder de plots para DVC (evita erro de output ausente) ──
    reports_dir = ROOT / "reports"
    for glob_name in ["confusion_matrix_placeholder", "roc_curve_placeholder"]:
        placeholder = reports_dir / f"{glob_name}.png"
        if not list(reports_dir.glob(f"{glob_name.split('_')[0]}_{glob_name.split('_')[1]}_*.png")) and not placeholder.exists():
            # Só cria se não existe nenhum PNG real do tipo
            pass  # PNG real criado pelos modelos avaliados com sucesso

    # Verificar se foi gerado ao menos um PNG de cada tipo
    cm_pngs = list(reports_dir.glob("confusion_matrix_*.png"))
    roc_pngs = list(reports_dir.glob("roc_curve_*.png"))
    if not cm_pngs:
        print("[evaluate] ⚠️  Nenhum confusion matrix gerado — todos os modelos falharam")
    if not roc_pngs:
        print("[evaluate] ⚠️  Nenhum ROC curve gerado — todos os modelos falharam")

    # ── Summary table ──────────────────────────────────────────────────────────
    if evaluation_results:
        df_results = pd.DataFrame(evaluation_results).T
        df_results = df_results[
            ["test_accuracy", "test_f1", "test_precision", "test_recall"]
        ]
        print("[evaluate] Resumo de Modelos:")
        print(df_results.to_string())
        print("\n[evaluate] ✅ Avaliação concluída!")


if __name__ == "__main__":
    evaluate()
