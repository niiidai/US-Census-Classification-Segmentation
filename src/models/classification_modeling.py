"""Production XGBoost classification training for the Census income dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


def load_modeling_artifacts(artifact_dir: Path) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Load engineered features, target, and Census weights."""
    X = pd.read_pickle(artifact_dir / "X.pkl")
    y = pd.read_pickle(artifact_dir / "y.pkl")
    weight_df = pd.read_pickle(artifact_dir / "weight.pkl")
    return X, y, weight_df.values.ravel()


def make_train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    weight: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, np.ndarray, np.ndarray, np.ndarray]:
    """Create stratified train, validation, and test splits."""
    X_train_full, X_test, y_train_full, y_test, weight_train_full, weight_test = train_test_split(
        X,
        y,
        weight,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    X_train, X_val, y_train, y_val, weight_train, weight_val = train_test_split(
        X_train_full,
        y_train_full,
        weight_train_full,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train_full,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, weight_train, weight_val, weight_test


def make_imbalance_aware_weights(y_train: pd.Series, weight_train: np.ndarray) -> np.ndarray:
    """Combine survey weights with class-balance weights."""
    class_weight_train = compute_sample_weight(class_weight="balanced", y=y_train)
    return weight_train * class_weight_train


def train_baseline_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    final_train_weight: np.ndarray,
    weight_val: np.ndarray,
) -> XGBClassifier:
    """Train a strong baseline XGBoost model."""
    model = XGBClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train,
        y_train,
        sample_weight=final_train_weight,
        eval_set=[(X_val, y_val)],
        sample_weight_eval_set=[weight_val],
        verbose=False,
    )
    return model


def evaluate_ranking(
    model: XGBClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    weight_val: np.ndarray,
) -> dict[str, float]:
    """Evaluate probability ranking quality."""
    proba = model.predict_proba(X_val)[:, 1]
    return {
        "roc_auc": roc_auc_score(y_val, proba, sample_weight=weight_val),
        "pr_auc": average_precision_score(y_val, proba, sample_weight=weight_val),
    }


def tune_xgb_with_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    final_train_weight: np.ndarray,
    weight_val: np.ndarray,
    n_trials: int = 30,
) -> optuna.Study:
    """Tune XGBoost hyperparameters using validation PR-AUC."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": 800,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": -1,
        }
        model = XGBClassifier(**params)
        model.fit(
            X_train,
            y_train,
            sample_weight=final_train_weight,
            eval_set=[(X_val, y_val)],
            sample_weight_eval_set=[weight_val],
            verbose=False,
        )
        val_proba = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, val_proba, sample_weight=weight_val)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study


def train_final_xgb(
    best_params: dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    final_train_weight: np.ndarray,
    weight_val: np.ndarray,
) -> XGBClassifier:
    """Train the final tuned XGBoost model."""
    model = XGBClassifier(
        n_estimators=800,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        **best_params,
    )
    model.fit(
        X_train,
        y_train,
        sample_weight=final_train_weight,
        eval_set=[(X_val, y_val)],
        sample_weight_eval_set=[weight_val],
        verbose=False,
    )
    return model


def tune_threshold(
    y_true: pd.Series,
    proba: np.ndarray,
    sample_weight: np.ndarray,
    metric: str = "f1",
) -> tuple[float, pd.DataFrame]:
    """Tune a probability threshold on validation predictions."""
    rows = []
    for threshold in np.arange(0.05, 0.95, 0.01):
        pred = (proba >= threshold).astype(int)
        precision = precision_score(y_true, pred, sample_weight=sample_weight, zero_division=0)
        recall = recall_score(y_true, pred, sample_weight=sample_weight, zero_division=0)
        f1 = f1_score(y_true, pred, sample_weight=sample_weight, zero_division=0)
        rows.append({"threshold": threshold, "precision": precision, "recall": recall, "f1": f1})

    threshold_df = pd.DataFrame(rows)
    best_threshold = float(threshold_df.loc[threshold_df[metric].idxmax(), "threshold"])
    return best_threshold, threshold_df


def evaluate_classifier(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    weight_test: np.ndarray,
    threshold: float,
) -> tuple[dict[str, float], np.ndarray]:
    """Evaluate final classifier on the test set."""
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(y_test, proba, sample_weight=weight_test),
        "pr_auc": average_precision_score(y_test, proba, sample_weight=weight_test),
        "precision": precision_score(y_test, pred, sample_weight=weight_test, zero_division=0),
        "recall": recall_score(y_test, pred, sample_weight=weight_test, zero_division=0),
        "f1": f1_score(y_test, pred, sample_weight=weight_test, zero_division=0),
    }
    print(classification_report(
        y_test,
        pred,
        sample_weight=weight_test,
        target_names=["<= 50000", "> 50000"],
        zero_division=0,
    ))
    return metrics, confusion_matrix(y_test, pred, sample_weight=weight_test)


def save_outputs(
    output_dir: Path,
    model: XGBClassifier,
    study: optuna.Study,
    threshold: float,
    test_metrics: dict[str, float],
    confusion: np.ndarray,
) -> None:
    """Save model and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "final_xgb_model.joblib")
    np.save(output_dir / "xgb_confusion_matrix.npy", confusion)
    metadata = {
        "best_params": study.best_params,
        "best_validation_pr_auc": float(study.best_value),
        "best_threshold": float(threshold),
        **{f"test_{k}": float(v) for k, v in test_metrics.items()},
    }
    with (output_dir / "xgb_model_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tuned XGBoost Census classifier.")
    parser.add_argument("--artifact-dir", type=Path, default=Path("../artifacts"))
    parser.add_argument("--output-dir", type=Path, default=Path("../model_artifacts"))
    parser.add_argument("--n-trials", type=int, default=30)
    args = parser.parse_args()

    X, y, weight = load_modeling_artifacts(args.artifact_dir)
    split = make_train_val_test_split(X, y, weight)
    X_train, X_val, X_test, y_train, y_val, y_test, weight_train, weight_val, weight_test = split
    final_train_weight = make_imbalance_aware_weights(y_train, weight_train)

    baseline = train_baseline_xgb(X_train, y_train, X_val, y_val, final_train_weight, weight_val)
    print("Baseline validation metrics:", evaluate_ranking(baseline, X_val, y_val, weight_val))

    study = tune_xgb_with_optuna(
        X_train,
        y_train,
        X_val,
        y_val,
        final_train_weight,
        weight_val,
        n_trials=args.n_trials,
    )
    final_model = train_final_xgb(
        study.best_params,
        X_train,
        y_train,
        X_val,
        y_val,
        final_train_weight,
        weight_val,
    )
    val_proba = final_model.predict_proba(X_val)[:, 1]
    threshold, threshold_df = tune_threshold(y_val, val_proba, weight_val)
    test_metrics, confusion = evaluate_classifier(final_model, X_test, y_test, weight_test, threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    threshold_df.to_csv(args.output_dir / "xgb_threshold_results.csv", index=False)
    save_outputs(args.output_dir, final_model, study, threshold, test_metrics, confusion)

    print("Best params:", study.best_params)
    print("Best threshold:", threshold)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
