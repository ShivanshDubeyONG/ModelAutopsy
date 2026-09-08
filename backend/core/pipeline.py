from pathlib import Path

import pandas as pd

from backend.core.loader import load_model
from engine.counterfactuals import find_counterfactual
from engine.diagnostics import discover_error_slices
from engine.explanations import (
    global_explanation,
    local_explanation,
)
from engine.findings import (
    build_findings,
    calculate_health_score,
)
from engine.metrics import calculate_metrics


def run_autopsy(
    model_path: str | Path,
    dataset_path: str | Path,
    target_column: str,
) -> dict:
    model = load_model(model_path)

    data = pd.read_csv(dataset_path)

    if target_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' "
            "was not found in the dataset."
        )

    X = data.drop(
        columns=[target_column]
    )

    y = data[target_column]

    predictions = model.predict(X)

    probabilities = None

    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(X)
        except Exception:
            probabilities = None

    metrics = calculate_metrics(
        y,
        predictions,
        probabilities,
    )

    error_slices = discover_error_slices(
        X,
        y,
        predictions,
    )

    importance = global_explanation(
        model,
        X,
    )

    local_index = _worst_prediction_index(
        y,
        predictions,
    )

    local = local_explanation(
        model,
        X,
        local_index,
    )

    counterfactual = find_counterfactual(
        model,
        X,
        local_index,
    )

    findings = build_findings(
        metrics,
        error_slices,
        importance,
    )

    health = calculate_health_score(
        metrics,
        findings,
    )

    return {
        "model": {
            "name": type(model).__name__,
            "problem_type": "classification",
        },
        "dataset": {
            "samples": len(data),
            "features": len(X.columns),
            "feature_names": list(X.columns),
            "target": target_column,
        },
        "health_score": health,
        "metrics": metrics,
        "error_analysis": error_slices,
        "feature_importance": importance,
        "representative_case": local,
        "counterfactual": counterfactual,
        "findings": findings,
    }


def _worst_prediction_index(
    y_true,
    y_pred,
) -> int:
    for index, (true, pred) in enumerate(
        zip(y_true, y_pred)
    ):
        if true != pred:
            return index

    return 0