from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(
    y_true: Any,
    y_pred: Any,
    probabilities: Any | None = None,
) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    result = {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
        ).tolist(),
        "labels": _native_list(np.unique(y_true)),
    }

    if probabilities is not None:
        try:
            probabilities = np.asarray(probabilities)

            if (
                probabilities.ndim == 2
                and probabilities.shape[1] == 2
            ):
                result["roc_auc"] = float(
                    roc_auc_score(
                        y_true,
                        probabilities[:, 1],
                    )
                )
        except (ValueError, TypeError):
            pass

    return result


def _native_list(values):
    return [
        int(v) if isinstance(v, np.integer)
        else float(v) if isinstance(v, np.floating)
        else v
        for v in values
    ]