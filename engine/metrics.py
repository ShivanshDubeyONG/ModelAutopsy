from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def infer_problem_type(
    y_true: Any,
    y_pred: Any,
) -> str:
    """
    Infer whether the output is classification or regression.

    Classification is preferred when the target is non-numeric or
    has a small number of unique values. Numeric targets with many
    distinct values are treated as regression.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.dtype.kind in {"O", "U", "S", "b"}:
        return "classification"

    unique_values = np.unique(y_true)

    if len(unique_values) <= 20:
        return "classification"

    return "regression"


def calculate_metrics(
    y_true: Any,
    y_pred: Any,
    probabilities: Any | None = None,
    problem_type: str | None = None,
) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if problem_type is None:
        problem_type = infer_problem_type(
            y_true,
            y_pred,
        )

    if problem_type == "regression":
        return _regression_metrics(
            y_true,
            y_pred,
        )

    return _classification_metrics(
        y_true,
        y_pred,
        probabilities,
    )


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: Any | None = None,
) -> dict:
    result = {
        "problem_type": "classification",
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
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
        "labels": _native_list(
            np.unique(y_true)
        ),
    }

    if probabilities is not None:
        try:
            probabilities = np.asarray(
                probabilities
            )

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
        except (
            ValueError,
            TypeError,
        ):
            pass

    return result


def _regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )

    result = {
        "problem_type": "regression",
        "mae": float(mae),
        "rmse": rmse,
        "r2": float(
            r2_score(
                y_true,
                y_pred,
            )
        ),
    }

    non_zero = y_true != 0

    if np.any(non_zero):
        mape = np.mean(
            np.abs(
                (
                    y_true[non_zero]
                    - y_pred[non_zero]
                )
                / y_true[non_zero]
            )
        )

        result["mape"] = float(
            mape
        )

    return result


def _native_list(values):
    return [
        int(v)
        if isinstance(v, np.integer)
        else float(v)
        if isinstance(v, np.floating)
        else bool(v)
        if isinstance(v, np.bool_)
        else v
        for v in values
    ]