from typing import Any

import numpy as np
import pandas as pd


def find_counterfactual(
    model: Any,
    X: pd.DataFrame,
    index: int,
    max_changes: int = 3,
) -> dict:
    """
    Find a simple counterfactual for a binary classification model.

    Multi-output models are intentionally rejected here because a
    counterfactual must have a clearly defined target output.
    """

    # Multi-output models need output-specific counterfactual logic.
    if hasattr(model, "estimators_"):
        estimators = getattr(
            model,
            "estimators_",
            None,
        )

        if estimators is not None:
            return {
                "found": False,
                "reason": (
                    "Counterfactual search for multi-output "
                    "models is not implemented yet."
                ),
            }

    if not hasattr(model, "predict_proba"):
        return {
            "found": False,
            "reason": "Model does not expose predict_proba().",
        }

    classes = list(
        getattr(model, "classes_", [])
    )

    if len(classes) != 2:
        return {
            "found": False,
            "reason": (
                "Counterfactual search currently "
                "requires binary classification."
            ),
        }

    if len(X) == 0:
        return {
            "found": False,
            "reason": "Dataset contains no rows.",
        }

    index = max(
        0,
        min(index, len(X) - 1),
    )

    original = X.iloc[[index]].copy()

    original_prediction = model.predict(
        original
    )[0]

    desired = (
        classes[0]
        if original_prediction == classes[1]
        else classes[1]
    )

    try:
        original_probability = float(
            model.predict_proba(original)[0][
                list(classes).index(
                    original_prediction
                )
            ]
        )
    except Exception:
        original_probability = None

    best = original.copy()

    changes = []

    for column in X.columns:
        series = X[column]

        # -----------------------------------------------------
        # Numeric feature
        # -----------------------------------------------------

        if pd.api.types.is_numeric_dtype(series):
            candidates = (
                series.quantile(
                    [0.1, 0.25, 0.5, 0.75, 0.9]
                )
                .drop_duplicates()
                .tolist()
            )

        # -----------------------------------------------------
        # Categorical feature
        # -----------------------------------------------------

        else:
            candidates = (
                series.dropna()
                .astype(str)
                .value_counts()
                .head(5)
                .index
                .tolist()
            )

        current_value = original.iloc[0][column]

        best_candidate = None
        best_probability = None

        for candidate in candidates:
            if candidate == current_value:
                continue

            candidate_row = original.copy()
            candidate_row[column] = candidate

            try:
                prediction = model.predict(
                    candidate_row
                )[0]

                if prediction != desired:
                    continue

                probabilities = model.predict_proba(
                    candidate_row
                )[0]

                desired_probability = float(
                    probabilities[
                        list(classes).index(desired)
                    ]
                )

                if (
                    best_probability is None
                    or desired_probability
                    > best_probability
                ):
                    best_probability = (
                        desired_probability
                    )
                    best_candidate = candidate

            except Exception:
                continue

        if best_candidate is not None:
            best[column] = best_candidate

            changes.append(
                {
                    "feature": column,
                    "from": _serializable(
                        current_value
                    ),
                    "to": _serializable(
                        best_candidate
                    ),
                    "desired_probability": round(
                        float(best_probability),
                        6,
                    ),
                }
            )

        if len(changes) >= max_changes:
            break

    final_prediction = model.predict(best)[0]

    found = final_prediction == desired

    return {
        "found": bool(found),
        "original_prediction": _serializable(
            original_prediction
        ),
        "desired_prediction": _serializable(
            desired
        ),
        "original_probability": (
            round(original_probability, 6)
            if original_probability is not None
            else None
        ),
        "changes": changes,
        "final_prediction": _serializable(
            final_prediction
        ),
    }


def _serializable(value):
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    return value