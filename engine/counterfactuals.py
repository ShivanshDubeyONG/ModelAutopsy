from typing import Any

import numpy as np
import pandas as pd


def find_counterfactual(
    model: Any,
    X: pd.DataFrame,
    index: int,
    max_changes: int = 3,
) -> dict:
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

    original = X.iloc[[index]].copy()

    original_prediction = model.predict(
        original
    )[0]

    desired = (
        classes[0]
        if original_prediction == classes[1]
        else classes[1]
    )

    candidates = {}

    for column in X.columns:
        series = X[column]

        if pd.api.types.is_numeric_dtype(series):
            candidates[column] = sorted(
                set(
                    float(series.quantile(q))
                    for q in np.linspace(
                        0,
                        1,
                        7,
                    )
                )
            )
        else:
            candidates[column] = list(
                series.dropna().unique()
            )

    current = original.copy()
    changes = {}

    for _ in range(max_changes):
        if model.predict(current)[0] == desired:
            break

        current_probability = _desired_probability(
            model,
            current,
            desired,
        )

        best_gain = 0
        best_feature = None
        best_value = None

        for column in X.columns:
            if column in changes:
                continue

            original_value = current.iloc[0][
                column
            ]

            for value in candidates[column]:
                if value == original_value:
                    continue

                trial = current.copy()
                trial[column] = value

                probability = _desired_probability(
                    model,
                    trial,
                    desired,
                )

                gain = (
                    probability
                    - current_probability
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = column
                    best_value = value

        if best_feature is None:
            break

        current[best_feature] = best_value
        changes[best_feature] = best_value

    final_prediction = model.predict(
        current
    )[0]

    return {
        "found": bool(
            final_prediction == desired
        ),
        "original_prediction": _native(
            original_prediction
        ),
        "desired_prediction": _native(
            desired
        ),
        "new_prediction": _native(
            final_prediction
        ),
        "changes": [
            {
                "feature": feature,
                "from": _native(
                    original.iloc[0][feature]
                ),
                "to": _native(value),
            }
            for feature, value
            in changes.items()
        ],
        "n_features_changed": len(changes),
    }


def _desired_probability(
    model,
    X,
    desired,
):
    probabilities = model.predict_proba(X)[0]
    classes = list(model.classes_)
    index = classes.index(desired)
    return float(probabilities[index])


def _native(value):
    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    return value