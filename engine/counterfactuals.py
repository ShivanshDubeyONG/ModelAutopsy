from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from engine.model_adapter import get_model_adapter


def _serializable(value: Any):
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    return value


def _candidate_values(
    X,
    feature,
    current,
):
    series = X[feature]

    if pd.api.types.is_numeric_dtype(series):
        values = series.dropna().astype(float)

        if len(values) == 0:
            return []

        candidates = list(
            np.unique(
                np.quantile(
                    values,
                    [
                        0.01,
                        0.05,
                        0.10,
                        0.20,
                        0.30,
                        0.40,
                        0.50,
                        0.60,
                        0.70,
                        0.80,
                        0.90,
                        0.95,
                        0.99,
                    ],
                )
            )
        )

        # Also include observed values around the current point.
        closest = sorted(
            values.unique(),
            key=lambda value: abs(
                float(value) - float(current)
            ),
        )[:12]

        candidates.extend(
            closest
        )

        return [
            float(value)
            for value in np.unique(candidates)
        ]

    modes = series.mode()

    values = list(
        series.dropna().unique()
    )

    if len(values) > 12:
        values = values[:12]

    if len(modes):
        values.insert(
            0,
            modes.iloc[0],
        )

    return list(
        dict.fromkeys(
            values
        )
    )


def _distance(
    X,
    feature,
    old,
    new,
):
    if old == new:
        return 0.0

    series = X[feature]

    if pd.api.types.is_numeric_dtype(series):
        std = float(
            series.std()
        )

        if not np.isfinite(std) or std <= 1e-9:
            std = 1.0

        return abs(
            float(new) - float(old)
        ) / std

    return 1.0


def _probability_for(
    adapter,
    row,
    target,
):
    probabilities = adapter.predict_proba(
        row
    )

    if probabilities is None:
        return None

    probabilities = np.asarray(
        probabilities
    )

    classes = adapter.classes_

    if classes is None:
        return None

    classes = list(classes)

    try:
        index = classes.index(
            target
        )

        if probabilities.ndim == 2:
            return float(
                probabilities[
                    0,
                    index,
                ]
            )

    except (
        ValueError,
        IndexError,
    ):
        pass

    return None


def _choose_target(
    adapter,
    row,
    original,
):
    """
    Choose the most plausible alternative class.

    If probabilities exist, use the runner-up class.
    Otherwise use another observed class.
    """

    probabilities = adapter.predict_proba(
        row
    )

    classes = adapter.classes_

    if (
        probabilities is not None
        and classes is not None
    ):
        probabilities = np.asarray(
            probabilities
        )

        classes = list(classes)

        if (
            probabilities.ndim == 2
            and probabilities.shape[1]
            == len(classes)
        ):
            ranking = np.argsort(
                probabilities[0]
            )[::-1]

            for position in ranking:
                candidate = classes[
                    int(position)
                ]

                if candidate != original:
                    return candidate

    if classes is not None:
        for candidate in classes:
            if candidate != original:
                return candidate

    return None


def _evaluate(
    adapter,
    row,
    target,
):
    prediction = adapter.predict(
        row
    )[0]

    probability = _probability_for(
        adapter,
        row,
        target,
    )

    return (
        prediction,
        probability,
    )


def _one_feature_search(
    adapter,
    X,
    row,
    original,
    target,
):
    results = []

    for feature in X.columns:
        current = row.iloc[0][feature]

        for candidate in _candidate_values(
            X,
            feature,
            current,
        ):
            if candidate == current:
                continue

            modified = row.copy()

            try:
                modified.loc[
                    modified.index[0],
                    feature,
                ] = candidate

                prediction, probability = _evaluate(
                    adapter,
                    modified,
                    target,
                )

                if prediction == target:
                    results.append(
                        {
                            "row": modified,
                            "changes": [
                                {
                                    "feature": feature,
                                    "from": _serializable(
                                        current
                                    ),
                                    "to": _serializable(
                                        candidate
                                    ),
                                    "distance": _distance(
                                        X,
                                        feature,
                                        current,
                                        candidate,
                                    ),
                                }
                            ],
                            "probability": probability,
                        }
                    )

            except Exception:
                continue

    if not results:
        return None

    results.sort(
        key=lambda item: (
            len(item["changes"]),
            sum(
                change["distance"]
                for change in item["changes"]
            ),
            -(
                item["probability"]
                if item["probability"]
                is not None
                else 0.0
            ),
        )
    )

    return results[0]


def _beam_search(
    adapter,
    X,
    row,
    target,
    max_changes=3,
):
    """
    Bounded search for combinations of feature changes.

    Keeps only the cheapest candidates at every level.
    """

    states = [
        (
            row.copy(),
            [],
            0.0,
        )
    ]

    features = list(X.columns)

    for depth in range(
        1,
        max_changes + 1,
    ):
        next_states = []

        for state_row, changes, cost in states:
            used = {
                change["feature"]
                for change in changes
            }

            remaining = [
                feature
                for feature in features
                if feature not in used
            ]

            for feature in remaining:
                current = state_row.iloc[0][
                    feature
                ]

                candidates = _candidate_values(
                    X,
                    feature,
                    current,
                )

                for candidate in candidates:
                    if candidate == current:
                        continue

                    modified = state_row.copy()

                    try:
                        modified.loc[
                            modified.index[0],
                            feature,
                        ] = candidate

                        distance = _distance(
                            X,
                            feature,
                            current,
                            candidate,
                        )

                        new_changes = (
                            changes
                            + [
                                {
                                    "feature": feature,
                                    "from": _serializable(
                                        current
                                    ),
                                    "to": _serializable(
                                        candidate
                                    ),
                                    "distance": distance,
                                }
                            ]
                        )

                        new_cost = (
                            cost + distance
                        )

                        prediction, probability = (
                            _evaluate(
                                adapter,
                                modified,
                                target,
                            )
                        )

                        if prediction == target:
                            return {
                                "row": modified,
                                "changes": new_changes,
                                "probability": probability,
                            }

                        next_states.append(
                            (
                                modified,
                                new_changes,
                                new_cost,
                            )
                        )

                    except Exception:
                        continue

        if not next_states:
            break

        next_states.sort(
            key=lambda item: item[2]
        )

        # Keep the search bounded.
        states = next_states[:80]

    return None


def find_counterfactual(
    model,
    X,
    index,
    output_index=None,
    desired_prediction=None,
    problem_type="classification",
):
    if len(X) == 0:
        return {
            "found": False,
            "reason": "Dataset contains no rows.",
        }

    index = max(
        0,
        min(
            int(index),
            len(X) - 1,
        ),
    )

    adapter = get_model_adapter(
        model,
        output_index=output_index,
    )

    row = X.iloc[[index]].copy()

    try:
        original = adapter.predict(row)[0]
    except Exception as exc:
        return {
            "found": False,
            "reason": (
                "Unable to evaluate original prediction: "
                f"{exc}"
            ),
        }

    target = desired_prediction

    # ---------------------------------------------------------
    # Regression
    # ---------------------------------------------------------

    if problem_type == "regression":
        if target is None:
            try:
                predictions = adapter.predict(X)

                lower = float(np.percentile(predictions, 25))
                upper = float(np.percentile(predictions, 75))
                current = float(original)

                if current <= lower:
                    target = upper
                else:
                    target = lower

            except Exception:
                return {
                    "found": False,
                    "original_prediction": _serializable(
                        original
                    ),
                    "reason": (
                        "Unable to determine a meaningful "
                        "alternative regression target."
                    ),
                }

    # ---------------------------------------------------------
    # Classification
    # ---------------------------------------------------------

    else:
        if target is None:
            target = _choose_target(
                adapter,
                row,
                original,
            )

        if target is None:
            return {
                "found": False,
                "original_prediction": _serializable(
                    original
                ),
                "reason": (
                    "No alternative target class "
                    "was available."
                ),
            }

    # ---------------------------------------------------------
    # One feature first
    # ---------------------------------------------------------

    result = _one_feature_search(
        adapter,
        X,
        row,
        original,
        target,
    )

    # ---------------------------------------------------------
    # Then bounded combinations
    # ---------------------------------------------------------

    if result is None:
        result = _beam_search(
            adapter,
            X,
            row,
            target,
            max_changes=3,
        )

    if result is None:
        return {
            "found": False,
            "original_prediction": _serializable(
                original
            ),
            "desired_prediction": _serializable(
                target
            ),
            "reason": (
                "No valid counterfactual was found "
                "within the observed feature distribution "
                "using up to three feature changes."
            ),
        }

    final_prediction = adapter.predict(
        result["row"]
    )[0]

    changes = []

    for change in result["changes"]:
        old_value = change["from"]
        new_value = change["to"]

        magnitude = None

        if isinstance(
            old_value,
            (int, float, np.integer, np.floating),
        ) and isinstance(
            new_value,
            (int, float, np.integer, np.floating),
        ):
            magnitude = abs(
                float(new_value)
                - float(old_value)
            )

        changes.append(
            {
                "feature": change["feature"],
                "from": _serializable(
                    old_value
                ),
                "to": _serializable(
                    new_value
                ),
                "magnitude": (
                    None
                    if magnitude is None
                    else round(
                        magnitude,
                        6,
                    )
                ),
            }
        )

    return {
        "found": True,
        "original_prediction": _serializable(
            original
        ),
        "final_prediction": _serializable(
            final_prediction
        ),
        "desired_prediction": _serializable(
            target
        ),
        "desired_probability": (
            None
            if result["probability"]
            is None
            else round(
                float(
                    result["probability"]
                ),
                6,
            )
        ),
        "changes": changes,
        "reason": (
            "Prediction changed using the smallest "
            "observed-distribution feature change found."
        ),
    }