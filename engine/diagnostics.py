from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def discover_error_slices(
    X: pd.DataFrame,
    y_true: Any,
    y_pred: Any,
    problem_type: str = "classification",
    probabilities: Any | None = None,
    min_size: int = 10,
    min_lift: float = 1.35,
    max_findings: int = 8,
) -> list[dict]:

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    if len(y_true) == 0:
        return []

    if problem_type == "regression":

        errors = np.abs(
            y_true.astype(float)
            - y_pred.astype(float)
        )

        baseline = float(
            errors.mean()
        )

        if baseline <= 1e-12:
            return []

        return _search_feature_slices(
            X,
            errors,
            baseline,
            "regression",
            min_size,
            min_lift,
            max_findings,
        )

    errors = (
        y_true != y_pred
    ).astype(float)

    baseline = float(
        errors.mean()
    )

    if baseline <= 1e-12:
        return []

    slices = _search_feature_slices(
        X,
        errors,
        baseline,
        "classification",
        min_size,
        min_lift,
        max_findings * 3,
    )

    slices.extend(
        _class_specific_slices(
            X,
            y_true,
            y_pred,
            min_size,
            min_lift,
            max_findings * 2,
        )
    )

    slices.extend(
        _confidence_slices(
            X,
            errors,
            probabilities,
            baseline,
            min_size,
            min_lift,
            max_findings * 2,
        )
    )

    return _rank_slices(
        slices,
        max_findings,
    )


def _search_feature_slices(
    X,
    errors,
    baseline,
    problem_type,
    min_size,
    min_lift,
    max_candidates,
):

    slices = []

    n = len(X)

    min_size = max(
        min_size,
        int(
            np.ceil(
                n * 0.02
            )
        ),
    )

    for column in X.columns:

        series = X[column]

        if pd.api.types.is_numeric_dtype(
            series
        ):

            if (
                series.nunique(
                    dropna=True
                )
                < 4
            ):
                masks = [
                    (
                        f"{column} = {value}",
                        series == value,
                    )
                    for value in (
                        series
                        .dropna()
                        .unique()
                    )
                ]

            else:

                quantiles = (
                    series
                    .quantile(
                        [
                            0.0,
                            0.25,
                            0.5,
                            0.75,
                            1.0,
                        ]
                    )
                    .drop_duplicates()
                    .to_numpy()
                )

                masks = [
                    (
                        (
                            f"{quantiles[i]:.3g} "
                            f"≤ {column} "
                            f"≤ {quantiles[i + 1]:.3g}"
                        ),
                        (
                            (series >= quantiles[i])
                            & (
                                series
                                <= quantiles[
                                    i + 1
                                ]
                            )
                        ),
                    )
                    for i in range(
                        len(quantiles) - 1
                    )
                ]

        else:

            top_values = (
                series
                .dropna()
                .value_counts()
                .head(8)
                .index
            )

            masks = [
                (
                    f"{column} = {value}",
                    series == value,
                )
                for value in top_values
            ]

        for condition, mask in masks:

            size = int(
                mask.sum()
            )

            if size < min_size:
                continue

            local_error = float(
                errors[mask].mean()
            )

            lift = (
                local_error
                / baseline
            )

            if lift < min_lift:
                continue

            slices.append(
                _slice_record(
                    column,
                    condition,
                    size,
                    local_error,
                    baseline,
                    lift,
                    problem_type,
                )
            )

    return _rank_slices(
        slices,
        max_candidates,
    )


def _class_specific_slices(
    X,
    y_true,
    y_pred,
    min_size,
    min_lift,
    max_candidates,
):

    slices = []

    errors = (
        y_true != y_pred
    )

    baseline = float(
        errors.mean()
    )

    if baseline <= 0:
        return []

    min_size = max(
        min_size,
        int(
            np.ceil(
                len(X) * 0.02
            )
        ),
    )

    labels = pd.unique(
        y_true
    )

    for label in labels:

        mask = (
            y_true == label
        )

        size = int(
            mask.sum()
        )

        if size < min_size:
            continue

        class_error = float(
            errors[mask].mean()
        )

        lift = (
            class_error
            / baseline
        )

        if lift < min_lift:
            continue

        slices.append(
            {
                "feature": "target_class",
                "condition": (
                    f"actual = {label}"
                ),
                "size": size,
                "error_rate": class_error,
                "baseline_error": baseline,
                "lift": float(lift),
                "problem_type": "classification",
                "kind": "class_specific",
            }
        )

    return _rank_slices(
        slices,
        max_candidates,
    )


def _confidence_slices(
    X,
    errors,
    probabilities,
    baseline,
    min_size,
    min_lift,
    max_candidates,
):

    if probabilities is None:
        return []

    try:

        probabilities = np.asarray(
            probabilities
        )

        if (
            probabilities.ndim != 2
            or probabilities.shape[0]
            != len(X)
        ):
            return []

        confidence = probabilities.max(
            axis=1
        )

    except Exception:
        return []

    slices = []

    min_size = max(
        min_size,
        int(
            np.ceil(
                len(X) * 0.02
            )
        ),
    )

    quantiles = np.quantile(
        confidence,
        [
            0.0,
            0.25,
            0.5,
            0.75,
            1.0,
        ],
    )

    for i in range(4):

        mask = (
            confidence
            >= quantiles[i]
        ) & (
            confidence
            <= quantiles[i + 1]
        )

        size = int(
            mask.sum()
        )

        if size < min_size:
            continue

        local_error = float(
            errors[mask].mean()
        )

        lift = (
            local_error
            / baseline
        )

        if lift < min_lift:
            continue

        slices.append(
            {
                "feature": "prediction_confidence",
                "condition": (
                    f"{quantiles[i]:.1%} "
                    f"≤ confidence ≤ "
                    f"{quantiles[i + 1]:.1%}"
                ),
                "size": size,
                "error_rate": local_error,
                "baseline_error": baseline,
                "lift": float(lift),
                "problem_type": "classification",
                "kind": "confidence",
            }
        )

    return _rank_slices(
        slices,
        max_candidates,
    )


def _slice_record(
    feature,
    condition,
    size,
    local_error,
    baseline,
    lift,
    problem_type,
):

    record = {
        "feature": feature,
        "condition": condition,
        "size": size,
        "baseline_error": baseline,
        "lift": float(lift),
        "problem_type": problem_type,
    }

    if problem_type == "regression":

        record["error"] = (
            local_error
        )

        record["error_metric"] = (
            "MAE"
        )

    else:

        record["error_rate"] = (
            local_error
        )

    return record


def _rank_slices(
    slices,
    limit,
):

    unique = {}

    for item in slices:

        key = (
            item.get("feature"),
            item.get("condition"),
        )

        if (
            key not in unique
            or item["lift"]
            > unique[key]["lift"]
        ):
            unique[key] = item

    return sorted(
        unique.values(),
        key=lambda item:
            item["lift"]
            * np.sqrt(
                max(
                    item["size"],
                    1,
                )
            ),
        reverse=True,
    )[:limit]