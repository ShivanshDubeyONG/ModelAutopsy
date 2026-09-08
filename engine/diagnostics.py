from typing import Any

import numpy as np
import pandas as pd


def discover_error_slices(
    X: pd.DataFrame,
    y_true: Any,
    y_pred: Any,
    min_size: int = 10,
    min_lift: float = 1.5,
    max_findings: int = 8,
) -> list[dict]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    errors = (y_true != y_pred).astype(float)
    baseline = float(errors.mean())

    if baseline == 0:
        return []

    slices = []

    for column in X.columns:
        series = X[column]

        if pd.api.types.is_numeric_dtype(series):
            if series.nunique() < 4:
                values = series.dropna().unique()

                for value in values:
                    mask = series == value
                    _add_slice(
                        slices,
                        column,
                        f"{column} = {value}",
                        mask,
                        errors,
                        baseline,
                        min_size,
                        min_lift,
                    )
            else:
                quantiles = series.quantile(
                    [0.0, 0.25, 0.5, 0.75, 1.0]
                ).unique()

                for low, high in zip(
                    quantiles[:-1],
                    quantiles[1:],
                ):
                    mask = (
                        (series >= low)
                        & (series <= high)
                    )

                    condition = (
                        f"{low:.3g} ≤ {column} ≤ {high:.3g}"
                    )

                    _add_slice(
                        slices,
                        column,
                        condition,
                        mask,
                        errors,
                        baseline,
                        min_size,
                        min_lift,
                    )

        else:
            for value in series.dropna().unique():
                mask = series == value

                _add_slice(
                    slices,
                    column,
                    f"{column} = {value}",
                    mask,
                    errors,
                    baseline,
                    min_size,
                    min_lift,
                )

    slices.sort(
        key=lambda item: (
            item["lift"] * item["size"]
        ),
        reverse=True,
    )

    return slices[:max_findings]


def _add_slice(
    output,
    feature,
    condition,
    mask,
    errors,
    baseline,
    min_size,
    min_lift,
):
    size = int(mask.sum())

    if size < min_size:
        return

    error_rate = float(errors[mask].mean())
    lift = error_rate / baseline

    if lift < min_lift:
        return

    output.append(
        {
            "feature": feature,
            "condition": condition,
            "size": size,
            "error_rate": error_rate,
            "baseline_error": baseline,
            "lift": float(lift),
        }
    )