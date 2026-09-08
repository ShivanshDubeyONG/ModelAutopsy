from typing import Any

import numpy as np
import pandas as pd


def global_explanation(
    model: Any,
    X: pd.DataFrame,
) -> dict:
    try:
        import shap

        sample = X

        if len(sample) > 500:
            sample = sample.sample(
                500,
                random_state=42,
            )

        explainer = shap.Explainer(
            model,
            sample,
        )

        values = explainer(sample)

        raw = np.asarray(values.values)

        if raw.ndim == 3:
            importance = np.abs(raw).mean(
                axis=(0, 2)
            )
        elif raw.ndim == 2:
            importance = np.abs(raw).mean(axis=0)
        else:
            importance = np.abs(raw)

        ranking = [
            {
                "feature": feature,
                "importance": float(score),
            }
            for feature, score in zip(
                X.columns,
                importance,
            )
        ]

        ranking.sort(
            key=lambda item: item["importance"],
            reverse=True,
        )

        return {
            "method": "SHAP",
            "features": ranking,
        }

    except Exception as exc:
        return {
            "method": "unavailable",
            "features": [],
            "error": str(exc),
        }


def local_explanation(
    model: Any,
    X: pd.DataFrame,
    index: int,
) -> dict:
    row = X.iloc[[index]]

    prediction = model.predict(row)[0]

    probability = None

    if hasattr(model, "predict_proba"):
        try:
            probability = float(
                np.max(
                    model.predict_proba(row)[0]
                )
            )
        except Exception:
            pass

    contributions = []

    try:
        import shap

        background = X

        if len(background) > 200:
            background = background.sample(
                200,
                random_state=42,
            )

        explainer = shap.Explainer(
            model,
            background,
        )

        explanation = explainer(row)
        values = np.asarray(
            explanation.values
        )[0]

        if values.ndim > 1:
            values = np.mean(
                np.abs(values),
                axis=-1,
            )

        for feature, value, contribution in zip(
            X.columns,
            row.iloc[0],
            values,
        ):
            contributions.append(
                {
                    "feature": feature,
                    "value": _native(value),
                    "contribution": float(
                        contribution
                    ),
                }
            )

        contributions.sort(
            key=lambda item: abs(
                item["contribution"]
            ),
            reverse=True,
        )

        method = "SHAP"

    except Exception:
        method = "prediction_only"

    return {
        "index": index,
        "prediction": _native(prediction),
        "probability": probability,
        "method": method,
        "contributions": contributions,
    }


def _native(value):
    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    return value