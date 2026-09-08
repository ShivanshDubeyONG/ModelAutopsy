from typing import Any

import numpy as np
import pandas as pd
import shap


def _normalise_shap_values(
    shap_values: Any,
    n_samples: int,
    n_features: int,
) -> np.ndarray:
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            values = np.asarray(shap_values[1])
        else:
            values = np.asarray(shap_values[0])
    else:
        values = np.asarray(shap_values)

        if values.ndim == 3:
            values = values[:, :, -1]

    if values.ndim != 2:
        raise ValueError(f"Unexpected SHAP output shape: {values.shape}")

    if values.shape == (n_features, n_samples):
        values = values.T

    if values.shape != (n_samples, n_features):
        raise ValueError(
            f"Unexpected SHAP matrix shape: {values.shape}; "
            f"expected {(n_samples, n_features)}"
        )

    return values


def _get_shap_values(explainer: Any, X: pd.DataFrame) -> Any:
    try:
        return explainer.shap_values(X)
    except Exception:
        return explainer.shap_values(
            X,
            check_additivity=False,
        )


def global_explanation(
    model: Any,
    X: pd.DataFrame,
) -> dict:
    feature_names = list(X.columns)

    try:
        explainer = shap.TreeExplainer(model)
        raw_values = _get_shap_values(explainer, X)

        values = _normalise_shap_values(
            raw_values,
            len(X),
            len(feature_names),
        )

        importance = np.abs(values).mean(axis=0)
        order = np.argsort(importance)[::-1]

        return {
            "method": "SHAP",
            "features": [
                {
                    "feature": feature_names[i],
                    "importance": float(importance[i]),
                }
                for i in order
            ],
        }

    except Exception as shap_error:
        if hasattr(model, "feature_importances_"):
            importance = np.asarray(model.feature_importances_)
            order = np.argsort(importance)[::-1]

            return {
                "method": "model_native",
                "features": [
                    {
                        "feature": feature_names[i],
                        "importance": float(importance[i]),
                    }
                    for i in order
                ],
                "warning": f"SHAP unavailable: {shap_error}",
            }

        return {
            "method": "unavailable",
            "features": [],
            "error": str(shap_error),
        }


def local_explanation(
    model: Any,
    X: pd.DataFrame,
    index: int,
) -> dict:
    """
    Explain one representative prediction.
    """

    if index < 0 or index >= len(X):
        raise IndexError(
            f"Prediction index {index} is out of range."
        )

    X_row = X.iloc[[index]]

    feature_names = list(X.columns)

    try:
        explainer = shap.TreeExplainer(model)

        raw_values = _get_shap_values(
            explainer,
            X_row,
        )

        values = _normalise_shap_values(
            raw_values,
            len(X_row),
            len(feature_names),
        )

        contributions = values[0]

        order = np.argsort(
            np.abs(contributions)
        )[::-1]

        prediction = model.predict(X_row)[0]

        probability = None

        if hasattr(model, "predict_proba"):
            try:
                probability = float(
                    model.predict_proba(X_row)[0][1]
                )
            except Exception:
                probability = None

        return {
            "index": int(index),
            "prediction": int(prediction),
            "probability": probability,
            "method": "SHAP",
            "contributions": [
                {
                    "feature": feature_names[i],
                    "value": X_row.iloc[0][feature_names[i]],
                    "contribution": float(
                        contributions[i]
                    ),
                }
                for i in order
            ],
        }

    except Exception:

        prediction = model.predict(X_row)[0]

        probability = None

        if hasattr(model, "predict_proba"):
            try:
                probability = float(
                    model.predict_proba(X_row)[0][1]
                )
            except Exception:
                probability = None

        return {
            "index": int(index),
            "prediction": int(prediction),
            "probability": probability,
            "method": "fallback",
            "contributions": [
                {
                    "feature": feature,
                    "value": X_row.iloc[0][feature],
                    "contribution": 0.0,
                }
                for feature in feature_names
            ],
        }
# Compatibility aliases used by the pipeline.
global_feature_importance = global_explanation