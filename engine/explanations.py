from typing import Any

import numpy as np


def _select_output_model(
    model,
    output_index: int | None = None,
):
    """
    Select the estimator responsible for one output.
    """

    # Model Autopsy mixed-output bundle
    if output_index is not None:
        output_names = getattr(
            model,
            "output_names",
            None,
        )

        if output_names is not None:
            if (
                0 <= output_index
                < len(output_names)
            ):
                output_name = output_names[
                    output_index
                ]

                if hasattr(
                    model,
                    "get_output_model",
                ):
                    return model.get_output_model(
                        output_name
                    )

    # sklearn multi-output estimators
    if output_index is not None:
        estimators = getattr(
            model,
            "estimators_",
            None,
        )

        if estimators is not None:
            try:
                return estimators[
                    output_index
                ]
            except (
                IndexError,
                TypeError,
            ):
                pass

    return model

def _to_serializable(value: Any):
    """
    Convert numpy/scalar values into JSON-safe Python values.
    """

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            str(key): _to_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _to_serializable(item)
            for item in value
        ]

    return value


def _normalise_shap_values(
    shap_values,
    n_features: int,
):
    """
    Normalise common SHAP output formats into a
    single feature x importance vector.
    """

    if isinstance(shap_values, list):
        if not shap_values:
            return None

        # Classification commonly returns one array per class.
        shap_values = shap_values[-1]

    values = np.asarray(shap_values)

    if values.ndim == 3:
        # samples x features x outputs
        values = values[:, :, -1]

    if values.ndim == 2:
        values = np.abs(values).mean(axis=0)

    elif values.ndim == 1:
        values = np.abs(values)

    else:
        return None

    if len(values) != n_features:
        return None

    return values


def global_explanation(
    model,
    X,
    output_index: int | None = None,
) -> dict:
    """
    Generate global feature importance.

    Uses SHAP when possible, then falls back to
    model-native feature_importances_ or coef_.
    """

    explanation_model = _select_output_model(
        model,
        output_index,
    )

    feature_names = list(X.columns)

    # ---------------------------------------------------------
    # SHAP
    # ---------------------------------------------------------

    try:
        import shap

        explainer = shap.TreeExplainer(
            explanation_model
        )

        shap_values = explainer.shap_values(X)

        values = _normalise_shap_values(
            shap_values,
            len(feature_names),
        )

        if values is not None:
            ranked = sorted(
                zip(feature_names, values),
                key=lambda item: abs(item[1]),
                reverse=True,
            )

            return {
                "method": "SHAP",
                "features": [
                    {
                        "feature": feature,
                        "importance": round(
                            float(importance),
                            6,
                        ),
                    }
                    for feature, importance in ranked
                ],
            }

    except Exception:
        pass

    # ---------------------------------------------------------
    # Native feature importance
    # ---------------------------------------------------------

    if hasattr(
        explanation_model,
        "feature_importances_",
    ):
        values = np.asarray(
            explanation_model.feature_importances_
        )

        if len(values) == len(feature_names):
            ranked = sorted(
                zip(feature_names, values),
                key=lambda item: abs(item[1]),
                reverse=True,
            )

            return {
                "method": "model_native",
                "features": [
                    {
                        "feature": feature,
                        "importance": round(
                            float(importance),
                            6,
                        ),
                    }
                    for feature, importance in ranked
                ],
            }

    # ---------------------------------------------------------
    # Linear coefficients
    # ---------------------------------------------------------

    if hasattr(
        explanation_model,
        "coef_",
    ):
        values = np.asarray(
            explanation_model.coef_
        )

        if values.ndim > 1:
            values = np.mean(
                np.abs(values),
                axis=0,
            )
        else:
            values = np.abs(values)

        if len(values) == len(feature_names):
            ranked = sorted(
                zip(feature_names, values),
                key=lambda item: abs(item[1]),
                reverse=True,
            )

            return {
                "method": "model_coefficients",
                "features": [
                    {
                        "feature": feature,
                        "importance": round(
                            float(importance),
                            6,
                        ),
                    }
                    for feature, importance in ranked
                ],
            }

    return {
        "method": "unavailable",
        "features": [],
        "reason": (
            "No supported global explanation method "
            "was available for this model."
        ),
    }


def local_explanation(
    model,
    X,
    index: int,
    output_index: int | None = None,
) -> dict:
    """
    Explain one representative prediction.
    """

    if len(X) == 0:
        return {
            "index": 0,
            "prediction": None,
            "features": {},
            "method": "unavailable",
        }

    index = max(
        0,
        min(index, len(X) - 1),
    )

    row = X.iloc[[index]]

    explanation_model = _select_output_model(
        model,
        output_index,
    )

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    try:
        raw_prediction = explanation_model.predict(
            row
        )

        prediction = _to_serializable(
            np.asarray(raw_prediction).reshape(-1)[0]
        )

    except Exception:
        prediction = None

    # ---------------------------------------------------------
    # Probability for binary classification
    # ---------------------------------------------------------

    probability = None

    if hasattr(
        explanation_model,
        "predict_proba",
    ):
        try:
            probabilities = np.asarray(
                explanation_model.predict_proba(row)
            )

            if (
                probabilities.ndim == 2
                and probabilities.shape[1] == 2
            ):
                probability = float(
                    probabilities[0][1]
                )

        except Exception:
            probability = None

    # ---------------------------------------------------------
    # SHAP local explanation
    # ---------------------------------------------------------

    try:
        import shap

        explainer = shap.TreeExplainer(
            explanation_model
        )

        shap_values = explainer.shap_values(
            row
        )

        if isinstance(
            shap_values,
            list,
        ):
            shap_values = shap_values[-1]

        values = np.asarray(
            shap_values
        )

        if values.ndim == 3:
            values = values[0, :, -1]

        elif values.ndim == 2:
            values = values[0]

        if len(values) == len(X.columns):
            feature_values = {
                feature: round(
                    float(value),
                    6,
                )
                for feature, value in zip(
                    X.columns,
                    values,
                )
            }

            return {
                "index": index,
                "prediction": prediction,
                "probability": probability,
                "features": feature_values,
                "method": "SHAP",
            }

    except Exception:
        pass

    # ---------------------------------------------------------
    # Native fallback
    # ---------------------------------------------------------

    return {
        "index": index,
        "prediction": prediction,
        "probability": probability,
        "features": {},
        "method": "unavailable",
        "reason": (
            "Local explanation was unavailable "
            "for this model."
        ),
    }


# Backward compatibility with older imports.
global_feature_importance = global_explanation