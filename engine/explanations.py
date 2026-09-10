from typing import Any

import numpy as np


def _select_output_model(
    model,
    output_index: int | None = None,
):
    """
    Select the estimator responsible for one output.
    """

    if output_index is not None:
        output_names = getattr(
            model,
            "output_names",
            None,
        )

        if output_names is not None:
            if 0 <= output_index < len(output_names):
                output_name = output_names[output_index]

                if hasattr(model, "get_output_model"):
                    return model.get_output_model(
                        output_name
                    )

    if output_index is not None:
        estimators = getattr(
            model,
            "estimators_",
            None,
        )

        if estimators is not None:
            try:
                return estimators[output_index]
            except (
                IndexError,
                TypeError,
            ):
                pass

    return model


def _to_serializable(value: Any):
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


def _global_shap_values(
    shap_values,
    n_features: int,
):
    """
    Convert common SHAP formats into one
    global feature-importance vector.
    """

    if isinstance(shap_values, list):
        if not shap_values:
            return None

        arrays = [
            np.asarray(value)
            for value in shap_values
        ]

        processed = []

        for array in arrays:
            if array.ndim == 2:
                processed.append(
                    np.abs(array).mean(axis=0)
                )

        if not processed:
            return None

        values = np.mean(
            processed,
            axis=0,
        )

    else:
        values = np.asarray(
            shap_values
        )

        if values.ndim == 3:
            # samples x features x outputs
            values = np.abs(values).mean(
                axis=(0, 2)
            )

        elif values.ndim == 2:
            values = np.abs(values).mean(
                axis=0
            )

        elif values.ndim == 1:
            values = np.abs(values)

        else:
            return None

    if len(values) != n_features:
        return None

    return values


def _local_shap_values(
    shap_values,
    class_index: int | None,
    n_features: int,
):
    """
    Extract one local SHAP vector.

    Handles:
        list[class] -> samples x features
        samples x features
        samples x features x classes
    """

    if isinstance(shap_values, list):
        if not shap_values:
            return None

        if class_index is None:
            class_index = len(shap_values) - 1

        class_index = max(
            0,
            min(
                class_index,
                len(shap_values) - 1,
            ),
        )

        values = np.asarray(
            shap_values[class_index]
        )

        if values.ndim == 2:
            values = values[0]

        elif values.ndim == 1:
            pass

        else:
            return None

    else:
        values = np.asarray(
            shap_values
        )

        if values.ndim == 3:
            if class_index is None:
                class_index = values.shape[2] - 1

            class_index = max(
                0,
                min(
                    class_index,
                    values.shape[2] - 1,
                ),
            )

            values = values[
                0,
                :,
                class_index,
            ]

        elif values.ndim == 2:
            values = values[0]

        elif values.ndim == 1:
            pass

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

    Priority:
        1. SHAP
        2. feature_importances_
        3. coef_
    """

    explanation_model = _select_output_model(
        model,
        output_index,
    )

    feature_names = list(
        X.columns
    )

    try:
        import shap

        explainer = shap.TreeExplainer(
            explanation_model
        )

        shap_values = explainer.shap_values(
            X
        )

        values = _global_shap_values(
            shap_values,
            len(feature_names),
        )

        if values is not None:
            ranked = sorted(
                zip(
                    feature_names,
                    values,
                ),
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

    if hasattr(
        explanation_model,
        "feature_importances_",
    ):
        values = np.asarray(
            explanation_model.feature_importances_
        )

        if len(values) == len(feature_names):
            ranked = sorted(
                zip(
                    feature_names,
                    values,
                ),
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
                zip(
                    feature_names,
                    values,
                ),
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
            "No supported global explanation "
            "method was available."
        ),
    }


def local_explanation(
    model,
    X,
    index: int,
    output_index: int | None = None,
    actual=None,
) -> dict:
    """
    Explain one representative prediction.

    The returned feature data contains BOTH:
        value        -> actual feature value
        contribution -> SHAP contribution
    """

    if len(X) == 0:
        return {
            "index": 0,
            "actual": _to_serializable(actual),
            "prediction": None,
            "probability": None,
            "probability_label": None,
            "features": {},
            "contributions": [],
            "method": "unavailable",
        }

    index = max(
        0,
        min(
            index,
            len(X) - 1,
        ),
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
            np.asarray(
                raw_prediction
            ).reshape(-1)[0]
        )

    except Exception:
        prediction = None

    # ---------------------------------------------------------
    # Probability
    # ---------------------------------------------------------

    probability = None
    probability_label = None
    class_index = None

    if hasattr(
        explanation_model,
        "predict_proba",
    ):
        try:
            probabilities = np.asarray(
                explanation_model.predict_proba(
                    row
                )
            )

            classes = getattr(
                explanation_model,
                "classes_",
                None,
            )

            if (
                probabilities.ndim == 2
                and probabilities.shape[1] > 0
                and classes is not None
            ):
                classes = list(classes)

                if prediction in classes:
                    class_index = classes.index(
                        prediction
                    )

                    probability = float(
                        probabilities[
                            0,
                            class_index,
                        ]
                    )

                    probability_label = _to_serializable(
                        prediction
                    )

        except Exception:
            pass

    # ---------------------------------------------------------
    # SHAP
    # ---------------------------------------------------------

    try:
        import shap

        explainer = shap.TreeExplainer(
            explanation_model
        )

        shap_values = explainer.shap_values(
            row
        )

        values = _local_shap_values(
            shap_values,
            class_index,
            len(X.columns),
        )

        if values is not None:
            contributions = []

            for feature, contribution in zip(
                X.columns,
                values,
            ):
                contributions.append(
                    {
                        "feature": feature,
                        "value": _to_serializable(
                            row.iloc[0][feature]
                        ),
                        "contribution": round(
                            float(contribution),
                            6,
                        ),
                    }
                )

            contributions.sort(
                key=lambda item: abs(
                    item["contribution"]
                ),
                reverse=True,
            )

            return {
                "index": index,
                "actual": _to_serializable(
                    actual
                ),
                "prediction": prediction,
                "probability": probability,
                "probability_label": probability_label,
                "features": {
                    item["feature"]: item[
                        "contribution"
                    ]
                    for item in contributions
                },
                "contributions": contributions,
                "method": "SHAP",
            }

    except Exception:
        pass

    return {
        "index": index,
        "actual": _to_serializable(
            actual
        ),
        "prediction": prediction,
        "probability": probability,
        "probability_label": probability_label,
        "features": {},
        "contributions": [],
        "method": "unavailable",
        "reason": (
            "Local explanation was unavailable "
            "for this model."
        ),
    }


global_feature_importance = global_explanation