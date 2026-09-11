from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _to_serializable(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            str(k): _to_serializable(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]

    return value


class _OutputView:
    """
    Small sklearn-compatible wrapper around ModelAdapter.

    This is important for sklearn.inspection.permutation_importance,
    especially when the uploaded model is a Pipeline.
    """

    def __init__(self, adapter):
        self.adapter = adapter

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return self.adapter.predict(X)

    def predict_proba(self, X):
        values = self.adapter.predict_proba(X)

        if values is None:
            raise AttributeError(
                "predict_proba unavailable"
            )

        return values


def _ranked_features(
    feature_names,
    values,
    method,
):
    values = np.asarray(values).reshape(-1)

    if len(values) != len(feature_names):
        return None

    ranked = sorted(
        zip(feature_names, values),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )

    return {
        "method": method,
        "features": [
            {
                "feature": feature,
                "importance": round(
                    float(value),
                    6,
                ),
            }
            for feature, value in ranked
        ],
    }


def _shap_global(
    adapter,
    X,
):
    try:
        import shap

        estimator = adapter.output_model

        explainer = shap.TreeExplainer(
            estimator
        )

        values = explainer.shap_values(X)

        if isinstance(values, list):
            arrays = []

            for value in values:
                array = np.asarray(value)

                if array.ndim == 2:
                    arrays.append(
                        np.abs(array).mean(axis=0)
                    )

            if not arrays:
                return None

            values = np.mean(
                arrays,
                axis=0,
            )

        else:
            values = np.asarray(values)

            if values.ndim == 3:
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

        return _ranked_features(
            list(X.columns),
            values,
            "SHAP",
        )

    except Exception:
        return None


def _native_importance(
    adapter,
    X,
):
    try:
        values = adapter.feature_importances_

        if values is None:
            return None

        return _ranked_features(
            list(X.columns),
            values,
            "model_native",
        )

    except Exception:
        return None


def _coefficient_importance(
    adapter,
    X,
):
    try:
        values = adapter.coef_

        if values is None:
            return None

        values = np.asarray(values)

        if values.ndim > 1:
            values = np.mean(
                np.abs(values),
                axis=0,
            )
        else:
            values = np.abs(values)

        return _ranked_features(
            list(X.columns),
            values,
            "model_coefficients",
        )

    except Exception:
        return None


def _permutation_importance(
    adapter,
    X,
    y,
    problem_type,
):
    """
    Model-agnostic global importance.

    Works with sklearn Pipelines because the complete
    uploaded model is called through adapter.predict().
    """

    try:
        from sklearn.inspection import permutation_importance

        estimator = _OutputView(adapter)

        scoring = (
            "neg_mean_absolute_error"
            if problem_type == "regression"
            else "accuracy"
        )

        result = permutation_importance(
            estimator,
            X,
            y,
            scoring=scoring,
            n_repeats=5,
            random_state=42,
            n_jobs=1,
        )

        return _ranked_features(
            list(X.columns),
            result.importances_mean,
            "permutation_importance",
        )

    except Exception:
        return None


def global_explanation(
    model,
    X,
    output_index=None,
    y=None,
    problem_type="classification",
):
    """
    Generate global feature importance.

    Priority:

        SHAP
        ↓
        native importance
        ↓
        coefficients
        ↓
        permutation importance
        ↓
        unavailable
    """

    from engine.model_adapter import get_model_adapter

    adapter = get_model_adapter(
        model,
        output_index=output_index,
    )

    # ---------------------------------------------------------
    # SHAP
    # ---------------------------------------------------------

    result = _shap_global(
        adapter,
        X,
    )

    if result is not None:
        return result

    # ---------------------------------------------------------
    # Native model importance
    # ---------------------------------------------------------

    result = _native_importance(
        adapter,
        X,
    )

    if result is not None:
        return result

    # ---------------------------------------------------------
    # Linear coefficients
    # ---------------------------------------------------------

    result = _coefficient_importance(
        adapter,
        X,
    )

    if result is not None:
        return result

    # ---------------------------------------------------------
    # Model-agnostic fallback
    # ---------------------------------------------------------

    if y is not None:
        result = _permutation_importance(
            adapter,
            X,
            y,
            problem_type,
        )

        if result is not None:
            return result

    return {
        "method": "unavailable",
        "features": [],
        "reason": (
            "No global explanation method "
            "could be computed."
        ),
    }


def _local_sensitivity(
    adapter,
    X,
    index,
):
    """
    Model-agnostic local explanation.

    Each feature is replaced by a typical value
    and the prediction change is measured.
    """

    row = X.iloc[[index]].copy()

    try:
        baseline = adapter.predict(row)[0]
    except Exception:
        return None

    contributions = []

    for feature in X.columns:
        modified = row.copy()

        series = X[feature]

        if pd.api.types.is_numeric_dtype(series):
            replacement = float(
                series.median()
            )
        else:
            modes = series.mode()

            if len(modes) == 0:
                continue

            replacement = modes.iloc[0]

        try:
            modified.loc[
                modified.index[0],
                feature,
            ] = replacement

            changed = adapter.predict(
                modified
            )[0]

            if (
                isinstance(
                    baseline,
                    (int, float, np.number),
                )
                and isinstance(
                    changed,
                    (int, float, np.number),
                )
            ):
                contribution = float(
                    baseline - changed
                )
            else:
                contribution = (
                    1.0
                    if baseline != changed
                    else 0.0
                )

            contributions.append(
                {
                    "feature": feature,
                    "value": _to_serializable(
                        row.iloc[0][feature]
                    ),
                    "contribution": round(
                        contribution,
                        6,
                    ),
                }
            )

        except Exception:
            continue

    contributions.sort(
        key=lambda item: abs(
            item["contribution"]
        ),
        reverse=True,
    )

    return contributions


def local_explanation(
    model,
    X,
    index,
    output_index=None,
    actual=None,
):
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
            int(index),
            len(X) - 1,
        ),
    )

    from engine.model_adapter import get_model_adapter

    adapter = get_model_adapter(
        model,
        output_index=output_index,
    )

    row = X.iloc[[index]]

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    try:
        prediction = _to_serializable(
            adapter.predict(row)[0]
        )
    except Exception:
        prediction = None

    # ---------------------------------------------------------
    # Probability
    # ---------------------------------------------------------

    probability = None
    probability_label = None

    probabilities = adapter.predict_proba(
        row
    )

    if probabilities is not None:
        try:
            probabilities = np.asarray(
                probabilities
            )

            classes = adapter.classes_

            if (
                probabilities.ndim == 2
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

                    probability_label = (
                        _to_serializable(
                            prediction
                        )
                    )

        except Exception:
            pass

    # ---------------------------------------------------------
    # SHAP local
    # ---------------------------------------------------------

    try:
        import shap

        explainer = shap.TreeExplainer(
            adapter.output_model
        )

        shap_values = explainer.shap_values(
            row
        )

        class_index = None

        if (
            probabilities is not None
            and adapter.classes_ is not None
            and prediction in list(
                adapter.classes_
            )
        ):
            class_index = list(
                adapter.classes_
            ).index(prediction)

        values = None

        if isinstance(shap_values, list):
            if shap_values:
                if class_index is None:
                    class_index = (
                        len(shap_values) - 1
                    )

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

        else:
            values = np.asarray(
                shap_values
            )

            if values.ndim == 3:
                if class_index is None:
                    class_index = (
                        values.shape[2] - 1
                    )

                values = values[
                    0,
                    :,
                    class_index,
                ]

        if values is not None:
            if values.ndim == 2:
                values = values[0]

            if (
                values.ndim == 1
                and len(values)
                == len(X.columns)
            ):
                contributions = [
                    {
                        "feature": feature,
                        "value": _to_serializable(
                            row.iloc[0][feature]
                        ),
                        "contribution": round(
                            float(value),
                            6,
                        ),
                    }
                    for feature, value in zip(
                        X.columns,
                        values,
                    )
                ]

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

    # ---------------------------------------------------------
    # Model-agnostic fallback
    # ---------------------------------------------------------

    contributions = _local_sensitivity(
        adapter,
        X,
        index,
    )

    if contributions is not None:
        return {
            "index": index,
            "actual": _to_serializable(actual),
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
            "method": "feature_sensitivity",
        }

    return {
        "index": index,
        "actual": _to_serializable(actual),
        "prediction": prediction,
        "probability": probability,
        "probability_label": probability_label,
        "features": {},
        "contributions": [],
        "method": "unavailable",
        "reason": (
            "Local explanation unavailable."
        ),
    }


global_feature_importance = global_explanation