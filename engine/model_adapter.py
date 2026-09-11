from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import warnings
import numpy as np
import pandas as pd


@dataclass
class ModelAdapter:
    model: Any
    output_index: int | None = None

    def _call_model(self, method, X):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "X has feature names, but StandardScaler "
                    "was fitted without feature names"
                ),
                category=UserWarning,
            )

            return method(X)

    @property
    def output_model(self):
        if self.output_index is None:
            return self.model

        output_names = getattr(
            self.model,
            "output_names",
            None,
        )

        if (
            output_names is not None
            and hasattr(
                self.model,
                "get_output_model",
            )
            and 0 <= self.output_index < len(output_names)
        ):
            return self.model.get_output_model(
                output_names[self.output_index]
            )

        estimators = getattr(
            self.model,
            "estimators_",
            None,
        )

        if estimators is not None:
            try:
                return estimators[self.output_index]
            except (IndexError, TypeError):
                pass

        return self.model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        array = np.asarray(
            self._call_model(
                self.model.predict,
                X,
            )
        )

        if (
            self.output_index is not None
            and array.ndim >= 2
        ):
            array = array[:, self.output_index]

        return array.reshape(-1)

    def predict_proba(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray | None:

        if not hasattr(
            self.model,
            "predict_proba",
        ):
            return None

        try:
            raw = self._call_model(
                self.model.predict_proba,
                X,
            )
        except Exception:
            return None

        if raw is None:
            return None

        if isinstance(raw, list):
            if self.output_index is None:
                return (
                    np.asarray(raw[0])
                    if raw
                    else None
                )

            if (
                0 <= self.output_index
                < len(raw)
            ):
                item = raw[self.output_index]

                return (
                    None
                    if item is None
                    else np.asarray(item)
                )

            return None

        array = np.asarray(raw)

        if self.output_index is None:
            return array

        if array.ndim == 3:
            if array.shape[2] > self.output_index:
                return array[
                    :,
                    :,
                    self.output_index,
                ]

            if array.shape[1] > self.output_index:
                return array[
                    :,
                    self.output_index,
                    :,
                ]

        return array

    def decision_function(self, X):
        if not hasattr(
            self.model,
            "decision_function",
        ):
            return None

        try:
            array = np.asarray(
                self._call_model(
                    self.model.decision_function,
                    X,
                )
            )
        except Exception:
            return None

        if (
            self.output_index is not None
            and array.ndim >= 2
        ):
            array = array[:, self.output_index]

        return array

    @property
    def classes_(self):
        classes = getattr(
            self.output_model,
            "classes_",
            None,
        )

        if classes is None:
            classes = getattr(
                self.model,
                "classes_",
                None,
            )

        return classes

    @property
    def feature_importances_(self):
        return getattr(
            self.output_model,
            "feature_importances_",
            None,
        )

    @property
    def coef_(self):
        return getattr(
            self.output_model,
            "coef_",
            None,
        )

    @property
    def is_pipeline(self):
        return (
            hasattr(self.model, "steps")
            and hasattr(
                self.model,
                "named_steps",
            )
        )

    @property
    def final_estimator(self):
        if self.is_pipeline:
            return self.model.steps[-1][1]

        return self.output_model

    @property
    def capabilities(self):
        return {
            "predict": hasattr(
                self.model,
                "predict",
            ),
            "predict_proba": hasattr(
                self.model,
                "predict_proba",
            ),
            "decision_function": hasattr(
                self.model,
                "decision_function",
            ),
            "feature_importances": hasattr(
                self.output_model,
                "feature_importances_",
            ),
            "coefficients": hasattr(
                self.output_model,
                "coef_",
            ),
            "pipeline": self.is_pipeline,
        }

    def describe(self):
        return {
            "type": type(
                self.model
            ).__name__,
            "output_model_type": type(
                self.output_model
            ).__name__,
            "final_estimator_type": type(
                self.final_estimator
            ).__name__,
            "output_index": self.output_index,
            "capabilities": self.capabilities,
        }


def get_model_adapter(
    model,
    output_index=None,
):
    if not callable(
        getattr(
            model,
            "predict",
            None,
        )
    ):
        raise TypeError(
            f"Unsupported model object: "
            f"{type(model).__name__}. "
            "Expected an object exposing predict()."
        )

    return ModelAdapter(
        model=model,
        output_index=output_index,
    )