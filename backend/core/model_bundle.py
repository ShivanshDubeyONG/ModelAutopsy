from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class MixedOutputModel:
    """
    Adapter for models that produce multiple outputs with
    different problem types.

    Example:

        Reference_Parameter -> regression model
        Validity_Label      -> classification model
    """

    def __init__(
        self,
        models: dict[str, Any],
        problem_types: dict[str, str],
    ):
        if not models:
            raise ValueError(
                "At least one output model is required."
            )

        if set(models) != set(problem_types):
            raise ValueError(
                "models and problem_types must contain "
                "the same output names."
            )

        for name, problem_type in problem_types.items():
            if problem_type not in {
                "classification",
                "regression",
            }:
                raise ValueError(
                    f"Unsupported problem type for "
                    f"'{name}': {problem_type}"
                )

        self.models = models
        self.problem_types = problem_types
        self.output_names = list(models.keys())

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return predictions in:

            samples x outputs
        """

        predictions = []

        for name in self.output_names:
            model = self.models[name]

            prediction = np.asarray(
                model.predict(X)
            ).reshape(-1)

            predictions.append(prediction)

        return np.column_stack(predictions)

    def predict_proba(self, X: pd.DataFrame) -> list:
        """
        Return classification probabilities for each output.

        Regression outputs return None.
        """

        probabilities = []

        for name in self.output_names:
            model = self.models[name]

            if (
                self.problem_types[name]
                != "classification"
            ):
                probabilities.append(None)
                continue

            if not hasattr(model, "predict_proba"):
                probabilities.append(None)
                continue

            try:
                probabilities.append(
                    model.predict_proba(X)
                )
            except Exception:
                probabilities.append(None)

        return probabilities

    def get_output_model(
        self,
        output_name: str,
    ):
        if output_name not in self.models:
            raise KeyError(
                f"Unknown output: '{output_name}'"
            )

        return self.models[output_name]


def save_model_bundle(
    path: str | Path,
    models: dict[str, Any],
    problem_types: dict[str, str],
) -> None:
    """
    Save a mixed-output model bundle.
    """

    bundle = {
        "type": "mixed_output",
        "models": models,
        "problem_types": problem_types,
    }

    joblib.dump(
        bundle,
        path,
    )


def load_model_bundle(
    path: str | Path,
) -> MixedOutputModel:
    """
    Load a mixed-output model bundle.
    """

    bundle = joblib.load(path)

    if not isinstance(bundle, dict):
        raise ValueError(
            "Invalid model bundle format."
        )

    if bundle.get("type") != "mixed_output":
        raise ValueError(
            "Model bundle is not a mixed-output bundle."
        )

    models = bundle.get("models")
    problem_types = bundle.get(
        "problem_types"
    )

    if not isinstance(models, dict):
        raise ValueError(
            "Model bundle is missing 'models'."
        )

    if not isinstance(problem_types, dict):
        raise ValueError(
            "Model bundle is missing 'problem_types'."
        )

    return MixedOutputModel(
        models=models,
        problem_types=problem_types,
    )