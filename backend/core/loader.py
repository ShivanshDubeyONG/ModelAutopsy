from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from backend.core.model_bundle import (
    load_model_bundle,
)


def _has_predict_api(
    obj: Any,
) -> bool:

    return callable(
        getattr(
            obj,
            "predict",
            None,
        )
    )


def _extract_estimator(
    obj: Any,
) -> Any:

    if _has_predict_api(
        obj
    ):
        return obj

    if isinstance(
        obj,
        dict,
    ):

        preferred_keys = (
            "model",
            "estimator",
            "classifier",
            "regressor",
            "pipeline",
            "model_object",
            "xgb_model",
        )

        for key in preferred_keys:

            if key not in obj:
                continue

            try:

                candidate = (
                    _extract_estimator(
                        obj[key]
                    )
                )

                if _has_predict_api(
                    candidate
                ):
                    return candidate

            except TypeError:
                pass

        for value in obj.values():

            try:

                candidate = (
                    _extract_estimator(
                        value
                    )
                )

                if _has_predict_api(
                    candidate
                ):
                    return candidate

            except TypeError:
                continue

    raise TypeError(
        "Unsupported model artifact. "
        f"Loaded object of type: "
        f"{type(obj).__name__}. "
        "Expected an estimator with "
        "predict(), or a supported "
        "model bundle."
    )


def load_model(
    model_path: str | Path,
):
    """
    Load a trusted ML artifact.

    Supported:
        .joblib
        .pkl
        .pickle

    NOTE:
    joblib/pickle deserialization can execute
    arbitrary code. Only trusted model artifacts
    should be loaded.

    Model Autopsy intentionally does NOT execute
    arbitrary uploaded .py files.
    """

    model_path = Path(
        model_path
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: "
            f"{model_path}"
        )

    if (
        model_path.suffix.lower()
        not in {
            ".joblib",
            ".pkl",
            ".pickle",
        }
    ):
        raise ValueError(
            "Unsupported model format. "
            "Use .joblib, .pkl, or .pickle."
        )

    loaded = joblib.load(
        model_path
    )

    # Model Autopsy mixed-output bundle.
    if (
        isinstance(
            loaded,
            dict,
        )
        and loaded.get(
            "type"
        )
        == "mixed_output"
    ):
        return load_model_bundle(
            model_path
        )

    # Normal estimator or common
    # third-party dictionary bundle.
    return _extract_estimator(
        loaded
    )