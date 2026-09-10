from pathlib import Path
from typing import Any

import joblib

from backend.core.model_bundle import load_model_bundle


def _has_predict_api(obj: Any) -> bool:
    """Return True if an object looks like a usable ML estimator."""
    return (
        hasattr(obj, "predict")
        or hasattr(obj, "predict_proba")
    )


def _extract_estimator(obj: Any) -> Any:
    """
    Extract an estimator from common model-bundle structures.

    Supports:
        - normal sklearn/xgboost estimators
        - Model Autopsy mixed-output bundles
        - dictionaries containing an estimator/model
        - nested dictionaries
    """

    if _has_predict_api(obj):
        return obj

    if isinstance(obj, dict):
        # Common model-bundle keys.
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
            if key in obj:
                candidate = _extract_estimator(obj[key])

                if _has_predict_api(candidate):
                    return candidate

        # Fall back to searching nested values.
        for value in obj.values():
            try:
                candidate = _extract_estimator(value)

                if _has_predict_api(candidate):
                    return candidate
            except (TypeError, ValueError):
                continue

    raise TypeError(
        "Unsupported model artifact. "
        f"Loaded object of type: {type(obj).__name__}. "
        "Expected an estimator with predict()/predict_proba(), "
        "or a supported model bundle."
    )


def load_model(
    model_path: str | Path,
):
    """
    Load a supported model artifact.

    Supports:
        - sklearn estimators
        - XGBoost / compatible estimators
        - normal joblib models
        - Model Autopsy mixed-output bundles
        - common dictionary-based model bundles
    """

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    loaded = joblib.load(model_path)

    # Model Autopsy's own bundle format.
    if (
        isinstance(loaded, dict)
        and loaded.get("type") == "mixed_output"
    ):
        return load_model_bundle(model_path)

    # Normal estimator or third-party model bundle.
    return _extract_estimator(loaded)