from pathlib import Path

import joblib

from backend.core.model_bundle import (
    MixedOutputModel,
    load_model_bundle,
)


def load_model(
    model_path: str | Path,
):
    """
    Load a supported model artifact.

    Supports:
        - normal sklearn/joblib models
        - Model Autopsy mixed-output bundles
    """

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    loaded = joblib.load(model_path)

    if (
        isinstance(loaded, dict)
        and loaded.get("type") == "mixed_output"
    ):
        return load_model_bundle(
            model_path
        )

    return loaded