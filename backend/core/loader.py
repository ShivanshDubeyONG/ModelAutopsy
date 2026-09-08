from pathlib import Path
from typing import Any

import joblib


SUPPORTED_EXTENSIONS = {".joblib", ".pkl"}


def load_model(model_path: str | Path) -> Any:
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Unsupported model format. Use .joblib or .pkl."
        )

    model = joblib.load(path)

    if not hasattr(model, "predict"):
        raise ValueError(
            "The uploaded object does not expose predict()."
        )

    return model