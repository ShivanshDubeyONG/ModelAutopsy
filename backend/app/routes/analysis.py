from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, UploadFile

from backend.core.pipeline import run_autopsy


router = APIRouter(
    prefix="/api",
    tags=["analysis"],
)

UPLOAD_DIR = Path("backend/uploads")
UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


@router.post("/analyze")
async def analyze(
    model: UploadFile = File(...),
    dataset: UploadFile = File(...),
    target_column: str = Form(...),
):
    model_extension = Path(
        model.filename or ""
    ).suffix.lower()

    dataset_extension = Path(
        dataset.filename or ""
    ).suffix.lower()

    if model_extension not in {
        ".joblib",
        ".pkl",
    }:
        return {
            "success": False,
            "error": (
                "Model must be .joblib or .pkl."
            ),
        }

    if dataset_extension != ".csv":
        return {
            "success": False,
            "error": "Dataset must be a CSV file.",
        }

    model_path = (
        UPLOAD_DIR
        / f"{uuid4().hex}{model_extension}"
    )

    dataset_path = (
        UPLOAD_DIR
        / f"{uuid4().hex}.csv"
    )

    model_path.write_bytes(
        await model.read()
    )

    dataset_path.write_bytes(
        await dataset.read()
    )

    try:
        report = run_autopsy(
            model_path,
            dataset_path,
            target_column,
        )

        return {
            "success": True,
            "report": report,
        }

    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
        }

    finally:
        model_path.unlink(
            missing_ok=True
        )

        dataset_path.unlink(
            missing_ok=True
        )