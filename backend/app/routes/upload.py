from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, UploadFile


router = APIRouter(
    prefix="/api/upload",
    tags=["upload"],
)

UPLOAD_DIR = Path("backend/uploads")
UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


@router.post("/")
async def upload_file(
    file: UploadFile = File(...),
):
    extension = Path(
        file.filename or ""
    ).suffix.lower()

    if extension not in {
        ".csv",
        ".joblib",
        ".pkl",
    }:
        return {
            "success": False,
            "error": "Unsupported file type.",
        }

    file_id = uuid4().hex
    destination = (
        UPLOAD_DIR
        / f"{file_id}{extension}"
    )

    content = await file.read()

    destination.write_bytes(content)

    return {
        "success": True,
        "file_id": file_id,
        "filename": file.filename,
        "path": str(destination),
    }