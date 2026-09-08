from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.routes.analysis import router as analysis_router
from backend.app.routes.upload import router as upload_router


app = FastAPI(
    title="Model Autopsy",
    description=(
        "Forensic debugging and explainability "
        "for machine learning models."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    analysis_router
)

app.include_router(
    upload_router
)


@app.get("/health")
def health():
    return {
        "status": "operational",
        "service": "model-autopsy",
    }