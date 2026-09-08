from pydantic import BaseModel


class AnalysisResponse(BaseModel):
    success: bool
    report: dict