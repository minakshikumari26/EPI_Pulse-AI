from pydantic import BaseModel, Field


class DiseaseInput(BaseModel):
    region: str = Field(..., examples=["Delhi"])
    cases: int = Field(..., ge=0, examples=[60])
    temperature: float = Field(..., examples=[32.0])
    humidity: float = Field(..., ge=0, examples=[88.0])
    rainfall: float = Field(..., ge=0, examples=[12.0])


class PredictionResponse(BaseModel):
    region: str
    risk_score: float
    alert: str

