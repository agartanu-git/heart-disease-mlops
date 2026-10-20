"""Pydantic schemas for the prediction API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    """A single patient record. Field ranges mirror the UCI dataset."""

    age: float = Field(..., ge=0, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=0, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=0, le=300, description="Resting blood pressure")
    chol: float = Field(..., ge=0, le=700, description="Serum cholesterol mg/dl")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results")
    thalach: float = Field(..., ge=0, le=250, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression")
    slope: int = Field(..., ge=0, le=3, description="Slope of peak exercise ST")
    ca: float = Field(..., ge=0, le=4, description="Number of major vessels (0-3)")
    thal: float = Field(..., ge=0, le=7, description="Thalassemia (3,6,7)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 63,
                "sex": 1,
                "cp": 1,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 2,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 3,
                "ca": 0,
                "thal": 6,
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = no disease, 1 = disease")
    label: str
    confidence: float = Field(..., description="Probability of predicted class")
    probability_disease: float


class BatchRequest(BaseModel):
    records: List[PatientFeatures]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: Optional[str] = None
