from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    symptoms: List[str] = Field(..., description="List of canonical symptom IDs", min_length=1)
    model_name: Optional[str] = Field("random_forest", description="Model algorithm identifier")
    top_n: Optional[int] = Field(5, ge=1, le=10, description="Number of top differential diagnosis items")

class AttributedSymptom(BaseModel):
    symptom_id: str
    name: str
    importance_score: float

class DiseasePrediction(BaseModel):
    rank: int
    disease: str
    confidence_percentage: float
    contributing_symptoms: List[AttributedSymptom]

class PredictResponse(BaseModel):
    model_used: str
    model_display_name: str
    predictions: List[DiseasePrediction]
    disclaimer: str

class SymptomItem(BaseModel):
    id: str
    name: str

class SymptomListResponse(BaseModel):
    total: int
    symptoms: List[SymptomItem]

class BenchmarkModelItem(BaseModel):
    name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    latency_ms: float
    train_time_sec: float

class BenchmarkResponse(BaseModel):
    models: Dict[str, BenchmarkModelItem]
