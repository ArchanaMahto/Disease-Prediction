import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.schemas import (
    PredictRequest, PredictResponse, SymptomListResponse, SymptomItem, BenchmarkResponse
)
from backend.predictor import predictor

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model artifacts into memory on startup
    predictor.load_artifacts()
    yield

app = FastAPI(
    title="Differential Diagnosis & Model Benchmark API",
    description="Decoupled REST API serving disease prediction probabilities, explainability, and ML model comparisons.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for decoupled frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/symptoms", response_model=SymptomListResponse)
async def get_symptoms():
    symptoms = predictor.get_all_symptoms()
    return SymptomListResponse(
        total=len(symptoms),
        symptoms=[SymptomItem(id=s["id"], name=s["name"]) for s in symptoms]
    )

@app.post("/api/predict", response_model=PredictResponse)
async def predict_disease(request: PredictRequest):
    result = predictor.predict(
        input_symptom_ids=request.symptoms,
        model_id=request.model_name or "random_forest",
        top_n=request.top_n or 5
    )
    return result

@app.get("/api/models/compare", response_model=BenchmarkResponse)
async def compare_models():
    benchmarks = predictor.get_benchmarks()
    return BenchmarkResponse(models=benchmarks)

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": predictor.is_loaded,
        "available_models": list(predictor.models.keys())
    }

# Mount static frontend files if directory exists
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(frontend_dir, "index.html"))
