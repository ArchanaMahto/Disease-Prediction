import os
import json
import numpy as np
import joblib
from typing import List, Dict, Any, Tuple
from fastapi import HTTPException

MODEL_DIR = "saved_model"

AVAILABLE_MODELS = {
    "naive_bayes": "Naive Bayes",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting"
}

DISCLAIMER_TEXT = (
    "Educational & Triage Aid Only. This system does not provide clinical medical diagnosis. "
    "If you are experiencing severe symptoms, please consult a licensed medical professional immediately."
)

class MedicalPredictor:
    def __init__(self):
        self.models = {}
        self.benchmarks = {}
        self.symptoms_list = []
        self.symptom_map = {}
        self.symptom_indices = {}
        self.is_loaded = False

    def load_artifacts(self):
        if not os.path.exists(MODEL_DIR):
            raise RuntimeError(f"Model directory '{MODEL_DIR}' does not exist. Run main.py first.")

        # Load symptoms metadata
        symptoms_path = os.path.join(MODEL_DIR, "symptoms.json")
        if not os.path.exists(symptoms_path):
            raise RuntimeError("symptoms.json metadata not found.")
            
        with open(symptoms_path, "r") as f:
            self.symptoms_list = json.load(f)
            
        self.symptom_map = {item["id"]: item["name"] for item in self.symptoms_list}
        self.symptom_indices = {item["id"]: idx for idx, item in enumerate(self.symptoms_list)}

        # Load benchmark metrics
        benchmarks_path = os.path.join(MODEL_DIR, "benchmarks.json")
        if os.path.exists(benchmarks_path):
            with open(benchmarks_path, "r") as f:
                self.benchmarks = json.load(f)

        # Load saved joblib models
        for model_id in AVAILABLE_MODELS.keys():
            model_file = os.path.join(MODEL_DIR, f"{model_id}.joblib")
            if os.path.exists(model_file):
                self.models[model_id] = joblib.load(model_file)
            else:
                print(f"Warning: Model file '{model_file}' not found.")

        self.is_loaded = True
        print(f"Loaded {len(self.models)} models and {len(self.symptoms_list)} symptoms.")

    def get_all_symptoms(self) -> List[Dict[str, str]]:
        return self.symptoms_list

    def get_benchmarks(self) -> Dict[str, Any]:
        return self.benchmarks

    def _extract_feature_contributions(self, model, model_id: str, selected_symptom_ids: List[str], disease: str) -> List[Dict[str, Any]]:
        contributions = []
        if not selected_symptom_ids:
            return contributions

        feature_importance_map = {}

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            for sym_id in selected_symptom_ids:
                idx = self.symptom_indices.get(sym_id)
                if idx is not None and idx < len(importances):
                    feature_importance_map[sym_id] = float(importances[idx])
        elif model_id == "naive_bayes" and hasattr(model, "feature_log_prob_"):
            # For Naive Bayes, estimate contribution via log likelihood for the predicted class
            classes = list(model.classes_)
            if disease in classes:
                class_idx = classes.index(disease)
                log_probs = model.feature_log_prob_[class_idx]
                for sym_id in selected_symptom_ids:
                    idx = self.symptom_indices.get(sym_id)
                    if idx is not None and idx < len(log_probs):
                        # Convert log prob to relative weight
                        feature_importance_map[sym_id] = float(np.exp(log_probs[idx]))

        # Normalize importance scores among selected symptoms
        total_importance = sum(feature_importance_map.values())
        if total_importance > 0:
            for sym_id in selected_symptom_ids:
                raw_score = feature_importance_map.get(sym_id, 0.0)
                norm_score = round(raw_score / total_importance, 3)
                contributions.append({
                    "symptom_id": sym_id,
                    "name": self.symptom_map.get(sym_id, sym_id),
                    "importance_score": norm_score
                })
        else:
            # Fallback uniform weighting if no importances available
            equal_score = round(1.0 / len(selected_symptom_ids), 3)
            for sym_id in selected_symptom_ids:
                contributions.append({
                    "symptom_id": sym_id,
                    "name": self.symptom_map.get(sym_id, sym_id),
                    "importance_score": equal_score
                })

        contributions.sort(key=lambda x: x["importance_score"], reverse=True)
        return contributions

    def predict(self, input_symptom_ids: List[str], model_id: str = "random_forest", top_n: int = 5) -> Dict[str, Any]:
        if not self.is_loaded:
            self.load_artifacts()

        if model_id not in self.models:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid model '{model_id}'. Available options: {list(AVAILABLE_MODELS.keys())}"
            )

        if not input_symptom_ids:
            raise HTTPException(status_code=400, detail="At least 1 symptom must be selected.")

        # Validate invalid/unknown symptom IDs
        invalid_symptoms = [s for s in input_symptom_ids if s not in self.symptom_indices]
        if invalid_symptoms:
            raise HTTPException(
                status_code=400, 
                detail=f"Unknown symptom ID(s): {invalid_symptoms}. Use /api/symptoms for valid list."
            )

        # Build binary feature vector
        vector = np.zeros(len(self.symptoms_list), dtype=int)
        for sym_id in input_symptom_ids:
            idx = self.symptom_indices[sym_id]
            vector[idx] = 1

        model = self.models[model_id]
        
        # Calculate probabilities
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([vector])[0]
            classes = model.classes_
            
            # Sort top_n diseases by probability descending
            top_indices = np.argsort(probabilities)[::-1][:top_n]
            
            predictions = []
            for rank_idx, idx in enumerate(top_indices, start=1):
                disease_name = classes[idx]
                prob = float(probabilities[idx])
                conf_pct = round(prob * 100, 2)
                
                # Extract feature contributions
                contributions = self._extract_feature_contributions(model, model_id, input_symptom_ids, disease_name)
                
                predictions.append({
                    "rank": rank_idx,
                    "disease": disease_name,
                    "confidence_percentage": conf_pct,
                    "contributing_symptoms": contributions
                })
        else:
            # Fallback for models without predict_proba
            pred_class = model.predict([vector])[0]
            contributions = self._extract_feature_contributions(model, model_id, input_symptom_ids, pred_class)
            predictions = [{
                "rank": 1,
                "disease": pred_class,
                "confidence_percentage": 100.0,
                "contributing_symptoms": contributions
            }]

        return {
            "model_used": model_id,
            "model_display_name": AVAILABLE_MODELS.get(model_id, model_id),
            "predictions": predictions,
            "disclaimer": DISCLAIMER_TEXT
        }

predictor = MedicalPredictor()
