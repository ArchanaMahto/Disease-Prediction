import os
import json
import time
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

DATASET_DIR = "dataset"
MODEL_DIR = "saved_model"

SYMPTOMS_LIST = [
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", "shivering", "chills", 
    "joint_pain", "stomach_pain", "acidity", "ulcers_on_tongue", "muscle_wasting", "vomiting", 
    "burning_micturition", "spotting_urination", "fatigue", "weight_gain", "anxiety", 
    "cold_hands_and_feets", "mood_swings", "weight_loss", "restlessness", "lethargy", 
    "patches_in_throat", "irregular_sugar_level", "cough", "high_fever", "sunken_eyes", 
    "breathlessness", "sweating", "dehydration", "indigestion", "headache", "yellowish_skin", 
    "dark_urine", "nausea", "loss_of_appetite", "pain_behind_the_eyes", "back_pain", 
    "constipation", "abdominal_pain", "diarrhoea", "mild_fever", "yellow_urine", 
    "yellowing_of_eyes", "acute_liver_failure", "fluid_overload", "swelling_of_stomach", 
    "swelled_lymph_nodes", "malaise", "blurred_and_distorted_vision", "phlegm", 
    "throat_irritation", "redness_of_eyes", "sinus_pressure", "runny_nose", "congestion", 
    "chest_pain", "weakness_in_limbs", "fast_heart_rate", "pain_during_bowel_movements", 
    "pain_in_anal_region", "bloody_stool", "irritation_in_anus", "neck_pain", "dizziness", 
    "cramps", "bruising", "obesity", "swollen_legs", "swollen_blood_vessels", 
    "puffy_face_and_eyes", "enlarged_thyroid", "brittle_nails", "swollen_extremeties", 
    "excessive_hunger", "extra_marital_contacts", "drying_and_tingling_lips", "slurred_speech", 
    "knee_pain", "hip_joint_pain", "muscle_weakness", "stiff_neck", "swelling_joints", 
    "movement_stiffness", "spinning_movements", "loss_of_balance", "unsteadiness", 
    "weakness_of_one_body_side", "loss_of_smell", "bladder_discomfort", "foul_smell_of_urine", 
    "continuous_feel_of_urine", "passage_of_gases", "internal_itching", "toxic_look_(typhos)", 
    "depression", "irritability", "muscle_pain", "altered_sensorium", "red_spots_over_body", 
    "belly_pain", "abnormal_menstruation", "dischromic_patches", "watering_from_eyes", 
    "increased_appetite", "polyuria", "family_history", "mucoid_sputum", "rusty_sputum", 
    "lack_of_concentration", "visual_disturbances", "receiving_blood_transfusion", 
    "receiving_unsterile_injections", "coma", "stomach_bleeding", "distention_of_abdomen", 
    "history_of_alcohol_consumption", "fluid_overload_1", "blood_in_sputum", 
    "prominent_veins_on_calf", "palpitations", "painful_walking", "pus_filled_blisters", 
    "blackheads", "scurring", "skin_peeling", "silver_like_dusting", "small_dents_in_nails", 
    "inflammatory_nails", "blister", "red_sore_around_nose", "yellow_crust_ooze"
]

def format_symptom_name(sym):
    return sym.replace("_", " ").title()

def run_fast_train():
    # Kill background task if running
    train_df = pd.read_csv(os.path.join(DATASET_DIR, "Training.csv"))
    test_df = pd.read_csv(os.path.join(DATASET_DIR, "Testing.csv"))
    
    X_train = train_df[SYMPTOMS_LIST]
    y_train = train_df["prognosis"]
    X_test = test_df[SYMPTOMS_LIST]
    y_test = test_df["prognosis"]
    
    models = {
        "naive_bayes": MultinomialNB(),
        "decision_tree": DecisionTreeClassifier(max_depth=15, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=25, max_depth=12, random_state=42),
        "gradient_boosting": HistGradientBoostingClassifier(max_iter=30, random_state=42)
    }
    
    benchmark_results = {}
    
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        predict_start = time.time()
        y_pred = model.predict(X_test)
        predict_time = (time.time() - predict_start) / len(X_test) * 1000  # ms per sample
        
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        
        joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.joblib"))
        
        benchmark_results[name] = {
            "name": name.replace("_", " ").title(),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "latency_ms": round(predict_time, 3),
            "train_time_sec": round(train_time, 3)
        }
        print(f"Done [{name}] -> Accuracy: {acc:.4f}, Latency: {predict_time:.3f}ms")
        
    with open(os.path.join(MODEL_DIR, "benchmarks.json"), "w") as f:
        json.dump(benchmark_results, f, indent=2)
        
    symptoms_metadata = [
        {"id": sym, "name": format_symptom_name(sym)}
        for sym in SYMPTOMS_LIST
    ]
    with open(os.path.join(MODEL_DIR, "symptoms.json"), "w") as f:
        json.dump(symptoms_metadata, f, indent=2)
        
    print("Fast training complete!")

if __name__ == "__main__":
    run_fast_train()
