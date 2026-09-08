import os
import json
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
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

DISEASE_PROFILES = {
    "Fungal infection": ["itching", "skin_rash", "nodal_skin_eruptions", "dischromic_patches"],
    "Allergy": ["continuous_sneezing", "shivering", "chills", "watering_from_eyes"],
    "GERD": ["acidity", "ulcers_on_tongue", "vomiting", "cough", "chest_pain"],
    "Chronic cholestasis": ["itching", "vomiting", "yellowish_skin", "nausea", "loss_of_appetite"],
    "Drug Reaction": ["itching", "skin_rash", "stomach_pain", "burning_micturition", "spotting_urination"],
    "Peptic ulcer disease": ["vomiting", "loss_of_appetite", "abdominal_pain", "passage_of_gases"],
    "AIDS": ["muscle_wasting", "patches_in_throat", "high_fever", "extra_marital_contacts"],
    "Diabetes": ["fatigue", "weight_loss", "restlessness", "lethargy", "irregular_sugar_level", "increased_appetite", "polyuria"],
    "Gastroenteritis": ["vomiting", "sunken_eyes", "dehydration", "diarrhoea"],
    "Bronchial Asthma": ["fatigue", "cough", "high_fever", "breathlessness", "mucoid_sputum"],
    "Hypertension": ["headache", "chest_pain", "dizziness", "loss_of_balance", "lack_of_concentration"],
    "Migraine": ["acidity", "indigestion", "headache", "blurred_and_distorted_vision", "excessive_hunger", "stiff_neck", "depression", "irritability"],
    "Cervical spondylosis": ["back_pain", "neck_pain", "dizziness", "loss_of_balance"],
    "Paralysis (brain hemorrhage)": ["vomiting", "headache", "weakness_in_limbs", "altered_sensorium"],
    "Jaundice": ["itching", "vomiting", "fatigue", "weight_loss", "high_fever", "yellowish_skin", "dark_urine"],
    "Malaria": ["chills", "vomiting", "high_fever", "sweating", "headache", "nausea", "muscle_pain"],
    "Chicken pox": ["itching", "skin_rash", "fatigue", "lethargy", "high_fever", "headache", "loss_of_appetite", "mild_fever", "swelled_lymph_nodes", "red_spots_over_body"],
    "Dengue": ["skin_rash", "chills", "joint_pain", "vomiting", "fatigue", "high_fever", "headache", "nausea", "loss_of_appetite", "pain_behind_the_eyes", "back_pain", "muscle_pain"],
    "Typhoid": ["chills", "vomiting", "fatigue", "high_fever", "headache", "nausea", "constipation", "abdominal_pain", "diarrhoea", "toxic_look_(typhos)", "belly_pain"],
    "hepatitis A": ["joint_pain", "vomiting", "fatigue", "high_fever", "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "abdominal_pain", "diarrhoea", "yellowing_of_eyes"],
    "Hepatitis B": ["itching", "fatigue", "lethargy", "yellowish_skin", "dark_urine", "loss_of_appetite", "abdominal_pain", "yellowing_of_eyes", "receiving_blood_transfusion", "receiving_unsterile_injections"],
    "Hepatitis C": ["fatigue", "yellowish_skin", "nausea", "loss_of_appetite", "yellowing_of_eyes", "family_history"],
    "Hepatitis D": ["joint_pain", "vomiting", "fatigue", "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "abdominal_pain", "yellowing_of_eyes"],
    "Hepatitis E": ["joint_pain", "vomiting", "fatigue", "high_fever", "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "abdominal_pain", "yellowing_of_eyes", "acute_liver_failure", "coma"],
    "Alcoholic hepatitis": ["vomiting", "yellowish_skin", "abdominal_pain", "swelling_of_stomach", "history_of_alcohol_consumption", "fluid_overload_1"],
    "Tuberculosis": ["chills", "vomiting", "fatigue", "weight_loss", "cough", "high_fever", "breathlessness", "sweating", "loss_of_appetite", "mild_fever", "phlegm", "swelled_lymph_nodes", "malaise", "phlegm", "blood_in_sputum"],
    "Common Cold": ["continuous_sneezing", "chills", "fatigue", "cough", "high_fever", "headache", "swelled_lymph_nodes", "malaise", "phlegm", "throat_irritation", "redness_of_eyes", "sinus_pressure", "runny_nose", "congestion", "chest_pain", "loss_of_smell"],
    "Pneumonia": ["chills", "fatigue", "cough", "high_fever", "breathlessness", "sweating", "malaise", "chest_pain", "fast_heart_rate", "rusty_sputum"],
    "Dimorphic hemmorhoids(piles)": ["constipation", "pain_during_bowel_movements", "pain_in_anal_region", "bloody_stool", "irritation_in_anus"],
    "Heart attack": ["vomiting", "breathlessness", "sweating", "chest_pain"],
    "Varicose veins": ["fatigue", "cramps", "bruising", "obesity", "swollen_legs", "swollen_blood_vessels", "prominent_veins_on_calf"],
    "Hypothyroidism": ["fatigue", "weight_gain", "cold_hands_and_feets", "mood_swings", "lethargy", "dizziness", "puffy_face_and_eyes", "enlarged_thyroid", "brittle_nails", "swollen_extremeties"],
    "Hyperthyroidism": ["fatigue", "mood_swings", "weight_loss", "restlessness", "sweating", "diarrhoea", "fast_heart_rate", "excessive_hunger", "muscle_weakness", "irritability", "abnormal_menstruation"],
    "Hypoglycemia": ["vomiting", "fatigue", "anxiety", "sweating", "headache", "nausea", "blurred_and_distorted_vision", "excessive_hunger", "drying_and_tingling_lips", "slurred_speech", "irritability", "palpitations"],
    "Osteoarthristis": ["joint_pain", "neck_pain", "knee_pain", "hip_joint_pain", "swelling_joints", "painful_walking"],
    "Arthritis": ["muscle_weakness", "stiff_neck", "swelling_joints", "movement_stiffness", "painful_walking"],
    "(vertigo) Paroymsal  Positional Vertigo": ["vomiting", "headache", "nausea", "spinning_movements", "loss_of_balance", "unsteadiness"],
    "Acne": ["skin_rash", "pus_filled_blisters", "blackheads", "scurring"],
    "Urinary tract infection": ["burning_micturition", "bladder_discomfort", "foul_smell_of_urine", "continuous_feel_of_urine"],
    "Psoriasis": ["skin_rash", "joint_pain", "skin_peeling", "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails"],
    "Impetigo": ["skin_rash", "high_fever", "blister", "red_sore_around_nose", "yellow_crust_ooze"]
}

def generate_datasets():
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    train_rows = []
    test_rows = []
    
    np.random.seed(42)
    
    for disease, primary_symptoms in DISEASE_PROFILES.items():
        # Generate 100 training samples per disease with minor noise
        for _ in range(100):
            row = {sym: 0 for sym in SYMPTOMS_LIST}
            # Primary symptoms present with high probability
            for sym in primary_symptoms:
                if np.random.rand() > 0.05: # 95% chance present
                    row[sym] = 1
            # Random non-primary symptoms with low probability (noise)
            for sym in SYMPTOMS_LIST:
                if sym not in primary_symptoms and np.random.rand() < 0.01:
                    row[sym] = 1
            row["prognosis"] = disease
            train_rows.append(row)
            
        # Generate 25 test samples per disease
        for _ in range(25):
            row = {sym: 0 for sym in SYMPTOMS_LIST}
            for sym in primary_symptoms:
                if np.random.rand() > 0.08:
                    row[sym] = 1
            for sym in SYMPTOMS_LIST:
                if sym not in primary_symptoms and np.random.rand() < 0.01:
                    row[sym] = 1
            row["prognosis"] = disease
            test_rows.append(row)
            
    df_train = pd.DataFrame(train_rows)
    df_test = pd.DataFrame(test_rows)
    
    df_train.to_csv(os.path.join(DATASET_DIR, "Training.csv"), index=False)
    df_test.to_csv(os.path.join(DATASET_DIR, "Testing.csv"), index=False)
    
    print(f"Generated Training.csv ({len(df_train)} rows) and Testing.csv ({len(df_test)} rows).")

def format_symptom_name(sym):
    return sym.replace("_", " ").title()

def train_and_benchmark():
    train_df = pd.read_csv(os.path.join(DATASET_DIR, "Training.csv"))
    test_df = pd.read_csv(os.path.join(DATASET_DIR, "Testing.csv"))
    
    X_train = train_df[SYMPTOMS_LIST]
    y_train = train_df["prognosis"]
    X_test = test_df[SYMPTOMS_LIST]
    y_test = test_df["prognosis"]
    
    models = {
        "naive_bayes": MultinomialNB(),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=30, random_state=42)
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
        print(f"Model [{name}] -> Accuracy: {acc:.4f}, F1: {f1:.4f}, Latency: {predict_time:.3f}ms")
        
    with open(os.path.join(MODEL_DIR, "benchmarks.json"), "w") as f:
        json.dump(benchmark_results, f, indent=2)
        
    symptoms_metadata = [
        {"id": sym, "name": format_symptom_name(sym)}
        for sym in SYMPTOMS_LIST
    ]
    with open(os.path.join(MODEL_DIR, "symptoms.json"), "w") as f:
        json.dump(symptoms_metadata, f, indent=2)
        
    print(f"Serialized all 4 models and metadata into '{MODEL_DIR}/'.")

if __name__ == "__main__":
    generate_datasets()
    train_and_benchmark()
