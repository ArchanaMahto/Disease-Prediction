# 🩺 MediPulse | Full-Stack Differential Diagnosis & ML Model Benchmark Engine

MediPulse is an interactive full-stack web application that predicts differential diagnoses based on user-selected symptoms, provides feature explainability, and benchmarks multiple machine learning algorithms side-by-side.

> [!WARNING]
> **Medical Disclaimer**: This application is strictly an educational triage aid and research tool. It does not provide clinical medical diagnoses. Always consult a licensed medical professional for health concerns.

---

## 🌟 Key Features

- **🔍 Searchable Symptom Picker**: Multi-select autocomplete interface with real-time fuzzy search across 132 canonical symptoms.
- **📊 Top 3–5 Differential Diagnosis**: Returns candidate disease predictions with probability confidence scores instead of a single verdict.
- **💡 Explainable AI (XAI)**: Displays key contributing symptoms for each predicted disease using feature importance attributions.
- **🎛️ Algorithm Switcher**: Dynamically switch predictions between **Random Forest**, **Naive Bayes**, **Decision Tree**, and **Gradient Boosting**.
- **📈 Model Benchmarking Dashboard**: Compare Accuracy, Precision, Recall, F1-Score, and inference latency metrics across all 4 scikit-learn models.
- **⚡ Decoupled REST API**: High-performance FastAPI backend delivering low-latency inferences (< 50ms).

---

## 🚀 Tech Stack

- **Backend**: Python 3.11, FastAPI, Pydantic v2, Uvicorn
- **Machine Learning**: Scikit-Learn, Joblib, NumPy, Pandas
- **Frontend**: Vanilla HTML5, Modern CSS3 (Glassmorphism, CSS Variables), ES6 JavaScript
- **Testing**: pytest, httpx

---

## 🛠️ Installation & Setup

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/YOUR_USERNAME/disease-prediction.git
cd disease-prediction

python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train Models & Generate Dataset
```bash
python main.py
```

### 4. Start Server
```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000`** in your browser.

---

## 🧪 Running Tests
```bash
python -m pytest tests/test_api.py -v
```

---

## 📄 License
MIT License
