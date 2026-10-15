# 🏥 Personal Health Guard
**Machine Learning Based Healthcare Prediction System**
*Team: Ranjith S, M Durga Prasad, Kishore Kumar M — Guided by Mrs. Saranya, AI&DS*

---

## 📋 Project Overview

Personal Health Guard is an ML-powered web application that predicts health risks (Normal / At Risk / Critical) based on user-provided health parameters. It uses:

- **Ensemble Learning**: AdaBoost + Random Forest + Gradient Boosting (Soft Voting)
- **MICE Imputation**: Multiple Imputation by Chained Equations for missing data
- **RFE Feature Selection**: Recursive Feature Elimination to select top 8 features
- **Flask REST API**: Backend prediction service
- **SQLite**: Persistent health record storage

---

## 🗂️ Project Structure

```
personal_health_guard/
├── backend/
│   ├── app.py              # Flask API server
│   ├── train_model.py      # ML model training script
│   └── models/             # Saved model artifacts (auto-generated)
│       ├── health_model.pkl
│       ├── imputer.pkl
│       ├── scaler.pkl
│       └── features.pkl
├── frontend/
│   └── templates/
│       └── index.html      # Full-stack single-file frontend
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the ML Model
```bash
cd backend
python train_model.py
```
This will:
- Generate synthetic health training data (2000 samples)
- Apply MICE imputation for missing values
- Apply RFE feature selection (selects top 8 features)
- Train ensemble model (AdaBoost + RF + GradientBoosting)
- Save model artifacts to `backend/models/`

### 3. Run the Application
```bash
cd backend
python app.py
```

### 4. Open in Browser
Navigate to: `http://localhost:5000`

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serve frontend UI |
| POST | `/api/predict` | Predict health risk |
| GET | `/api/history` | Get last 20 assessments |
| GET | `/api/stats` | Dashboard statistics |
| GET | `/api/health` | Server health check |

### POST /api/predict — Request Body
```json
{
  "age": 45,
  "gender": 1,
  "bmi": 28.5,
  "blood_pressure_sys": 135,
  "blood_pressure_dia": 85,
  "blood_sugar": 145,
  "heart_rate": 80,
  "cholesterol": 210,
  "smoking": 0,
  "physical_activity": 3
}
```

### POST /api/predict — Response
```json
{
  "prediction": 1,
  "label": "At Risk",
  "color": "#f59e0b",
  "confidence": 72.4,
  "probabilities": {
    "Normal": 18.2,
    "At Risk": 72.4,
    "Critical": 9.4
  },
  "health_score": 65,
  "advice": [
    {
      "icon": "🩸",
      "title": "Blood Sugar Alert",
      "text": "Monitor carbohydrate intake..."
    }
  ],
  "timestamp": "2024-01-15T10:30:00"
}
```

---

## 🤖 ML Pipeline

```
Raw Input Data
     │
     ▼
MICE Imputation ──── Handle missing values
     │
     ▼
StandardScaler ───── Feature normalization
     │
     ▼
RFE Feature Sel. ─── Select top 8 features
     │
     ▼
Ensemble Model
  ├── AdaBoost (n=100, lr=0.5)
  ├── Random Forest (n=100)
  └── Gradient Boosting (n=100, lr=0.1)
     │  Soft Voting
     ▼
Prediction: Normal / At Risk / Critical
+ Probability scores + Health advice
```

---

## 📊 System Modules

1. **Data Collection** — User health parameter inputs
2. **Data Preprocessing** — MICE imputation + StandardScaler
3. **Feature Selection** — RFE (8 of 10 features selected)
4. **ML Prediction Engine** — Ensemble AdaBoost model
5. **Web Interface** — Flask + HTML/CSS/JS
6. **Health Record Storage** — SQLite database

---

## 🛠️ Tech Stack

- Python 3.8+ | Flask | Flask-CORS
- Scikit-learn (AdaBoost, RandomForest, GradientBoosting, RFE, IterativeImputer)
- NumPy | Pandas | Joblib
- SQLite3
- HTML5 | CSS3 (custom properties) | Vanilla JavaScript
- Google Fonts (Syne + DM Sans)
