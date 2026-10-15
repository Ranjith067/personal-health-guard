"""
Personal Health Guard - Flask Backend API
Machine Learning Based Healthcare Prediction System
"""

from flask import Flask, request, jsonify, render_template
try:
    from flask_cors import CORS
except ImportError:
    CORS = None  # optional; install flask-cors for cross-origin support
import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime
import sqlite3

app = Flask(__name__, template_folder='../frontend/templates', static_folder='../frontend/static')
if CORS: CORS(app)

# ─── Load ML Artifacts ────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

model    = joblib.load(os.path.join(MODEL_DIR, 'health_model.pkl'))
imputer  = joblib.load(os.path.join(MODEL_DIR, 'imputer.pkl'))
scaler   = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
features = joblib.load(os.path.join(MODEL_DIR, 'features.pkl'))

RISK_LABELS    = {0: 'Normal', 1: 'At Risk', 2: 'Critical'}
RISK_COLORS    = {0: '#22c55e', 1: '#f59e0b', 2: '#ef4444'}
RISK_ICONS     = {0: '✅', 1: '⚠️', 2: '🚨'}

ALL_FEATURES = ['age', 'gender', 'bmi', 'blood_pressure_sys', 'blood_pressure_dia',
                'blood_sugar', 'heart_rate', 'cholesterol', 'smoking', 'physical_activity']

# ─── Database Setup ────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), 'health_records.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        age INTEGER,
        gender TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS health_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        age INTEGER,
        gender INTEGER,
        bmi REAL,
        blood_pressure_sys INTEGER,
        blood_pressure_dia INTEGER,
        blood_sugar INTEGER,
        heart_rate INTEGER,
        cholesterol INTEGER,
        smoking INTEGER,
        physical_activity INTEGER,
        prediction INTEGER,
        confidence REAL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )''')
    conn.commit()
    conn.close()

init_db()

# ─── Helper Functions ──────────────────────────────────────────────────────────
def get_health_advice(prediction, data):
    advice = []
    
    if data.get('blood_sugar', 0) > 140:
        advice.append({'icon': '🍬', 'title': 'Blood Sugar Alert', 'text': 'Monitor carbohydrate intake and consult a diabetologist. Consider HbA1c testing.'})
    if data.get('blood_pressure_sys', 0) > 130:
        advice.append({'icon': '💊', 'title': 'Blood Pressure', 'text': 'Reduce sodium intake, increase physical activity, and monitor BP regularly.'})
    if data.get('bmi', 0) > 30:
        advice.append({'icon': '🏃', 'title': 'Weight Management', 'text': 'A 5–10% weight reduction can significantly reduce health risks. Consider a dietitian consultation.'})
    if data.get('cholesterol', 0) > 200:
        advice.append({'icon': '🫀', 'title': 'Cholesterol', 'text': 'Limit saturated fats, increase omega-3 intake, and consider statins if advised by your doctor.'})
    if data.get('smoking', 0) == 1:
        advice.append({'icon': '🚭', 'title': 'Smoking Cessation', 'text': 'Quitting smoking is the single most impactful health change you can make.'})
    if data.get('physical_activity', 7) < 3:
        advice.append({'icon': '🧘', 'title': 'Exercise', 'text': 'Aim for at least 150 minutes of moderate exercise weekly (WHO recommendation).'})
    if data.get('heart_rate', 0) > 100:
        advice.append({'icon': '❤️', 'title': 'Heart Rate', 'text': 'Elevated resting heart rate detected. Deep breathing exercises and stress management can help.'})
    
    if prediction == 2 and not advice:
        advice.append({'icon': '🏥', 'title': 'Urgent Consultation', 'text': 'Critical health markers detected. Please consult a healthcare professional immediately.'})
    elif prediction == 0 and not advice:
        advice.append({'icon': '🌟', 'title': 'Great Health!', 'text': 'Your health parameters look excellent! Keep up your healthy lifestyle habits.'})
    
    return advice

def calculate_health_score(data):
    score = 100
    if data.get('blood_sugar', 90) > 200: score -= 25
    elif data.get('blood_sugar', 90) > 140: score -= 12
    if data.get('blood_pressure_sys', 120) > 140: score -= 20
    elif data.get('blood_pressure_sys', 120) > 130: score -= 10
    if data.get('bmi', 22) > 35: score -= 15
    elif data.get('bmi', 22) > 30: score -= 8
    if data.get('cholesterol', 150) > 240: score -= 15
    elif data.get('cholesterol', 150) > 200: score -= 8
    if data.get('smoking', 0): score -= 15
    if data.get('heart_rate', 75) > 100: score -= 10
    pa = data.get('physical_activity', 3)
    score += min(pa * 2, 10)
    return max(0, min(100, score))

# ─── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Build full feature vector
        row = {f: data.get(f, np.nan) for f in ALL_FEATURES}
        df_full = pd.DataFrame([row])
        
        # Impute missing values
        df_imputed = pd.DataFrame(imputer.transform(df_full), columns=ALL_FEATURES)
        
        # Select RFE features
        df_selected = df_imputed[features]
        
        # Scale
        X_scaled = scaler.transform(df_selected)
        
        # Predict
        prediction = int(model.predict(X_scaled)[0])
        probabilities = model.predict_proba(X_scaled)[0].tolist()
        confidence = float(max(probabilities))
        
        health_score = calculate_health_score(data)
        advice = get_health_advice(prediction, data)
        
        # Save to DB
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''INSERT INTO health_records 
            (age, gender, bmi, blood_pressure_sys, blood_pressure_dia, blood_sugar,
             heart_rate, cholesterol, smoking, physical_activity, prediction, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (data.get('age'), data.get('gender'), data.get('bmi'),
             data.get('blood_pressure_sys'), data.get('blood_pressure_dia'),
             data.get('blood_sugar'), data.get('heart_rate'), data.get('cholesterol'),
             data.get('smoking'), data.get('physical_activity'), prediction, confidence))
        conn.commit()
        record_id = c.lastrowid
        conn.close()
        
        return jsonify({
            'record_id': record_id,
            'prediction': prediction,
            'label': RISK_LABELS[prediction],
            'color': RISK_COLORS[prediction],
            'icon': RISK_ICONS[prediction],
            'confidence': round(confidence * 100, 1),
            'probabilities': {
                'Normal': round(probabilities[0] * 100, 1),
                'At Risk': round(probabilities[1] * 100, 1),
                'Critical': round(probabilities[2] * 100, 1),
            },
            'health_score': health_score,
            'advice': advice,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''SELECT id, age, bmi, blood_pressure_sys, blood_sugar, heart_rate,
                     cholesterol, prediction, confidence, timestamp
                     FROM health_records ORDER BY timestamp DESC LIMIT 20''')
        rows = c.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            history.append({
                'id': row[0],
                'age': row[1],
                'bmi': row[2],
                'blood_pressure_sys': row[3],
                'blood_sugar': row[4],
                'heart_rate': row[5],
                'cholesterol': row[6],
                'prediction': row[7],
                'label': RISK_LABELS.get(row[7], 'Unknown'),
                'color': RISK_COLORS.get(row[7], '#888'),
                'confidence': round((row[8] or 0) * 100, 1),
                'timestamp': row[9]
            })
        
        return jsonify({'history': history})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM health_records')
        total = c.fetchone()[0]
        c.execute('SELECT prediction, COUNT(*) FROM health_records GROUP BY prediction')
        dist = dict(c.fetchall())
        c.execute('SELECT AVG(bmi), AVG(blood_sugar), AVG(blood_pressure_sys), AVG(heart_rate) FROM health_records')
        avgs = c.fetchone()
        conn.close()
        
        return jsonify({
            'total_assessments': total,
            'distribution': {
                'Normal': dist.get(0, 0),
                'At Risk': dist.get(1, 0),
                'Critical': dist.get(2, 0),
            },
            'averages': {
                'bmi': round(avgs[0] or 0, 1),
                'blood_sugar': round(avgs[1] or 0, 1),
                'blood_pressure': round(avgs[2] or 0, 1),
                'heart_rate': round(avgs[3] or 0, 1),
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'model': 'Personal Health Guard v1.0', 'features': features})

if __name__ == '__main__':
    print("🏥 Personal Health Guard Backend Starting...")
    print(f"   Model features: {features}")
    app.run(debug=True, host='0.0.0.0', port=5000)
