"""
Personal Health Guard - ML Model Training
Uses ensemble learning with AdaBoost for disease prediction.
Handles missing values with MICE-like imputation.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer  # MICE equivalent
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

np.random.seed(42)

# ─── Generate Synthetic Health Dataset ────────────────────────────────────────
def generate_health_data(n=2000):
    """Generate realistic synthetic health data for training."""
    age = np.random.randint(18, 80, n)
    gender = np.random.choice([0, 1], n)  # 0=Female, 1=Male
    bmi = np.round(np.random.normal(27, 6, n).clip(15, 50), 1)
    blood_pressure_sys = np.random.randint(90, 180, n)
    blood_pressure_dia = np.random.randint(60, 110, n)
    blood_sugar = np.random.randint(70, 280, n)
    heart_rate = np.random.randint(55, 120, n)
    cholesterol = np.random.randint(120, 300, n)
    smoking = np.random.choice([0, 1], n, p=[0.7, 0.3])
    physical_activity = np.random.randint(0, 7, n)  # days/week
    
    # Risk scoring (0=Normal, 1=At Risk, 2=Critical)
    risk_score = np.zeros(n)
    
    # Diabetes risk
    risk_score += (blood_sugar > 200).astype(int) * 2
    risk_score += ((blood_sugar > 140) & (blood_sugar <= 200)).astype(int)
    risk_score += (bmi > 30).astype(int)
    risk_score += (age > 45).astype(int) * 0.5
    
    # Heart disease risk
    risk_score += (blood_pressure_sys > 140).astype(int)
    risk_score += (cholesterol > 240).astype(int)
    risk_score += smoking.astype(int)
    risk_score += (heart_rate > 100).astype(int)
    risk_score -= physical_activity * 0.2
    
    # Classify
    labels = np.where(risk_score < 1.5, 0, np.where(risk_score < 3.5, 1, 2))
    
    # Randomly introduce missing values (5%)
    df = pd.DataFrame({
        'age': age,
        'gender': gender,
        'bmi': bmi,
        'blood_pressure_sys': blood_pressure_sys,
        'blood_pressure_dia': blood_pressure_dia,
        'blood_sugar': blood_sugar,
        'heart_rate': heart_rate,
        'cholesterol': cholesterol,
        'smoking': smoking,
        'physical_activity': physical_activity,
    })
    
    # Add 5% missing values randomly
    for col in ['bmi', 'cholesterol', 'blood_sugar']:
        mask = np.random.random(n) < 0.05
        df.loc[mask, col] = np.nan
    
    df['risk_label'] = labels
    return df


# ─── Train Pipeline ────────────────────────────────────────────────────────────
def train():
    print("📊 Generating health dataset...")
    df = generate_health_data(2000)
    
    X = df.drop('risk_label', axis=1)
    y = df['risk_label'].astype(int)
    
    print(f"   Dataset shape: {X.shape}")
    print(f"   Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Step 1: MICE-based imputation for missing values
    print("\n🔧 Applying MICE imputation for missing values...")
    imputer = IterativeImputer(max_iter=10, random_state=42)
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Step 2: Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Step 3: Feature selection via RFE
    print("🎯 Selecting features via Recursive Feature Elimination...")
    base_clf = RandomForestClassifier(n_estimators=50, random_state=42)
    rfe = RFE(estimator=base_clf, n_features_to_select=8, step=1)
    rfe.fit(X_scaled, y)
    selected_features = X.columns[rfe.support_].tolist()
    print(f"   Selected features: {selected_features}")
    X_selected = X_imputed[selected_features]
    X_scaled_sel = scaler.fit_transform(X_selected)
    
    # Step 4: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_sel, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Step 5: Ensemble model with AdaBoost + Voting
    print("\n🤖 Training ensemble model...")
    adaboost = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    
    ensemble = VotingClassifier(
        estimators=[('ada', adaboost), ('rf', rf), ('gb', gb)],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)
    
    # Step 6: Evaluate
    y_pred = ensemble.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'At Risk', 'Critical']))
    
    # Step 7: Save artifacts
    os.makedirs('models', exist_ok=True)
    joblib.dump(ensemble, 'models/health_model.pkl')
    joblib.dump(imputer, 'models/imputer.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(selected_features, 'models/features.pkl')
    
    # Save full scaler with all features for input preprocessing
    scaler_full = StandardScaler()
    scaler_full.fit(imputer.transform(X))
    joblib.dump(scaler_full, 'models/scaler_full.pkl')
    
    print("\n💾 Model artifacts saved to models/")
    return acc, selected_features


if __name__ == '__main__':
    acc, features = train()
    print(f"\n🏥 Personal Health Guard model ready! Accuracy: {acc*100:.1f}%")
