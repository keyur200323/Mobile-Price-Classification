"""
STEP 7: Machine Learning Model Training (Random Forest vs. Support Vector Machine)
"""

import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=" * 70)
print("🤖 STEP 7: MACHINE LEARNING - APPLY 2 ALGORITHMS (BASELINE)")
print("   Algorithm 1: Random Forest Classifier (Tree Ensemble)")
print("   Algorithm 2: Support Vector Classifier (Kernel SVM)")
print("=" * 70)

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ---------------------------------------------------------
# 7.1 LOAD SELECTED DATASET
# ---------------------------------------------------------
input_path = os.path.join("data", "mobile_selected.csv")
if not os.path.exists(input_path):
    print(f"❌ Error: {input_path} not found. Please run Step 6 first!")
    exit()

df = pd.read_csv(input_path)
print(f"📂 Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

X = df.drop(columns=['price_range'])
y = df['price_range']
feature_names = X.columns.tolist()

# ---------------------------------------------------------
# 7.2 TRAIN-TEST SPLIT (80/20 STRATIFIED)
# ---------------------------------------------------------
print("\n" + "-" * 70)
print("1. Performing Stratified Train-Test Split (80% / 20%)...")
print("-" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"  Training samples: {X_train.shape[0]}")
print(f"  Testing samples:  {X_test.shape[0]}")
print(f"  Features count:   {X_train.shape[1]}")

# ---------------------------------------------------------
# 7.3 FEATURE SCALING (StandardScaler)
# ---------------------------------------------------------
print("\n" + "-" * 70)
print("2. Fitting StandardScaler & Transforming Data...")
print("-" * 70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for clean column tracking
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)

# Save scaler
scaler_file = os.path.join("models", "scaler.pkl")
joblib.dump(scaler, scaler_file)
print(f"  ✅ Fitted StandardScaler saved to '{scaler_file}'")

# Save split datasets for tuning and evaluation steps
joblib.dump((X_train_scaled_df, X_test_scaled_df, y_train, y_test, feature_names), 
            os.path.join("data", "train_test_data.pkl"))

# ---------------------------------------------------------
# 7.4 ALGORITHM 1: RANDOM FOREST CLASSIFIER (BASELINE)
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("🌲 TRAINING ALGORITHM 1: RANDOM FOREST CLASSIFIER (BASELINE)")
print("=" * 70)

rf_baseline = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_baseline.fit(X_train_scaled_df, y_train)

rf_train_acc = accuracy_score(y_train, rf_baseline.predict(X_train_scaled_df)) * 100
rf_test_pred = rf_baseline.predict(X_test_scaled_df)
rf_test_acc = accuracy_score(y_test, rf_test_pred) * 100

print(f"Random Forest Training Accuracy: {rf_train_acc:.2f}%")
print(f"Random Forest Testing Accuracy:  {rf_test_acc:.2f}%")

target_names = ['Low Cost (0)', 'Medium Cost (1)', 'High Cost (2)', 'Very High Cost (3)']
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_test_pred, target_names=target_names))

# ---------------------------------------------------------
# 7.5 ALGORITHM 2: SUPPORT VECTOR MACHINE (BASELINE)
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("🎯 TRAINING ALGORITHM 2: SUPPORT VECTOR CLASSIFIER (BASELINE)")
print("=" * 70)

svm_baseline = SVC(probability=True, random_state=42)
svm_baseline.fit(X_train_scaled_df, y_train)

svm_train_acc = accuracy_score(y_train, svm_baseline.predict(X_train_scaled_df)) * 100
svm_test_pred = svm_baseline.predict(X_test_scaled_df)
svm_test_acc = accuracy_score(y_test, svm_test_pred) * 100

print(f"SVM Training Accuracy: {svm_train_acc:.2f}%")
print(f"SVM Testing Accuracy:  {svm_test_acc:.2f}%")

print("\nSVM Classification Report:")
print(classification_report(y_test, svm_test_pred, target_names=target_names))

# ---------------------------------------------------------
# 7.6 BASELINE SUMMARY COMPARISON
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("📊 BASELINE PERFORMANCE SUMMARY")
print("=" * 70)

summary_df = pd.DataFrame({
    'Model': ['Random Forest (Baseline)', 'SVM Classifier (Baseline)'],
    'Train Accuracy (%)': [round(rf_train_acc, 2), round(svm_train_acc, 2)],
    'Test Accuracy (%)': [round(rf_test_acc, 2), round(svm_test_acc, 2)],
    'Train-Test Overfit Gap (%)': [round(rf_train_acc - rf_test_acc, 2), round(svm_train_acc - svm_test_acc, 2)]
})
print(summary_df.to_string(index=False))

print("\n" + "=" * 70)
print("✅ STEP 7 COMPLETE! Random Forest & SVM successfully trained.")
print("=" * 70)
print("➡️ Ready for STEP 8: Hyperparameter Tuning & Regularization")

