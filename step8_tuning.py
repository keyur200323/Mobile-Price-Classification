"""
STEP 8: Hyperparameter Tuning & Model Optimization (RF vs. SVM)
"""

import pandas as pd
import numpy as np
import os
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

print("=" * 70)
print("🔧 STEP 8: HYPERPARAMETER TUNING & REGULARIZATION")
print("=" * 70)

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)

# ---------------------------------------------------------
# 8.1 LOAD TRAIN/TEST SPLITS FROM STEP 7
# ---------------------------------------------------------
data_path = os.path.join("data", "train_test_data.pkl")
if not os.path.exists(data_path):
    print(f"❌ Error: {data_path} not found. Please run Step 7 first!")
    exit()

X_train, X_test, y_train, y_test, feature_names = joblib.load(data_path)
print(f"📂 Loaded training samples: {X_train.shape[0]}, testing samples: {X_test.shape[0]}")
print(f"📋 Features ({len(feature_names)}): {feature_names}")

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
target_names = ['Low Cost (0)', 'Medium Cost (1)', 'High Cost (2)', 'Very High Cost (3)']

# ---------------------------------------------------------
# 8.2 TUNING ALGORITHM 1: RANDOM FOREST
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("🌲 1. TUNING RANDOM FOREST CLASSIFIER (RandomizedSearchCV)...")
print("=" * 70)

rf_param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 15, 20, 25, 30, None],
    'min_samples_split': [2, 5, 8, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False]
}

rf_random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=rf_param_grid,
    n_iter=60,
    cv=cv_strategy,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_random_search.fit(X_train, y_train)

rf_best = rf_random_search.best_estimator_
rf_best_params = rf_random_search.best_params_
rf_cv_score = rf_random_search.best_score_ * 100

rf_tuned_train_acc = accuracy_score(y_train, rf_best.predict(X_train)) * 100
rf_tuned_test_pred = rf_best.predict(X_test)
rf_tuned_test_acc = accuracy_score(y_test, rf_tuned_test_pred) * 100

print(f"\n  ✅ Best Random Forest Hyperparameters:")
for param, val in rf_best_params.items():
    print(f"     • {param:20s}: {val}")
print(f"  ✅ Best 5-Fold Cross-Validation Accuracy: {rf_cv_score:.2f}%")
print(f"  ✅ Tuned Test Set Accuracy:               {rf_tuned_test_acc:.2f}%")

# ---------------------------------------------------------
# 8.3 TUNING ALGORITHM 2: SUPPORT VECTOR MACHINE (SVM)
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("🎯 2. TUNING SUPPORT VECTOR CLASSIFIER (GridSearchCV)...")
print("=" * 70)

svm_param_grid = {
    'C': [0.1, 1, 5, 10, 50, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['linear', 'rbf', 'poly'],
    'degree': [2, 3]  # Only used when kernel='poly'
}

svm_grid_search = GridSearchCV(
    estimator=SVC(probability=True, random_state=42),
    param_grid=svm_param_grid,
    cv=cv_strategy,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

svm_grid_search.fit(X_train, y_train)

svm_best = svm_grid_search.best_estimator_
svm_best_params = svm_grid_search.best_params_
svm_cv_score = svm_grid_search.best_score_ * 100

svm_tuned_train_acc = accuracy_score(y_train, svm_best.predict(X_train)) * 100
svm_tuned_test_pred = svm_best.predict(X_test)
svm_tuned_test_acc = accuracy_score(y_test, svm_tuned_test_pred) * 100

print(f"\n  ✅ Best SVM Hyperparameters:")
for param, val in svm_best_params.items():
    print(f"     • {param:20s}: {val}")
print(f"  ✅ Best 5-Fold Cross-Validation Accuracy: {svm_cv_score:.2f}%")
print(f"  ✅ Tuned Test Set Accuracy:               {svm_tuned_test_acc:.2f}%")

# ---------------------------------------------------------
# 8.4 SAVE BEST TUNED MODELS & METADATA
# ---------------------------------------------------------
print("\n" + "-" * 70)
print("Saving Optimized Models and Metadata...")
print("-" * 70)

rf_model_path = os.path.join("models", "rf_model.pkl")
svm_model_path = os.path.join("models", "svm_model.pkl")

joblib.dump(rf_best, rf_model_path)
joblib.dump(svm_best, svm_model_path)

print(f"  ✅ Saved Tuned Random Forest: '{rf_model_path}'")
print(f"  ✅ Saved Tuned SVM Classifier: '{svm_model_path}'")

# Serialize parameters cleanly for JSON
def sanitize_params(d):
    clean = {}
    for k, v in d.items():
        if isinstance(v, (np.integer, int)):
            clean[k] = int(v)
        elif isinstance(v, (np.floating, float)):
            clean[k] = float(v)
        elif v is None:
            clean[k] = None
        else:
            clean[k] = str(v)
    return clean

metadata = {
    'selected_features': feature_names,
    'rf_test_accuracy': float(rf_tuned_test_acc),
    'svm_test_accuracy': float(svm_tuned_test_acc),
    'rf_cv_accuracy': float(rf_cv_score),
    'svm_cv_accuracy': float(svm_cv_score),
    'rf_best_params': sanitize_params(rf_best_params),
    'svm_best_params': sanitize_params(svm_best_params)
}

metadata_path = os.path.join("models", "model_metadata.json")
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"  ✅ Saved Metadata: '{metadata_path}'")

# ---------------------------------------------------------
# 8.5 VISUALIZE TUNING PERFORMANCE
# ---------------------------------------------------------
print("\nGenerating Tuning Comparison Visualization...")
fig, ax = plt.subplots(figsize=(10, 6))

models = ['Random Forest', 'SVM Classifier']
cv_scores = [rf_cv_score, svm_cv_score]
test_scores = [rf_tuned_test_acc, svm_tuned_test_acc]

x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, cv_scores, width, label='5-Fold CV Accuracy (%)', color='#3498db', edgecolor='black')
bars2 = ax.bar(x + width/2, test_scores, width, label='Holdout Test Accuracy (%)', color='#2ecc71', edgecolor='black')

for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.8, f"{h:.2f}%", ha='center', fontweight='bold', fontsize=11)

for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.8, f"{h:.2f}%", ha='center', fontweight='bold', fontsize=11)

ax.set_ylabel("Accuracy Score (%)", fontsize=12)
ax.set_title("Tuned Models - Cross-Validation vs. Test Accuracy", fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12, fontweight='bold')
ax.set_ylim(0, 115)
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("visualizations/07_hyperparameter_tuning.png", dpi=150)
plt.close()
print("  ✅ Saved 'visualizations/07_hyperparameter_tuning.png'")

print("\n" + "=" * 70)
print("✅ STEP 8 COMPLETE! Hyperparameter tuning finished successfully.")
print("=" * 70)
print("➡️ Ready for STEP 9: Comprehensive Model Evaluation & Metric Comparison")

