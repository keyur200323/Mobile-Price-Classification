"""
STEP 9: Comprehensive Model Evaluation (Random Forest vs. Support Vector Machine)
"""

import pandas as pd
import numpy as np
import os
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay,
    cohen_kappa_score, log_loss, classification_report
)
from sklearn.preprocessing import label_binarize

print("=" * 75)
print("📈 STEP 9: COMPREHENSIVE MODEL EVALUATION & COMPARISON")
print("   Algorithm 1: Random Forest Classifier (Tuned)")
print("   Algorithm 2: Support Vector Classifier (Tuned)")
print("=" * 75)

# Ensure directories exist
os.makedirs("visualizations", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ---------------------------------------------------------
# 9.1 LOAD MODELS AND TEST DATA
# ---------------------------------------------------------
data_path = os.path.join("data", "train_test_data.pkl")
rf_path = os.path.join("models", "rf_model.pkl")
svm_path = os.path.join("models", "svm_model.pkl")

if not os.path.exists(data_path) or not os.path.exists(rf_path) or not os.path.exists(svm_path):
    print("❌ Error: Missing models or test data. Please run Step 7 and Step 8 first!")
    exit()

X_train, X_test, y_train, y_test, feature_names = joblib.load(data_path)
rf_model = joblib.load(rf_path)
svm_model = joblib.load(svm_path)

print(f"📂 Loaded Test Set: {X_test.shape[0]} samples × {X_test.shape[1]} features")

price_labels = {0: 'Low Cost', 1: 'Medium Cost', 2: 'High Cost', 3: 'Very High Cost'}
class_names = [price_labels[i] for i in range(4)]

# ---------------------------------------------------------
# 9.2 GENERATE PREDICTIONS & PROBABILITIES
# ---------------------------------------------------------
print("\nGenerating model predictions and probability distributions...")

# Random Forest Predictions
rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)

# SVM Predictions
svm_pred = svm_model.predict(X_test)
svm_proba = svm_model.predict_proba(X_test)

# ---------------------------------------------------------
# 9.3 CALCULATE COMPREHENSIVE METRICS
# ---------------------------------------------------------
# Binarize targets for multi-class ROC-AUC
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])

rf_metrics = {
    'Accuracy': accuracy_score(y_test, rf_pred),
    'Precision (Macro)': precision_score(y_test, rf_pred, average='macro'),
    'Precision (Weighted)': precision_score(y_test, rf_pred, average='weighted'),
    'Recall (Macro)': recall_score(y_test, rf_pred, average='macro'),
    'Recall (Weighted)': recall_score(y_test, rf_pred, average='weighted'),
    'F1-Score (Macro)': f1_score(y_test, rf_pred, average='macro'),
    'F1-Score (Weighted)': f1_score(y_test, rf_pred, average='weighted'),
    'ROC-AUC (Macro OVR)': roc_auc_score(y_test_bin, rf_proba, multi_class='ovr', average='macro'),
    'Cohen Kappa': cohen_kappa_score(y_test, rf_pred),
    'Log Loss': log_loss(y_test, rf_proba)
}

svm_metrics = {
    'Accuracy': accuracy_score(y_test, svm_pred),
    'Precision (Macro)': precision_score(y_test, svm_pred, average='macro'),
    'Precision (Weighted)': precision_score(y_test, svm_pred, average='weighted'),
    'Recall (Macro)': recall_score(y_test, svm_pred, average='macro'),
    'Recall (Weighted)': recall_score(y_test, svm_pred, average='weighted'),
    'F1-Score (Macro)': f1_score(y_test, svm_pred, average='macro'),
    'F1-Score (Weighted)': f1_score(y_test, svm_pred, average='weighted'),
    'ROC-AUC (Macro OVR)': roc_auc_score(y_test_bin, svm_proba, multi_class='ovr', average='macro'),
    'Cohen Kappa': cohen_kappa_score(y_test, svm_pred),
    'Log Loss': log_loss(y_test, svm_proba)
}

# ---------------------------------------------------------
# 9.4 DISPLAY FINAL COMPARISON TABLE
# ---------------------------------------------------------
comparison_df = pd.DataFrame({
    'Metric': list(rf_metrics.keys()),
    'Random Forest': [round(v, 4) for v in rf_metrics.values()],
    'SVM Classifier': [round(v, 4) for v in svm_metrics.values()]
})

def pick_winner(row):
    m = row['Metric']
    rf_v = row['Random Forest']
    svm_v = row['SVM Classifier']
    if m == 'Log Loss':  # Lower is better
        return '🏆 RF' if rf_v < svm_v else ('🏆 SVM' if svm_v < rf_v else 'Tie')
    else:               # Higher is better
        return '🏆 RF' if rf_v > svm_v else ('🏆 SVM' if svm_v > rf_v else 'Tie')

comparison_df['Winner'] = comparison_df.apply(pick_winner, axis=1)

print("\n" + "=" * 75)
print("📊 FINAL MULTI-METRIC PERFORMANCE COMPARISON")
print("=" * 75)
print(comparison_df.to_string(index=False))

# ---------------------------------------------------------
# 9.5 VISUALIZATION 1: CONFUSION MATRICES SIDE BY SIDE
# ---------------------------------------------------------
print("\n1. Generating Confusion Matrices Plot...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

cm_rf = confusion_matrix(y_test, rf_pred)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=class_names)
disp_rf.plot(ax=axes[0], cmap='Blues', colorbar=False, values_format='d')
axes[0].set_title(f"Random Forest Confusion Matrix\n(Accuracy: {rf_metrics['Accuracy']*100:.2f}%)", fontsize=13, fontweight='bold')
axes[0].set_xticklabels(class_names, rotation=15)

cm_svm = confusion_matrix(y_test, svm_pred)
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=class_names)
disp_svm.plot(ax=axes[1], cmap='Greens', colorbar=False, values_format='d')
axes[1].set_title(f"SVM Classifier Confusion Matrix\n(Accuracy: {svm_metrics['Accuracy']*100:.2f}%)", fontsize=13, fontweight='bold')
axes[1].set_xticklabels(class_names, rotation=15)

plt.suptitle("Confusion Matrix Comparison on Holdout Test Set", fontsize=16, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig("visualizations/08_confusion_matrices.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved 'visualizations/08_confusion_matrices.png'")

# ---------------------------------------------------------
# 9.6 VISUALIZATION 2: MULTI-CLASS ONE-VS-REST ROC CURVES
# ---------------------------------------------------------
print("\n2. Generating Multi-Class ROC Curves...")
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']

# Random Forest ROC
for i in range(4):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], rf_proba[:, i])
    roc_score = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color=colors[i], lw=2, label=f'{class_names[i]} (AUC = {roc_score:.3f})')

axes[0].plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6)
axes[0].set_title(f"Random Forest OvR ROC Curves\n(Macro AUC = {rf_metrics['ROC-AUC (Macro OVR)']:.3f})", fontsize=13, fontweight='bold')
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend(loc="lower right", fontsize=10)
axes[0].grid(True, linestyle='--', alpha=0.4)

# SVM ROC
for i in range(4):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], svm_proba[:, i])
    roc_score = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, color=colors[i], lw=2, label=f'{class_names[i]} (AUC = {roc_score:.3f})')

axes[1].plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6)
axes[1].set_title(f"SVM Classifier OvR ROC Curves\n(Macro AUC = {svm_metrics['ROC-AUC (Macro OVR)']:.3f})", fontsize=13, fontweight='bold')
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend(loc="lower right", fontsize=10)
axes[1].grid(True, linestyle='--', alpha=0.4)

plt.suptitle("One-vs-Rest (OvR) ROC Curves by Mobile Price Tier", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("visualizations/09_multiclass_roc_curves.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved 'visualizations/09_multiclass_roc_curves.png'")

# ---------------------------------------------------------
# 9.7 VISUALIZATION 3: CORE METRICS BAR CHART
# ---------------------------------------------------------
print("\n3. Generating Core Metrics Bar Plot...")
fig, ax = plt.subplots(figsize=(12, 6))

bar_metrics = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)', 'ROC-AUC (Macro OVR)']
rf_bar_vals = [rf_metrics[m] for m in bar_metrics]
svm_bar_vals = [svm_metrics[m] for m in bar_metrics]

x = np.arange(len(bar_metrics))
width = 0.35

bars1 = ax.bar(x - width/2, rf_bar_vals, width, label='Random Forest', color='#3498db', edgecolor='black', alpha=0.85)
bars2 = ax.bar(x + width/2, svm_bar_vals, width, label='SVM Classifier', color='#2ecc71', edgecolor='black', alpha=0.85)

for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel("Score (0.0 - 1.0)", fontsize=12)
ax.set_title("Random Forest vs. SVM - Performance Metrics Comparison", fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(bar_metrics, fontsize=11, fontweight='bold')
ax.set_ylim(0, 1.15)
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig("visualizations/10_metrics_bar_comparison.png", dpi=150)
plt.close()
print("  ✅ Saved 'visualizations/10_metrics_bar_comparison.png'")

# ---------------------------------------------------------
# 9.8 VISUALIZATION 4: POLAR RADAR CHART
# ---------------------------------------------------------
print("\n4. Generating Polar Radar Chart...")
radar_categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Kappa']
rf_radar_vals = [
    rf_metrics['Accuracy'], rf_metrics['Precision (Macro)'],
    rf_metrics['Recall (Macro)'], rf_metrics['F1-Score (Macro)'],
    rf_metrics['ROC-AUC (Macro OVR)'], rf_metrics['Cohen Kappa']
]
svm_radar_vals = [
    svm_metrics['Accuracy'], svm_metrics['Precision (Macro)'],
    svm_metrics['Recall (Macro)'], svm_metrics['F1-Score (Macro)'],
    svm_metrics['ROC-AUC (Macro OVR)'], svm_metrics['Cohen Kappa']
]

angles = np.linspace(0, 2 * np.pi, len(radar_categories), endpoint=False).tolist()
rf_radar_vals += rf_radar_vals[:1]
svm_radar_vals += svm_radar_vals[:1]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.fill(angles, rf_radar_vals, alpha=0.25, color='#3498db')
ax.plot(angles, rf_radar_vals, 'o-', linewidth=2.5, color='#3498db', label='Random Forest', markersize=7)

ax.fill(angles, svm_radar_vals, alpha=0.25, color='#2ecc71')
ax.plot(angles, svm_radar_vals, 'o-', linewidth=2.5, color='#2ecc71', label='SVM Classifier', markersize=7)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_categories, fontsize=11, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.set_title("Multidimensional Performance Radar Profile", fontsize=15, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=11)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("visualizations/11_radar_chart_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved 'visualizations/11_radar_chart_comparison.png'")

# ---------------------------------------------------------
# 9.9 VISUALIZATION 5: PER-CLASS F1-SCORE BREAKDOWN
# ---------------------------------------------------------
print("\n5. Generating Per-Class F1-Score Breakdown...")
fig, ax = plt.subplots(figsize=(10, 5))

rf_rep = classification_report(y_test, rf_pred, target_names=class_names, output_dict=True)
svm_rep = classification_report(y_test, svm_pred, target_names=class_names, output_dict=True)

rf_f1s = [rf_rep[cls]['f1-score'] for cls in class_names]
svm_f1s = [svm_rep[cls]['f1-score'] for cls in class_names]

x = np.arange(len(class_names))
width = 0.35

ax.bar(x - width/2, rf_f1s, width, label='Random Forest', color='#3498db', edgecolor='black')
ax.bar(x + width/2, svm_f1s, width, label='SVM Classifier', color='#2ecc71', edgecolor='black')

for i in range(len(class_names)):
    ax.text(i - width/2, rf_f1s[i] + 0.01, f"{rf_f1s[i]:.2f}", ha='center', fontweight='bold', fontsize=10)
    ax.text(i + width/2, svm_f1s[i] + 0.01, f"{svm_f1s[i]:.2f}", ha='center', fontweight='bold', fontsize=10)

ax.set_ylabel("F1-Score", fontsize=12)
ax.set_title("Per-Class F1-Score Breakdown", fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(class_names, fontsize=11, fontweight='bold')
ax.set_ylim(0, 1.15)
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig("visualizations/12_per_class_performance.png", dpi=150)
plt.close()
print("  ✅ Saved 'visualizations/12_per_class_performance.png'")

# ---------------------------------------------------------
# 9.10 DECLARE WINNER & UPDATE METADATA
# ---------------------------------------------------------
rf_overall = np.mean([rf_metrics['Accuracy'], rf_metrics['F1-Score (Macro)'], rf_metrics['ROC-AUC (Macro OVR)']])
svm_overall = np.mean([svm_metrics['Accuracy'], svm_metrics['F1-Score (Macro)'], svm_metrics['ROC-AUC (Macro OVR)']])

winner_name = "SVM Classifier" if svm_overall >= rf_overall else "Random Forest"
winner_margin = abs(svm_overall - rf_overall) * 100

print("\n" + "=" * 75)
print("🏆 OFFICIAL VERDICT")
print("=" * 75)
print(f"Random Forest Composite Score: {rf_overall*100:.2f}%")
print(f"SVM Classifier Composite Score: {svm_overall*100:.2f}%")
print(f"\n🥇 OVERALL WINNER: {winner_name} (outperformed by {winner_margin:.2f}%)")

# Update metadata file for GUI application
metadata_path = os.path.join("models", "model_metadata.json")
with open(metadata_path, 'r') as f:
    meta = json.load(f)

meta['rf_metrics'] = {k: round(v, 4) for k, v in rf_metrics.items()}
meta['svm_metrics'] = {k: round(v, 4) for k, v in svm_metrics.items()}
meta['winner'] = winner_name
meta['price_labels'] = price_labels

with open(metadata_path, 'w') as f:
    json.dump(meta, f, indent=2)

print(f"💾 Updated '{metadata_path}' with full metric suite.")

print("\n" + "=" * 75)
print("✅ STEP 9 COMPLETE! All evaluation metrics & 5 visualization charts generated.")
print("=" * 75)
print("➡️ Ready for the final step: STEP 10 / GUI Application (Interactive Tkinter Desktop App)")

