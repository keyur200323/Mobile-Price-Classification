"""
STEP 4: Data Augmentation using SMOTE (Synthetic Minority Over-sampling)
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from collections import Counter

print("=" * 70)
print("🔄 STEP 4: DATA AUGMENTATION (SMOTE BALANCING & GENERATION)")
print("=" * 70)

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)

# ---------------------------------------------------------
# 4.1 LOAD ENGINEERED DATASET FROM STEP 3
# ---------------------------------------------------------
input_path = os.path.join("data", "mobile_engineered.csv")
if not os.path.exists(input_path):
    print(f"❌ Error: {input_path} not found. Please run Step 3 first!")
    exit()

df = pd.read_csv(input_path)
print(f"📂 Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# ---------------------------------------------------------
# 4.2 SEPARATE FEATURES AND TARGET
# ---------------------------------------------------------
X = df.drop(columns=['price_range'])
y = df['price_range']

class_names = {0: 'Low Cost', 1: 'Medium Cost', 2: 'High Cost', 3: 'Very High Cost'}

before_counts = Counter(y)
print("\n" + "-" * 70)
print("Class Distribution Before Augmentation:")
print("-" * 70)
for cls_idx, count in sorted(before_counts.items()):
    pct = (count / len(y)) * 100
    print(f"  Class {cls_idx} ({class_names[cls_idx]:15s}): {count:5d} samples ({pct:5.2f}%)")

# ---------------------------------------------------------
# 4.3 APPLY SMOTE FOR MULTI-CLASS SYNTHETIC AUGMENTATION
# ---------------------------------------------------------
print("\n" + "-" * 70)
print("Applying Synthetic Data Augmentation via SMOTE (k_neighbors=5)...")
print("-" * 70)

# Apply SMOTE to generate synthetic feature representations along class margins
smote = SMOTE(k_neighbors=5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

after_counts = Counter(y_resampled)
print("\nClass Distribution After Augmentation:")
for cls_idx, count in sorted(after_counts.items()):
    pct = (count / len(y_resampled)) * 100
    print(f"  Class {cls_idx} ({class_names[cls_idx]:15s}): {count:5d} samples ({pct:5.2f}%)")

# ---------------------------------------------------------
# 4.4 VISUALIZE BEFORE VS AFTER AUGMENTATION
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
palette = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']

# Before Plot
axes[0].bar(
    [class_names[k] for k in sorted(before_counts.keys())],
    [before_counts[k] for k in sorted(before_counts.keys())],
    color=palette, edgecolor='black', alpha=0.85
)
axes[0].set_title("Class Distribution (Before SMOTE)", fontsize=13, fontweight='bold')
axes[0].set_ylabel("Number of Samples")
axes[0].set_ylim(0, max(after_counts.values()) + 100)
for i, k in enumerate(sorted(before_counts.keys())):
    axes[0].text(i, before_counts[k] + 10, str(before_counts[k]), ha='center', fontweight='bold')

# After Plot
axes[1].bar(
    [class_names[k] for k in sorted(after_counts.keys())],
    [after_counts[k] for k in sorted(after_counts.keys())],
    color=palette, edgecolor='black', alpha=0.85
)
axes[1].set_title("Class Distribution (After SMOTE Augmentation)", fontsize=13, fontweight='bold')
axes[1].set_ylabel("Number of Samples")
axes[1].set_ylim(0, max(after_counts.values()) + 100)
for i, k in enumerate(sorted(after_counts.keys())):
    axes[1].text(i, after_counts[k] + 10, str(after_counts[k]), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig("visualizations/augmentation_comparison.png", dpi=150)
plt.close()
print("\n📊 Saved 'visualizations/augmentation_comparison.png'")

# ---------------------------------------------------------
# 4.5 RECOMBINE AND SAVE AUGMENTED DATASET
# ---------------------------------------------------------
df_augmented = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), 
                          pd.Series(y_resampled, name='price_range')], axis=1)

output_path = os.path.join("data", "mobile_augmented.csv")
df_augmented.to_csv(output_path, index=False)

print("\n" + "=" * 70)
print(f"✅ STEP 4 COMPLETE! Augmented dataset saved to '{output_path}'")
print(f"📊 Dimensions: {df_augmented.shape[0]} rows × {df_augmented.shape[1]} columns")
print("=" * 70)
print("➡️ Ready for STEP 5: Exploratory Data Visualization")

