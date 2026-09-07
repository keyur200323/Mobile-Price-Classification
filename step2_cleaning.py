"""
STEP 2: Data Preparation - Clean and Validate
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("=" * 70)
print("🧹 STEP 2: DATA PREPARATION - CLEAN AND VALIDATE")
print("=" * 70)

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)

# ---------------------------------------------------------
# 2.1 LOAD DATASET FROM STEP 1
# ---------------------------------------------------------
data_path = os.path.join("data", "mobile_train.csv")
if not os.path.exists(data_path):
    print(f"❌ Error: {data_path} not found. Please run Step 1 first!")
    exit()

df = pd.read_csv(data_path)
print(f"📂 Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# ---------------------------------------------------------
# 2.2 MISSING VALUES & DUPLICATES CHECK
# ---------------------------------------------------------
print("\n" + "-" * 70)
print("1. Checking Missing Values & Duplicates")
print("-" * 70)

missing_counts = df.isnull().sum()
total_missing = missing_counts.sum()
print(f"Total missing values: {total_missing}")

duplicates = df.duplicated().sum()
print(f"Total duplicate rows: {duplicates}")

if duplicates > 0:
    df = df.drop_duplicates()
    print(f"✅ Dropped {duplicates} duplicate rows.")

# Plot missing values heatmap (useful for assignment report)
plt.figure(figsize=(10, 4))
sns.heatmap(df.isnull(), cbar=False, cmap='Blues', yticklabels=False)
plt.title("Missing Values Heatmap (Clean Check)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("visualizations/missing_values_heatmap.png", dpi=150)
plt.close()
print("📊 Saved 'visualizations/missing_values_heatmap.png'")

# ---------------------------------------------------------
# 2.3 DOMAIN-SPECIFIC DATA VALIDATION
# ---------------------------------------------------------
print("\n" + "-" * 70)
print("2. Domain Validation & Correcting Inconsistent Values")
print("-" * 70)

# In real mobile hardware data:
# - sc_w (screen width in cm) cannot be 0 cm
# - px_height (resolution height) cannot be 0 px
zero_sc_w = (df['sc_w'] == 0).sum()
zero_px_h = (df['px_height'] == 0).sum()

print(f"Found {zero_sc_w} rows where screen width (sc_w) = 0 cm (physically invalid)")
print(f"Found {zero_px_h} rows where pixel height (px_height) = 0 px (physically invalid)")

# Replace 0s with column medians for those physical dimensions
if zero_sc_w > 0:
    median_sc_w = df[df['sc_w'] > 0]['sc_w'].median()
    df['sc_w'] = df['sc_w'].replace(0, median_sc_w)
    print(f"  ✅ Replaced 0 values in 'sc_w' with median ({median_sc_w:.1f} cm)")

if zero_px_h > 0:
    median_px_h = df[df['px_height'] > 0]['px_height'].median()
    df['px_height'] = df['px_height'].replace(0, median_px_h)
    print(f"  ✅ Replaced 0 values in 'px_height' with median ({median_px_h:.1f} px)")

# ---------------------------------------------------------
# 2.4 OUTLIER DETECTION & HANDLING (IQR WINSORIZATION)
# ---------------------------------------------------------
print("\n" + "-" * 70)
print("3. Outlier Analysis & Winsorization (IQR Capping)")
print("-" * 70)

continuous_cols = [
    'battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep',
    'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width',
    'ram', 'sc_h', 'sc_w', 'talk_time'
]

# Plot boxplots before capping
fig, axes = plt.subplots(2, 7, figsize=(22, 8))
axes = axes.flatten()

for i, col in enumerate(continuous_cols):
    axes[i].boxplot(df[col], patch_artist=True,
                    boxprops=dict(facecolor='#a8dadc', color='#1d3557'),
                    medianprops=dict(color='#e63946', linewidth=2))
    axes[i].set_title(col, fontsize=10, fontweight='bold')

plt.suptitle("Outlier Detection Before Capping (Box Plots)", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("visualizations/outlier_boxplots_before.png", dpi=150, bbox_inches='tight')
plt.close()
print("📊 Saved 'visualizations/outlier_boxplots_before.png'")

# Winsorize outliers (clip between Q1 - 1.5*IQR and Q3 + 1.5*IQR)
df_clean = df.copy()
outlier_summary = []

for col in continuous_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_count = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
    
    # Cap values
    df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
    
    outlier_summary.append({
        'Feature': col,
        'Outliers Detected': outliers_count,
        'Lower Bound': round(lower_bound, 2),
        'Upper Bound': round(upper_bound, 2)
    })

outlier_df = pd.DataFrame(outlier_summary)
print("\nOutlier Detection & Capping Summary:")
print(outlier_df.to_string(index=False))

# ---------------------------------------------------------
# 2.5 SAVE CLEANED DATASET
# ---------------------------------------------------------
output_file = os.path.join("data", "mobile_cleaned.csv")
df_clean.to_csv(output_file, index=False)

print("\n" + "=" * 70)
print(f"✅ STEP 2 COMPLETE! Cleaned dataset saved to '{output_file}'")
print(f"📊 Final Clean Dimensions: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
print("=" * 70)
print("➡️ Ready for STEP 3: Data Wrangling & Feature Engineering")

