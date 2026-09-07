"""
STEP 3: Data Wrangling & Feature Engineering
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("🔧 STEP 3: DATA WRANGLING & FEATURE ENGINEERING")
print("=" * 70)

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)

# ---------------------------------------------------------
# 3.1 LOAD CLEANED DATASET
# ---------------------------------------------------------
input_path = os.path.join("data", "mobile_cleaned.csv")
if not os.path.exists(input_path):
    print(f"❌ Error: {input_path} not found. Please run Step 2 first!")
    exit()

df = pd.read_csv(input_path)
print(f"📂 Loaded cleaned dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# ---------------------------------------------------------
# 3.2 FEATURE ENGINEERING
# ---------------------------------------------------------
print("\n" + "-" * 70)
print("Engineering New Domain Features...")
print("-" * 70)

df_eng = df.copy()

# 1. Total Camera Megapixels
df_eng['total_camera_mp'] = df_eng['pc'] + df_eng['fc']
print("  ✅ Created 'total_camera_mp' (Primary + Front Camera)")

# 2. Screen Display Area (in cm²)
df_eng['screen_area'] = df_eng['sc_h'] * df_eng['sc_w']
print("  ✅ Created 'screen_area' (Screen Height × Screen Width)")

# 3. Screen Aspect Ratio
df_eng['aspect_ratio'] = (df_eng['px_width'] / (df_eng['px_height'] + 1)).round(3)
print("  ✅ Created 'aspect_ratio' (Pixel Width / Pixel Height)")

# 4. Pixel Density (Approximation of Screen Sharpness)
diagonal_px = np.sqrt(df_eng['px_width']**2 + df_eng['px_height']**2)
df_eng['pixel_density'] = (diagonal_px / (df_eng['sc_h'] + 1)).round(2)
print("  ✅ Created 'pixel_density' (Resolution Sharpness Index)")

# 5. RAM Distribution Per CPU Core
df_eng['ram_per_core'] = (df_eng['ram'] / df_eng['n_cores']).round(2)
print("  ✅ Created 'ram_per_core' (RAM / Number of Cores)")

# 6. Battery Power per Unit of Mobile Weight
df_eng['battery_wt_ratio'] = (df_eng['battery_power'] / df_eng['mobile_wt']).round(2)
print("  ✅ Created 'battery_wt_ratio' (Battery Capacity / Weight)")

# 7. Internal Storage to RAM Capacity Ratio
df_eng['storage_ram_ratio'] = (df_eng['int_memory'] / ((df_eng['ram'] / 1024) + 0.1)).round(2)
print("  ✅ Created 'storage_ram_ratio' (Storage / RAM in GB)")

# 8. Composite Connectivity Score (Sum of wireless/hardware sensors)
df_eng['connectivity_score'] = (
    df_eng['blue'] + df_eng['dual_sim'] + df_eng['four_g'] + 
    df_eng['three_g'] + df_eng['touch_screen'] + df_eng['wifi']
)
print("  ✅ Created 'connectivity_score' (Sum of 6 feature flags)")

# 9. Composite Computational Performance Score
df_eng['performance_score'] = (
    (df_eng['ram'] * df_eng['clock_speed'] * df_eng['n_cores']) / 1000
).round(2)
print("  ✅ Created 'performance_score' (RAM × Clock Speed × Cores / 1000)")

# 10. RAM Tier Category (Binning)
def ram_tier(ram):
    if ram < 1000:
        return 0  # Budget Tier
    elif ram < 2000:
        return 1  # Mid Tier
    elif ram < 3000:
        return 2  # Upper Tier
    else:
        return 3  # Flagship Tier

df_eng['ram_category'] = df_eng['ram'].apply(ram_tier)
print("  ✅ Created 'ram_category' (Binned 0-3)")

# 11. Battery Tier Category (Binning)
def battery_tier(bp):
    if bp < 1000:
        return 0  # Standard
    elif bp < 1500:
        return 1  # Moderate
    else:
        return 2  # Heavy Duty

df_eng['battery_category'] = df_eng['battery_power'].apply(battery_tier)
print("  ✅ Created 'battery_category' (Binned 0-2)")

# ---------------------------------------------------------
# 3.3 EVALUATE NEW FEATURES' CORRELATION WITH TARGET
# ---------------------------------------------------------
print("\n" + "-" * 70)
print("Correlation of Features with Target (price_range):")
print("-" * 70)

correlations = df_eng.corr()['price_range'].sort_values(ascending=False)
corr_table = pd.DataFrame({
    'Feature': correlations.index,
    'Correlation with price_range': correlations.values.round(4)
})
print(corr_table.to_string(index=False))

# Plot top correlations
plt.figure(figsize=(10, 8))
top_corr = correlations.drop('price_range')
colors = ['#e63946' if c > 0 else '#457b9d' for c in top_corr.values]
top_corr.plot(kind='barh', color=colors)
plt.title("Correlation of All Features with Price Range", fontsize=14, fontweight='bold')
plt.xlabel("Pearson Correlation Coefficient")
plt.tight_layout()
plt.savefig("visualizations/feature_correlations.png", dpi=150)
plt.close()
print("\n📊 Saved 'visualizations/feature_correlations.png'")

# ---------------------------------------------------------
# 3.4 SAVE ENGINEERED DATASET
# ---------------------------------------------------------
output_path = os.path.join("data", "mobile_engineered.csv")
df_eng.to_csv(output_path, index=False)

print("\n" + "=" * 70)
print(f"✅ STEP 3 COMPLETE! Engineered dataset saved to '{output_path}'")
print(f"📊 Dataset expanded from 21 columns to {df_eng.shape[1]} columns!")
print("=" * 70)
print("➡️ Ready for STEP 4: Data Augmentation (SMOTE Balancing)")
