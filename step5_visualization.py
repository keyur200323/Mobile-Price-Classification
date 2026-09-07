"""
STEP 5: Comprehensive Data Visualization
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("📊 STEP 5: COMPREHENSIVE DATA VISUALIZATION")
print("=" * 70)

# Ensure visualizations directory exists
os.makedirs("visualizations", exist_ok=True)

# ---------------------------------------------------------
# 5.1 LOAD AUGMENTED DATASET
# ---------------------------------------------------------
input_path = os.path.join("data", "mobile_augmented.csv")
if not os.path.exists(input_path):
    print(f"❌ Error: {input_path} not found. Please run Step 4 first!")
    exit()

df = pd.read_csv(input_path)
print(f"📂 Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

price_labels = {0: 'Low Cost', 1: 'Medium Cost', 2: 'High Cost', 3: 'Very High Cost'}
palette = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']

# ---------------------------------------------------------
# 5.2 PLOT 1: DISTRIBUTION OF KEY CONTINUOUS FEATURES
# ---------------------------------------------------------
print("\nGenerating Plot 1: Key Feature Distributions...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
key_features = ['ram', 'battery_power', 'pixel_density', 'int_memory', 'mobile_wt', 'performance_score']

for i, col in enumerate(key_features):
    ax = axes[i // 3, i % 3]
    for cls_idx in range(4):
        subset = df[df['price_range'] == cls_idx][col]
        sns.kdeplot(subset, ax=ax, label=price_labels[cls_idx], color=palette[cls_idx], fill=True, alpha=0.25)
    ax.set_title(f"Distribution of {col}", fontsize=12, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)

plt.suptitle("Key Feature Density Distributions by Price Range", fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig("visualizations/01_feature_distributions.png", dpi=150)
plt.close()
print("  ✅ Saved 'visualizations/01_feature_distributions.png'")

# ---------------------------------------------------------
# 5.3 PLOT 2: CORRELATION HEATMAP
# ---------------------------------------------------------
print("\nGenerating Plot 2: Correlation Heatmap...")
plt.figure(figsize=(18, 14))
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(
    corr_matrix, mask=mask, cmap='coolwarm', vmin=-1, vmax=1,
    annot=True, fmt='.2f', annot_kws={'size': 7}, cbar_kws={'shrink': 0.8},
    linewidths=0.3, linecolor='white'
)
plt.title("Full Feature Correlation Heatmap", fontsize=16, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("visualizations/02_correlation_heatmap.png", dpi=150)
plt.close()
print("  ✅ Saved 'visualizations/02_correlation_heatmap.png'")

# ---------------------------------------------------------
# 5.4 PLOT 3: VIOLIN PLOTS FOR TOP DIFFERENTIATING FEATURES
# ---------------------------------------------------------
print("\nGenerating Plot 3: Violin Plots for Price Drivers...")
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
violin_features = ['ram', 'battery_power', 'performance_score', 'ram_per_core']

for i, col in enumerate(violin_features):
    sns.violinplot(
        x='price_range', y=col, data=df, ax=axes[i],
        palette=palette, inner='quartile'
    )
    axes[i].set_xticklabels([price_labels[k] for k in range(4)], rotation=15)
    axes[i].set_title(f"{col} Across Tiers", fontsize=12, fontweight='bold')
    axes[i].set_xlabel("Price Range")

plt.suptitle("Distribution Spread across Price Categories (Violin Plots)", fontsize=16, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig("visualizations/03_violin_plots.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved 'visualizations/03_violin_plots.png'")

# ---------------------------------------------------------
# 5.5 PLOT 4: SCATTER PLOT - MULTIVARIATE RELATIONSHIPS
# ---------------------------------------------------------
print("\nGenerating Plot 4: Scatter Plot (RAM vs Battery vs Resolution)...")
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(
    df['ram'], df['battery_power'], 
    c=df['price_range'], cmap='viridis', 
    s=df['px_width'] / 30, alpha=0.6, edgecolors='black', linewidth=0.5
)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_ticks([0, 1, 2, 3])
cbar.set_ticklabels(['0 - Low', '1 - Med', '2 - High', '3 - Flagship'])
cbar.set_label('Price Range Tier', fontsize=11)

ax.set_title("RAM vs. Battery Power (Bubble Size = Pixel Width)", fontsize=14, fontweight='bold')
ax.set_xlabel("RAM (MB)", fontsize=11)
ax.set_ylabel("Battery Power (mAh)", fontsize=11)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("visualizations/04_multivariate_scatter.png", dpi=150)
plt.close()
print("  ✅ Saved 'visualizations/04_multivariate_scatter.png'")

# ---------------------------------------------------------
# 5.6 PLOT 5: CONNECTIVITY & BINARY FEATURES BREAKDOWN
# ---------------------------------------------------------
print("\nGenerating Plot 5: Binary & Connectivity Feature Analysis...")
binary_cols = ['three_g', 'four_g', 'dual_sim', 'wifi', 'touch_screen']
fig, axes = plt.subplots(1, len(binary_cols), figsize=(20, 4))

for i, col in enumerate(binary_cols):
    ct = pd.crosstab(df['price_range'], df[col], normalize='index') * 100
    ct.plot(kind='bar', stacked=True, ax=axes[i], color=['#95a5a6', '#2ecc71'], edgecolor='black', alpha=0.85)
    axes[i].set_title(f"{col.upper()} Distribution", fontsize=11, fontweight='bold')
    axes[i].set_ylabel("Percentage (%)")
    axes[i].set_xlabel("Price Range")
    axes[i].set_xticklabels([f"Tier {k}" for k in range(4)], rotation=0)
    axes[i].legend(["No", "Yes"], fontsize=8, loc='upper right')

plt.suptitle("Presence of Hardware Capabilities Across Price Tiers", fontsize=15, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig("visualizations/05_binary_features_breakdown.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved 'visualizations/05_binary_features_breakdown.png'")

print("\n" + "=" * 70)
print("✅ STEP 5 COMPLETE! 5 High-Quality Visualizations Generated in 'visualizations/'")
print("=" * 70)
print("➡️ Ready for STEP 6: Feature Selection")
