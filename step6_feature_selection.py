"""
STEP 6: Feature Selection using Multiple Methods & Consensus Ranking
"""

import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import rankdata

print("=" * 70)
print("🎯 STEP 6: FEATURE SELECTION (MULTI-METHOD CONSENSUS)")
print("=" * 70)

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)

# ---------------------------------------------------------
# 6.1 LOAD AUGMENTED DATASET
# ---------------------------------------------------------
input_path = os.path.join("data", "mobile_augmented.csv")
if not os.path.exists(input_path):
    print(f"❌ Error: {input_path} not found. Please run Step 4 first!")
    exit()

df = pd.read_csv(input_path)
print(f"📂 Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

X = df.drop(columns=['price_range'])
y = df['price_range']
feature_names = X.columns.tolist()

# ---------------------------------------------------------
# 6.2 METHOD 1: ANOVA F-TEST (SelectKBest)
# ---------------------------------------------------------
print("\n" + "-" * 70)
print("1. Computing ANOVA F-Statistic Scores...")
print("-" * 70)

selector_f = SelectKBest(score_func=f_classif, k='all')
selector_f.fit(X, y)

f_scores = selector_f.scores_
f_scores = np.nan_to_num(f_scores)  # Replace any potential NaNs with 0

# ---------------------------------------------------------
# 6.3 METHOD 2: MUTUAL INFORMATION CLASSIFICATION
# ---------------------------------------------------------
print("\n" + "-" * 70)
print("2. Computing Mutual Information Entropy Scores...")
print("-" * 70)

mi_scores = mutual_info_classif(X, y, random_state=42)

# ---------------------------------------------------------
# 6.4 METHOD 3: RANDOM FOREST FEATURE IMPORTANCE
# ---------------------------------------------------------
print("\n" + "-" * 70)
print("3. Computing Random Forest Mean Impurity Reduction...")
print("-" * 70)

rf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
rf.fit(X, y)
rf_scores = rf.feature_importances_

# ---------------------------------------------------------
# 6.5 CONSENSUS AGGREGATE RANKING
# ---------------------------------------------------------
print("\n" + "-" * 70)
print("4. Computing Aggregate Consensus Ranks...")
print("-" * 70)

# Lower rank number = better score (1 is best)
rank_df = pd.DataFrame({'Feature': feature_names})
rank_df['F_Score'] = f_scores
rank_df['F_Rank'] = rankdata(-f_scores)

rank_df['MI_Score'] = mi_scores
rank_df['MI_Rank'] = rankdata(-mi_scores)

rank_df['RF_Score'] = rf_scores
rank_df['RF_Rank'] = rankdata(-rf_scores)

# Consensus rank = average of ranks from all 3 methods
rank_df['Avg_Rank'] = rank_df[['F_Rank', 'MI_Rank', 'RF_Rank']].mean(axis=1).round(2)
rank_df = rank_df.sort_values(by='Avg_Rank', ascending=True).reset_index(drop=True)

print("\nFull Feature Ranking Table:")
print(rank_df[['Feature', 'F_Rank', 'MI_Rank', 'RF_Rank', 'Avg_Rank']].to_string(index=False))

# ---------------------------------------------------------
# 6.6 SELECT TOP N FEATURES
# ---------------------------------------------------------
TOP_K = 14
selected_features = rank_df.head(TOP_K)['Feature'].tolist()

print("\n" + "=" * 70)
print(f"🏆 TOP {TOP_K} SELECTED FEATURES FOR MACHINE LEARNING:")
print("=" * 70)
for i, feat in enumerate(selected_features, 1):
    avg_rk = rank_df.loc[rank_df['Feature'] == feat, 'Avg_Rank'].values[0]
    print(f"  {i:2d}. {feat:25s} (Consensus Rank: {avg_rk:.2f})")

# ---------------------------------------------------------
# 6.7 VISUALIZE FEATURE IMPORTANCES
# ---------------------------------------------------------
print("\nGenerating Feature Selection Comparison Visualization...")
fig, axes = plt.subplots(1, 3, figsize=(22, 10))

# 1. F-Score Plot
top_f = rank_df.sort_values('F_Score', ascending=False).head(14)
axes[0].barh(range(len(top_f)), top_f['F_Score'], color='#3498db', edgecolor='black', alpha=0.85)
axes[0].set_yticks(range(len(top_f)))
axes[0].set_yticklabels(top_f['Feature'], fontsize=10)
axes[0].set_title("ANOVA F-Test (Variance Ratio)", fontsize=13, fontweight='bold')
axes[0].invert_yaxis()
axes[0].set_xlabel("F-Score")

# 2. MI-Score Plot
top_mi = rank_df.sort_values('MI_Score', ascending=False).head(14)
axes[1].barh(range(len(top_mi)), top_mi['MI_Score'], color='#2ecc71', edgecolor='black', alpha=0.85)
axes[1].set_yticks(range(len(top_mi)))
axes[1].set_yticklabels(top_mi['Feature'], fontsize=10)
axes[1].set_title("Mutual Information (Dependency Score)", fontsize=13, fontweight='bold')
axes[1].invert_yaxis()
axes[1].set_xlabel("MI Score")

# 3. RF Importance Plot
top_rf = rank_df.sort_values('RF_Score', ascending=False).head(14)
axes[2].barh(range(len(top_rf)), top_rf['RF_Score'], color='#e74c3c', edgecolor='black', alpha=0.85)
axes[2].set_yticks(range(len(top_rf)))
axes[2].set_yticklabels(top_rf['Feature'], fontsize=10)
axes[2].set_title("Random Forest Gini Importance", fontsize=13, fontweight='bold')
axes[2].invert_yaxis()
axes[2].set_xlabel("Mean Impurity Reduction")

plt.suptitle("Feature Selection - Method Comparison", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("visualizations/06_feature_selection_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Saved 'visualizations/06_feature_selection_comparison.png'")

# ---------------------------------------------------------
# 6.8 SAVE SELECTED SUBSET & METADATA
# ---------------------------------------------------------
df_selected = df[selected_features + ['price_range']]
output_data_path = os.path.join("data", "mobile_selected.csv")
df_selected.to_csv(output_data_path, index=False)

json_path = os.path.join("data", "selected_features.json")
with open(json_path, 'w') as f:
    json.dump({'selected_features': selected_features, 'top_k': TOP_K}, f, indent=2)

print("\n" + "=" * 70)
print(f"✅ STEP 6 COMPLETE! Selected dataset saved to '{output_data_path}'")
print(f"💾 Feature configuration saved to '{json_path}'")
print("=" * 70)
print("➡️ Ready for STEP 7: Machine Learning Model Training (RF vs XGBoost)")

