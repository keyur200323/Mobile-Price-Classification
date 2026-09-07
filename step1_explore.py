"""
STEP 1: Load and Explore the Mobile Price Dataset (Local File)
"""

import pandas as pd
import numpy as np
import os

print("=" * 70)
print("📱 STEP 1: RAW DATASET - COLLECT AND UNDERSTAND")
print("=" * 70)

# ---------------------------------------------------------
# 1.1 LOAD LOCAL DATASET
# ---------------------------------------------------------
possible_paths = [
    os.path.join("data", "mobile_train.csv"),
    os.path.join("data", "train.csv"),
    "mobile_train.csv",
    "train.csv"
]

file_path = None
for p in possible_paths:
    if os.path.exists(p):
        file_path = p
        break

if file_path is None:
    print("\n❌ Could not find dataset file!")
    print("Please make sure 'train.csv' or 'mobile_train.csv' is placed inside the 'data/' folder.")
    exit()

print(f"\n📂 Loading dataset from: {file_path}")
df = pd.read_csv(file_path)

# Make sure standard copy exists in data/mobile_train.csv
os.makedirs("data", exist_ok=True)
standard_path = os.path.join("data", "mobile_train.csv")
if file_path != standard_path:
    df.to_csv(standard_path, index=False)

print(f"✅ Successfully loaded! Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# ---------------------------------------------------------
# 1.2 BASIC INFORMATION
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("📊 FIRST 5 ROWS")
print("=" * 70)
print(df.head().to_string())

print("\n" + "=" * 70)
print("📊 LAST 5 ROWS")
print("=" * 70)
print(df.tail().to_string())

# ---------------------------------------------------------
# 1.3 DATA TYPES AND NULLS
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("📊 COLUMN INFO")
print("=" * 70)
print(df.info())

# ---------------------------------------------------------
# 1.4 STATISTICAL SUMMARY
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("📊 STATISTICAL SUMMARY")
print("=" * 70)
print(df.describe().round(2).to_string())

# ---------------------------------------------------------
# 1.5 UNIQUE VALUES
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("📊 UNIQUE VALUES PER COLUMN")
print("=" * 70)
for col in df.columns:
    n = df[col].nunique()
    if n <= 10:
        print(f"  {col:15s}: {n:3d} unique → {sorted(df[col].unique())}")
    else:
        print(f"  {col:15s}: {n:3d} unique → min={df[col].min()}, max={df[col].max()}")

# ---------------------------------------------------------
# 1.6 TARGET VARIABLE DISTRIBUTION
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("🎯 TARGET VARIABLE: price_range")
print("=" * 70)

price_labels = {0: 'Low Cost', 1: 'Medium Cost', 2: 'High Cost', 3: 'Very High Cost'}
target_counts = df['price_range'].value_counts().sort_index()

for val, count in target_counts.items():
    pct = count / len(df) * 100
    bar = '█' * int(pct)
    print(f"  Class {val} ({price_labels[val]:15s}): {count:5d} ({pct:5.1f}%) {bar}")

# ---------------------------------------------------------
# 1.7 CHECK MISSING & DUPLICATES
# ---------------------------------------------------------
missing_count = df.isnull().sum().sum()
duplicate_count = df.duplicated().sum()

print("\n" + "=" * 70)
print("🔍 QUALITY CHECKS")
print("=" * 70)
print(f"  Missing values: {missing_count} {'✅' if missing_count == 0 else '⚠️'}")
print(f"  Duplicate rows: {duplicate_count} {'✅' if duplicate_count == 0 else '⚠️'}")

print("\n" + "=" * 70)
print("✅ STEP 1 COMPLETE!")
print("=" * 70)
print("➡️ Ready for STEP 2: Data Cleaning & Validation")