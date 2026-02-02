#!/usr/bin/env python3
"""Data exploration for the WEAR-ME insulin resistance dataset."""
import pandas as pd
import numpy as np

# Load data - has 2 header rows
df = pd.read_csv('data.csv', header=[0,1])

# Flatten multi-level columns
print("=== RAW COLUMN STRUCTURE ===")
print(f"Shape: {df.shape}")
print(f"Column levels: {df.columns.nlevels}")
print()

# Read with single header (row 1 = actual column names)
df = pd.read_csv('data.csv', skiprows=[0])
print(f"Shape after skip: {df.shape}")
print(f"\nColumns ({len(df.columns)}):")
for i, c in enumerate(df.columns):
    print(f"  {i}: {c}")

print(f"\n=== TARGET VARIABLES ===")
for target in ['True_hba1c', 'True_HOMA_IR']:
    print(f"\n{target}:")
    print(f"  dtype: {df[target].dtype}")
    print(f"  nulls: {df[target].isna().sum()}")
    print(f"  stats: {df[target].describe()}")

print(f"\n=== MISSING VALUES ===")
missing = df.isnull().sum()
missing = missing[missing > 0]
print(missing.sort_values(ascending=False))

print(f"\n=== DATA TYPES ===")
print(df.dtypes.value_counts())

print(f"\n=== SEX DISTRIBUTION ===")
print(df['sex'].value_counts())

print(f"\n=== CATEGORICAL COLUMNS ===")
for col in df.columns:
    if df[col].dtype == 'object':
        print(f"{col}: {df[col].unique()[:10]}")

print(f"\n=== FEATURE GROUPS ===")
demo_cols = ['age', 'sex', 'bmi']
fitbit_cols = [c for c in df.columns if any(x in c.lower() for x in ['heart rate', 'hrv', 'step', 'sleep', 'azm'])]
label_cols = ['True_hba1c', 'True_HOMA_IR', 'True_IR_Class', 'True_Diabetes_2_Class', 'True_Normoglycemic_2_Class', 'True_Diabetes_3_Class']
blood_cols = [c for c in df.columns if c not in demo_cols + fitbit_cols + label_cols + ['Participant_id']]

print(f"Demographics ({len(demo_cols)}): {demo_cols}")
print(f"Fitbit ({len(fitbit_cols)}): {fitbit_cols}")
print(f"Blood biomarkers ({len(blood_cols)}): {blood_cols}")
print(f"Labels ({len(label_cols)}): {label_cols}")

# Correlations with targets
print(f"\n=== TOP CORRELATIONS WITH True_HOMA_IR ===")
numeric_df = df.select_dtypes(include=[np.number])
corr_homa = numeric_df.corr()['True_HOMA_IR'].abs().sort_values(ascending=False)
print(corr_homa.head(20))

print(f"\n=== TOP CORRELATIONS WITH True_hba1c ===")
corr_hba1c = numeric_df.corr()['True_hba1c'].abs().sort_values(ascending=False)
print(corr_hba1c.head(20))
