#!/usr/bin/env python3
"""Quick baseline check - what's the ceiling for this data?"""
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data.csv', skiprows=[0])
sex_map = {'Female': 0, 'Male': 1}
df['sex_num'] = df['sex'].map(lambda x: sex_map.get(x, 0.5))
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].median())

# Feature engineering
df['trig_hdl'] = df['triglycerides'] / (df['hdl'] + 0.1)
df['glucose_bmi'] = df['glucose'] * df['bmi']
df['glucose_trig'] = df['glucose'] * df['triglycerides'] / 1000
df['ldl_hdl'] = df['ldl'] / (df['hdl'] + 0.1)
df['glucose_sq'] = df['glucose'] ** 2 / 1000
df['bmi_sq'] = df['bmi'] ** 2
df['log_trig'] = np.log1p(df['triglycerides'])
df['log_glucose'] = np.log1p(df['glucose'])
df['glucose_insulin_proxy'] = df['glucose'] * df['trig_hdl']
df['bmi_age'] = df['bmi'] * df['age']
df['rhr_hrv'] = df['Resting Heart Rate (mean)'] / (df['HRV (mean)'] + 0.1)

demo = ['age', 'bmi', 'sex_num']
fitbit = [c for c in df.columns if any(x in c.lower() for x in ['heart rate', 'hrv', 'step', 'sleep', 'azm'])]
label_cols = ['True_hba1c', 'True_HOMA_IR', 'True_IR_Class', 'True_Diabetes_2_Class', 'True_Normoglycemic_2_Class', 'True_Diabetes_3_Class']
blood = [c for c in df.columns if c not in demo + fitbit + label_cols + ['Participant_id', 'sex', 'sex_num'] 
         and c not in ['trig_hdl','glucose_bmi','glucose_trig','ldl_hdl','glucose_sq','bmi_sq','log_trig','log_glucose','glucose_insulin_proxy','bmi_age','rhr_hrv']]
eng = ['trig_hdl','glucose_bmi','glucose_trig','ldl_hdl','glucose_sq','bmi_sq','log_trig','log_glucose','glucose_insulin_proxy','bmi_age','rhr_hrv']

all_cols = demo + fitbit + blood + eng
dw_cols = demo + fitbit + ['bmi_sq', 'bmi_age', 'rhr_hrv']

all_cols = [c for c in all_cols if c in df.columns]
dw_cols = [c for c in dw_cols if c in df.columns]

mask = df[['True_HOMA_IR', 'True_hba1c']].notna().all(axis=1)
df = df[mask]

for col in all_cols:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(df[col].median())

kf = KFold(5, shuffle=True, random_state=42)

models = {
    'Ridge': Ridge(alpha=1.0),
    'HistGBR': HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=6, random_state=42),
    'HistGBR_deep': HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.03, max_depth=8, random_state=42),
    'RF': RandomForestRegressor(n_estimators=500, max_depth=12, random_state=42),
    'MLP_sk': MLPRegressor(hidden_layer_sizes=(256,128,64), max_iter=1000, random_state=42),
}

for target in ['True_HOMA_IR', 'True_hba1c']:
    for feat_name, feat_cols in [('all', all_cols), ('demo_wearable', dw_cols)]:
        print(f"\n{target} | {feat_name} ({len(feat_cols)} features)")
        X = df[feat_cols].values
        y = df[target].values
        
        for mname, model in models.items():
            pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
            scores = cross_val_score(pipe, X, y, cv=kf, scoring='r2')
            
            # Also try with log target
            y_log = np.log1p(y)
            scores_log = cross_val_score(pipe, X, y_log, cv=kf, scoring='r2')
            
            print(f"  {mname:15s}: R²={scores.mean():.4f}±{scores.std():.4f} | log_R²={scores_log.mean():.4f}±{scores_log.std():.4f}")
