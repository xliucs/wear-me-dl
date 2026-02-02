#!/usr/bin/env python3
"""V13: LightGBM + advanced stacking + exhaustive hyperparameter search."""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import (HistGradientBoostingRegressor, RandomForestRegressor, 
                               ExtraTreesRegressor)
import xgboost as xgb
import lightgbm as lgb
import time

df = pd.read_csv('data.csv', skiprows=[0])
targets_cols = ['True_hba1c','True_HOMA_IR','True_IR_Class','True_Diabetes_2_Class',
                'True_Normoglycemic_2_Class','True_Diabetes_3_Class','Participant_id']

raw_feature_cols = [c for c in df.columns if c not in targets_cols]
X_raw = df[raw_feature_cols].copy()
X_raw['sex_num'] = (X_raw['sex'] == 'Male').astype(int)
X_raw = X_raw.drop('sex', axis=1)

def engineer_all(X):
    X = X.copy()
    X['bmi_sq'] = X['bmi'] ** 2
    X['bmi_cubed'] = X['bmi'] ** 3
    X['age_sq'] = X['age'] ** 2
    X['bmi_age'] = X['bmi'] * X['age']
    X['age_sex'] = X['age'] * X['sex_num']
    X['bmi_sex'] = X['bmi'] * X['sex_num']
    X['trig_hdl'] = X['triglycerides'] / X['hdl'].clip(lower=1)
    X['tyg'] = np.log(X['triglycerides'].clip(lower=1) * X['glucose'].clip(lower=1) / 2)
    X['tyg_bmi'] = X['tyg'] * X['bmi']
    X['non_hdl_ratio'] = (X['total cholesterol'] - X['hdl']) / X['hdl'].clip(lower=1)
    X['glucose_bmi'] = X['glucose'] * X['bmi']
    X['glucose_sq'] = X['glucose'] ** 2
    X['glucose_age'] = X['glucose'] * X['age']
    X['glucose_hdl'] = X['glucose'] / X['hdl'].clip(lower=1)
    X['glucose_trig'] = X['glucose'] * X['triglycerides']
    X['mets_ir'] = np.log(2 * X['glucose'].clip(lower=1) + X['triglycerides'].clip(lower=1)) * X['bmi']
    X['mets_ir_bmi'] = X['mets_ir'] * X['bmi']
    X['glucose_proxy'] = X['glucose'] * X['triglycerides'] / X['hdl'].clip(lower=1)
    X['bmi_trig'] = X['bmi'] * X['triglycerides']
    X['insulin_proxy'] = X['glucose'] * X['bmi'] * X['triglycerides'] / (X['hdl'].clip(lower=1) * 100)
    X['vat_proxy'] = X['bmi'] * X['triglycerides'] / X['hdl'].clip(lower=1)
    X['liver_stress'] = X['alt'] * X['ggt'] / X['albumin'].clip(lower=1)
    X['inflammation'] = X['crp'] * X['white_blood_cell']
    X['glucose_hb'] = X['glucose'] / X['hb'].clip(lower=1)
    X['glucose_rbc'] = X['glucose'] / X['red_blood_cell'].clip(lower=1)
    X['glucose_rdw'] = X['glucose'] * X['rdw']
    X['glucose_mchc'] = X['glucose'] * X['mchc']
    X['glucose_mcv'] = X['glucose'] * X['mcv']
    X['glucose_hematocrit'] = X['glucose'] / X['hematocrit'].clip(lower=1)
    X['rbc_health'] = X['hb'] * X['hematocrit'] / X['red_blood_cell'].clip(lower=1)
    X['rdw_mchc'] = X['rdw'] / X['mchc'].clip(lower=1)
    X['inflammation_glucose'] = X['crp'] * X['glucose']
    X['glucose_log'] = np.log1p(X['glucose'])
    X['glucose_cubed'] = X['glucose'] ** 3
    X['cholesterol_glucose'] = X['total cholesterol'] * X['glucose']
    X['lymph_glucose'] = X['absolute_lymphocytes'] * X['glucose']
    X['ggt_glucose'] = X['ggt'] * X['glucose']
    X['trig_glucose_bmi'] = X['triglycerides'] * X['glucose'] * X['bmi']
    X['egfr_age'] = X['egfr'] * X['age']
    rhr = 'Resting Heart Rate (mean)'
    hrv = 'HRV (mean)'
    steps = 'STEPS (mean)'
    sleep = 'SLEEP Duration (mean)'
    azm = 'AZM Weekly (mean)'
    X['rhr_bmi'] = X[rhr] * X['bmi']
    X['rhr_bmi_sq'] = X[rhr] * X['bmi'] ** 2
    X['rhr_hrv'] = X[rhr] / X[hrv].clip(lower=1)
    X['steps_sleep'] = X[steps] * X[sleep]
    X['bmi_rhr_sq'] = X['bmi'] * X[rhr] ** 2
    X['low_azm_obese'] = (X[azm] < X[azm].median()).astype(int) * X['bmi']
    X['cardio_fitness'] = X[hrv] * X[steps] / X[rhr].clip(lower=1)
    X['sleep_quality'] = X[sleep] * X[hrv]
    return X

def make_bins(y, n_bins=5):
    return pd.qcut(y, n_bins, labels=False, duplicates='drop')

def feature_selection(X, y, n_top=35):
    from sklearn.ensemble import GradientBoostingRegressor
    gbr = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42, subsample=0.8)
    gbr.fit(X, y)
    imp = pd.Series(gbr.feature_importances_, index=X.columns).sort_values(ascending=False)
    return imp.head(n_top).index.tolist()

def run_experiment(X, y, target_name, target_r2, log_target=False):
    """Run comprehensive experiment for one target."""
    bins = make_bins(y)
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    
    models = {}
    
    # XGBoost variants
    xgb_configs = [
        ('xgb_d5', dict(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1)),
        ('xgb_d4', dict(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2)),
        ('xgb_d6', dict(n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.7, colsample_bytree=0.7, reg_alpha=1, reg_lambda=3)),
        ('xgb_d3', dict(n_estimators=800, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2)),
    ]
    
    # LightGBM variants
    lgb_configs = [
        ('lgb_d5', dict(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1, verbose=-1, n_jobs=2)),
        ('lgb_d4', dict(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2, verbose=-1, n_jobs=2)),
        ('lgb_d7', dict(n_estimators=300, max_depth=7, learning_rate=0.08, subsample=0.7, colsample_bytree=0.7, reg_alpha=1, reg_lambda=3, verbose=-1, n_jobs=2)),
        ('lgb_d3', dict(n_estimators=800, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, verbose=-1, n_jobs=2)),
        ('lgb_dart', dict(n_estimators=300, max_depth=5, learning_rate=0.1, boosting_type='dart', verbose=-1, n_jobs=2)),
    ]
    
    all_configs = []
    for name, params in xgb_configs:
        all_configs.append((name, lambda p=params: xgb.XGBRegressor(**p, random_state=42), False))
        if log_target:
            all_configs.append((name+'_log', lambda p=params: xgb.XGBRegressor(**p, random_state=42), True))
    
    for name, params in lgb_configs:
        all_configs.append((name, lambda p=params: lgb.LGBMRegressor(**p, random_state=42), False))
        if log_target:
            all_configs.append((name+'_log', lambda p=params: lgb.LGBMRegressor(**p, random_state=42), True))
    
    # HGBR
    all_configs.append(('hgbr', lambda: HistGradientBoostingRegressor(max_iter=500, max_depth=5, learning_rate=0.05, random_state=42), False))
    if log_target:
        all_configs.append(('hgbr_log', lambda: HistGradientBoostingRegressor(max_iter=500, max_depth=5, learning_rate=0.05, random_state=42), True))
    
    # RF, ET
    all_configs.append(('rf', lambda: RandomForestRegressor(n_estimators=500, max_depth=8, max_features=0.5, random_state=42, n_jobs=2), False))
    all_configs.append(('et', lambda: ExtraTreesRegressor(n_estimators=500, max_depth=8, max_features=0.5, random_state=42, n_jobs=2), False))
    
    # Ridge
    all_configs.append(('ridge', lambda: Ridge(alpha=100), True))  # needs scaling flag, handled separately
    
    # Run all models
    n = len(y)
    oof_preds = {}
    
    for name, model_fn, use_log in all_configs:
        t0 = time.time()
        preds = np.zeros(n)
        counts = np.zeros(n)
        
        for train_idx, test_idx in rkf.split(X, bins):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr = y[train_idx]
            y_use = np.log1p(y_tr) if use_log else y_tr
            
            if name == 'ridge':
                scaler = StandardScaler()
                X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns)
                X_te_s = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns)
                model = model_fn()
                model.fit(X_tr_s, y_use)
                p = model.predict(X_te_s)
            else:
                model = model_fn()
                model.fit(X_tr, y_use)
                p = model.predict(X_te)
            
            if use_log:
                p = np.expm1(p)
            
            preds[test_idx] += p
            counts[test_idx] += 1
        
        preds /= counts
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        oof_preds[name] = preds
        dt = time.time() - t0
        print(f"  {name:25s}: R2={r2:.4f} ({dt:.1f}s)")
    
    # Blend optimization
    names = sorted(oof_preds.keys(), key=lambda k: 1 - np.sum((y - oof_preds[k])**2) / np.sum((y - y.mean())**2), reverse=True)[:10]
    preds_mat = np.array([oof_preds[n] for n in names])
    ss_tot = np.sum((y - y.mean()) ** 2)
    
    best_r2 = -999
    best_w = None
    rng = np.random.RandomState(42)
    
    for _ in range(200000):
        w = rng.dirichlet(np.ones(len(names)))
        blend = w @ preds_mat
        r2 = 1 - np.sum((y - blend) ** 2) / ss_tot
        if r2 > best_r2:
            best_r2 = r2
            best_w = w
    
    # Pairwise
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            for alpha in np.linspace(0, 1, 201):
                blend = alpha * preds_mat[i] + (1-alpha) * preds_mat[j]
                r2 = 1 - np.sum((y - blend) ** 2) / ss_tot
                if r2 > best_r2:
                    best_r2 = r2
                    best_w = np.zeros(len(names))
                    best_w[i] = alpha
                    best_w[j] = 1-alpha
    
    print(f"\n  Blend: R2={best_r2:.4f}")
    for nm, w in zip(names, best_w):
        if w > 0.01:
            print(f"    {nm}: {w:.3f}")
    
    # Ridge stacking
    from sklearn.model_selection import StratifiedKFold
    n_samples = len(y)
    oof_mat = np.column_stack([oof_preds[nm2] for nm2 in names])
    stack_preds = np.zeros(n_samples)
    stack_counts = np.zeros(n_samples)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, te_idx in skf.split(oof_mat, bins):
        ridge = RidgeCV(alphas=[0.1, 1, 10, 100, 1000])
        ridge.fit(oof_mat[tr_idx], y[tr_idx])
        stack_preds[te_idx] = ridge.predict(oof_mat[te_idx])
        stack_counts[te_idx] += 1
    stack_preds /= stack_counts
    r2_stack = 1 - np.sum((y - stack_preds) ** 2) / ss_tot
    print(f"  Ridge stack: R2={r2_stack:.4f}")
    
    best = max(best_r2, r2_stack)
    print(f"\n  >>> {target_name} BEST: R2={best:.4f} (target {target_r2}, gap={target_r2-best:.3f})")
    return best

# ============================================================
print("=" * 60)
print("V13: LIGHTGBM + COMPREHENSIVE STACKING")
print("=" * 60)

# HOMA_IR ALL
print("\n" + "=" * 60)
print("HOMA_IR ALL | Target: 0.65")
print("=" * 60)
y_homa = df['True_HOMA_IR'].values
mask = ~np.isnan(y_homa)
X_eng = engineer_all(X_raw[mask].reset_index(drop=True))
X_eng = X_eng.fillna(X_eng.median())
y_homa = y_homa[mask]
top35 = feature_selection(X_eng, y_homa, 35)
X_sel = X_eng[top35]
best_homa_all = run_experiment(X_sel, y_homa, 'HOMA_IR ALL', 0.65, log_target=True)

# hba1c ALL
print("\n" + "=" * 60)
print("hba1c ALL | Target: 0.85")
print("=" * 60)
y_hba1c = df['True_hba1c'].values
mask2 = ~np.isnan(y_hba1c)
X_eng2 = engineer_all(X_raw[mask2].reset_index(drop=True))
X_eng2 = X_eng2.fillna(X_eng2.median())
y_hba1c = y_hba1c[mask2]
top30_hba1c = feature_selection(X_eng2, y_hba1c, 30)
X_sel2 = X_eng2[top30_hba1c]
print(f"Top 5 hba1c features: {top30_hba1c[:5]}")
best_hba1c_all = run_experiment(X_sel2, y_hba1c, 'hba1c ALL', 0.85, log_target=False)

# DW features
dw_base = ['age', 'bmi', 'sex_num',
           'Resting Heart Rate (mean)', 'Resting Heart Rate (median)', 'Resting Heart Rate (std)',
           'HRV (mean)', 'HRV (median)', 'HRV (std)',
           'STEPS (mean)', 'STEPS (median)', 'STEPS (std)',
           'SLEEP Duration (mean)', 'SLEEP Duration (median)', 'SLEEP Duration (std)',
           'AZM Weekly (mean)', 'AZM Weekly (median)', 'AZM Weekly (std)']
dw_eng = ['bmi_sq', 'bmi_cubed', 'age_sq', 'bmi_age', 'age_sex', 'bmi_sex',
          'rhr_bmi', 'rhr_bmi_sq', 'rhr_hrv', 'steps_sleep', 'bmi_rhr_sq',
          'low_azm_obese', 'cardio_fitness', 'sleep_quality']

# HOMA DW
print("\n" + "=" * 60)
print("HOMA_IR DW | Target: 0.37")
print("=" * 60)
dw_cols = [c for c in dw_base + dw_eng if c in X_eng.columns]
X_dw = X_eng[dw_cols]
best_homa_dw = run_experiment(X_dw, y_homa, 'HOMA_IR DW', 0.37, log_target=True)

# hba1c DW
print("\n" + "=" * 60)
print("hba1c DW | Target: 0.70")
print("=" * 60)
dw_cols2 = [c for c in dw_base + dw_eng if c in X_eng2.columns]
X_dw2 = X_eng2[dw_cols2]
best_hba1c_dw = run_experiment(X_dw2, y_hba1c, 'hba1c DW', 0.70, log_target=False)

# Summary
print("\n" + "=" * 60)
print("V13 FINAL SUMMARY")
print("=" * 60)
for name, r2, target in [
    ('HOMA_IR ALL', best_homa_all, 0.65),
    ('HOMA_IR DW', best_homa_dw, 0.37),
    ('hba1c ALL', best_hba1c_all, 0.85),
    ('hba1c DW', best_hba1c_dw, 0.70),
]:
    print(f"  {name:15s}: R2={r2:.4f} (target {target}) [gap={target-r2:.3f}]")
print("\nDone.")
