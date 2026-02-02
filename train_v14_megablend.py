#!/usr/bin/env python3
"""V14: Mega-blend combining all model families + hyperparameter sweep."""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import (GradientBoostingRegressor, HistGradientBoostingRegressor, 
                               RandomForestRegressor, ExtraTreesRegressor)
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
    sleep_col = 'SLEEP Duration (mean)'
    azm = 'AZM Weekly (mean)'
    X['rhr_bmi'] = X[rhr] * X['bmi']
    X['rhr_bmi_sq'] = X[rhr] * X['bmi'] ** 2
    X['rhr_hrv'] = X[rhr] / X[hrv].clip(lower=1)
    X['steps_sleep'] = X[steps] * X[sleep_col]
    X['bmi_rhr_sq'] = X['bmi'] * X[rhr] ** 2
    X['low_azm_obese'] = (X[azm] < X[azm].median()).astype(int) * X['bmi']
    X['cardio_fitness'] = X[hrv] * X[steps] / X[rhr].clip(lower=1)
    X['sleep_quality'] = X[sleep_col] * X[hrv]
    return X

def make_bins(y, n_bins=5):
    return pd.qcut(y, n_bins, labels=False, duplicates='drop')

def feature_selection(X, y, n_top=35):
    gbr = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42, subsample=0.8)
    gbr.fit(X, y)
    imp = pd.Series(gbr.feature_importances_, index=X.columns).sort_values(ascending=False)
    return imp.head(n_top).index.tolist()

def run_all_models_cv(X, y, log_target=False, n_repeats=5):
    """Run comprehensive model set with 5 repeats for more stable estimates."""
    bins = make_bins(y)
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=42)
    
    # Comprehensive model configs
    configs = [
        # XGBoost sweep
        ('xgb_d5_mse', lambda: xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1, random_state=42), False),
        ('xgb_d4_mse', lambda: xgb.XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2, random_state=42), False),
        ('xgb_d6_mse', lambda: xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.7, colsample_bytree=0.7, reg_alpha=1, reg_lambda=3, random_state=42), False),
        ('xgb_d3_mse', lambda: xgb.XGBRegressor(n_estimators=800, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2, random_state=42), False),
        ('xgb_d5_mae', lambda: xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, objective='reg:absoluteerror', random_state=42), False),
        # LightGBM
        ('lgb_d5', lambda: lgb.LGBMRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, verbose=-1, n_jobs=2, random_state=42), False),
        ('lgb_dart', lambda: lgb.LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.1, boosting_type='dart', verbose=-1, n_jobs=2, random_state=42), False),
        ('lgb_d7', lambda: lgb.LGBMRegressor(n_estimators=300, max_depth=7, learning_rate=0.08, subsample=0.7, colsample_bytree=0.7, verbose=-1, n_jobs=2, random_state=42), False),
        # HGBR
        ('hgbr', lambda: HistGradientBoostingRegressor(max_iter=500, max_depth=5, learning_rate=0.05, random_state=42), False),
        # RF, ET
        ('rf', lambda: RandomForestRegressor(n_estimators=500, max_depth=8, max_features=0.5, random_state=42, n_jobs=2), False),
        ('et', lambda: ExtraTreesRegressor(n_estimators=500, max_depth=8, max_features=0.5, random_state=42, n_jobs=2), False),
    ]
    
    if log_target:
        log_configs = []
        for name, fn, _ in configs[:5]:  # XGB + HGBR with log
            log_configs.append((name + '_log', fn, True))
        log_configs.append(('hgbr_log', lambda: HistGradientBoostingRegressor(max_iter=500, max_depth=5, learning_rate=0.05, random_state=42), True))
        log_configs.append(('lgb_d7_log', lambda: lgb.LGBMRegressor(n_estimators=300, max_depth=7, learning_rate=0.08, subsample=0.7, colsample_bytree=0.7, verbose=-1, n_jobs=2, random_state=42), True))
        configs.extend(log_configs)
    
    n_samples = len(y)
    oof = {}
    
    for name, model_fn, use_log in configs:
        t0 = time.time()
        preds = np.zeros(n_samples)
        counts = np.zeros(n_samples)
        
        for train_idx, test_idx in rkf.split(X, bins):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr = y[train_idx]
            y_use = np.log1p(y_tr) if use_log else y_tr
            
            model = model_fn()
            model.fit(X_tr, y_use)
            p = model.predict(X_te)
            if use_log:
                p = np.expm1(p)
            
            preds[test_idx] += p
            counts[test_idx] += 1
        
        preds /= counts
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - np.sum((y - preds) ** 2) / ss_tot
        oof[name] = preds
        print(f"  {name:25s}: R2={r2:.4f} ({time.time()-t0:.1f}s)")
    
    # Mega blend optimization
    ss_tot = np.sum((y - y.mean()) ** 2)
    names = sorted(oof.keys(), key=lambda k: 1 - np.sum((y - oof[k])**2)/ss_tot, reverse=True)[:12]
    preds_mat = np.array([oof[nm] for nm in names])
    
    best_r2 = -999
    best_w = None
    rng = np.random.RandomState(42)
    
    for _ in range(500000):
        w = rng.dirichlet(np.ones(len(names)))
        blend = w @ preds_mat
        r2 = 1 - np.sum((y - blend) ** 2) / ss_tot
        if r2 > best_r2:
            best_r2 = r2
            best_w = w
    
    # Pairwise + triplet
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
    
    # Triplet blends for top 5
    for i in range(min(5, len(names))):
        for j in range(i+1, min(5, len(names))):
            for k in range(j+1, min(5, len(names))):
                for a in np.linspace(0.1, 0.8, 15):
                    for b in np.linspace(0.1, 0.8-a, 10):
                        c = 1 - a - b
                        if c > 0:
                            blend = a * preds_mat[i] + b * preds_mat[j] + c * preds_mat[k]
                            r2 = 1 - np.sum((y - blend) ** 2) / ss_tot
                            if r2 > best_r2:
                                best_r2 = r2
                                best_w = np.zeros(len(names))
                                best_w[i] = a
                                best_w[j] = b
                                best_w[k] = c
    
    print(f"\n  Mega blend: R2={best_r2:.4f}")
    for nm, w in zip(names, best_w):
        if w > 0.01:
            print(f"    {nm}: {w:.3f}")
    
    # Ridge stack
    oof_mat = np.column_stack([oof[nm] for nm in names])
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
    
    return max(best_r2, r2_stack)

# ============================================================
print("=" * 60)
print("V14: MEGA-BLEND (5 REPEATS, 500K SEARCH)")
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

# Try different feature counts
for n_top in [30, 35, 40]:
    top_feats = feature_selection(X_eng, y_homa, n_top)
    X_sel = X_eng[top_feats]
    print(f"\n--- Top {n_top} features ---")
    best = run_all_models_cv(X_sel, y_homa, log_target=True, n_repeats=5)
    print(f"  >>> Top {n_top} BEST: {best:.4f}")

# hba1c ALL
print("\n" + "=" * 60)
print("hba1c ALL | Target: 0.85")
print("=" * 60)
y_hba1c = df['True_hba1c'].values
mask2 = ~np.isnan(y_hba1c)
X_eng2 = engineer_all(X_raw[mask2].reset_index(drop=True))
X_eng2 = X_eng2.fillna(X_eng2.median())
y_hba1c = y_hba1c[mask2]

for n_top in [20, 25, 30]:
    top_feats = feature_selection(X_eng2, y_hba1c, n_top)
    X_sel2 = X_eng2[top_feats]
    print(f"\n--- Top {n_top} features ---")
    best = run_all_models_cv(X_sel2, y_hba1c, log_target=False, n_repeats=5)
    print(f"  >>> Top {n_top} BEST: {best:.4f}")

print("\nDone.")
