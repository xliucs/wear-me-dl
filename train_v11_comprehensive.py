#!/usr/bin/env python3
"""V11: Comprehensive 4-target optimization with target-specific feature engineering."""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor, 
                               HistGradientBoostingRegressor, ExtraTreesRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import xgboost as xgb
from scipy.stats import pearsonr
import os, sys, time

# Set HF_TOKEN env var for TabPFN access
# os.environ['HF_TOKEN'] = 'your_token_here'

# ============================================================
# DATA LOADING
# ============================================================
df = pd.read_csv('data.csv', skiprows=[0])
targets_cols = ['True_hba1c','True_HOMA_IR','True_IR_Class','True_Diabetes_2_Class',
                'True_Normoglycemic_2_Class','True_Diabetes_3_Class','Participant_id']

raw_feature_cols = [c for c in df.columns if c not in targets_cols]
X_raw = df[raw_feature_cols].copy()
X_raw['sex_num'] = (X_raw['sex'] == 'Male').astype(int)
X_raw = X_raw.drop('sex', axis=1)

# ============================================================
# FEATURE ENGINEERING — COMMON
# ============================================================
def engineer_features(X, target='homa'):
    """Engineer features, with some target-specific ones."""
    X = X.copy()
    
    # Basic interactions
    X['bmi_sq'] = X['bmi'] ** 2
    X['bmi_cubed'] = X['bmi'] ** 3
    X['age_sq'] = X['age'] ** 2
    X['bmi_age'] = X['bmi'] * X['age']
    X['age_sex'] = X['age'] * X['sex_num']
    X['bmi_sex'] = X['bmi'] * X['sex_num']
    
    # Metabolic
    if 'triglycerides' in X.columns and 'hdl' in X.columns:
        X['trig_hdl'] = X['triglycerides'] / X['hdl'].clip(lower=1)
        X['tyg'] = np.log(X['triglycerides'].clip(lower=1) * X['glucose'].clip(lower=1) / 2)
        X['tyg_bmi'] = X['tyg'] * X['bmi']
        X['non_hdl_ratio'] = (X['total cholesterol'] - X['hdl']) / X['hdl'].clip(lower=1)
    
    if 'glucose' in X.columns:
        X['glucose_bmi'] = X['glucose'] * X['bmi']
        X['glucose_sq'] = X['glucose'] ** 2
        X['glucose_age'] = X['glucose'] * X['age']
        X['glucose_hdl'] = X['glucose'] / X['hdl'].clip(lower=1)
        X['glucose_trig'] = X['glucose'] * X['triglycerides']
        
    # HOMA-IR specific
    if target == 'homa':
        X['mets_ir'] = np.log(2 * X['glucose'].clip(lower=1) + X['triglycerides'].clip(lower=1)) * X['bmi']
        X['mets_ir_bmi'] = X['mets_ir'] * X['bmi']
        X['glucose_proxy'] = X['glucose'] * X['triglycerides'] / X['hdl'].clip(lower=1)
        X['bmi_trig'] = X['bmi'] * X['triglycerides']
        X['insulin_proxy'] = X['glucose'] * X['bmi'] * X['triglycerides'] / (X['hdl'].clip(lower=1) * 100)
        X['vat_proxy'] = X['bmi'] * X['triglycerides'] / X['hdl'].clip(lower=1)
        X['liver_stress'] = X['alt'] * X['ggt'] / (X['albumin'].clip(lower=1))
        X['inflammation'] = X['crp'] * X['white_blood_cell']
    
    # hba1c specific — red blood cell & glycation features
    if target == 'hba1c':
        X['glucose_hb'] = X['glucose'] / X['hb'].clip(lower=1)  # glycation ratio proxy
        X['glucose_rbc'] = X['glucose'] / X['red_blood_cell'].clip(lower=1)
        X['glucose_rdw'] = X['glucose'] * X['rdw']
        X['glucose_mchc'] = X['glucose'] * X['mchc']
        X['glucose_mcv'] = X['glucose'] * X['mcv']
        X['glucose_hematocrit'] = X['glucose'] / X['hematocrit'].clip(lower=1)
        X['age_glucose'] = X['age'] * X['glucose']
        X['age_rdw'] = X['age'] * X['rdw']
        X['rbc_health'] = X['hb'] * X['hematocrit'] / X['red_blood_cell'].clip(lower=1)
        X['rdw_mchc'] = X['rdw'] / X['mchc'].clip(lower=1)
        X['inflammation_glucose'] = X['crp'] * X['glucose']
        X['glucose_log'] = np.log1p(X['glucose'])
        X['glucose_cubed'] = X['glucose'] ** 3
        X['cholesterol_glucose'] = X['total cholesterol'] * X['glucose']
        X['lymph_glucose'] = X['absolute_lymphocytes'] * X['glucose']
        X['egfr_age'] = X['egfr'] * X['age']
        X['calcium_glucose'] = X['calcium'] * X['glucose']
        X['globulin_glucose'] = X['globulin'] * X['glucose']
        X['ggt_glucose'] = X['ggt'] * X['glucose']
        X['trig_glucose_bmi'] = X['triglycerides'] * X['glucose'] * X['bmi']
        # Metabolic syndrome proxy
        X['mets_ir'] = np.log(2 * X['glucose'].clip(lower=1) + X['triglycerides'].clip(lower=1)) * X['bmi']
        X['glucose_proxy'] = X['glucose'] * X['triglycerides'] / X['hdl'].clip(lower=1)
    
    # Wearable interactions (for both targets)
    rhr = 'Resting Heart Rate (mean)'
    hrv = 'HRV (mean)'
    steps = 'STEPS (mean)'
    sleep = 'SLEEP Duration (mean)'
    azm = 'AZM Weekly (mean)'
    
    if rhr in X.columns:
        X['rhr_bmi'] = X[rhr] * X['bmi']
        X['rhr_bmi_sq'] = X[rhr] * X['bmi'] ** 2
        X['rhr_hrv'] = X[rhr] / X[hrv].clip(lower=1)
        X['steps_sleep'] = X[steps] * X[sleep]
        X['bmi_rhr_sq'] = X['bmi'] * X[rhr] ** 2
        X['low_azm_obese'] = (X[azm] < X[azm].median()).astype(int) * X['bmi']
        X['cardio_fitness'] = X[hrv] * X[steps] / X[rhr].clip(lower=1)
        X['sleep_quality'] = X[sleep] * X[hrv]
    
    return X

# ============================================================
# DW FEATURES
# ============================================================
dw_base = ['age', 'bmi', 'sex_num',
           'Resting Heart Rate (mean)', 'Resting Heart Rate (median)', 'Resting Heart Rate (std)',
           'HRV (mean)', 'HRV (median)', 'HRV (std)',
           'STEPS (mean)', 'STEPS (median)', 'STEPS (std)',
           'SLEEP Duration (mean)', 'SLEEP Duration (median)', 'SLEEP Duration (std)',
           'AZM Weekly (mean)', 'AZM Weekly (median)', 'AZM Weekly (std)']

def get_dw_features(X_eng):
    """Get DW feature columns from engineered dataframe."""
    dw_eng = ['bmi_sq', 'bmi_cubed', 'age_sq', 'bmi_age', 'age_sex', 'bmi_sex',
              'rhr_bmi', 'rhr_bmi_sq', 'rhr_hrv', 'steps_sleep', 'bmi_rhr_sq',
              'low_azm_obese', 'cardio_fitness', 'sleep_quality']
    return [c for c in dw_base + dw_eng if c in X_eng.columns]

# ============================================================
# CV INFRASTRUCTURE
# ============================================================
def make_bins(y, n_bins=5):
    """Bin continuous target for stratified splits."""
    return pd.qcut(y, n_bins, labels=False, duplicates='drop')

def run_cv(X, y, model_fn, n_splits=5, n_repeats=3, scale=True, log_target=False):
    """Run repeated stratified CV, return mean R²."""
    bins = make_bins(y)
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    
    all_preds = np.zeros(len(y))
    all_counts = np.zeros(len(y))
    
    for train_idx, test_idx in rkf.split(X, bins):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        
        if scale:
            scaler = StandardScaler()
            X_tr = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
            X_te = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns, index=X_te.index)
        
        y_tr_use = np.log1p(y_tr) if log_target else y_tr
        
        model = model_fn()
        model.fit(X_tr, y_tr_use)
        preds = model.predict(X_te)
        
        if log_target:
            preds = np.expm1(preds)
        
        all_preds[test_idx] += preds
        all_counts[test_idx] += 1
    
    avg_preds = all_preds / all_counts
    ss_res = np.sum((y - avg_preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2, avg_preds

def run_cv_multi(X, y, model_configs, n_splits=5, n_repeats=3):
    """Run multiple models, return dict of (r2, preds)."""
    results = {}
    for name, config in model_configs.items():
        t0 = time.time()
        model_fn = config['fn']
        scale = config.get('scale', True)
        log_t = config.get('log', False)
        r2, preds = run_cv(X, y, model_fn, n_splits, n_repeats, scale, log_t)
        dt = time.time() - t0
        results[name] = {'r2': r2, 'preds': preds}
        print(f"  {name:30s}: R2={r2:.4f} ({dt:.1f}s)")
    return results

def optimize_blend(results, y, n_search=200000, top_k=10):
    """Find optimal blend weights via Dirichlet search."""
    names = sorted(results.keys(), key=lambda k: results[k]['r2'], reverse=True)[:top_k]
    preds_mat = np.array([results[n]['preds'] for n in names])
    
    best_r2 = -999
    best_weights = None
    
    rng = np.random.RandomState(42)
    ss_tot = np.sum((y - y.mean()) ** 2)
    
    for _ in range(n_search):
        w = rng.dirichlet(np.ones(len(names)))
        blend = w @ preds_mat
        ss_res = np.sum((y - blend) ** 2)
        r2 = 1 - ss_res / ss_tot
        if r2 > best_r2:
            best_r2 = r2
            best_weights = w
    
    # Also try each pair
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            for alpha in np.linspace(0, 1, 101):
                blend = alpha * preds_mat[i] + (1-alpha) * preds_mat[j]
                ss_res = np.sum((y - blend) ** 2)
                r2 = 1 - ss_res / ss_tot
                if r2 > best_r2:
                    best_r2 = r2
                    best_weights = np.zeros(len(names))
                    best_weights[i] = alpha
                    best_weights[j] = 1-alpha
    
    print(f"  Blend R2={best_r2:.4f}")
    for n, w in zip(names, best_weights):
        if w > 0.01:
            print(f"    {n}: {w:.3f}")
    
    return best_r2

def feature_selection(X, y, n_top=35):
    """Select top features by GBR importance."""
    gbr = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42, subsample=0.8)
    gbr.fit(X, y)
    imp = pd.Series(gbr.feature_importances_, index=X.columns).sort_values(ascending=False)
    return imp.head(n_top).index.tolist()

# ============================================================
# MODEL CONFIGS
# ============================================================
def homa_all_models():
    return {
        'xgb_d5_mse': {'fn': lambda: xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1, random_state=42), 'scale': False},
        'xgb_d4_mse': {'fn': lambda: xgb.XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2, random_state=42), 'scale': False},
        'xgb_d5_mae': {'fn': lambda: xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, objective='reg:absoluteerror', random_state=42), 'scale': False},
        'xgb_d5_log': {'fn': lambda: xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42), 'scale': False, 'log': True},
        'xgb_d3_log': {'fn': lambda: xgb.XGBRegressor(n_estimators=800, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, random_state=42), 'scale': False, 'log': True},
        'xgb_d6_mse': {'fn': lambda: xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.7, colsample_bytree=0.7, reg_alpha=1, reg_lambda=3, random_state=42), 'scale': False},
        'hgbr_wide': {'fn': lambda: HistGradientBoostingRegressor(max_iter=500, max_depth=5, learning_rate=0.05, max_leaf_nodes=31, random_state=42), 'scale': False},
        'hgbr_log': {'fn': lambda: HistGradientBoostingRegressor(max_iter=500, max_depth=5, learning_rate=0.05, random_state=42), 'scale': False, 'log': True},
        'rf_500': {'fn': lambda: RandomForestRegressor(n_estimators=500, max_depth=8, max_features=0.5, random_state=42, n_jobs=-1), 'scale': False},
        'et_500': {'fn': lambda: ExtraTreesRegressor(n_estimators=500, max_depth=8, max_features=0.5, random_state=42, n_jobs=-1), 'scale': False},
        'ridge_100': {'fn': lambda: Ridge(alpha=100), 'scale': True},
        'ridge_10': {'fn': lambda: Ridge(alpha=10), 'scale': True},
    }

def hba1c_all_models():
    """Models tuned for hba1c which has different characteristics."""
    return {
        'xgb_d4_mse': {'fn': lambda: xgb.XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2, random_state=42), 'scale': False},
        'xgb_d5_mse': {'fn': lambda: xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42), 'scale': False},
        'xgb_d3_mse': {'fn': lambda: xgb.XGBRegressor(n_estimators=800, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, reg_alpha=1, reg_lambda=3, random_state=42), 'scale': False},
        'xgb_d4_mae': {'fn': lambda: xgb.XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, objective='reg:absoluteerror', random_state=42), 'scale': False},
        'xgb_d5_huber': {'fn': lambda: xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, objective='reg:pseudohubererror', random_state=42), 'scale': False},
        'hgbr': {'fn': lambda: HistGradientBoostingRegressor(max_iter=500, max_depth=5, learning_rate=0.05, random_state=42), 'scale': False},
        'hgbr_log': {'fn': lambda: HistGradientBoostingRegressor(max_iter=500, max_depth=5, learning_rate=0.05, random_state=42), 'scale': False, 'log': True},
        'rf_500': {'fn': lambda: RandomForestRegressor(n_estimators=500, max_depth=8, max_features=0.5, random_state=42, n_jobs=-1), 'scale': False},
        'et_500': {'fn': lambda: ExtraTreesRegressor(n_estimators=500, max_depth=8, max_features=0.5, random_state=42, n_jobs=-1), 'scale': False},
        'ridge_10': {'fn': lambda: Ridge(alpha=10), 'scale': True},
        'ridge_100': {'fn': lambda: Ridge(alpha=100), 'scale': True},
        'elastic': {'fn': lambda: ElasticNet(alpha=0.01, l1_ratio=0.5), 'scale': True},
    }

def dw_models():
    return {
        'xgb_d3': {'fn': lambda: xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=1, reg_lambda=3, random_state=42), 'scale': False},
        'xgb_d4': {'fn': lambda: xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=1, reg_lambda=3, random_state=42), 'scale': False},
        'xgb_d3_log': {'fn': lambda: xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42), 'scale': False, 'log': True},
        'ridge_50': {'fn': lambda: Ridge(alpha=50), 'scale': True},
        'ridge_100': {'fn': lambda: Ridge(alpha=100), 'scale': True},
        'ridge_200': {'fn': lambda: Ridge(alpha=200), 'scale': True},
        'ridge_500': {'fn': lambda: Ridge(alpha=500), 'scale': True},
        'lasso_01': {'fn': lambda: Lasso(alpha=0.1), 'scale': True},
        'elastic': {'fn': lambda: ElasticNet(alpha=0.05, l1_ratio=0.5), 'scale': True},
        'rf': {'fn': lambda: RandomForestRegressor(n_estimators=500, max_depth=4, max_features=0.5, random_state=42, n_jobs=-1), 'scale': False},
        'knn_20': {'fn': lambda: KNeighborsRegressor(n_neighbors=20, weights='distance'), 'scale': True},
        'knn_30': {'fn': lambda: KNeighborsRegressor(n_neighbors=30, weights='distance'), 'scale': True},
        'svr': {'fn': lambda: SVR(kernel='rbf', C=1.0, epsilon=0.1), 'scale': True},
    }

# ============================================================
# MAIN EXPERIMENTS
# ============================================================
print("=" * 60)
print("V11: COMPREHENSIVE 4-TARGET OPTIMIZATION")
print("=" * 60)

results_summary = {}

# --- 1. HOMA_IR ALL ---
print("\n" + "=" * 60)
print("1. HOMA_IR ALL | Target: 0.65")
print("=" * 60)

y_homa = df['True_HOMA_IR'].values
mask_homa = ~np.isnan(y_homa)
X_homa_eng = engineer_features(X_raw[mask_homa].reset_index(drop=True), target='homa')
X_homa_eng = X_homa_eng.fillna(X_homa_eng.median())
y_homa = y_homa[mask_homa]

# Feature selection with multiple top-k
for n_top in [25, 30, 35, 40]:
    top_feats = feature_selection(X_homa_eng, y_homa, n_top=n_top)
    X_sel = X_homa_eng[top_feats]
    
    # Quick test with best model
    r2, _ = run_cv(X_sel, y_homa, 
                   lambda: xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, 
                                            subsample=0.8, colsample_bytree=0.8, random_state=42),
                   scale=False)
    print(f"  Top {n_top} features: R2={r2:.4f}")

# Use best feature count
best_n = 35  # usually best
top_feats_homa = feature_selection(X_homa_eng, y_homa, n_top=best_n)
X_homa_sel = X_homa_eng[top_feats_homa]

print(f"\nRunning {len(homa_all_models())} models on top {best_n} features...")
results_homa = run_cv_multi(X_homa_sel, y_homa, homa_all_models())
best_homa_all = optimize_blend(results_homa, y_homa)
results_summary['HOMA_IR ALL'] = best_homa_all

# --- 2. HOMA_IR DW ---
print("\n" + "=" * 60)
print("2. HOMA_IR DW | Target: 0.37")
print("=" * 60)

dw_feats = get_dw_features(X_homa_eng)
X_homa_dw = X_homa_eng[dw_feats]
print(f"  DW features: {len(dw_feats)}")

results_homa_dw = run_cv_multi(X_homa_dw, y_homa, dw_models())

# Skip TabPFN (hangs in some CV configurations)

best_homa_dw = optimize_blend(results_homa_dw, y_homa)
results_summary['HOMA_IR DW'] = best_homa_dw

# --- 3. hba1c ALL ---
print("\n" + "=" * 60)
print("3. hba1c ALL | Target: 0.85")
print("=" * 60)

y_hba1c = df['True_hba1c'].values
mask_hba1c = ~np.isnan(y_hba1c)
X_hba1c_eng = engineer_features(X_raw[mask_hba1c].reset_index(drop=True), target='hba1c')
X_hba1c_eng = X_hba1c_eng.fillna(X_hba1c_eng.median())
y_hba1c = y_hba1c[mask_hba1c]

# Feature selection
for n_top in [20, 25, 30, 35, 40]:
    top_feats = feature_selection(X_hba1c_eng, y_hba1c, n_top=n_top)
    X_sel = X_hba1c_eng[top_feats]
    r2, _ = run_cv(X_sel, y_hba1c,
                   lambda: xgb.XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
                                            subsample=0.8, colsample_bytree=0.8, random_state=42),
                   scale=False)
    print(f"  Top {n_top} features: R2={r2:.4f}")

# Find best n_top
best_n_hba1c = 30  
top_feats_hba1c = feature_selection(X_hba1c_eng, y_hba1c, n_top=best_n_hba1c)
X_hba1c_sel = X_hba1c_eng[top_feats_hba1c]

print(f"\nTop features for hba1c:")
for f in top_feats_hba1c[:10]:
    print(f"  {f}")

print(f"\nRunning {len(hba1c_all_models())} models on top {best_n_hba1c} features...")
results_hba1c = run_cv_multi(X_hba1c_sel, y_hba1c, hba1c_all_models())
best_hba1c_all = optimize_blend(results_hba1c, y_hba1c)
results_summary['hba1c ALL'] = best_hba1c_all

# --- 4. hba1c DW ---
print("\n" + "=" * 60)
print("4. hba1c DW | Target: 0.70")
print("=" * 60)

dw_feats_hba1c = get_dw_features(X_hba1c_eng)
X_hba1c_dw = X_hba1c_eng[dw_feats_hba1c]
print(f"  DW features: {len(dw_feats_hba1c)}")

results_hba1c_dw = run_cv_multi(X_hba1c_dw, y_hba1c, dw_models())

# Skip TabPFN (hangs in some CV configurations)

best_hba1c_dw = optimize_blend(results_hba1c_dw, y_hba1c)
results_summary['hba1c DW'] = best_hba1c_dw

# ============================================================
# FINAL SUMMARY
# ============================================================
targets = {'HOMA_IR ALL': 0.65, 'HOMA_IR DW': 0.37, 'hba1c ALL': 0.85, 'hba1c DW': 0.70}

print("\n" + "=" * 60)
print("V11 FINAL SUMMARY")
print("=" * 60)
for name, target in targets.items():
    r2 = results_summary.get(name, 0)
    gap = target - r2
    status = "✓" if r2 >= target else "✗"
    print(f"  {status} {name:15s}: R2={r2:.4f} (target {target}) [gap={gap:.3f}]")

print("\nDone.")
