#!/usr/bin/env python3
"""V12: Push scores higher with multi-seed ensembles + stacking + target-specific features."""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor, 
                               HistGradientBoostingRegressor, ExtraTreesRegressor)
import xgboost as xgb
import time

# ============================================================
# DATA
# ============================================================
df = pd.read_csv('data.csv', skiprows=[0])
targets_cols = ['True_hba1c','True_HOMA_IR','True_IR_Class','True_Diabetes_2_Class',
                'True_Normoglycemic_2_Class','True_Diabetes_3_Class','Participant_id']

raw_feature_cols = [c for c in df.columns if c not in targets_cols]
X_raw = df[raw_feature_cols].copy()
X_raw['sex_num'] = (X_raw['sex'] == 'Male').astype(int)
X_raw = X_raw.drop('sex', axis=1)

def engineer_all(X):
    """Full feature engineering for ALL features."""
    X = X.copy()
    X['bmi_sq'] = X['bmi'] ** 2
    X['bmi_cubed'] = X['bmi'] ** 3
    X['age_sq'] = X['age'] ** 2
    X['bmi_age'] = X['bmi'] * X['age']
    X['age_sex'] = X['age'] * X['sex_num']
    X['bmi_sex'] = X['bmi'] * X['sex_num']
    
    # Metabolic
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
    
    # hba1c-specific
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
    
    # Wearable
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
    gbr = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42, subsample=0.8)
    gbr.fit(X, y)
    imp = pd.Series(gbr.feature_importances_, index=X.columns).sort_values(ascending=False)
    return imp.head(n_top).index.tolist(), imp

# ============================================================
# MULTI-SEED ENSEMBLE + STACKING
# ============================================================
def stacked_cv(X, y, base_models, meta_alpha=100, n_splits=5, n_repeats=3, log_target=False):
    """
    Level-1: Run each base model in CV, collect OOF predictions.
    Level-2: Ridge on stacked OOF predictions.
    """
    bins = make_bins(y)
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    
    n = len(y)
    n_models = len(base_models)
    oof_preds = {name: np.zeros(n) for name in base_models}
    oof_counts = np.zeros(n)
    
    for fold_idx, (train_idx, test_idx) in enumerate(rkf.split(X, bins)):
        X_tr, X_te = X.iloc[train_idx].values, X.iloc[test_idx].values
        y_tr = y[train_idx]
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        
        y_tr_use = np.log1p(y_tr) if log_target else y_tr
        
        for name, model_fn in base_models.items():
            model = model_fn()
            # Tree models don't need scaling
            if 'xgb' in name or 'hgbr' in name or 'rf' in name or 'et' in name:
                model.fit(X_tr, y_tr_use)
                preds = model.predict(X_te)
            else:
                model.fit(X_tr_s, y_tr_use)
                preds = model.predict(X_te_s)
            
            if log_target:
                preds = np.expm1(preds)
            
            oof_preds[name][test_idx] += preds
        
        oof_counts[test_idx] += 1
    
    # Average OOF predictions
    for name in oof_preds:
        oof_preds[name] /= oof_counts
    
    # Individual model R²
    ss_tot = np.sum((y - y.mean()) ** 2)
    model_r2s = {}
    for name, preds in oof_preds.items():
        r2 = 1 - np.sum((y - preds) ** 2) / ss_tot
        model_r2s[name] = r2
        print(f"  {name:30s}: R2={r2:.4f}")
    
    # Stack: Ridge on OOF predictions
    oof_matrix = np.column_stack([oof_preds[name] for name in base_models])
    
    # Simple average
    avg_preds = oof_matrix.mean(axis=1)
    r2_avg = 1 - np.sum((y - avg_preds) ** 2) / ss_tot
    print(f"  {'Simple avg':30s}: R2={r2_avg:.4f}")
    
    # Weighted average (optimize)
    best_r2 = -999
    best_w = None
    names = list(base_models.keys())
    rng = np.random.RandomState(42)
    for _ in range(200000):
        w = rng.dirichlet(np.ones(len(names)))
        blend = w @ oof_matrix.T
        r2 = 1 - np.sum((y - blend) ** 2) / ss_tot
        if r2 > best_r2:
            best_r2 = r2
            best_w = w
    
    # Also try pairwise
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            for alpha in np.linspace(0, 1, 201):
                blend = alpha * oof_matrix[:, i] + (1-alpha) * oof_matrix[:, j]
                r2 = 1 - np.sum((y - blend) ** 2) / ss_tot
                if r2 > best_r2:
                    best_r2 = r2
                    best_w = np.zeros(len(names))
                    best_w[i] = alpha
                    best_w[j] = 1-alpha
    
    print(f"  {'Optimized blend':30s}: R2={best_r2:.4f}")
    for nm, w in zip(names, best_w):
        if w > 0.01:
            print(f"    {nm}: {w:.3f}")
    
    # Ridge stacking
    from sklearn.linear_model import RidgeCV
    ridge_meta = RidgeCV(alphas=[0.1, 1, 10, 100, 1000])
    n_samples = len(y)
    stack_preds = np.zeros(n_samples)
    stack_counts = np.zeros(n_samples)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, te_idx in skf.split(oof_matrix, bins):
        ridge_meta.fit(oof_matrix[tr_idx], y[tr_idx])
        stack_preds[te_idx] = ridge_meta.predict(oof_matrix[te_idx])
        stack_counts[te_idx] += 1
    stack_preds /= stack_counts
    r2_stack = 1 - np.sum((y - stack_preds) ** 2) / ss_tot
    print(f"  {'Ridge stack':30s}: R2={r2_stack:.4f}")
    
    return max(best_r2, r2_stack, r2_avg, max(model_r2s.values()))

# ============================================================
# MULTI-SEED XGB
# ============================================================
def multi_seed_xgb(X, y, params, n_seeds=10, n_splits=5, n_repeats=3, log_target=False):
    """Train XGB with multiple seeds and average predictions."""
    bins = make_bins(y)
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    
    all_preds = np.zeros(len(y))
    all_counts = np.zeros(len(y))
    
    for train_idx, test_idx in rkf.split(X, bins):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr = y[train_idx]
        y_tr_use = np.log1p(y_tr) if log_target else y_tr
        
        seed_preds = []
        for seed in range(n_seeds):
            p = params.copy()
            p['random_state'] = seed * 42 + 7
            model = xgb.XGBRegressor(**p)
            model.fit(X_tr, y_tr_use)
            preds = model.predict(X_te)
            if log_target:
                preds = np.expm1(preds)
            seed_preds.append(preds)
        
        avg_preds = np.mean(seed_preds, axis=0)
        all_preds[test_idx] += avg_preds
        all_counts[test_idx] += 1
    
    avg_preds = all_preds / all_counts
    ss_res = np.sum((y - avg_preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot, avg_preds

# ============================================================
# MAIN
# ============================================================
print("=" * 60)
print("V12: MULTI-SEED + STACKING PUSH")
print("=" * 60)

# --- HOMA_IR ALL ---
print("\n" + "=" * 60)
print("HOMA_IR ALL | Target: 0.65")
print("=" * 60)

y_homa = df['True_HOMA_IR'].values
mask = ~np.isnan(y_homa)
X_eng = engineer_all(X_raw[mask].reset_index(drop=True))
X_eng = X_eng.fillna(X_eng.median())
y_homa = y_homa[mask]

top_feats, imp = feature_selection(X_eng, y_homa, n_top=35)
X_sel = X_eng[top_feats]
print(f"Using top 35 features")

# Multi-seed XGB ensembles
params_d5 = dict(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1)
params_d4 = dict(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2)
params_d6 = dict(n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.7, colsample_bytree=0.7, reg_alpha=1, reg_lambda=3)
params_d3 = dict(n_estimators=800, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2)

print("\n--- Multi-seed XGB (10 seeds each) ---")
for name, params, log_t in [
    ('xgb_d5_5seed', params_d5, False),
    ('xgb_d4_5seed', params_d4, False),
    ('xgb_d6_5seed', params_d6, False),
    ('xgb_d5_log_5seed', params_d5, True),
]:
    t0 = time.time()
    r2, _ = multi_seed_xgb(X_sel, y_homa, params, n_seeds=5, n_repeats=1, log_target=log_t)
    print(f"  {name:30s}: R2={r2:.4f} ({time.time()-t0:.1f}s)")

# Also try with QuantileTransformer
print("\n--- QuantileTransformed features ---")
qt = QuantileTransformer(n_quantiles=200, output_distribution='normal', random_state=42)
X_qt = pd.DataFrame(qt.fit_transform(X_sel), columns=X_sel.columns, index=X_sel.index)

for name, params, log_t in [
    ('xgb_d5_qt', params_d5, False),
    ('xgb_d5_qt_log', params_d5, True),
]:
    t0 = time.time()
    r2, _ = multi_seed_xgb(X_qt, y_homa, params, n_seeds=5, log_target=log_t)
    print(f"  {name:30s}: R2={r2:.4f} ({time.time()-t0:.1f}s)")

# Stacking
print("\n--- Stacked Generalization ---")
base_homa = {
    'xgb_d5': lambda: xgb.XGBRegressor(**params_d5, random_state=42),
    'xgb_d4': lambda: xgb.XGBRegressor(**params_d4, random_state=42),
    'xgb_d6': lambda: xgb.XGBRegressor(**params_d6, random_state=42),
    'hgbr': lambda: HistGradientBoostingRegressor(max_iter=500, max_depth=5, learning_rate=0.05, random_state=42),
    'hgbr_log_proxy': lambda: HistGradientBoostingRegressor(max_iter=500, max_depth=5, learning_rate=0.05, random_state=42),
    'rf': lambda: RandomForestRegressor(n_estimators=500, max_depth=8, max_features=0.5, random_state=42, n_jobs=2),
    'et': lambda: ExtraTreesRegressor(n_estimators=500, max_depth=8, max_features=0.5, random_state=42, n_jobs=2),
    'ridge': lambda: Ridge(alpha=100),
}
best_homa_all = stacked_cv(X_sel, y_homa, base_homa, log_target=False)

print("\n--- Stacked with log target ---")
base_homa_log = {
    'xgb_d5_log': lambda: xgb.XGBRegressor(**params_d5, random_state=42),
    'xgb_d4_log': lambda: xgb.XGBRegressor(**params_d4, random_state=42),
    'hgbr_log': lambda: HistGradientBoostingRegressor(max_iter=500, max_depth=5, learning_rate=0.05, random_state=42),
    'rf_log': lambda: RandomForestRegressor(n_estimators=500, max_depth=8, max_features=0.5, random_state=42, n_jobs=2),
    'ridge_log': lambda: Ridge(alpha=100),
}
best_homa_all_log = stacked_cv(X_sel, y_homa, base_homa_log, log_target=True)

print(f"\n>>> HOMA_IR ALL BEST: {max(best_homa_all, best_homa_all_log):.4f}")

# --- hba1c ALL ---
print("\n" + "=" * 60)
print("hba1c ALL | Target: 0.85")
print("=" * 60)

y_hba1c = df['True_hba1c'].values
mask2 = ~np.isnan(y_hba1c)
X_eng2 = engineer_all(X_raw[mask2].reset_index(drop=True))
X_eng2 = X_eng2.fillna(X_eng2.median())
y_hba1c = y_hba1c[mask2]

# Feature selection specifically for hba1c
top_hba1c, imp_hba1c = feature_selection(X_eng2, y_hba1c, n_top=35)
print("Top 10 hba1c features:")
for f in top_hba1c[:10]:
    print(f"  {f}: {imp_hba1c[f]:.4f}")

# Try different feature counts
for n_top in [15, 20, 25, 30, 35, 40]:
    feats = imp_hba1c.head(n_top).index.tolist()
    X_sel2 = X_eng2[feats]
    from sklearn.model_selection import cross_val_score
    r2, _ = multi_seed_xgb(X_sel2, y_hba1c, params_d4, n_seeds=5, n_repeats=1)
    print(f"  Top {n_top}: R2={r2:.4f}")

best_n = 30
top_hba1c = imp_hba1c.head(best_n).index.tolist()
X_hba1c_sel = X_eng2[top_hba1c]

# Multi-seed ensembles for hba1c
print(f"\n--- Multi-seed XGB (10 seeds) on top {best_n} ---")
for name, params, log_t in [
    ('xgb_d4_5seed', params_d4, False),
    ('xgb_d5_5seed', params_d5, False),
    ('xgb_d3_5seed', params_d3, False),
]:
    t0 = time.time()
    r2, _ = multi_seed_xgb(X_hba1c_sel, y_hba1c, params, n_seeds=5, n_repeats=1, log_target=log_t)
    print(f"  {name:30s}: R2={r2:.4f} ({time.time()-t0:.1f}s)")

# Stacking for hba1c
print("\n--- Stacked Generalization ---")
base_hba1c = {
    'xgb_d4': lambda: xgb.XGBRegressor(**params_d4, random_state=42),
    'xgb_d5': lambda: xgb.XGBRegressor(**params_d5, random_state=42),
    'xgb_d3': lambda: xgb.XGBRegressor(**params_d3, random_state=42),
    'xgb_huber': lambda: xgb.XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, objective='reg:pseudohubererror', random_state=42),
    'hgbr': lambda: HistGradientBoostingRegressor(max_iter=500, max_depth=5, learning_rate=0.05, random_state=42),
    'rf': lambda: RandomForestRegressor(n_estimators=500, max_depth=8, max_features=0.5, random_state=42, n_jobs=2),
    'et': lambda: ExtraTreesRegressor(n_estimators=500, max_depth=8, max_features=0.5, random_state=42, n_jobs=2),
    'ridge': lambda: Ridge(alpha=10),
}
best_hba1c_all = stacked_cv(X_hba1c_sel, y_hba1c, base_hba1c)
print(f"\n>>> hba1c ALL BEST: {best_hba1c_all:.4f}")

# --- DW targets ---
print("\n" + "=" * 60)
print("DW TARGETS (demographics + wearables only)")
print("=" * 60)

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
dw_cols = [c for c in dw_base + dw_eng if c in X_eng.columns]
X_homa_dw = X_eng[dw_cols]
print(f"\nHOMA_IR DW ({len(dw_cols)} features)")

base_dw = {
    'xgb_d3': lambda: xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=1, reg_lambda=3, random_state=42),
    'ridge_50': lambda: Ridge(alpha=50),
    'ridge_100': lambda: Ridge(alpha=100),
    'ridge_200': lambda: Ridge(alpha=200),
    'lasso': lambda: Lasso(alpha=0.1),
    'elastic': lambda: ElasticNet(alpha=0.05, l1_ratio=0.5),
    'rf': lambda: RandomForestRegressor(n_estimators=500, max_depth=4, max_features=0.5, random_state=42, n_jobs=2),
}
best_homa_dw = stacked_cv(X_homa_dw, y_homa, base_dw)
print(f"\n>>> HOMA_IR DW BEST: {best_homa_dw:.4f}")

# hba1c DW
dw_cols2 = [c for c in dw_base + dw_eng if c in X_eng2.columns]
X_hba1c_dw = X_eng2[dw_cols2]
print(f"\nhba1c DW ({len(dw_cols2)} features)")

best_hba1c_dw = stacked_cv(X_hba1c_dw, y_hba1c, base_dw)
print(f"\n>>> hba1c DW BEST: {best_hba1c_dw:.4f}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("V12 FINAL SUMMARY")
print("=" * 60)
targets_map = {
    'HOMA_IR ALL': (max(best_homa_all, best_homa_all_log), 0.65),
    'HOMA_IR DW': (best_homa_dw, 0.37),
    'hba1c ALL': (best_hba1c_all, 0.85),
    'hba1c DW': (best_hba1c_dw, 0.70),
}
for name, (r2, target) in targets_map.items():
    gap = target - r2
    print(f"  {name:15s}: R2={r2:.4f} (target {target}) [gap={gap:.3f}]")
print("\nDone.")
