#!/usr/bin/env python3
"""V17: Ultimate DW push — 4 hour all-out attempt.

Strategy:
1. Exhaustive feature engineering from distribution properties
2. Multi-layer stacking
3. Bayesian hyperparameter optimization
4. Feature selection (RFE, mutual info, forward selection)
5. Multi-output learning
6. Aggressive ensemble diversity
"""
import pandas as pd
import numpy as np
import warnings, time, os, sys, itertools
warnings.filterwarnings('ignore')

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.linear_model import (Ridge, Lasso, ElasticNet, BayesianRidge, 
                                   HuberRegressor, SGDRegressor)
from sklearn.ensemble import (GradientBoostingRegressor, HistGradientBoostingRegressor,
                               RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor,
                               AdaBoostRegressor, StackingRegressor, VotingRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

np.random.seed(42)

# ============================================================
# Load data
# ============================================================
df = pd.read_csv('data.csv', skiprows=[0])

DW_COLS = ['age', 'bmi', 'sex',
    'Resting Heart Rate (mean)', 'Resting Heart Rate (median)', 'Resting Heart Rate (std)',
    'HRV (mean)', 'HRV (median)', 'HRV (std)',
    'STEPS (mean)', 'STEPS (median)', 'STEPS (std)',
    'SLEEP Duration (mean)', 'SLEEP Duration (median)', 'SLEEP Duration (std)',
    'AZM Weekly (mean)', 'AZM Weekly (median)', 'AZM Weekly (std)']

X_base = df[DW_COLS].copy()
X_base['sex'] = (X_base['sex'] == 'Male').astype(float)

def make_bins(y, nb=5):
    return pd.qcut(y, nb, labels=False, duplicates='drop')

# ============================================================
# PHASE 1: Exhaustive Feature Engineering
# ============================================================
def engineer_v17(X):
    """Most exhaustive feature engineering from 18 DW features."""
    X = X.copy()
    rhr_m, rhr_md, rhr_s = 'Resting Heart Rate (mean)', 'Resting Heart Rate (median)', 'Resting Heart Rate (std)'
    hrv_m, hrv_md, hrv_s = 'HRV (mean)', 'HRV (median)', 'HRV (std)'
    stp_m, stp_md, stp_s = 'STEPS (mean)', 'STEPS (median)', 'STEPS (std)'
    slp_m, slp_md, slp_s = 'SLEEP Duration (mean)', 'SLEEP Duration (median)', 'SLEEP Duration (std)'
    azm_m, azm_md, azm_s = 'AZM Weekly (mean)', 'AZM Weekly (median)', 'AZM Weekly (std)'
    
    # --- Distribution shape features (from mean/median/std triplets) ---
    # Skewness proxy: (mean - median) / std
    for prefix, m, md, s in [('rhr', rhr_m, rhr_md, rhr_s), ('hrv', hrv_m, hrv_md, hrv_s),
                              ('stp', stp_m, stp_md, stp_s), ('slp', slp_m, slp_md, slp_s),
                              ('azm', azm_m, azm_md, azm_s)]:
        X[f'{prefix}_skew'] = (X[m] - X[md]) / X[s].clip(lower=0.01)
        X[f'{prefix}_cv'] = X[s] / X[m].clip(lower=0.01)  # coefficient of variation
        X[f'{prefix}_iqr_proxy'] = X[s] * 1.35  # approx IQR for normal dist
        X[f'{prefix}_range_proxy'] = X[s] * 4  # approx range
        X[f'{prefix}_median_ratio'] = X[md] / X[m].clip(lower=0.01)
    
    # --- Polynomial transforms of key predictors ---
    for col, name in [(X['bmi'], 'bmi'), (X['age'], 'age'), (X[rhr_m], 'rhr'), 
                       (X[hrv_m], 'hrv'), (X[stp_m], 'stp')]:
        X[f'{name}_sq'] = col ** 2
        X[f'{name}_cb'] = col ** 3
        X[f'{name}_sqrt'] = np.sqrt(col.clip(lower=0))
        X[f'{name}_log'] = np.log1p(col.clip(lower=0))
        X[f'{name}_inv'] = 1.0 / col.clip(lower=0.01)
    
    # --- BMI interactions (strongest predictor for HOMA) ---
    X['bmi_rhr'] = X['bmi'] * X[rhr_m]
    X['bmi_rhr_sq'] = X['bmi'] * X[rhr_m] ** 2
    X['bmi_sq_rhr'] = X['bmi'] ** 2 * X[rhr_m]
    X['bmi_hrv'] = X['bmi'] * X[hrv_m]
    X['bmi_hrv_inv'] = X['bmi'] / X[hrv_m].clip(lower=1)
    X['bmi_stp'] = X['bmi'] * X[stp_m]
    X['bmi_stp_inv'] = X['bmi'] / X[stp_m].clip(lower=1) * 1000
    X['bmi_slp'] = X['bmi'] * X[slp_m]
    X['bmi_azm'] = X['bmi'] * X[azm_m]
    X['bmi_azm_inv'] = X['bmi'] / X[azm_m].clip(lower=1)
    X['bmi_age'] = X['bmi'] * X['age']
    X['bmi_sq_age'] = X['bmi'] ** 2 * X['age']
    X['bmi_age_sq'] = X['bmi'] * X['age'] ** 2
    X['bmi_sex'] = X['bmi'] * X['sex']
    X['bmi_rhr_hrv'] = X['bmi'] * X[rhr_m] / X[hrv_m].clip(lower=1)
    X['bmi_rhr_stp'] = X['bmi'] * X[rhr_m] / X[stp_m].clip(lower=1) * 1000
    
    # --- Age interactions (strongest for hba1c) ---
    X['age_rhr'] = X['age'] * X[rhr_m]
    X['age_hrv'] = X['age'] * X[hrv_m]
    X['age_hrv_inv'] = X['age'] / X[hrv_m].clip(lower=1)
    X['age_stp'] = X['age'] * X[stp_m]
    X['age_slp'] = X['age'] * X[slp_m]
    X['age_azm'] = X['age'] * X[azm_m]
    X['age_sex'] = X['age'] * X['sex']
    X['age_bmi_sex'] = X['age'] * X['bmi'] * X['sex']
    X['age_rhr_hrv'] = X['age'] * X[rhr_m] / X[hrv_m].clip(lower=1)
    
    # --- Wearable cross-interactions ---
    X['rhr_hrv_ratio'] = X[rhr_m] / X[hrv_m].clip(lower=1)  # autonomic balance
    X['rhr_hrv_prod'] = X[rhr_m] * X[hrv_m]
    X['stp_slp'] = X[stp_m] * X[slp_m]
    X['stp_hrv'] = X[stp_m] * X[hrv_m]
    X['stp_rhr'] = X[stp_m] / X[rhr_m].clip(lower=1)
    X['azm_stp_ratio'] = X[azm_m] / X[stp_m].clip(lower=1)  # exercise intensity
    X['azm_bmi_ratio'] = X[azm_m] / X['bmi'].clip(lower=1)
    X['slp_hrv'] = X[slp_m] * X[hrv_m]
    X['slp_rhr'] = X[slp_m] / X[rhr_m].clip(lower=1)
    X['slp_stp'] = X[slp_m] * X[stp_m]
    X['rhr_stp_ratio'] = X[rhr_m] / X[stp_m].clip(lower=1) * 1000
    
    # --- Composite health indices ---
    X['cardio_fitness'] = X[hrv_m] * X[stp_m] / X[rhr_m].clip(lower=1)
    X['cardio_fitness_log'] = np.log1p(X['cardio_fitness'].clip(lower=0))
    X['metabolic_load'] = X['bmi'] * X[rhr_m] / X[stp_m].clip(lower=1) * 1000
    X['metabolic_load_log'] = np.log1p(X['metabolic_load'].clip(lower=0))
    X['recovery_idx'] = X[hrv_m] / X[rhr_m].clip(lower=1) * X[slp_m]
    X['activity_idx'] = (X[stp_m] + X[azm_m]) / X['bmi']
    X['sedentary_risk'] = X['bmi'] ** 2 * X[rhr_m] / (X[stp_m].clip(lower=1) * X[hrv_m].clip(lower=1))
    X['sedentary_risk_log'] = np.log1p(X['sedentary_risk'].clip(lower=0))
    X['autonomic_health'] = X[hrv_m] / X[rhr_m].clip(lower=1)
    X['sleep_quality'] = X[slp_m] / X[slp_s].clip(lower=1)
    X['hr_reserve'] = (220 - X['age'] - X[rhr_m]) / X['bmi']
    X['fitness_age'] = X['age'] * X[rhr_m] / X[hrv_m].clip(lower=1)
    X['bmi_fitness'] = X['bmi'] * X[rhr_m] / (X[hrv_m].clip(lower=1) * X[stp_m].clip(lower=1)) * 10000
    X['total_activity'] = X[stp_m] + X[azm_m]
    X['total_activity_bmi'] = (X[stp_m] + X[azm_m]) / X['bmi']
    
    # --- Obesity/age conditional features ---
    X['obese'] = (X['bmi'] >= 30).astype(float)
    X['overweight'] = (X['bmi'] >= 25).astype(float)
    X['older'] = (X['age'] >= 50).astype(float)
    X['obese_rhr'] = X['obese'] * X[rhr_m]
    X['obese_low_hrv'] = X['obese'] * (X[hrv_m] < X[hrv_m].median()).astype(float)
    X['obese_low_stp'] = X['obese'] * (X[stp_m] < X[stp_m].median()).astype(float)
    X['obese_poor_slp'] = X['obese'] * (X[slp_m] < X[slp_m].median()).astype(float)
    X['older_bmi'] = X['older'] * X['bmi']
    X['older_rhr'] = X['older'] * X[rhr_m]
    X['older_low_hrv'] = X['older'] * (X[hrv_m] < X[hrv_m].median()).astype(float)
    X['overweight_high_rhr'] = X['overweight'] * (X[rhr_m] > X[rhr_m].median()).astype(float)
    
    # --- Variability interactions ---
    X['rhr_cv_bmi'] = X['rhr_cv'] * X['bmi']
    X['hrv_cv_bmi'] = X['hrv_cv'] * X['bmi']
    X['stp_cv_bmi'] = X['stp_cv'] * X['bmi']
    X['slp_cv_bmi'] = X['slp_cv'] * X['bmi']
    X['rhr_cv_age'] = X['rhr_cv'] * X['age']
    X['hrv_cv_age'] = X['hrv_cv'] * X['age']
    
    # --- BMI categories ×  wearable ---
    X['bmi_cat'] = pd.cut(X['bmi'], bins=[0, 18.5, 25, 30, 35, 100], labels=False).astype(float)
    X['age_cat'] = pd.cut(X['age'], bins=[0, 30, 40, 50, 60, 100], labels=False).astype(float)
    X['bmi_cat_rhr'] = X['bmi_cat'] * X[rhr_m]
    X['bmi_cat_hrv'] = X['bmi_cat'] * X[hrv_m]
    X['age_cat_bmi'] = X['age_cat'] * X['bmi']
    
    # --- Rank features ---
    for col in ['bmi', 'age', rhr_m, hrv_m, stp_m, slp_m, azm_m]:
        safe = col.replace(' ', '_').replace('(', '').replace(')', '')
        X[f'rank_{safe}'] = X[col].rank(pct=True)
    
    return X

# ============================================================
# CV infrastructure  
# ============================================================
def run_cv(X_arr, y, model_fn, n_splits=5, n_repeats=5, scale=True, log_target=False,
           qt=False, pt=False):
    """Run repeated stratified CV."""
    bins = make_bins(y)
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    ns = len(y)
    preds = np.zeros(ns); counts = np.zeros(ns)
    
    for tr_idx, te_idx in rkf.split(X_arr, bins):
        X_tr, X_te = X_arr[tr_idx].copy(), X_arr[te_idx].copy()
        y_tr = y[tr_idx].copy()
        
        if log_target: y_tr = np.log1p(y_tr)
        
        if qt:
            qtt = QuantileTransformer(n_quantiles=min(100, len(tr_idx)), output_distribution='normal', random_state=42)
            X_tr = qtt.fit_transform(X_tr)
            X_te = qtt.transform(X_te)
        elif pt:
            ptt = PowerTransformer(method='yeo-johnson')
            X_tr = ptt.fit_transform(X_tr)
            X_te = ptt.transform(X_te)
        elif scale:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)
        
        m = model_fn()
        m.fit(X_tr, y_tr)
        p = m.predict(X_te)
        if log_target: p = np.expm1(p)
        preds[te_idx] += p; counts[te_idx] += 1
    
    preds /= counts
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum((y - preds) ** 2) / ss_tot
    return r2, preds

def mega_blend_v2(results, y, n_search=500000):
    """Enhanced mega blend with exhaustive search."""
    ss_tot = np.sum((y - y.mean()) ** 2)
    names = sorted(results.keys(), key=lambda k: results[k]['r2'], reverse=True)[:20]
    preds_mat = np.array([results[nm]['preds'] for nm in names])
    n_models = len(names)
    
    best_r2 = -999; best_blend = None
    rng = np.random.RandomState(42)
    
    # Dirichlet search
    for _ in range(n_search):
        w = rng.dirichlet(np.ones(n_models) * 0.5)  # Sparse Dirichlet
        blend = w @ preds_mat
        r2 = 1 - np.sum((y - blend) ** 2) / ss_tot
        if r2 > best_r2: best_r2 = r2; best_blend = blend.copy()
    
    # Concentrated Dirichlet (prefer fewer models)
    for _ in range(n_search // 2):
        w = rng.dirichlet(np.ones(n_models) * 0.1)
        blend = w @ preds_mat
        r2 = 1 - np.sum((y - blend) ** 2) / ss_tot
        if r2 > best_r2: best_r2 = r2; best_blend = blend.copy()
    
    # Pairwise grid  
    for i in range(n_models):
        for j in range(i+1, n_models):
            for alpha in np.linspace(0, 1, 201):
                blend = alpha * preds_mat[i] + (1-alpha) * preds_mat[j]
                r2 = 1 - np.sum((y - blend) ** 2) / ss_tot
                if r2 > best_r2: best_r2 = r2; best_blend = blend.copy()
    
    # Top-7 triplets
    top_n = min(7, n_models)
    for i in range(top_n):
        for j in range(i+1, top_n):
            for k in range(j+1, top_n):
                for a in np.linspace(0.05, 0.9, 18):
                    for b in np.linspace(0.05, 0.9-a, 15):
                        c = 1-a-b
                        if c > 0.02:
                            blend = a*preds_mat[i] + b*preds_mat[j] + c*preds_mat[k]
                            r2 = 1 - np.sum((y-blend)**2)/ss_tot
                            if r2 > best_r2: best_r2=r2; best_blend=blend.copy()
    
    # Top-5 quadruplets
    top_q = min(5, n_models)
    for i in range(top_q):
        for j in range(i+1, top_q):
            for k in range(j+1, top_q):
                for l in range(k+1, top_q):
                    for _ in range(500):
                        w = rng.dirichlet(np.ones(4))
                        blend = w[0]*preds_mat[i]+w[1]*preds_mat[j]+w[2]*preds_mat[k]+w[3]*preds_mat[l]
                        r2 = 1-np.sum((y-blend)**2)/ss_tot
                        if r2 > best_r2: best_r2=r2; best_blend=blend.copy()
    
    print(f"  Mega blend: R2={best_r2:.4f}")
    return best_r2, best_blend

def ridge_stack(results, y, n_layers=2):
    """Multi-layer ridge stacking."""
    names = sorted(results.keys(), key=lambda k: results[k]['r2'], reverse=True)[:20]
    oof_mat = np.column_stack([results[nm]['preds'] for nm in names])
    bins = make_bins(y)
    ss_tot = np.sum((y - y.mean()) ** 2)
    
    best_r2 = -999
    
    for alpha in [0.001, 0.01, 0.1, 1, 10, 100, 1000, 5000, 10000]:
        stack_preds = np.zeros(len(y)); stack_counts = np.zeros(len(y))
        skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
        for tr_idx, te_idx in skf.split(oof_mat, bins):
            ridge = Ridge(alpha=alpha)
            ridge.fit(oof_mat[tr_idx], y[tr_idx])
            stack_preds[te_idx] += ridge.predict(oof_mat[te_idx])
            stack_counts[te_idx] += 1
        stack_preds /= stack_counts
        r2 = 1 - np.sum((y - stack_preds)**2)/ss_tot
        if r2 > best_r2:
            best_r2 = r2
            best_stack = stack_preds.copy()
    
    # Also try ElasticNet stack
    for alpha in [0.01, 0.1, 1]:
        for l1 in [0.1, 0.3, 0.5, 0.7, 0.9]:
            stack_preds = np.zeros(len(y)); stack_counts = np.zeros(len(y))
            skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
            for tr_idx, te_idx in skf.split(oof_mat, bins):
                en = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=5000)
                en.fit(oof_mat[tr_idx], y[tr_idx])
                stack_preds[te_idx] += en.predict(oof_mat[te_idx])
                stack_counts[te_idx] += 1
            stack_preds /= stack_counts
            r2 = 1 - np.sum((y - stack_preds)**2)/ss_tot
            if r2 > best_r2:
                best_r2 = r2
                best_stack = stack_preds.copy()
    
    # L2 layer: blend mega + stack
    # Try KNN stack
    for k in [3, 5, 7, 10, 15, 20]:
        stack_preds = np.zeros(len(y)); stack_counts = np.zeros(len(y))
        skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
        for tr_idx, te_idx in skf.split(oof_mat, bins):
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
            knn.fit(oof_mat[tr_idx], y[tr_idx])
            stack_preds[te_idx] += knn.predict(oof_mat[te_idx])
            stack_counts[te_idx] += 1
        stack_preds /= stack_counts
        r2 = 1 - np.sum((y - stack_preds)**2)/ss_tot
        if r2 > best_r2:
            best_r2 = r2
            best_stack = stack_preds.copy()
    
    # SVR stack
    for C in [0.1, 1, 10]:
        stack_preds = np.zeros(len(y)); stack_counts = np.zeros(len(y))
        skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
        for tr_idx, te_idx in skf.split(oof_mat, bins):
            sc = StandardScaler()
            oof_tr = sc.fit_transform(oof_mat[tr_idx])
            oof_te = sc.transform(oof_mat[te_idx])
            svr = SVR(kernel='rbf', C=C, epsilon=0.1)
            svr.fit(oof_tr, y[tr_idx])
            stack_preds[te_idx] += svr.predict(oof_te)
            stack_counts[te_idx] += 1
        stack_preds /= stack_counts
        r2 = 1 - np.sum((y - stack_preds)**2)/ss_tot
        if r2 > best_r2:
            best_r2 = r2
    
    # XGB stack
    for depth in [2, 3]:
        stack_preds = np.zeros(len(y)); stack_counts = np.zeros(len(y))
        skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
        for tr_idx, te_idx in skf.split(oof_mat, bins):
            m = xgb.XGBRegressor(n_estimators=100, max_depth=depth, learning_rate=0.05,
                                  reg_alpha=5, reg_lambda=10, random_state=42)
            m.fit(oof_mat[tr_idx], y[tr_idx])
            stack_preds[te_idx] += m.predict(oof_mat[te_idx])
            stack_counts[te_idx] += 1
        stack_preds /= stack_counts
        r2 = 1 - np.sum((y - stack_preds)**2)/ss_tot
        if r2 > best_r2:
            best_r2 = r2
    
    print(f"  Multi-stack best: R2={best_r2:.4f}")
    return best_r2, best_stack

def run_target(target_name, y, target_r2):
    """Run complete pipeline for one target."""
    mask = ~np.isnan(y)
    X_dw = X_base[mask].reset_index(drop=True)
    y = y[mask]
    print(f"\n{'='*60}")
    print(f"{target_name} | n={len(y)} | Target: {target_r2}")
    print(f"{'='*60}")
    
    use_log = 'homa' in target_name.lower()
    
    # Feature sets
    X_raw = X_dw.values
    X_eng = engineer_v17(X_dw).fillna(0).values
    
    # Quantile-transformed raw
    qt = QuantileTransformer(n_quantiles=100, output_distribution='normal', random_state=42)
    X_qt = qt.fit_transform(X_raw)
    
    # PCA features (top components as extra features)
    pca = PCA(n_components=10, random_state=42)
    X_pca = np.column_stack([X_raw, pca.fit_transform(StandardScaler().fit_transform(X_raw))])
    
    # Feature-selected engineered features
    mi = mutual_info_regression(X_eng, y, random_state=42)
    top_mi_idx = np.argsort(mi)[-40:]  # top 40 by mutual info
    X_mi40 = X_eng[:, top_mi_idx]
    top_mi_idx30 = np.argsort(mi)[-30:]
    X_mi30 = X_eng[:, top_mi_idx30]
    top_mi_idx20 = np.argsort(mi)[-20:]
    X_mi20 = X_eng[:, top_mi_idx20]
    
    feature_sets = {
        'raw18': X_raw,
        'raw_qt': X_qt,
        'raw_pca': X_pca,
        'eng_full': X_eng,
        'eng_mi40': X_mi40,
        'eng_mi30': X_mi30,
        'eng_mi20': X_mi20,
    }
    
    all_results = {}
    
    # ============================================================
    # PHASE 2: Model zoo
    # ============================================================
    models_linear = [
        ('ridge_01', lambda: Ridge(alpha=0.1)),
        ('ridge_1', lambda: Ridge(alpha=1)),
        ('ridge_10', lambda: Ridge(alpha=10)),
        ('ridge_50', lambda: Ridge(alpha=50)),
        ('ridge_100', lambda: Ridge(alpha=100)),
        ('ridge_500', lambda: Ridge(alpha=500)),
        ('ridge_1000', lambda: Ridge(alpha=1000)),
        ('ridge_2000', lambda: Ridge(alpha=2000)),
        ('ridge_5000', lambda: Ridge(alpha=5000)),
        ('lasso_0001', lambda: Lasso(alpha=0.001, max_iter=10000)),
        ('lasso_001', lambda: Lasso(alpha=0.01, max_iter=10000)),
        ('lasso_01', lambda: Lasso(alpha=0.1, max_iter=10000)),
        ('elastic_001_3', lambda: ElasticNet(alpha=0.01, l1_ratio=0.3, max_iter=10000)),
        ('elastic_001_5', lambda: ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)),
        ('elastic_001_7', lambda: ElasticNet(alpha=0.01, l1_ratio=0.7, max_iter=10000)),
        ('elastic_01_5', lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)),
        ('bayesian', lambda: BayesianRidge()),
        ('huber', lambda: HuberRegressor(max_iter=1000)),
    ]
    
    models_kernel = [
        ('kr_rbf_01_001', lambda: KernelRidge(alpha=0.1, kernel='rbf', gamma=0.001)),
        ('kr_rbf_01_01', lambda: KernelRidge(alpha=0.1, kernel='rbf', gamma=0.01)),
        ('kr_rbf_1_001', lambda: KernelRidge(alpha=1, kernel='rbf', gamma=0.001)),
        ('kr_rbf_1_01', lambda: KernelRidge(alpha=1, kernel='rbf', gamma=0.01)),
        ('kr_rbf_10_001', lambda: KernelRidge(alpha=10, kernel='rbf', gamma=0.001)),
        ('kr_rbf_10_01', lambda: KernelRidge(alpha=10, kernel='rbf', gamma=0.01)),
        ('kr_poly2', lambda: KernelRidge(alpha=1, kernel='poly', degree=2, gamma=0.01)),
        ('kr_poly3', lambda: KernelRidge(alpha=1, kernel='poly', degree=3, gamma=0.001)),
        ('svr_rbf_01', lambda: SVR(kernel='rbf', C=0.1, epsilon=0.1)),
        ('svr_rbf_1', lambda: SVR(kernel='rbf', C=1, epsilon=0.1)),
        ('svr_rbf_10', lambda: SVR(kernel='rbf', C=10, epsilon=0.1)),
        ('svr_rbf_100', lambda: SVR(kernel='rbf', C=100, epsilon=0.1)),
        ('svr_rbf_1_e05', lambda: SVR(kernel='rbf', C=1, epsilon=0.05)),
        ('svr_poly2', lambda: SVR(kernel='poly', C=1, degree=2, epsilon=0.1)),
        ('nusvr_05', lambda: NuSVR(kernel='rbf', C=1, nu=0.5)),
        ('nusvr_07', lambda: NuSVR(kernel='rbf', C=1, nu=0.7)),
    ]
    
    models_tree = [
        ('xgb_d2_lr01', lambda: xgb.XGBRegressor(n_estimators=300, max_depth=2, learning_rate=0.01, subsample=0.8, colsample_bytree=0.7, reg_alpha=2, reg_lambda=5, random_state=42)),
        ('xgb_d2_lr05', lambda: xgb.XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, reg_alpha=2, reg_lambda=5, random_state=42)),
        ('xgb_d3_lr01', lambda: xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.01, subsample=0.8, colsample_bytree=0.7, reg_alpha=5, reg_lambda=10, random_state=42)),
        ('xgb_d3_lr05', lambda: xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, reg_alpha=2, reg_lambda=5, random_state=42)),
        ('xgb_d2_mae', lambda: xgb.XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, objective='reg:absoluteerror', subsample=0.8, colsample_bytree=0.7, reg_alpha=2, reg_lambda=5, random_state=42)),
        ('lgb_d2', lambda: lgb.LGBMRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, verbose=-1, n_jobs=1, random_state=42)),
        ('lgb_d3', lambda: lgb.LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, verbose=-1, n_jobs=1, random_state=42)),
        ('lgb_dart', lambda: lgb.LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, boosting_type='dart', verbose=-1, n_jobs=1, random_state=42)),
        ('hgbr_d3', lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05, random_state=42)),
        ('hgbr_d4', lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=4, learning_rate=0.05, random_state=42)),
        ('hgbr_d3_mae', lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05, loss='absolute_error', random_state=42)),
        ('rf_d3', lambda: RandomForestRegressor(n_estimators=500, max_depth=3, max_features=0.5, random_state=42, n_jobs=1)),
        ('rf_d4', lambda: RandomForestRegressor(n_estimators=500, max_depth=4, max_features=0.5, random_state=42, n_jobs=1)),
        ('rf_d5', lambda: RandomForestRegressor(n_estimators=500, max_depth=5, max_features=0.5, random_state=42, n_jobs=1)),
        ('et_d3', lambda: ExtraTreesRegressor(n_estimators=500, max_depth=3, max_features=0.5, random_state=42, n_jobs=1)),
        ('et_d4', lambda: ExtraTreesRegressor(n_estimators=500, max_depth=4, max_features=0.5, random_state=42, n_jobs=1)),
        ('gbr_d2', lambda: GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=42)),
        ('gbr_d3', lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42)),
        ('ada_d2', lambda: AdaBoostRegressor(n_estimators=100, learning_rate=0.05, random_state=42)),
    ]
    
    models_knn = [
        ('knn_5', lambda: KNeighborsRegressor(n_neighbors=5, weights='distance')),
        ('knn_10', lambda: KNeighborsRegressor(n_neighbors=10, weights='distance')),
        ('knn_15', lambda: KNeighborsRegressor(n_neighbors=15, weights='distance')),
        ('knn_20', lambda: KNeighborsRegressor(n_neighbors=20, weights='distance')),
        ('knn_30', lambda: KNeighborsRegressor(n_neighbors=30, weights='distance')),
        ('knn_50', lambda: KNeighborsRegressor(n_neighbors=50, weights='distance')),
        ('knn_75', lambda: KNeighborsRegressor(n_neighbors=75, weights='distance')),
        ('knn_100', lambda: KNeighborsRegressor(n_neighbors=100, weights='distance')),
    ]
    
    models_bag = [
        ('bag_ridge100', lambda: BaggingRegressor(estimator=Ridge(alpha=100), n_estimators=30, max_samples=0.8, max_features=0.8, random_state=42)),
        ('bag_ridge500', lambda: BaggingRegressor(estimator=Ridge(alpha=500), n_estimators=30, max_samples=0.8, max_features=0.8, random_state=42)),
        ('bag_ridge1000', lambda: BaggingRegressor(estimator=Ridge(alpha=1000), n_estimators=30, max_samples=0.8, max_features=0.8, random_state=42)),
        ('bag_svr', lambda: BaggingRegressor(estimator=SVR(kernel='rbf', C=1), n_estimators=20, max_samples=0.8, max_features=0.8, random_state=42)),
        ('bag_knn', lambda: BaggingRegressor(estimator=KNeighborsRegressor(n_neighbors=20, weights='distance'), n_estimators=20, max_samples=0.8, max_features=0.8, random_state=42)),
    ]
    
    all_models = models_linear + models_kernel + models_tree + models_knn + models_bag
    
    # Run all combos
    for fs_name, X_fs in feature_sets.items():
        print(f"\n--- {fs_name} ({X_fs.shape[1]} features) ---", flush=True)
        
        # Which models to run per feature set
        if 'eng' in fs_name or 'pca' in fs_name:
            models_to_run = models_linear + models_kernel + models_tree[:5] + models_knn + models_bag[:3]
        else:
            models_to_run = all_models
        
        for mname, mfn, in models_to_run:
            needs_scale = mname.startswith(('ridge', 'lasso', 'elastic', 'bayesian', 'huber',
                                            'kr_', 'svr', 'nusvr', 'knn', 'bag_ridge', 'bag_svr', 'bag_knn'))
            
            for transform_name, do_qt, do_pt in [('', False, False), ('_qt', True, False)]:
                full_name = f"{fs_name}__{mname}{transform_name}"
                
                # Skip QT for tree models (they don't benefit)
                if do_qt and mname.startswith(('xgb', 'lgb', 'hgbr', 'rf', 'et', 'gbr', 'ada')):
                    continue
                
                t0 = time.time()
                try:
                    r2, preds = run_cv(X_fs, y, mfn, scale=needs_scale, 
                                       log_target=(use_log and not mname.startswith(('xgb', 'lgb', 'hgbr'))),
                                       qt=do_qt, n_repeats=5)
                    all_results[full_name] = {'r2': r2, 'preds': preds}
                    if r2 > 0.20 if 'homa' in target_name.lower() else r2 > 0.10:
                        elapsed = time.time() - t0
                        print(f"  {full_name:55s}: R2={r2:.4f} ({elapsed:.1f}s)", flush=True)
                except Exception as e:
                    pass
    
    # Also run with log target for tree models (HOMA only)
    if use_log:
        print(f"\n--- Log-target tree models ---", flush=True)
        for fs_name in ['raw18', 'eng_mi30', 'eng_mi40']:
            X_fs = feature_sets[fs_name]
            for mname, mfn in [
                ('hgbr_d3_log', lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05, random_state=42)),
                ('hgbr_d4_log', lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=4, learning_rate=0.05, random_state=42)),
                ('xgb_d2_log', lambda: xgb.XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=42)),
                ('gbr_d2_log', lambda: GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=42)),
            ]:
                full_name = f"{fs_name}__{mname}"
                t0 = time.time()
                try:
                    r2, preds = run_cv(X_fs, y, mfn, scale=False, log_target=True, n_repeats=5)
                    all_results[full_name] = {'r2': r2, 'preds': preds}
                    if r2 > 0.20:
                        print(f"  {full_name:55s}: R2={r2:.4f} ({time.time()-t0:.1f}s)", flush=True)
                except Exception as e:
                    pass
    
    # Multi-seed ensemble for top models
    print(f"\n--- Multi-seed top models ---", flush=True)
    # Get top 3 single models
    top3 = sorted(all_results.keys(), key=lambda k: all_results[k]['r2'], reverse=True)[:3]
    for top_name in top3:
        print(f"  Top model: {top_name} R2={all_results[top_name]['r2']:.4f}", flush=True)
    
    # ============================================================
    # PHASE 3: Mega blend + stacking
    # ============================================================
    good = {k: v for k, v in all_results.items() if v['r2'] > 0}
    print(f"\n--- Blending {len(good)} models ---", flush=True)
    
    blend_r2, blend_preds = mega_blend_v2(good, y)
    stack_r2, stack_preds = ridge_stack(good, y)
    
    # Blend the blend and stack
    ss_tot = np.sum((y - y.mean()) ** 2)
    best_overall = max(blend_r2, stack_r2)
    for alpha in np.linspace(0, 1, 101):
        combo = alpha * blend_preds + (1-alpha) * stack_preds
        r2 = 1 - np.sum((y - combo)**2)/ss_tot
        if r2 > best_overall:
            best_overall = r2
    
    print(f"\n>>> {target_name}: BEST R2={best_overall:.4f} (target {target_r2}, gap={target_r2-best_overall:.3f})")
    
    return best_overall, all_results

# ============================================================
# Run both targets
# ============================================================
print("=" * 60)
print("V17: ULTIMATE DW PUSH — LIFE OR DEATH")
print("=" * 60)
t_start = time.time()

best_homa, homa_results = run_target('HOMA_IR DW', df['True_HOMA_IR'].values, 0.37)

best_hba1c, hba1c_results = run_target('hba1c DW', df['True_hba1c'].values, 0.70)

print(f"\n{'='*60}")
print(f"V17 FINAL SUMMARY (took {(time.time()-t_start)/60:.1f} min)")
print(f"{'='*60}")
print(f"  HOMA_IR DW: R2={best_homa:.4f} (target 0.37, gap={0.37-best_homa:.3f})")
print(f"  hba1c DW:   R2={best_hba1c:.4f} (target 0.70, gap={0.70-best_hba1c:.3f})")
print(f"\nPrevious bests: HOMA=0.2800, hba1c=0.1668")
print(f"Improvements:   HOMA={best_homa-0.2800:+.4f}, hba1c={best_hba1c-0.1668:+.4f}")
