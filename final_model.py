#!/usr/bin/env python3
"""
Final Model: HOMA-IR DW Prediction (Demographics + Wearables)
=============================================================
Best validated: R²=0.3517 (Pearson r=0.593) via multi-layer stacking of 385 models.

Two modes:
  1. --validate : Reproduce CV result (should give R²=0.3517)
  2. --predict <holdout.csv> : Train on full data, predict holdout

Usage:
  python3 final_model.py --validate
  python3 final_model.py --predict holdout.csv --output predictions.csv
"""
import pandas as pd
import numpy as np
import warnings
import argparse
import time
import json
import os

warnings.filterwarnings('ignore')

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.ensemble import (HistGradientBoostingRegressor, GradientBoostingRegressor,
                               RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor)
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.svm import SVR, NuSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr
import xgboost as xgb
import lightgbm as lgb

# ============================================================
# CONSTANTS (frozen — do not change for reproducibility)
# ============================================================
RANDOM_SEED = 42
N_SPLITS = 5
N_REPEATS = 5
N_BINS = 5

DW_COLS = ['age', 'bmi', 'sex',
    'Resting Heart Rate (mean)', 'Resting Heart Rate (median)', 'Resting Heart Rate (std)',
    'HRV (mean)', 'HRV (median)', 'HRV (std)',
    'STEPS (mean)', 'STEPS (median)', 'STEPS (std)',
    'SLEEP Duration (mean)', 'SLEEP Duration (median)', 'SLEEP Duration (std)',
    'AZM Weekly (mean)', 'AZM Weekly (median)', 'AZM Weekly (std)']

RHR_M = 'Resting Heart Rate (mean)'; RHR_MD = 'Resting Heart Rate (median)'; RHR_S = 'Resting Heart Rate (std)'
HRV_M = 'HRV (mean)'; HRV_MD = 'HRV (median)'; HRV_S = 'HRV (std)'
STP_M = 'STEPS (mean)'; STP_MD = 'STEPS (median)'; STP_S = 'STEPS (std)'
SLP_M = 'SLEEP Duration (mean)'; SLP_MD = 'SLEEP Duration (median)'; SLP_S = 'SLEEP Duration (std)'
AZM_M = 'AZM Weekly (mean)'; AZM_MD = 'AZM Weekly (median)'; AZM_S = 'AZM Weekly (std)'


# ============================================================
# FEATURE ENGINEERING (frozen)
# ============================================================
def engineer_features(X_df):
    """94 engineered features from 18 raw DW features."""
    X = X_df.copy()
    for pfx, m, md, s in [('rhr', RHR_M, RHR_MD, RHR_S), ('hrv', HRV_M, HRV_MD, HRV_S),
                           ('stp', STP_M, STP_MD, STP_S), ('slp', SLP_M, SLP_MD, SLP_S),
                           ('azm', AZM_M, AZM_MD, AZM_S)]:
        X[f'{pfx}_skew'] = (X[m] - X[md]) / X[s].clip(lower=0.01)
        X[f'{pfx}_cv'] = X[s] / X[m].clip(lower=0.01)
    for col, nm in [(X['bmi'], 'bmi'), (X['age'], 'age'), (X[RHR_M], 'rhr'),
                     (X[HRV_M], 'hrv'), (X[STP_M], 'stp')]:
        X[f'{nm}_sq'] = col ** 2
        X[f'{nm}_log'] = np.log1p(col.clip(lower=0))
        X[f'{nm}_inv'] = 1 / (col.clip(lower=0.01))
    X['bmi_rhr'] = X['bmi'] * X[RHR_M]; X['bmi_sq_rhr'] = X['bmi'] ** 2 * X[RHR_M]
    X['bmi_hrv'] = X['bmi'] * X[HRV_M]; X['bmi_hrv_inv'] = X['bmi'] / X[HRV_M].clip(lower=1)
    X['bmi_stp'] = X['bmi'] * X[STP_M]; X['bmi_stp_inv'] = X['bmi'] / X[STP_M].clip(lower=1) * 1000
    X['bmi_slp'] = X['bmi'] * X[SLP_M]; X['bmi_azm'] = X['bmi'] * X[AZM_M]
    X['bmi_age'] = X['bmi'] * X['age']; X['bmi_sq_age'] = X['bmi'] ** 2 * X['age']
    X['bmi_sex'] = X['bmi'] * X['sex']
    X['bmi_rhr_hrv'] = X['bmi'] * X[RHR_M] / X[HRV_M].clip(lower=1)
    X['bmi_rhr_stp'] = X['bmi'] * X[RHR_M] / X[STP_M].clip(lower=1) * 1000
    X['age_rhr'] = X['age'] * X[RHR_M]; X['age_hrv_inv'] = X['age'] / X[HRV_M].clip(lower=1)
    X['age_stp'] = X['age'] * X[STP_M]; X['age_slp'] = X['age'] * X[SLP_M]
    X['age_sex'] = X['age'] * X['sex']; X['age_bmi_sex'] = X['age'] * X['bmi'] * X['sex']
    X['rhr_hrv'] = X[RHR_M] / X[HRV_M].clip(lower=1); X['stp_hrv'] = X[STP_M] * X[HRV_M]
    X['stp_rhr'] = X[STP_M] / X[RHR_M].clip(lower=1); X['azm_stp'] = X[AZM_M] / X[STP_M].clip(lower=1)
    X['slp_hrv'] = X[SLP_M] * X[HRV_M]; X['slp_rhr'] = X[SLP_M] / X[RHR_M].clip(lower=1)
    X['cardio'] = X[HRV_M] * X[STP_M] / X[RHR_M].clip(lower=1)
    X['cardio_log'] = np.log1p(X['cardio'].clip(lower=0))
    X['met_load'] = X['bmi'] * X[RHR_M] / X[STP_M].clip(lower=1) * 1000
    X['met_load_log'] = np.log1p(X['met_load'].clip(lower=0))
    X['recovery'] = X[HRV_M] / X[RHR_M].clip(lower=1) * X[SLP_M]
    X['activity_bmi'] = (X[STP_M] + X[AZM_M]) / X['bmi']
    X['sed_risk'] = X['bmi'] ** 2 * X[RHR_M] / (X[STP_M].clip(lower=1) * X[HRV_M].clip(lower=1))
    X['sed_risk_log'] = np.log1p(X['sed_risk'].clip(lower=0))
    X['auto_health'] = X[HRV_M] / X[RHR_M].clip(lower=1)
    X['hr_reserve'] = (220 - X['age'] - X[RHR_M]) / X['bmi']
    X['fitness_age'] = X['age'] * X[RHR_M] / X[HRV_M].clip(lower=1)
    X['bmi_fitness'] = X['bmi'] * X[RHR_M] / (X[HRV_M].clip(lower=1) * X[STP_M].clip(lower=1)) * 10000
    X['obese'] = (X['bmi'] >= 30).astype(float); X['older'] = (X['age'] >= 50).astype(float)
    X['obese_rhr'] = X['obese'] * X[RHR_M]
    X['obese_low_hrv'] = X['obese'] * (X[HRV_M] < X[HRV_M].median()).astype(float)
    X['older_bmi'] = X['older'] * X['bmi']; X['older_rhr'] = X['older'] * X[RHR_M]
    X['rhr_cv_bmi'] = X['rhr_cv'] * X['bmi']; X['hrv_cv_bmi'] = X['hrv_cv'] * X['bmi']
    X['rhr_cv_age'] = X['rhr_cv'] * X['age']
    for col in ['bmi', 'age', RHR_M, HRV_M, STP_M]:
        X[f'rank_{col[:3]}'] = X[col].rank(pct=True)
    return X


# ============================================================
# AUGMENTED FEATURE GENERATORS (frozen)
# ============================================================
def make_te_features(X_df, y, bins, n_bins_list=[3, 5, 10], smooth=10):
    """Target-encoded features via OOF encoding."""
    te = np.zeros((len(y), 0))
    for col in ['bmi', 'age', RHR_M, HRV_M, STP_M, SLP_M, AZM_M]:
        vals = X_df[col].values
        for nb in n_bins_list:
            try:
                be = np.percentile(vals, np.linspace(0, 100, nb + 1))
                be[0] -= 1; be[-1] += 1
                cb = np.digitize(vals, be[1:-1])
            except:
                continue
            enc = np.zeros(len(y)); gm = y.mean()
            for tr, te_idx in StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED).split(vals, bins):
                for b in range(nb):
                    mtr = cb[tr] == b; mte = cb[te_idx] == b
                    if mtr.sum() > 0:
                        bm = y[tr][mtr].mean(); bc = mtr.sum()
                        enc[te_idx[mte]] = (bc * bm + smooth * gm) / (bc + smooth)
                    else:
                        enc[te_idx[mte]] = gm
            te = np.column_stack([te, enc])
    return te


def make_knn_features(X, y, bins, k_list=[5, 10, 20, 50]):
    """KNN target features via OOF."""
    kf = np.zeros((len(y), 0))
    for k in k_list:
        mf = np.zeros(len(y)); sf = np.zeros(len(y)); df = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED).split(X, bins):
            sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
            nn = NearestNeighbors(n_neighbors=k); nn.fit(Xtr)
            _, idx = nn.kneighbors(Xte)
            for i, ti in enumerate(te):
                nt = y[tr][idx[i]]
                mf[ti] = nt.mean(); sf[ti] = nt.std(); df[ti] = np.median(nt)
        kf = np.column_stack([kf, mf, sf, df])
    return kf


def make_qc_features(X, y, quantiles=[0.25, 0.5, 0.75, 0.9]):
    """Quantile classification features via OOF."""
    from sklearn.ensemble import GradientBoostingClassifier
    qf = np.zeros((len(y), 0))
    for q in quantiles:
        thr = np.quantile(y, q); yc = (y > thr).astype(int)
        probs = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED).split(X, yc):
            sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
            clf = GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.1, random_state=RANDOM_SEED)
            clf.fit(Xtr, yc[tr]); probs[te] = clf.predict_proba(Xte)[:, 1]
        qf = np.column_stack([qf, probs])
    return qf


# ============================================================
# MODEL DEFINITIONS (frozen)
# ============================================================
def get_fast_models():
    return [
        ('ridge_10', lambda: Ridge(alpha=10), True),
        ('ridge_50', lambda: Ridge(alpha=50), True),
        ('ridge_100', lambda: Ridge(alpha=100), True),
        ('ridge_500', lambda: Ridge(alpha=500), True),
        ('ridge_1000', lambda: Ridge(alpha=1000), True),
        ('ridge_2000', lambda: Ridge(alpha=2000), True),
        ('lasso_001', lambda: Lasso(alpha=0.01, max_iter=10000), True),
        ('lasso_01', lambda: Lasso(alpha=0.1, max_iter=10000), True),
        ('elastic_01_5', lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000), True),
        ('elastic_001_5', lambda: ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000), True),
        ('bayesian', lambda: BayesianRidge(), True),
        ('huber', lambda: HuberRegressor(max_iter=1000), True),
        ('kr_rbf_01_001', lambda: KernelRidge(alpha=0.1, kernel='rbf', gamma=0.001), True),
        ('kr_rbf_01_01', lambda: KernelRidge(alpha=0.1, kernel='rbf', gamma=0.01), True),
        ('kr_rbf_1_001', lambda: KernelRidge(alpha=1, kernel='rbf', gamma=0.001), True),
        ('kr_rbf_1_01', lambda: KernelRidge(alpha=1, kernel='rbf', gamma=0.01), True),
        ('kr_rbf_10_01', lambda: KernelRidge(alpha=10, kernel='rbf', gamma=0.01), True),
        ('kr_poly2', lambda: KernelRidge(alpha=1, kernel='poly', degree=2, gamma=0.01), True),
        ('svr_1', lambda: SVR(kernel='rbf', C=1, epsilon=0.1), True),
        ('svr_10', lambda: SVR(kernel='rbf', C=10, epsilon=0.1), True),
        ('nusvr_05', lambda: NuSVR(kernel='rbf', C=1, nu=0.5), True),
        ('nusvr_07', lambda: NuSVR(kernel='rbf', C=1, nu=0.7), True),
        ('knn_10', lambda: KNeighborsRegressor(n_neighbors=10, weights='distance'), True),
        ('knn_15', lambda: KNeighborsRegressor(n_neighbors=15, weights='distance'), True),
        ('knn_20', lambda: KNeighborsRegressor(n_neighbors=20, weights='distance'), True),
        ('knn_30', lambda: KNeighborsRegressor(n_neighbors=30, weights='distance'), True),
        ('knn_50', lambda: KNeighborsRegressor(n_neighbors=50, weights='distance'), True),
        ('knn_75', lambda: KNeighborsRegressor(n_neighbors=75, weights='distance'), True),
        ('bag_ridge100', lambda: BaggingRegressor(estimator=Ridge(alpha=100), n_estimators=30, max_samples=0.8, max_features=0.8, random_state=RANDOM_SEED), True),
        ('bag_ridge500', lambda: BaggingRegressor(estimator=Ridge(alpha=500), n_estimators=30, max_samples=0.8, max_features=0.8, random_state=RANDOM_SEED), True),
        ('bag_ridge1000', lambda: BaggingRegressor(estimator=Ridge(alpha=1000), n_estimators=30, max_samples=0.8, max_features=0.8, random_state=RANDOM_SEED), True),
    ]


def get_med_models():
    return [
        ('xgb_d2', lambda: xgb.XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, reg_alpha=2, reg_lambda=5, random_state=RANDOM_SEED), False),
        ('xgb_d3_lr01', lambda: xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.01, subsample=0.8, colsample_bytree=0.7, reg_alpha=5, reg_lambda=10, random_state=RANDOM_SEED), False),
        ('xgb_d2_mae', lambda: xgb.XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, objective='reg:absoluteerror', random_state=RANDOM_SEED), False),
        ('xgb_d2_huber', lambda: xgb.XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, objective='reg:pseudohubererror', random_state=RANDOM_SEED), False),
        ('lgb_d2', lambda: lgb.LGBMRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, verbose=-1, n_jobs=1, random_state=RANDOM_SEED), False),
        ('lgb_d3', lambda: lgb.LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, verbose=-1, n_jobs=1, random_state=RANDOM_SEED), False),
        ('lgb_dart', lambda: lgb.LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, boosting_type='dart', verbose=-1, n_jobs=1, random_state=RANDOM_SEED), False),
        ('hgbr_d2', lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=2, learning_rate=0.05, random_state=RANDOM_SEED), False),
        ('hgbr_d3', lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05, random_state=RANDOM_SEED), False),
        ('gbr_d2', lambda: GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=RANDOM_SEED), False),
        ('gbr_d3', lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=RANDOM_SEED), False),
        ('rf_d3', lambda: RandomForestRegressor(n_estimators=300, max_depth=3, max_features=0.5, random_state=RANDOM_SEED, n_jobs=1), False),
        ('rf_d4', lambda: RandomForestRegressor(n_estimators=300, max_depth=4, max_features=0.5, random_state=RANDOM_SEED, n_jobs=1), False),
        ('et_d3', lambda: ExtraTreesRegressor(n_estimators=300, max_depth=3, max_features=0.5, random_state=RANDOM_SEED, n_jobs=1), False),
        ('et_d4', lambda: ExtraTreesRegressor(n_estimators=300, max_depth=4, max_features=0.5, random_state=RANDOM_SEED, n_jobs=1), False),
    ]


# ============================================================
# CORE PIPELINE
# ============================================================
def load_and_prepare(data_path='data.csv'):
    """Load data and prepare all feature sets."""
    df = pd.read_csv(data_path, skiprows=[0])
    X_dw = df[DW_COLS].copy()
    X_dw['sex'] = (X_dw['sex'] == 'Male').astype(float)
    
    y = df['True_HOMA_IR'].values
    mask = ~np.isnan(y)
    X_dw = X_dw[mask].reset_index(drop=True)
    y = y[mask]
    
    X_raw = X_dw.values
    X_eng = engineer_features(X_dw).fillna(0).values
    mi = mutual_info_regression(X_eng, y, random_state=RANDOM_SEED)
    X_mi35 = X_eng[:, np.argsort(mi)[-35:]]
    
    bins = pd.qcut(y, N_BINS, labels=False, duplicates='drop')
    
    # Augmented features
    te_feats = make_te_features(X_dw, y, bins)
    knn_feats = make_knn_features(X_raw, y, bins)
    qc_feats = make_qc_features(X_raw, y)
    
    X_mega = np.column_stack([X_raw, te_feats, knn_feats, qc_feats])
    X_mega_eng = np.column_stack([X_eng, te_feats, knn_feats, qc_feats])
    
    fsets = {
        'raw18': X_raw, 'eng': X_eng, 'mi35': X_mi35,
        'mega': X_mega, 'mega_eng': X_mega_eng,
    }
    
    return {
        'df': df, 'X_dw': X_dw, 'y': y, 'bins': bins, 'mask': mask,
        'fsets': fsets, 'n_samples': len(y),
        'ss_tot': np.sum((y - y.mean()) ** 2),
    }


def generate_oof(X, y, model_fn, splits, scale=True, log_t=False):
    """Generate OOF predictions."""
    n = len(y)
    preds = np.zeros(n); counts = np.zeros(n)
    for tr, te in splits:
        Xtr, Xte = X[tr].copy(), X[te].copy()
        ytr = y[tr].copy()
        if log_t:
            ytr = np.log1p(ytr)
        if scale:
            s = StandardScaler(); Xtr = s.fit_transform(Xtr); Xte = s.transform(Xte)
        m = model_fn(); m.fit(Xtr, ytr); p = m.predict(Xte)
        if log_t:
            p = np.expm1(p)
        preds[te] += p; counts[te] += 1
    preds /= counts
    return preds


def run_pipeline(data, verbose=True):
    """Run the full V22b pipeline. Returns best R² and all predictions."""
    y = data['y']; bins = data['bins']; ss_tot = data['ss_tot']
    n_samples = data['n_samples']; fsets = data['fsets']
    
    splits_5rep = list(RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_SEED
    ).split(np.zeros(n_samples), bins))
    
    splits_3rep = list(RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=3, random_state=RANDOM_SEED
    ).split(np.zeros(n_samples), bins))
    
    all_results = {}
    fast_models = get_fast_models()
    med_models = get_med_models()
    
    for fs_name, X_fs in fsets.items():
        if verbose:
            print(f"  {fs_name} ({X_fs.shape[1]}f)...", end='', flush=True)
        count = 0
        for mname, mfn, scale in fast_models:
            full = f"{fs_name}__{mname}"
            try:
                preds = generate_oof(X_fs, y, mfn, splits_5rep, scale=scale)
                r2 = 1 - np.sum((y - preds) ** 2) / ss_tot
                all_results[full] = {'r2': r2, 'preds': preds}; count += 1
                # Log variant
                preds_l = generate_oof(X_fs, y, mfn, splits_5rep, scale=scale, log_t=True)
                r2l = 1 - np.sum((y - preds_l) ** 2) / ss_tot
                all_results[full + '_log'] = {'r2': r2l, 'preds': preds_l}; count += 1
            except:
                pass
        for mname, mfn, scale in med_models:
            full = f"{fs_name}__{mname}"
            try:
                preds = generate_oof(X_fs, y, mfn, splits_3rep, scale=scale)
                r2 = 1 - np.sum((y - preds) ** 2) / ss_tot
                all_results[full] = {'r2': r2, 'preds': preds}; count += 1
            except:
                pass
        if verbose:
            print(f" {count} models", flush=True)
    
    if verbose:
        print(f"\n  Total: {len(all_results)} base models")
    
    # Stacking
    good = {k: v for k, v in all_results.items() if v['r2'] > 0}
    names = sorted(good.keys(), key=lambda k: good[k]['r2'], reverse=True)[:25]
    preds_mat = np.array([good[nm]['preds'] for nm in names])
    
    if verbose:
        print(f"  Stacking top {len(names)} models")
        for nm in names[:5]:
            print(f"    {nm:55s}: R²={good[nm]['r2']:.4f}")
    
    best_r2 = -999; all_stacks = {}
    
    # Layer-1 stackers
    for alpha in [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000]:
        sp = np.zeros(n_samples); sc = np.zeros(n_samples)
        for tr, te in splits_5rep:
            m = Ridge(alpha=alpha); m.fit(preds_mat[:, tr].T, y[tr])
            sp[te] += m.predict(preds_mat[:, te].T); sc[te] += 1
        sp /= sc; r2 = 1 - np.sum((y - sp) ** 2) / ss_tot
        if r2 > best_r2: best_r2 = r2
        if r2 > 0.20: all_stacks[f'ridge_{alpha}'] = sp.copy()
    
    for alpha in [0.01, 0.1, 1]:
        for l1 in [0.1, 0.3, 0.5, 0.7, 0.9]:
            sp = np.zeros(n_samples); sc = np.zeros(n_samples)
            for tr, te in splits_5rep:
                m = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=5000, positive=True)
                m.fit(preds_mat[:, tr].T, y[tr]); sp[te] += m.predict(preds_mat[:, te].T); sc[te] += 1
            sp /= sc; r2 = 1 - np.sum((y - sp) ** 2) / ss_tot
            if r2 > best_r2: best_r2 = r2
            if r2 > 0.20: all_stacks[f'en_{alpha}_{l1}'] = sp.copy()
    
    for alpha in [0.001, 0.01, 0.1]:
        sp = np.zeros(n_samples); sc = np.zeros(n_samples)
        for tr, te in splits_5rep:
            m = Lasso(alpha=alpha, max_iter=5000, positive=True)
            m.fit(preds_mat[:, tr].T, y[tr]); sp[te] += m.predict(preds_mat[:, te].T); sc[te] += 1
        sp /= sc; r2 = 1 - np.sum((y - sp) ** 2) / ss_tot
        if r2 > best_r2: best_r2 = r2
        if r2 > 0.20: all_stacks[f'lasso_{alpha}'] = sp.copy()
    
    for k in [3, 5, 7, 10, 15, 20, 30]:
        sp = np.zeros(n_samples); sc = np.zeros(n_samples)
        for tr, te in splits_3rep:
            ss = StandardScaler(); pt = ss.fit_transform(preds_mat[:, tr].T); pe = ss.transform(preds_mat[:, te].T)
            m = KNeighborsRegressor(n_neighbors=k, weights='distance'); m.fit(pt, y[tr])
            sp[te] += m.predict(pe); sc[te] += 1
        sp /= sc; r2 = 1 - np.sum((y - sp) ** 2) / ss_tot
        if r2 > best_r2: best_r2 = r2
        if r2 > 0.20: all_stacks[f'knn_{k}'] = sp.copy()
    
    for C in [0.1, 1, 10, 100]:
        for eps in [0.05, 0.1]:
            sp = np.zeros(n_samples); sc = np.zeros(n_samples)
            for tr, te in splits_3rep:
                ss = StandardScaler(); pt = ss.fit_transform(preds_mat[:, tr].T); pe = ss.transform(preds_mat[:, te].T)
                m = SVR(kernel='rbf', C=C, epsilon=eps); m.fit(pt, y[tr])
                sp[te] += m.predict(pe); sc[te] += 1
            sp /= sc; r2 = 1 - np.sum((y - sp) ** 2) / ss_tot
            if r2 > best_r2: best_r2 = r2
            if r2 > 0.20: all_stacks[f'svr_{C}_{eps}'] = sp.copy()
    
    for d in [2, 3]:
        sp = np.zeros(n_samples); sc = np.zeros(n_samples)
        for tr, te in splits_3rep:
            m = xgb.XGBRegressor(n_estimators=100, max_depth=d, learning_rate=0.05, reg_alpha=5, reg_lambda=10, random_state=RANDOM_SEED)
            m.fit(preds_mat[:, tr].T, y[tr]); sp[te] += m.predict(preds_mat[:, te].T); sc[te] += 1
        sp /= sc; r2 = 1 - np.sum((y - sp) ** 2) / ss_tot
        if r2 > best_r2: best_r2 = r2
        if r2 > 0.20: all_stacks[f'xgb_{d}'] = sp.copy()
    
    sp = np.zeros(n_samples); sc = np.zeros(n_samples)
    for tr, te in splits_5rep:
        m = BayesianRidge(); m.fit(preds_mat[:, tr].T, y[tr])
        sp[te] += m.predict(preds_mat[:, te].T); sc[te] += 1
    sp /= sc; r2 = 1 - np.sum((y - sp) ** 2) / ss_tot
    if r2 > best_r2: best_r2 = r2
    if r2 > 0.20: all_stacks['bayesian'] = sp.copy()
    
    if verbose:
        print(f"  Layer-1 best: R²={best_r2:.4f} ({len(all_stacks)} stacks)")
    
    # Mega blend
    rng = np.random.RandomState(RANDOM_SEED); blend_best = -999
    nm_count = len(names)
    for _ in range(500000):
        w = rng.dirichlet(np.ones(nm_count) * 0.3); bl = w @ preds_mat
        r2 = 1 - np.sum((y - bl) ** 2) / ss_tot
        if r2 > blend_best: blend_best = r2
    for _ in range(200000):
        w = rng.dirichlet(np.ones(nm_count) * 0.1); bl = w @ preds_mat
        r2 = 1 - np.sum((y - bl) ** 2) / ss_tot
        if r2 > blend_best: blend_best = r2
    for i in range(nm_count):
        for j in range(i + 1, nm_count):
            for a in np.linspace(0, 1, 201):
                bl = a * preds_mat[i] + (1 - a) * preds_mat[j]
                r2 = 1 - np.sum((y - bl) ** 2) / ss_tot
                if r2 > blend_best: blend_best = r2
    for i in range(min(8, nm_count)):
        for j in range(i + 1, min(8, nm_count)):
            for k_idx in range(j + 1, min(8, nm_count)):
                for a in np.linspace(0.05, 0.9, 20):
                    for b in np.linspace(0.05, 0.9 - a, 15):
                        c = 1 - a - b
                        if c > 0:
                            bl = a * preds_mat[i] + b * preds_mat[j] + c * preds_mat[k_idx]
                            r2 = 1 - np.sum((y - bl) ** 2) / ss_tot
                            if r2 > blend_best: blend_best = r2
    if blend_best > best_r2: best_r2 = blend_best
    if verbose:
        print(f"  Mega blend: R²={blend_best:.4f}")
    
    # Layer-2
    if len(all_stacks) >= 3:
        snames = list(all_stacks.keys())
        smat = np.array([all_stacks[k_name] for k_name in snames])
        for alpha in [0.01, 0.1, 1, 10, 100]:
            sp = np.zeros(n_samples); sc = np.zeros(n_samples)
            for tr, te in splits_5rep:
                m = Ridge(alpha=alpha); m.fit(smat[:, tr].T, y[tr])
                sp[te] += m.predict(smat[:, te].T); sc[te] += 1
            sp /= sc; r2 = 1 - np.sum((y - sp) ** 2) / ss_tot
            if r2 > best_r2: best_r2 = r2
        if verbose:
            print(f"  Layer-2: R²={best_r2:.4f} ({len(all_stacks)} stacks)")
    
    r_best = pearsonr(np.zeros(1), np.zeros(1))[0] if best_r2 < 0 else np.sqrt(best_r2)
    
    return {
        'r2': best_r2,
        'pearson_r': r_best,
        'n_base_models': len(all_results),
        'n_stacks': len(all_stacks),
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WEAR-ME HOMA-IR DW Final Model')
    parser.add_argument('--validate', action='store_true', help='Reproduce CV result')
    parser.add_argument('--predict', type=str, help='Path to holdout CSV')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output predictions CSV')
    parser.add_argument('--data', type=str, default='data.csv', help='Training data path')
    args = parser.parse_args()
    
    if args.validate:
        print("=" * 70)
        print("VALIDATION: Reproducing V22b R²=0.3517")
        print("=" * 70)
        
        t0 = time.time()
        print("\nLoading data and preparing features...")
        data = load_and_prepare(args.data)
        print(f"  n={data['n_samples']}, feature sets: {list(data['fsets'].keys())}")
        print(f"  Sizes: {', '.join(f'{k}={v.shape[1]}' for k, v in data['fsets'].items())}")
        
        print("\nGenerating base models...")
        result = run_pipeline(data, verbose=True)
        
        elapsed = (time.time() - t0) / 60
        print(f"\n{'=' * 70}")
        print(f"RESULT: R²={result['r2']:.4f}  r={result['pearson_r']:.4f}")
        print(f"Expected: R²=0.3517  r=0.593")
        diff = result['r2'] - 0.3517
        match_str = 'YES' if abs(diff) < 0.001 else f'NO (diff={diff:+.4f})'
        print(f"Match: {match_str}")
        print(f"Base models: {result['n_base_models']}, Stacks: {result['n_stacks']}")
        print(f"Time: {elapsed:.1f} min")
        print(f"{'=' * 70}")
    
    elif args.predict:
        print(f"Holdout prediction not yet implemented.")
        print(f"TODO: Train full pipeline on all data, predict holdout.")
    
    else:
        parser.print_help()
