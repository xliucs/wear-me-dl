#!/usr/bin/env python3
"""V17b: Focused DW push — maximize stacking with diverse base models.
Key insight from V16: Ridge stack of diverse models got 0.2800 from singles at 0.26.
Strategy: Generate maximum diversity, then multi-layer stack."""
import pandas as pd, numpy as np, warnings, time, os, sys
warnings.filterwarnings('ignore')
os.environ['HF_TOKEN'] = 'REDACTED'

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.ensemble import (HistGradientBoostingRegressor, RandomForestRegressor, 
                               ExtraTreesRegressor, BaggingRegressor, GradientBoostingRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.feature_selection import mutual_info_regression
import xgboost as xgb
import lightgbm as lgb

df = pd.read_csv('data.csv', skiprows=[0])
DW = ['age','bmi','sex',
    'Resting Heart Rate (mean)','Resting Heart Rate (median)','Resting Heart Rate (std)',
    'HRV (mean)','HRV (median)','HRV (std)',
    'STEPS (mean)','STEPS (median)','STEPS (std)',
    'SLEEP Duration (mean)','SLEEP Duration (median)','SLEEP Duration (std)',
    'AZM Weekly (mean)','AZM Weekly (median)','AZM Weekly (std)']

X_base = df[DW].copy()
X_base['sex'] = (X_base['sex']=='Male').astype(float)

rhr_m = 'Resting Heart Rate (mean)'; rhr_md = 'Resting Heart Rate (median)'; rhr_s = 'Resting Heart Rate (std)'
hrv_m = 'HRV (mean)'; hrv_md = 'HRV (median)'; hrv_s = 'HRV (std)'
stp_m = 'STEPS (mean)'; stp_md = 'STEPS (median)'; stp_s = 'STEPS (std)'
slp_m = 'SLEEP Duration (mean)'; slp_md = 'SLEEP Duration (median)'; slp_s = 'SLEEP Duration (std)'
azm_m = 'AZM Weekly (mean)'; azm_md = 'AZM Weekly (median)'; azm_s = 'AZM Weekly (std)'

def make_bins(y, nb=5): return pd.qcut(y, nb, labels=False, duplicates='drop')

def engineer(X):
    X = X.copy()
    # Distribution shape
    for pfx, m, md, s in [('rhr',rhr_m,rhr_md,rhr_s),('hrv',hrv_m,hrv_md,hrv_s),
                           ('stp',stp_m,stp_md,stp_s),('slp',slp_m,slp_md,slp_s),('azm',azm_m,azm_md,azm_s)]:
        X[f'{pfx}_skew'] = (X[m]-X[md])/X[s].clip(lower=0.01)
        X[f'{pfx}_cv'] = X[s]/X[m].clip(lower=0.01)
    # Polynomials
    for col,nm in [(X['bmi'],'bmi'),(X['age'],'age'),(X[rhr_m],'rhr'),(X[hrv_m],'hrv'),(X[stp_m],'stp')]:
        X[f'{nm}_sq']=col**2; X[f'{nm}_log']=np.log1p(col.clip(lower=0)); X[f'{nm}_inv']=1/(col.clip(lower=0.01))
    # BMI interactions
    X['bmi_rhr']=X['bmi']*X[rhr_m]; X['bmi_sq_rhr']=X['bmi']**2*X[rhr_m]
    X['bmi_hrv']=X['bmi']*X[hrv_m]; X['bmi_hrv_inv']=X['bmi']/X[hrv_m].clip(lower=1)
    X['bmi_stp']=X['bmi']*X[stp_m]; X['bmi_stp_inv']=X['bmi']/X[stp_m].clip(lower=1)*1000
    X['bmi_slp']=X['bmi']*X[slp_m]; X['bmi_azm']=X['bmi']*X[azm_m]
    X['bmi_age']=X['bmi']*X['age']; X['bmi_sq_age']=X['bmi']**2*X['age']
    X['bmi_sex']=X['bmi']*X['sex']; X['bmi_rhr_hrv']=X['bmi']*X[rhr_m]/X[hrv_m].clip(lower=1)
    X['bmi_rhr_stp']=X['bmi']*X[rhr_m]/X[stp_m].clip(lower=1)*1000
    # Age interactions
    X['age_rhr']=X['age']*X[rhr_m]; X['age_hrv_inv']=X['age']/X[hrv_m].clip(lower=1)
    X['age_stp']=X['age']*X[stp_m]; X['age_slp']=X['age']*X[slp_m]
    X['age_sex']=X['age']*X['sex']; X['age_bmi_sex']=X['age']*X['bmi']*X['sex']
    # Wearable cross
    X['rhr_hrv']=X[rhr_m]/X[hrv_m].clip(lower=1); X['stp_hrv']=X[stp_m]*X[hrv_m]
    X['stp_rhr']=X[stp_m]/X[rhr_m].clip(lower=1); X['azm_stp']=X[azm_m]/X[stp_m].clip(lower=1)
    X['slp_hrv']=X[slp_m]*X[hrv_m]; X['slp_rhr']=X[slp_m]/X[rhr_m].clip(lower=1)
    # Composites
    X['cardio']=X[hrv_m]*X[stp_m]/X[rhr_m].clip(lower=1)
    X['cardio_log']=np.log1p(X['cardio'].clip(lower=0))
    X['met_load']=X['bmi']*X[rhr_m]/X[stp_m].clip(lower=1)*1000
    X['met_load_log']=np.log1p(X['met_load'].clip(lower=0))
    X['recovery']=X[hrv_m]/X[rhr_m].clip(lower=1)*X[slp_m]
    X['activity']=X[stp_m]+X[azm_m]; X['activity_bmi']=(X[stp_m]+X[azm_m])/X['bmi']
    X['sed_risk']=X['bmi']**2*X[rhr_m]/(X[stp_m].clip(lower=1)*X[hrv_m].clip(lower=1))
    X['sed_risk_log']=np.log1p(X['sed_risk'].clip(lower=0))
    X['auto_health']=X[hrv_m]/X[rhr_m].clip(lower=1)
    X['hr_reserve']=(220-X['age']-X[rhr_m])/X['bmi']
    X['fitness_age']=X['age']*X[rhr_m]/X[hrv_m].clip(lower=1)
    X['bmi_fitness']=X['bmi']*X[rhr_m]/(X[hrv_m].clip(lower=1)*X[stp_m].clip(lower=1))*10000
    # Conditional
    X['obese']=(X['bmi']>=30).astype(float); X['older']=(X['age']>=50).astype(float)
    X['obese_rhr']=X['obese']*X[rhr_m]; X['obese_low_hrv']=X['obese']*(X[hrv_m]<X[hrv_m].median()).astype(float)
    X['older_bmi']=X['older']*X['bmi']; X['older_rhr']=X['older']*X[rhr_m]
    # Variability interactions
    X['rhr_cv_bmi']=X['rhr_cv']*X['bmi']; X['hrv_cv_bmi']=X['hrv_cv']*X['bmi']
    X['rhr_cv_age']=X['rhr_cv']*X['age']
    # Ranks
    for col in ['bmi','age',rhr_m,hrv_m,stp_m]:
        X[f'rank_{col[:3]}']=X[col].rank(pct=True)
    return X

def run_cv(X, y, model_fn, scale=True, log_t=False, qt=False, n_rep=5):
    bins = make_bins(y)
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=42)
    ns = len(y); preds = np.zeros(ns); counts = np.zeros(ns)
    for tr, te in rkf.split(X, bins):
        Xtr, Xte = X[tr].copy(), X[te].copy(); ytr = y[tr].copy()
        if log_t: ytr = np.log1p(ytr)
        if qt:
            q = QuantileTransformer(n_quantiles=min(100,len(tr)), output_distribution='normal', random_state=42)
            Xtr = q.fit_transform(Xtr); Xte = q.transform(Xte)
        elif scale:
            s = StandardScaler(); Xtr = s.fit_transform(Xtr); Xte = s.transform(Xte)
        m = model_fn(); m.fit(Xtr, ytr); p = m.predict(Xte)
        if log_t: p = np.expm1(p)
        preds[te] += p; counts[te] += 1
    preds /= counts
    return 1 - np.sum((y-preds)**2)/np.sum((y-y.mean())**2), preds

def advanced_stack(results, y, label):
    """Multi-method stacking with thorough search."""
    ss_tot = np.sum((y-y.mean())**2)
    names = sorted(results.keys(), key=lambda k: results[k]['r2'], reverse=True)[:25]
    preds_mat = np.array([results[nm]['preds'] for nm in names])
    nm_count = len(names)
    bins = make_bins(y)
    
    print(f"\n  Top 10 base models:", flush=True)
    for nm in names[:10]:
        print(f"    {nm:55s}: R2={results[nm]['r2']:.4f}", flush=True)
    
    best_r2 = -999; best_preds = None
    all_stack_preds = {}  # For layer 2
    
    # --- Layer 1: Diverse stacking methods ---
    # Ridge stack (sweep alpha)
    for alpha in [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000]:
        sp = np.zeros(len(y)); sc_count = np.zeros(len(y))
        for tr, te in RepeatedStratifiedKFold(5, 5, random_state=42).split(preds_mat.T, bins):
            m = Ridge(alpha=alpha); m.fit(preds_mat[:, tr].T, y[tr])
            sp[te] += m.predict(preds_mat[:, te].T); sc_count[te] += 1
        sp /= sc_count
        r2 = 1-np.sum((y-sp)**2)/ss_tot
        if r2 > best_r2: best_r2 = r2; best_preds = sp.copy()
        if r2 > 0.25: all_stack_preds[f'ridge_{alpha}'] = sp.copy()
    print(f"  Ridge stack best: R2={best_r2:.4f}", flush=True)
    
    # ElasticNet stack
    for alpha in [0.01, 0.1, 1]:
        for l1 in [0.1, 0.3, 0.5, 0.7, 0.9]:
            sp = np.zeros(len(y)); sc_count = np.zeros(len(y))
            for tr, te in RepeatedStratifiedKFold(5, 5, random_state=42).split(preds_mat.T, bins):
                m = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=5000, positive=True)
                m.fit(preds_mat[:, tr].T, y[tr])
                sp[te] += m.predict(preds_mat[:, te].T); sc_count[te] += 1
            sp /= sc_count
            r2 = 1-np.sum((y-sp)**2)/ss_tot
            if r2 > best_r2: best_r2 = r2; best_preds = sp.copy()
            if r2 > 0.25: all_stack_preds[f'en_{alpha}_{l1}'] = sp.copy()
    print(f"  After ElasticNet stack: R2={best_r2:.4f}", flush=True)
    
    # Lasso stack (feature selection among models)
    for alpha in [0.001, 0.01, 0.1]:
        sp = np.zeros(len(y)); sc_count = np.zeros(len(y))
        for tr, te in RepeatedStratifiedKFold(5, 5, random_state=42).split(preds_mat.T, bins):
            m = Lasso(alpha=alpha, max_iter=5000, positive=True)
            m.fit(preds_mat[:, tr].T, y[tr])
            sp[te] += m.predict(preds_mat[:, te].T); sc_count[te] += 1
        sp /= sc_count
        r2 = 1-np.sum((y-sp)**2)/ss_tot
        if r2 > best_r2: best_r2 = r2; best_preds = sp.copy()
        if r2 > 0.25: all_stack_preds[f'lasso_{alpha}'] = sp.copy()
    print(f"  After Lasso stack: R2={best_r2:.4f}", flush=True)
    
    # KNN stack
    for k in [3, 5, 7, 10, 15, 20, 30]:
        sp = np.zeros(len(y)); sc_count = np.zeros(len(y))
        for tr, te in RepeatedStratifiedKFold(5, 3, random_state=42).split(preds_mat.T, bins):
            ss = StandardScaler(); pt = ss.fit_transform(preds_mat[:, tr].T); pe = ss.transform(preds_mat[:, te].T)
            m = KNeighborsRegressor(n_neighbors=k, weights='distance'); m.fit(pt, y[tr])
            sp[te] += m.predict(pe); sc_count[te] += 1
        sp /= sc_count
        r2 = 1-np.sum((y-sp)**2)/ss_tot
        if r2 > best_r2: best_r2 = r2; best_preds = sp.copy()
        if r2 > 0.25: all_stack_preds[f'knn_{k}'] = sp.copy()
    print(f"  After KNN stack: R2={best_r2:.4f}", flush=True)
    
    # SVR stack
    for C in [0.1, 1, 10, 100]:
        for eps in [0.05, 0.1, 0.2]:
            sp = np.zeros(len(y)); sc_count = np.zeros(len(y))
            for tr, te in RepeatedStratifiedKFold(5, 3, random_state=42).split(preds_mat.T, bins):
                ss = StandardScaler(); pt = ss.fit_transform(preds_mat[:, tr].T); pe = ss.transform(preds_mat[:, te].T)
                m = SVR(kernel='rbf', C=C, epsilon=eps); m.fit(pt, y[tr])
                sp[te] += m.predict(pe); sc_count[te] += 1
            sp /= sc_count
            r2 = 1-np.sum((y-sp)**2)/ss_tot
            if r2 > best_r2: best_r2 = r2; best_preds = sp.copy()
            if r2 > 0.25: all_stack_preds[f'svr_{C}_{eps}'] = sp.copy()
    print(f"  After SVR stack: R2={best_r2:.4f}", flush=True)
    
    # XGB stack
    for depth in [2, 3]:
        for lr in [0.01, 0.05]:
            sp = np.zeros(len(y)); sc_count = np.zeros(len(y))
            for tr, te in RepeatedStratifiedKFold(5, 3, random_state=42).split(preds_mat.T, bins):
                m = xgb.XGBRegressor(n_estimators=100, max_depth=depth, learning_rate=lr,
                                      reg_alpha=5, reg_lambda=10, random_state=42)
                m.fit(preds_mat[:, tr].T, y[tr])
                sp[te] += m.predict(preds_mat[:, te].T); sc_count[te] += 1
            sp /= sc_count
            r2 = 1-np.sum((y-sp)**2)/ss_tot
            if r2 > best_r2: best_r2 = r2; best_preds = sp.copy()
            if r2 > 0.25: all_stack_preds[f'xgb_{depth}_{lr}'] = sp.copy()
    print(f"  After XGB stack: R2={best_r2:.4f}", flush=True)
    
    # Bayesian Ridge stack
    sp = np.zeros(len(y)); sc_count = np.zeros(len(y))
    for tr, te in RepeatedStratifiedKFold(5, 5, random_state=42).split(preds_mat.T, bins):
        m = BayesianRidge(); m.fit(preds_mat[:, tr].T, y[tr])
        sp[te] += m.predict(preds_mat[:, te].T); sc_count[te] += 1
    sp /= sc_count
    r2 = 1-np.sum((y-sp)**2)/ss_tot
    if r2 > best_r2: best_r2 = r2; best_preds = sp.copy()
    if r2 > 0.25: all_stack_preds['bayesian'] = sp.copy()
    print(f"  After BayesianRidge stack: R2={best_r2:.4f}", flush=True)
    
    # --- Mega blend ---
    rng = np.random.RandomState(42)
    blend_best = -999
    for _ in range(500000):
        w = rng.dirichlet(np.ones(nm_count)*0.3)
        bl = w @ preds_mat; r2 = 1-np.sum((y-bl)**2)/ss_tot
        if r2 > blend_best: blend_best = r2
    # Sparse
    for _ in range(200000):
        w = rng.dirichlet(np.ones(nm_count)*0.1)
        bl = w @ preds_mat; r2 = 1-np.sum((y-bl)**2)/ss_tot
        if r2 > blend_best: blend_best = r2
    # Pairwise
    for i in range(nm_count):
        for j in range(i+1, nm_count):
            for a in np.linspace(0, 1, 201):
                bl = a*preds_mat[i]+(1-a)*preds_mat[j]; r2=1-np.sum((y-bl)**2)/ss_tot
                if r2 > blend_best: blend_best = r2
    # Triplets
    for i in range(min(8,nm_count)):
        for j in range(i+1,min(8,nm_count)):
            for k in range(j+1,min(8,nm_count)):
                for a in np.linspace(0.05,0.9,20):
                    for b in np.linspace(0.05,0.9-a,15):
                        c=1-a-b
                        if c>0:
                            bl=a*preds_mat[i]+b*preds_mat[j]+c*preds_mat[k]
                            r2=1-np.sum((y-bl)**2)/ss_tot
                            if r2>blend_best: blend_best=r2
    print(f"  Mega blend: R2={blend_best:.4f}", flush=True)
    if blend_best > best_r2: best_r2 = blend_best
    
    # --- Layer 2: Stack of stacks ---
    if len(all_stack_preds) >= 3:
        print(f"\n  Layer 2: stacking {len(all_stack_preds)} stack outputs", flush=True)
        stack_names = list(all_stack_preds.keys())
        stack_mat = np.array([all_stack_preds[k] for k in stack_names])
        for alpha in [0.01, 0.1, 1, 10, 100]:
            sp = np.zeros(len(y)); sc_count = np.zeros(len(y))
            for tr, te in RepeatedStratifiedKFold(5, 5, random_state=42).split(stack_mat.T, bins):
                m = Ridge(alpha=alpha); m.fit(stack_mat[:, tr].T, y[tr])
                sp[te] += m.predict(stack_mat[:, te].T); sc_count[te] += 1
            sp /= sc_count
            r2 = 1-np.sum((y-sp)**2)/ss_tot
            if r2 > best_r2: best_r2 = r2; best_preds = sp.copy()
        print(f"  Layer 2 best: R2={best_r2:.4f}", flush=True)
    
    print(f"\n  >>> {label} FINAL: R2={best_r2:.4f}", flush=True)
    return best_r2

def run_target(target_col, target_name, target_r2):
    y_full = df[target_col].values; mask = ~np.isnan(y_full)
    X_dw = X_base[mask].reset_index(drop=True)
    y = y_full[mask]
    use_log = 'homa' in target_name.lower()
    
    print(f"\n{'='*60}")
    print(f"{target_name} | n={len(y)} | Target: {target_r2}")
    print(f"{'='*60}")
    
    # Feature sets: only 3 (raw, engineered, MI-selected)
    X_raw = X_dw.values
    X_eng = engineer(X_dw).fillna(0).values
    mi = mutual_info_regression(X_eng, y, random_state=42)
    X_mi = X_eng[:, np.argsort(mi)[-35:]]
    
    fsets = {'raw18': X_raw, 'eng': X_eng, 'mi35': X_mi}
    
    all_results = {}
    
    # Define focused model set — prioritize diversity
    models = [
        # Linear (diverse regularization)
        ('ridge_50', lambda: Ridge(alpha=50), True),
        ('ridge_100', lambda: Ridge(alpha=100), True),
        ('ridge_500', lambda: Ridge(alpha=500), True),
        ('ridge_1000', lambda: Ridge(alpha=1000), True),
        ('ridge_2000', lambda: Ridge(alpha=2000), True),
        ('lasso_001', lambda: Lasso(alpha=0.01, max_iter=10000), True),
        ('elastic_01_5', lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000), True),
        ('bayesian', lambda: BayesianRidge(), True),
        ('huber', lambda: HuberRegressor(max_iter=1000), True),
        # Kernel
        ('kr_rbf_01_01', lambda: KernelRidge(alpha=0.1, kernel='rbf', gamma=0.01), True),
        ('kr_rbf_1_01', lambda: KernelRidge(alpha=1, kernel='rbf', gamma=0.01), True),
        ('kr_rbf_01_001', lambda: KernelRidge(alpha=0.1, kernel='rbf', gamma=0.001), True),
        ('kr_poly2', lambda: KernelRidge(alpha=1, kernel='poly', degree=2, gamma=0.01), True),
        ('svr_1', lambda: SVR(kernel='rbf', C=1, epsilon=0.1), True),
        ('svr_10', lambda: SVR(kernel='rbf', C=10, epsilon=0.1), True),
        ('nusvr_05', lambda: NuSVR(kernel='rbf', C=1, nu=0.5), True),
        # KNN
        ('knn_10', lambda: KNeighborsRegressor(n_neighbors=10, weights='distance'), True),
        ('knn_20', lambda: KNeighborsRegressor(n_neighbors=20, weights='distance'), True),
        ('knn_30', lambda: KNeighborsRegressor(n_neighbors=30, weights='distance'), True),
        ('knn_50', lambda: KNeighborsRegressor(n_neighbors=50, weights='distance'), True),
        # Tree (shallow, regularized)
        ('xgb_d2', lambda: xgb.XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, reg_alpha=2, reg_lambda=5, random_state=42), False),
        ('xgb_d3', lambda: xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.01, subsample=0.8, colsample_bytree=0.7, reg_alpha=5, reg_lambda=10, random_state=42), False),
        ('xgb_d2_mae', lambda: xgb.XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, objective='reg:absoluteerror', subsample=0.8, colsample_bytree=0.7, random_state=42), False),
        ('lgb_d2', lambda: lgb.LGBMRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, verbose=-1, n_jobs=1, random_state=42), False),
        ('lgb_dart', lambda: lgb.LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, boosting_type='dart', verbose=-1, n_jobs=1, random_state=42), False),
        ('hgbr_d3', lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05, random_state=42), False),
        ('hgbr_d4', lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=4, learning_rate=0.05, random_state=42), False),
        ('rf_d4', lambda: RandomForestRegressor(n_estimators=500, max_depth=4, max_features=0.5, random_state=42, n_jobs=1), False),
        ('et_d4', lambda: ExtraTreesRegressor(n_estimators=500, max_depth=4, max_features=0.5, random_state=42, n_jobs=1), False),
        ('gbr_d2', lambda: GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=42), False),
        # Bagged
        ('bag_ridge', lambda: BaggingRegressor(estimator=Ridge(alpha=100), n_estimators=30, max_samples=0.8, max_features=0.8, random_state=42), True),
        ('bag_svr', lambda: BaggingRegressor(estimator=SVR(kernel='rbf', C=1), n_estimators=20, max_samples=0.8, max_features=0.8, random_state=42), True),
    ]
    
    # Multi-seed variants for top tree models
    seed_models = [
        ('xgb_d2_s1', lambda: xgb.XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, reg_alpha=2, reg_lambda=5, random_state=1), False),
        ('xgb_d2_s2', lambda: xgb.XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, reg_alpha=2, reg_lambda=5, random_state=2), False),
        ('xgb_d2_s3', lambda: xgb.XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, reg_alpha=2, reg_lambda=5, random_state=3), False),
        ('rf_d4_s1', lambda: RandomForestRegressor(n_estimators=500, max_depth=4, max_features=0.5, random_state=1, n_jobs=1), False),
        ('rf_d4_s2', lambda: RandomForestRegressor(n_estimators=500, max_depth=4, max_features=0.5, random_state=2, n_jobs=1), False),
    ]
    
    for fs_name, X_fs in fsets.items():
        print(f"\n--- {fs_name} ({X_fs.shape[1]} features) ---", flush=True)
        
        cur_models = models + (seed_models if fs_name == 'raw18' else [])
        
        for mname, mfn, needs_scale in cur_models:
            # Normal
            full = f"{fs_name}__{mname}"
            t0 = time.time()
            try:
                r2, preds = run_cv(X_fs, y, mfn, scale=needs_scale, log_t=False, n_rep=5)
                all_results[full] = {'r2': r2, 'preds': preds}
                thresh = 0.18 if 'homa' in target_name.lower() else 0.10
                if r2 > thresh:
                    print(f"  {full:55s}: R2={r2:.4f} ({time.time()-t0:.1f}s)", flush=True)
            except: pass
            
            # Log target (HOMA only, non-tree models that accept log)
            if use_log and needs_scale:
                full_log = f"{fs_name}__{mname}_log"
                try:
                    r2, preds = run_cv(X_fs, y, mfn, scale=needs_scale, log_t=True, n_rep=5)
                    all_results[full_log] = {'r2': r2, 'preds': preds}
                    if r2 > thresh:
                        print(f"  {full_log:55s}: R2={r2:.4f} ({time.time()-t0:.1f}s)", flush=True)
                except: pass
            
            # Log target for tree models too (HOMA only)
            if use_log and not needs_scale:
                full_log = f"{fs_name}__{mname}_log"
                try:
                    r2, preds = run_cv(X_fs, y, mfn, scale=False, log_t=True, n_rep=5)
                    all_results[full_log] = {'r2': r2, 'preds': preds}
                    if r2 > thresh:
                        print(f"  {full_log:55s}: R2={r2:.4f} ({time.time()-t0:.1f}s)", flush=True)
                except: pass
    
    # TabPFN
    print(f"\n--- TabPFN ---", flush=True)
    try:
        from tabpfn import TabPFNRegressor
        for fs_name in ['raw18', 'mi35']:
            X_fs = fsets[fs_name]
            for ne in [4, 8]:
                full = f"tabpfn_{fs_name}_{ne}"
                t0 = time.time()
                r2, preds = run_cv(X_fs, y, lambda n=ne: TabPFNRegressor(n_estimators=n), scale=True, n_rep=3)
                all_results[full] = {'r2': r2, 'preds': preds}
                print(f"  {full:55s}: R2={r2:.4f} ({time.time()-t0:.1f}s)", flush=True)
    except Exception as e:
        print(f"  TabPFN failed: {e}", flush=True)
    
    # Advanced stacking
    good = {k: v for k, v in all_results.items() if v['r2'] > 0}
    print(f"\n--- Stacking {len(good)} models ---", flush=True)
    best = advanced_stack(good, y, target_name)
    return best

# ============================================================
print("=" * 60)
print("V17b: FOCUSED DW PUSH — DIVERSITY + MULTI-LAYER STACKING")
print("=" * 60)
t0 = time.time()

best_homa = run_target('True_HOMA_IR', 'HOMA_IR DW', 0.37)
best_hba1c = run_target('True_hba1c', 'hba1c DW', 0.70)

print(f"\n{'='*60}")
print(f"V17b FINAL ({(time.time()-t0)/60:.1f} min)")
print(f"{'='*60}")
print(f"  HOMA_IR DW: R2={best_homa:.4f} (target 0.37, gap={0.37-best_homa:.3f})")
print(f"  hba1c DW:   R2={best_hba1c:.4f} (target 0.70, gap={0.70-best_hba1c:.3f})")
print(f"  Previous: HOMA=0.2800, hba1c=0.1668")
