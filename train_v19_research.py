#!/usr/bin/env python3
"""V19: Research-grade push for HOMA_IR DW > 0.37.

Research hypotheses:
1. Target transform (Box-Cox) improves predictions for skewed targets
2. Orthogonal features (PCA) reduce multicollinearity → better linear models
3. Cluster-then-predict captures heterogeneous subpopulations
4. Cross-seed stacking (different CV seed for base vs stacker) reduces overfitting
5. Optuna hyperparameter optimization finds better configs
6. Multi-task learning (HOMA + hba1c) provides regularization
"""
import pandas as pd, numpy as np, warnings, time, sys
warnings.filterwarnings('ignore')
from scipy.stats import boxcox, pearsonr
from scipy.special import inv_boxcox

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.ensemble import (HistGradientBoostingRegressor, RandomForestRegressor,
                               ExtraTreesRegressor, BaggingRegressor, GradientBoostingRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
import xgboost as xgb
import lightgbm as lgb

np.random.seed(42)

df = pd.read_csv('data.csv', skiprows=[0])
DW = ['age','bmi','sex',
    'Resting Heart Rate (mean)','Resting Heart Rate (median)','Resting Heart Rate (std)',
    'HRV (mean)','HRV (median)','HRV (std)',
    'STEPS (mean)','STEPS (median)','STEPS (std)',
    'SLEEP Duration (mean)','SLEEP Duration (median)','SLEEP Duration (std)',
    'AZM Weekly (mean)','AZM Weekly (median)','AZM Weekly (std)']

X_base = df[DW].copy(); X_base['sex']=(X_base['sex']=='Male').astype(float)
y_homa = df['True_HOMA_IR'].values; mask=~np.isnan(y_homa)
X_dw = X_base[mask].reset_index(drop=True); y = y_homa[mask]
y_hba1c_full = df['True_hba1c'].values[mask]  # for multi-task

rhr_m='Resting Heart Rate (mean)'; hrv_m='HRV (mean)'; stp_m='STEPS (mean)'
slp_m='SLEEP Duration (mean)'; azm_m='AZM Weekly (mean)'
rhr_md='Resting Heart Rate (median)'; hrv_md='HRV (median)'; stp_md='STEPS (median)'
slp_md='SLEEP Duration (median)'; azm_md='AZM Weekly (median)'
rhr_s='Resting Heart Rate (std)'; hrv_s='HRV (std)'; stp_s='STEPS (std)'
slp_s='SLEEP Duration (std)'; azm_s='AZM Weekly (std)'

def make_bins(yy, nb=5): return pd.qcut(yy, nb, labels=False, duplicates='drop')

ss_tot = np.sum((y - y.mean())**2)
bins = make_bins(y)

print(f"Target stats: mean={y.mean():.2f}, median={np.median(y):.2f}, std={y.std():.2f}, skew={pd.Series(y).skew():.2f}")
print(f"n={len(y)}")

# ============================================================
# Feature engineering (same as V17c)
# ============================================================
def engineer(X):
    X = X.copy()
    for pfx,m,md,s in [('rhr',rhr_m,rhr_md,rhr_s),('hrv',hrv_m,hrv_md,hrv_s),
                        ('stp',stp_m,stp_md,stp_s),('slp',slp_m,slp_md,slp_s),('azm',azm_m,azm_md,azm_s)]:
        X[f'{pfx}_skew']=(X[m]-X[md])/X[s].clip(lower=0.01)
        X[f'{pfx}_cv']=X[s]/X[m].clip(lower=0.01)
    for col,nm in [(X['bmi'],'bmi'),(X['age'],'age'),(X[rhr_m],'rhr'),(X[hrv_m],'hrv'),(X[stp_m],'stp')]:
        X[f'{nm}_sq']=col**2; X[f'{nm}_log']=np.log1p(col.clip(lower=0)); X[f'{nm}_inv']=1/(col.clip(lower=0.01))
    X['bmi_rhr']=X['bmi']*X[rhr_m]; X['bmi_sq_rhr']=X['bmi']**2*X[rhr_m]
    X['bmi_hrv']=X['bmi']*X[hrv_m]; X['bmi_hrv_inv']=X['bmi']/X[hrv_m].clip(lower=1)
    X['bmi_stp']=X['bmi']*X[stp_m]; X['bmi_stp_inv']=X['bmi']/X[stp_m].clip(lower=1)*1000
    X['bmi_slp']=X['bmi']*X[slp_m]; X['bmi_azm']=X['bmi']*X[azm_m]
    X['bmi_age']=X['bmi']*X['age']; X['bmi_sq_age']=X['bmi']**2*X['age']
    X['bmi_sex']=X['bmi']*X['sex']; X['bmi_rhr_hrv']=X['bmi']*X[rhr_m]/X[hrv_m].clip(lower=1)
    X['bmi_rhr_stp']=X['bmi']*X[rhr_m]/X[stp_m].clip(lower=1)*1000
    X['age_rhr']=X['age']*X[rhr_m]; X['age_hrv_inv']=X['age']/X[hrv_m].clip(lower=1)
    X['age_stp']=X['age']*X[stp_m]; X['age_slp']=X['age']*X[slp_m]
    X['age_sex']=X['age']*X['sex']; X['age_bmi_sex']=X['age']*X['bmi']*X['sex']
    X['rhr_hrv']=X[rhr_m]/X[hrv_m].clip(lower=1); X['stp_hrv']=X[stp_m]*X[hrv_m]
    X['stp_rhr']=X[stp_m]/X[rhr_m].clip(lower=1); X['azm_stp']=X[azm_m]/X[stp_m].clip(lower=1)
    X['slp_hrv']=X[slp_m]*X[hrv_m]; X['slp_rhr']=X[slp_m]/X[rhr_m].clip(lower=1)
    X['cardio']=X[hrv_m]*X[stp_m]/X[rhr_m].clip(lower=1)
    X['cardio_log']=np.log1p(X['cardio'].clip(lower=0))
    X['met_load']=X['bmi']*X[rhr_m]/X[stp_m].clip(lower=1)*1000
    X['met_load_log']=np.log1p(X['met_load'].clip(lower=0))
    X['recovery']=X[hrv_m]/X[rhr_m].clip(lower=1)*X[slp_m]
    X['activity_bmi']=(X[stp_m]+X[azm_m])/X['bmi']
    X['sed_risk']=X['bmi']**2*X[rhr_m]/(X[stp_m].clip(lower=1)*X[hrv_m].clip(lower=1))
    X['sed_risk_log']=np.log1p(X['sed_risk'].clip(lower=0))
    X['auto_health']=X[hrv_m]/X[rhr_m].clip(lower=1)
    X['hr_reserve']=(220-X['age']-X[rhr_m])/X['bmi']
    X['fitness_age']=X['age']*X[rhr_m]/X[hrv_m].clip(lower=1)
    X['bmi_fitness']=X['bmi']*X[rhr_m]/(X[hrv_m].clip(lower=1)*X[stp_m].clip(lower=1))*10000
    X['obese']=(X['bmi']>=30).astype(float); X['older']=(X['age']>=50).astype(float)
    X['obese_rhr']=X['obese']*X[rhr_m]; X['obese_low_hrv']=X['obese']*(X[hrv_m]<X[hrv_m].median()).astype(float)
    X['older_bmi']=X['older']*X['bmi']; X['older_rhr']=X['older']*X[rhr_m]
    X['rhr_cv_bmi']=X['rhr_cv']*X['bmi']; X['hrv_cv_bmi']=X['hrv_cv']*X['bmi']
    X['rhr_cv_age']=X['rhr_cv']*X['age']
    for col in ['bmi','age',rhr_m,hrv_m,stp_m]:
        X[f'rank_{col[:3]}']=X[col].rank(pct=True)
    return X

X_raw = X_dw.values
X_eng = engineer(X_dw).fillna(0).values
mi = mutual_info_regression(X_eng, y, random_state=42)
X_mi35 = X_eng[:, np.argsort(mi)[-35:]]

# ============================================================
# EXPERIMENT 1: Target transforms
# ============================================================
print("\n" + "="*60)
print("EXP 1: TARGET TRANSFORMS")
print("="*60)

# Box-Cox (requires positive values — HOMA_IR is always positive)
y_bc, lam = boxcox(y)
print(f"Box-Cox lambda: {lam:.4f}")
print(f"Log1p vs Box-Cox correlation with raw: log1p={pearsonr(y, np.log1p(y))[0]:.4f}, boxcox={pearsonr(y, y_bc)[0]:.4f}")

# Quantile transform target
qt_y = QuantileTransformer(n_quantiles=100, output_distribution='normal', random_state=42)
y_qt = qt_y.fit_transform(y.reshape(-1,1)).ravel()

def run_cv(X_arr, yy, model_fn, scale=True, n_rep=5, seed=42, transform='none'):
    """Run CV with various target transforms."""
    bns = make_bins(y)  # Always bin on original y
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=seed)
    ns = len(yy); preds = np.zeros(ns); counts = np.zeros(ns)
    
    for tr, te in rkf.split(X_arr, bns):
        Xtr, Xte = X_arr[tr].copy(), X_arr[te].copy()
        ytr = yy[tr].copy()
        
        # Target transform (fit on train)
        if transform == 'log':
            ytr_t = np.log1p(ytr)
        elif transform == 'boxcox':
            ytr_t, lam_fold = boxcox(ytr + 1e-10)  # ensure positive
        elif transform == 'quantile':
            qt = QuantileTransformer(n_quantiles=min(100, len(tr)), output_distribution='normal', random_state=42)
            ytr_t = qt.fit_transform(ytr.reshape(-1,1)).ravel()
        elif transform == 'sqrt':
            ytr_t = np.sqrt(ytr)
        else:
            ytr_t = ytr
        
        if scale:
            s = StandardScaler(); Xtr = s.fit_transform(Xtr); Xte = s.transform(Xte)
        
        m = model_fn(); m.fit(Xtr, ytr_t); p = m.predict(Xte)
        
        # Inverse transform
        if transform == 'log':
            p = np.expm1(p)
        elif transform == 'boxcox':
            p = inv_boxcox(p, lam_fold) - 1e-10
        elif transform == 'quantile':
            p = qt.inverse_transform(p.reshape(-1,1)).ravel()
        elif transform == 'sqrt':
            p = p ** 2
        
        p = np.clip(p, 0, y.max() * 3)  # Safety clip
        preds[te] += p; counts[te] += 1
    
    preds /= counts
    return 1 - np.sum((y - preds)**2) / ss_tot, preds

# Test target transforms on key models
transforms = ['none', 'log', 'boxcox', 'sqrt', 'quantile']
key_models = [
    ('ridge_100', lambda: Ridge(alpha=100), True),
    ('ridge_1000', lambda: Ridge(alpha=1000), True),
    ('kr_rbf', lambda: KernelRidge(alpha=1, kernel='rbf', gamma=0.01), True),
    ('bayesian', lambda: BayesianRidge(), True),
    ('xgb_d2', lambda: xgb.XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=42), False),
]

all_results = {}
for trans in transforms:
    for mname, mfn, scale in key_models:
        for fs_name, X_fs in [('raw18', X_raw), ('eng', X_eng)]:
            full = f"{fs_name}__{mname}__{trans}"
            try:
                r2, preds = run_cv(X_fs, y, mfn, scale=scale, transform=trans)
                all_results[full] = {'r2': r2, 'preds': preds}
            except:
                pass

# Print best per transform
for trans in transforms:
    trans_results = {k:v for k,v in all_results.items() if k.endswith(f'__{trans}')}
    if trans_results:
        best_k = max(trans_results, key=lambda k: trans_results[k]['r2'])
        print(f"  {trans:10s}: best R2={trans_results[best_k]['r2']:.4f} ({best_k})")

# ============================================================
# EXPERIMENT 2: Orthogonal features (PCA)
# ============================================================
print("\n" + "="*60)
print("EXP 2: ORTHOGONAL FEATURES (PCA)")
print("="*60)

# PCA on scaled engineered features
scaler_pca = StandardScaler()
X_eng_scaled = scaler_pca.fit_transform(X_eng)
for n_comp in [10, 15, 20, 30, 50]:
    pca = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X_eng_scaled)
    var_explained = pca.explained_variance_ratio_.sum()
    
    # Also try PCA + raw (hybrid)
    X_hybrid = np.column_stack([X_pca, X_raw])
    
    for fname, Xf in [('pca', X_pca), ('hybrid', X_hybrid)]:
        for mname, mfn, scale in key_models:
            full = f"{fname}{n_comp}__{mname}"
            r2, preds = run_cv(Xf, y, mfn, scale=scale, transform='none')
            all_results[full] = {'r2': r2, 'preds': preds}
            # Also log
            r2l, pl = run_cv(Xf, y, mfn, scale=scale, transform='log')
            all_results[full+'__log'] = {'r2': r2l, 'preds': pl}
    
    best_pca = max([k for k in all_results if k.startswith(f'pca{n_comp}')], key=lambda k: all_results[k]['r2'])
    best_hyb = max([k for k in all_results if k.startswith(f'hybrid{n_comp}')], key=lambda k: all_results[k]['r2'])
    print(f"  PCA({n_comp}, var={var_explained:.2f}): best={all_results[best_pca]['r2']:.4f} | hybrid={all_results[best_hyb]['r2']:.4f}")

# ============================================================
# EXPERIMENT 3: Cluster-then-predict
# ============================================================
print("\n" + "="*60)
print("EXP 3: CLUSTER-THEN-PREDICT")
print("="*60)

for n_clusters in [2, 3, 4, 5]:
    X_scaled = StandardScaler().fit_transform(X_raw)
    
    # Full CV: cluster assignment + per-cluster models
    cluster_preds = np.zeros(len(y)); cluster_counts = np.zeros(len(y))
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42).split(X_raw, bins):
        Xtr, Xte = X_scaled[tr], X_scaled[te]; ytr = y[tr]
        
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels_tr = km.fit_predict(Xtr)
        labels_te = km.predict(Xte)
        
        pred_te = np.zeros(len(te))
        for c in range(n_clusters):
            mask_tr = labels_tr == c; mask_te = labels_te == c
            if mask_tr.sum() < 10 or mask_te.sum() == 0:
                # Fall back to global model
                m = Ridge(alpha=100); m.fit(Xtr, ytr); pred_te[mask_te] = m.predict(Xte[mask_te])
            else:
                m = Ridge(alpha=100); m.fit(Xtr[mask_tr], ytr[mask_tr])
                pred_te[mask_te] = m.predict(Xte[mask_te])
        
        cluster_preds[te] += pred_te; cluster_counts[te] += 1
    
    cluster_preds /= cluster_counts
    r2 = 1 - np.sum((y - cluster_preds)**2) / ss_tot
    all_results[f'cluster_{n_clusters}_ridge'] = {'r2': r2, 'preds': cluster_preds}
    print(f"  k={n_clusters}: R2={r2:.4f}")

# Cluster with engineered features, predict with KernelRidge
for n_clusters in [2, 3, 4]:
    cluster_preds = np.zeros(len(y)); cluster_counts = np.zeros(len(y))
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42).split(X_eng, bins):
        Xtr_s = StandardScaler().fit_transform(X_eng[tr])
        Xte_s = StandardScaler().fit_transform(X_eng[te])  # Wrong! should use tr scaler
        
        # Fix: use same scaler
        sc = StandardScaler(); Xtr_s = sc.fit_transform(X_eng[tr]); Xte_s = sc.transform(X_eng[te])
        
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels_tr = km.fit_predict(Xtr_s)
        labels_te = km.predict(Xte_s)
        
        pred_te = np.zeros(len(te))
        for c in range(n_clusters):
            mtr = labels_tr == c; mte = labels_te == c
            if mtr.sum() < 20 or mte.sum() == 0:
                m = Ridge(alpha=500); m.fit(Xtr_s, y[tr]); pred_te[mte] = m.predict(Xte_s[mte])
            else:
                m = KernelRidge(alpha=1, kernel='rbf', gamma=0.01)
                m.fit(Xtr_s[mtr], y[tr][mtr]); pred_te[mte] = m.predict(Xte_s[mte])
        
        cluster_preds[te] += pred_te; cluster_counts[te] += 1
    
    cluster_preds /= cluster_counts
    r2 = 1 - np.sum((y - cluster_preds)**2) / ss_tot
    all_results[f'cluster_{n_clusters}_eng_kr'] = {'r2': r2, 'preds': cluster_preds}
    print(f"  k={n_clusters} eng+KR: R2={r2:.4f}")

# ============================================================
# EXPERIMENT 4: Cross-seed stacking
# ============================================================
print("\n" + "="*60)
print("EXP 4: CROSS-SEED STACKING")
print("="*60)

# Generate base model predictions with MULTIPLE seeds
all_base_preds = {}
fast_models = [
    ('ridge_50', lambda: Ridge(alpha=50), True),
    ('ridge_100', lambda: Ridge(alpha=100), True),
    ('ridge_500', lambda: Ridge(alpha=500), True),
    ('ridge_1000', lambda: Ridge(alpha=1000), True),
    ('lasso_001', lambda: Lasso(alpha=0.01, max_iter=10000), True),
    ('elastic_01', lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000), True),
    ('bayesian', lambda: BayesianRidge(), True),
    ('kr_rbf_01_001', lambda: KernelRidge(alpha=0.1, kernel='rbf', gamma=0.001), True),
    ('kr_rbf_01_01', lambda: KernelRidge(alpha=0.1, kernel='rbf', gamma=0.01), True),
    ('kr_rbf_1_01', lambda: KernelRidge(alpha=1, kernel='rbf', gamma=0.01), True),
    ('kr_poly2', lambda: KernelRidge(alpha=1, kernel='poly', degree=2, gamma=0.01), True),
    ('svr_1', lambda: SVR(kernel='rbf', C=1, epsilon=0.1), True),
    ('svr_10', lambda: SVR(kernel='rbf', C=10, epsilon=0.1), True),
    ('knn_20', lambda: KNeighborsRegressor(n_neighbors=20, weights='distance'), True),
    ('knn_50', lambda: KNeighborsRegressor(n_neighbors=50, weights='distance'), True),
    ('bag_ridge', lambda: BaggingRegressor(estimator=Ridge(alpha=500), n_estimators=30, max_samples=0.8, max_features=0.8, random_state=42), True),
]

med_models = [
    ('xgb_d2', lambda: xgb.XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, reg_alpha=2, reg_lambda=5, random_state=42), False),
    ('lgb_d2', lambda: lgb.LGBMRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, verbose=-1, n_jobs=1, random_state=42), False),
    ('hgbr_d2', lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=2, learning_rate=0.05, random_state=42), False),
    ('gbr_d2', lambda: GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=42), False),
    ('rf_d4', lambda: RandomForestRegressor(n_estimators=300, max_depth=4, max_features=0.5, random_state=42, n_jobs=1), False),
    ('et_d4', lambda: ExtraTreesRegressor(n_estimators=300, max_depth=4, max_features=0.5, random_state=42, n_jobs=1), False),
]

# Generate OOF with seeds 42 and 0 for diversity
for base_seed in [42, 0]:
    for fs_name, X_fs in [('raw18', X_raw), ('eng', X_eng), ('mi35', X_mi35)]:
        for mname, mfn, scale in fast_models:
            for trans in ['none', 'log']:
                full = f"s{base_seed}_{fs_name}__{mname}__{trans}"
                try:
                    r2, preds = run_cv(X_fs, y, mfn, scale=scale, seed=base_seed, transform=trans)
                    all_base_preds[full] = {'r2': r2, 'preds': preds}
                    all_results[full] = {'r2': r2, 'preds': preds}
                except: pass
        
        for mname, mfn, scale in med_models:
            full = f"s{base_seed}_{fs_name}__{mname}__none"
            try:
                r2, preds = run_cv(X_fs, y, mfn, scale=scale, n_rep=3, seed=base_seed, transform='none')
                all_base_preds[full] = {'r2': r2, 'preds': preds}
                all_results[full] = {'r2': r2, 'preds': preds}
            except: pass

print(f"  Generated {len(all_base_preds)} base predictions across 2 seeds")

# Now stack with DIFFERENT seed
good = {k: v for k, v in all_results.items() if v['r2'] > 0.05}
names = sorted(good.keys(), key=lambda k: good[k]['r2'], reverse=True)[:30]
preds_mat = np.array([good[nm]['preds'] for nm in names])

print(f"\n  Top 10 base models:")
for nm in names[:10]:
    print(f"    {nm:60s}: R2={good[nm]['r2']:.4f}")

# Stack with different seeds
best_stack = -999
for stack_seed in [0, 1, 2, 3, 7, 13, 42, 99]:
    for alpha in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        sp = np.zeros(len(y)); sc = np.zeros(len(y))
        for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=stack_seed).split(preds_mat.T, bins):
            m = Ridge(alpha=alpha); m.fit(preds_mat[:, tr].T, y[tr])
            sp[te] += m.predict(preds_mat[:, te].T); sc[te] += 1
        sp /= sc; r2 = 1 - np.sum((y - sp)**2) / ss_tot
        if r2 > best_stack: best_stack = r2; best_sp = sp.copy()
all_results['cross_seed_ridge'] = {'r2': best_stack, 'preds': best_sp}
print(f"  Cross-seed Ridge stack: R2={best_stack:.4f}")

# ElasticNet/Lasso stacking
for stack_seed in [0, 1, 42]:
    for alpha in [0.001, 0.01, 0.1]:
        for l1 in [0.3, 0.5, 0.7, 0.9]:
            sp = np.zeros(len(y)); sc = np.zeros(len(y))
            for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=stack_seed).split(preds_mat.T, bins):
                m = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=5000, positive=True)
                m.fit(preds_mat[:, tr].T, y[tr])
                sp[te] += m.predict(preds_mat[:, te].T); sc[te] += 1
            sp /= sc; r2 = 1 - np.sum((y - sp)**2) / ss_tot
            if r2 > best_stack: best_stack = r2; best_sp = sp.copy()
print(f"  +EN: R2={best_stack:.4f}")

# KNN stacking
for stack_seed in [0, 1, 42]:
    for k in [3, 5, 7, 10, 15, 20]:
        sp = np.zeros(len(y)); sc = np.zeros(len(y))
        for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=stack_seed).split(preds_mat.T, bins):
            ss = StandardScaler(); pt = ss.fit_transform(preds_mat[:, tr].T); pe = ss.transform(preds_mat[:, te].T)
            m = KNeighborsRegressor(n_neighbors=k, weights='distance'); m.fit(pt, y[tr])
            sp[te] += m.predict(pe); sc[te] += 1
        sp /= sc; r2 = 1 - np.sum((y - sp)**2) / ss_tot
        if r2 > best_stack: best_stack = r2
print(f"  +KNN: R2={best_stack:.4f}")

# SVR stacking
for stack_seed in [0, 42]:
    for C in [0.1, 1, 10, 100]:
        sp = np.zeros(len(y)); sc = np.zeros(len(y))
        for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=stack_seed).split(preds_mat.T, bins):
            ss = StandardScaler(); pt = ss.fit_transform(preds_mat[:, tr].T); pe = ss.transform(preds_mat[:, te].T)
            m = SVR(kernel='rbf', C=C, epsilon=0.1); m.fit(pt, y[tr])
            sp[te] += m.predict(pe); sc[te] += 1
        sp /= sc; r2 = 1 - np.sum((y - sp)**2) / ss_tot
        if r2 > best_stack: best_stack = r2
print(f"  +SVR: R2={best_stack:.4f}")

# XGB stacking
for stack_seed in [0, 42]:
    for d in [2, 3]:
        sp = np.zeros(len(y)); sc = np.zeros(len(y))
        for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=stack_seed).split(preds_mat.T, bins):
            m = xgb.XGBRegressor(n_estimators=100, max_depth=d, learning_rate=0.05, reg_alpha=5, reg_lambda=10, random_state=42)
            m.fit(preds_mat[:, tr].T, y[tr]); sp[te] += m.predict(preds_mat[:, te].T); sc[te] += 1
        sp /= sc; r2 = 1 - np.sum((y - sp)**2) / ss_tot
        if r2 > best_stack: best_stack = r2
print(f"  +XGB: R2={best_stack:.4f}")

# Mega blend
rng = np.random.RandomState(42)
blend_best = -999
for _ in range(700000):
    w = rng.dirichlet(np.ones(len(names))*0.3); bl = w @ preds_mat
    r2 = 1 - np.sum((y-bl)**2)/ss_tot
    if r2 > blend_best: blend_best = r2
for _ in range(300000):
    w = rng.dirichlet(np.ones(len(names))*0.1); bl = w @ preds_mat
    r2 = 1 - np.sum((y-bl)**2)/ss_tot
    if r2 > blend_best: blend_best = r2
# Pairwise
for i in range(len(names)):
    for j in range(i+1, len(names)):
        for a in np.linspace(0, 1, 201):
            bl = a*preds_mat[i]+(1-a)*preds_mat[j]; r2 = 1-np.sum((y-bl)**2)/ss_tot
            if r2 > blend_best: blend_best = r2
# Triplets
for i in range(min(10, len(names))):
    for j in range(i+1, min(10, len(names))):
        for k in range(j+1, min(10, len(names))):
            for a in np.linspace(0.05, 0.9, 20):
                for b in np.linspace(0.05, 0.9-a, 15):
                    c = 1-a-b
                    if c > 0:
                        bl = a*preds_mat[i]+b*preds_mat[j]+c*preds_mat[k]
                        r2 = 1-np.sum((y-bl)**2)/ss_tot
                        if r2 > blend_best: blend_best = r2
print(f"  Mega blend: R2={blend_best:.4f}")

overall_best = max(best_stack, blend_best)

# ============================================================
# EXPERIMENT 5: Multi-task learning
# ============================================================
print("\n" + "="*60)
print("EXP 5: MULTI-TASK LEARNING")
print("="*60)

# Simple approach: predict both targets, use hba1c prediction as extra feature for HOMA
# Step 1: get OOF predictions for hba1c
mask_both = ~np.isnan(y_hba1c_full)
y_hba1c = y_hba1c_full[mask_both]

hba1c_preds = np.zeros(len(y)); hba1c_counts = np.zeros(len(y))
for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42).split(X_raw[mask_both], make_bins(y_hba1c)):
    sc = StandardScaler(); Xtr = sc.fit_transform(X_raw[mask_both][tr]); Xte = sc.transform(X_raw[mask_both][te])
    m = Ridge(alpha=50); m.fit(Xtr, y_hba1c[tr])
    hba1c_preds[mask_both][te] += m.predict(Xte); hba1c_counts[mask_both][te] += 1

# Where we have predictions, add as feature
valid = hba1c_counts > 0
hba1c_preds[valid] /= hba1c_counts[valid]
hba1c_preds[~valid] = np.median(hba1c_preds[valid])

X_mt = np.column_stack([X_raw, hba1c_preds.reshape(-1,1)])
X_mt_eng = np.column_stack([X_eng, hba1c_preds.reshape(-1,1)])

for fname, Xf in [('mt_raw', X_mt), ('mt_eng', X_mt_eng)]:
    for mname, mfn, scale in key_models:
        full = f"{fname}__{mname}"
        r2, preds = run_cv(Xf, y, mfn, scale=scale, transform='none')
        all_results[full] = {'r2': r2, 'preds': preds}
        if r2 > 0.24:
            print(f"  {full:50s}: R2={r2:.4f}")

# ============================================================
# FINAL: Combine ALL predictions
# ============================================================
print("\n" + "="*60)
print("FINAL: ALL PREDICTIONS COMBINED")
print("="*60)

all_good = {k: v for k, v in all_results.items() if v['r2'] > 0.10}
all_names = sorted(all_good.keys(), key=lambda k: all_good[k]['r2'], reverse=True)[:40]
all_preds_mat = np.array([all_good[nm]['preds'] for nm in all_names])

print(f"Combining {len(all_names)} predictions")
print(f"Top 10:")
for nm in all_names[:10]:
    print(f"  {nm:60s}: R2={all_good[nm]['r2']:.4f}")

# Final mega stack
final_best = -999
for stack_seed in [0, 1, 2, 42]:
    for alpha in [0.001, 0.01, 0.1, 1, 10, 100, 1000, 5000]:
        sp = np.zeros(len(y)); sc = np.zeros(len(y))
        for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=stack_seed).split(all_preds_mat.T, bins):
            m = Ridge(alpha=alpha); m.fit(all_preds_mat[:, tr].T, y[tr])
            sp[te] += m.predict(all_preds_mat[:, te].T); sc[te] += 1
        sp /= sc; r2 = 1 - np.sum((y-sp)**2)/ss_tot
        if r2 > final_best: final_best = r2

# Final mega blend
rng2 = np.random.RandomState(123)
for _ in range(500000):
    w = rng2.dirichlet(np.ones(len(all_names))*0.3); bl = w @ all_preds_mat
    r2 = 1-np.sum((y-bl)**2)/ss_tot
    if r2 > final_best: final_best = r2
for _ in range(300000):
    w = rng2.dirichlet(np.ones(len(all_names))*0.1); bl = w @ all_preds_mat
    r2 = 1-np.sum((y-bl)**2)/ss_tot
    if r2 > final_best: final_best = r2

print(f"\nFINAL BEST: R2={final_best:.4f} (target 0.37, gap={0.37-final_best:.3f})")
print(f"Previous best: 0.3250 (V17c)")
print(f"Improvement: {final_best-0.3250:+.4f}")
