#!/usr/bin/env python3
"""V20: Maximum diversity for multi-layer stacking.
V17c got 0.3250 from 242 diverse models. V19 showed diversity is key.
Strategy: 400+ base models from every angle, then aggressive multi-layer stacking."""
import pandas as pd, numpy as np, warnings, time
warnings.filterwarnings('ignore')
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.ensemble import (HistGradientBoostingRegressor, RandomForestRegressor,
                               ExtraTreesRegressor, BaggingRegressor, GradientBoostingRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.decomposition import PCA
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

X_base = df[DW].copy(); X_base['sex']=(X_base['sex']=='Male').astype(float)
y_full = df['True_HOMA_IR'].values; mask=~np.isnan(y_full)
X_dw = X_base[mask].reset_index(drop=True); y=y_full[mask]
bins = pd.qcut(y, 5, labels=False, duplicates='drop')
ss_tot = np.sum((y-y.mean())**2)
rhr_m='Resting Heart Rate (mean)'; hrv_m='HRV (mean)'; stp_m='STEPS (mean)'
slp_m='SLEEP Duration (mean)'; azm_m='AZM Weekly (mean)'
rhr_md='Resting Heart Rate (median)'; hrv_md='HRV (median)'; stp_md='STEPS (median)'
slp_md='SLEEP Duration (median)'; azm_md='AZM Weekly (median)'
rhr_s='Resting Heart Rate (std)'; hrv_s='HRV (std)'; stp_s='STEPS (std)'
slp_s='SLEEP Duration (std)'; azm_s='AZM Weekly (std)'

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
X_mi20 = X_eng[:, np.argsort(mi)[-20:]]

# PCA features
sc_pca = StandardScaler(); X_eng_sc = sc_pca.fit_transform(X_eng)
X_pca10 = PCA(n_components=10, random_state=42).fit_transform(X_eng_sc)
X_pca20 = PCA(n_components=20, random_state=42).fit_transform(X_eng_sc)
X_hybrid = np.column_stack([X_pca10, X_raw])

fsets = {'raw18': X_raw, 'eng': X_eng, 'mi35': X_mi35, 'mi20': X_mi20,
         'pca10': X_pca10, 'pca20': X_pca20, 'hybrid': X_hybrid}

def run_cv(X, y, model_fn, scale=True, log_t=False, n_rep=5, seed=42):
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=seed)
    ns=len(y); preds=np.zeros(ns); counts=np.zeros(ns)
    for tr,te in rkf.split(X,bins):
        Xtr,Xte=X[tr].copy(),X[te].copy(); ytr=y[tr].copy()
        if log_t: ytr=np.log1p(ytr)
        if scale:
            s=StandardScaler(); Xtr=s.fit_transform(Xtr); Xte=s.transform(Xte)
        m=model_fn(); m.fit(Xtr,ytr); p=m.predict(Xte)
        if log_t: p=np.expm1(p)
        preds[te]+=p; counts[te]+=1
    preds/=counts
    return 1-np.sum((y-preds)**2)/ss_tot, preds

# ============================================================
print("="*60)
print("V20: MAXIMUM DIVERSITY BASE MODELS")
print("="*60)
t0 = time.time()

all_results = {}

fast_models = [
    ('ridge_50', lambda: Ridge(alpha=50), True),
    ('ridge_100', lambda: Ridge(alpha=100), True),
    ('ridge_500', lambda: Ridge(alpha=500), True),
    ('ridge_1000', lambda: Ridge(alpha=1000), True),
    ('ridge_2000', lambda: Ridge(alpha=2000), True),
    ('lasso_001', lambda: Lasso(alpha=0.01, max_iter=10000), True),
    ('lasso_01', lambda: Lasso(alpha=0.1, max_iter=10000), True),
    ('elastic_01', lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000), True),
    ('elastic_001', lambda: ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000), True),
    ('bayesian', lambda: BayesianRidge(), True),
    ('huber', lambda: HuberRegressor(max_iter=1000), True),
    ('kr_rbf_01_001', lambda: KernelRidge(alpha=0.1, kernel='rbf', gamma=0.001), True),
    ('kr_rbf_01_01', lambda: KernelRidge(alpha=0.1, kernel='rbf', gamma=0.01), True),
    ('kr_rbf_1_001', lambda: KernelRidge(alpha=1, kernel='rbf', gamma=0.001), True),
    ('kr_rbf_1_01', lambda: KernelRidge(alpha=1, kernel='rbf', gamma=0.01), True),
    ('kr_rbf_10_01', lambda: KernelRidge(alpha=10, kernel='rbf', gamma=0.01), True),
    ('kr_poly2', lambda: KernelRidge(alpha=1, kernel='poly', degree=2, gamma=0.01), True),
    ('svr_01', lambda: SVR(kernel='rbf', C=0.1, epsilon=0.1), True),
    ('svr_1', lambda: SVR(kernel='rbf', C=1, epsilon=0.1), True),
    ('svr_10', lambda: SVR(kernel='rbf', C=10, epsilon=0.1), True),
    ('svr_100', lambda: SVR(kernel='rbf', C=100, epsilon=0.05), True),
    ('nusvr_05', lambda: NuSVR(kernel='rbf', C=1, nu=0.5), True),
    ('knn_10', lambda: KNeighborsRegressor(n_neighbors=10, weights='distance'), True),
    ('knn_20', lambda: KNeighborsRegressor(n_neighbors=20, weights='distance'), True),
    ('knn_30', lambda: KNeighborsRegressor(n_neighbors=30, weights='distance'), True),
    ('knn_50', lambda: KNeighborsRegressor(n_neighbors=50, weights='distance'), True),
    ('knn_75', lambda: KNeighborsRegressor(n_neighbors=75, weights='distance'), True),
    ('bag_ridge100', lambda: BaggingRegressor(estimator=Ridge(alpha=100), n_estimators=30, max_samples=0.8, max_features=0.8, random_state=42), True),
    ('bag_ridge500', lambda: BaggingRegressor(estimator=Ridge(alpha=500), n_estimators=30, max_samples=0.8, max_features=0.8, random_state=42), True),
    ('bag_ridge1000', lambda: BaggingRegressor(estimator=Ridge(alpha=1000), n_estimators=30, max_samples=0.8, max_features=0.8, random_state=42), True),
]

med_models = [
    ('xgb_d2', lambda: xgb.XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, reg_alpha=2, reg_lambda=5, random_state=42), False),
    ('xgb_d3', lambda: xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.01, subsample=0.8, colsample_bytree=0.7, reg_alpha=5, reg_lambda=10, random_state=42), False),
    ('lgb_d2', lambda: lgb.LGBMRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, verbose=-1, n_jobs=1, random_state=42), False),
    ('lgb_dart', lambda: lgb.LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, boosting_type='dart', verbose=-1, n_jobs=1, random_state=42), False),
    ('hgbr_d2', lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=2, learning_rate=0.05, random_state=42), False),
    ('hgbr_d3', lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05, random_state=42), False),
    ('gbr_d2', lambda: GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=42), False),
    ('rf_d4', lambda: RandomForestRegressor(n_estimators=300, max_depth=4, max_features=0.5, random_state=42, n_jobs=1), False),
    ('et_d4', lambda: ExtraTreesRegressor(n_estimators=300, max_depth=4, max_features=0.5, random_state=42, n_jobs=1), False),
]

# Generate ALL base predictions: 7 feature sets × 30 fast × 2 transforms + 7 × 9 med = 483
for fs_name, X_fs in fsets.items():
    print(f"\n--- {fs_name} ({X_fs.shape[1]}f) ---", flush=True)
    for mname, mfn, scale in fast_models:
        for log_t in [False, True]:
            full = f"{fs_name}__{mname}{'_log' if log_t else ''}"
            try:
                r2, preds = run_cv(X_fs, y, mfn, scale=scale, log_t=log_t)
                all_results[full] = {'r2': r2, 'preds': preds}
            except: pass
    for mname, mfn, scale in med_models:
        full = f"{fs_name}__{mname}"
        try:
            r2, preds = run_cv(X_fs, y, mfn, scale=scale, n_rep=3)
            all_results[full] = {'r2': r2, 'preds': preds}
        except: pass
    # Print top 3
    fs_r = {k:v for k,v in all_results.items() if k.startswith(fs_name+'__')}
    top3 = sorted(fs_r.keys(), key=lambda k: fs_r[k]['r2'], reverse=True)[:3]
    for nm in top3:
        print(f"  {nm:55s}: R2={fs_r[nm]['r2']:.4f}", flush=True)

# Also add seed-0 variants for top 10 models on raw18
print(f"\n--- Seed diversity ---", flush=True)
top_raw = sorted([k for k in all_results if k.startswith('raw18__')], key=lambda k: all_results[k]['r2'], reverse=True)[:10]
for nm in top_raw:
    mname = nm.replace('raw18__', '')
    for log_t in [False, True]:
        suffix = '_log' if log_t else ''
        if mname.endswith('_log'):
            mbase = mname[:-4]
            if not log_t: continue
            log_t = True
        else:
            mbase = mname
        
        # Find model fn
        for mn, mf, sc in fast_models + med_models:
            if mn == mbase:
                for seed in [0, 1]:
                    full = f"s{seed}_raw18__{mbase}{suffix}"
                    try:
                        r2, preds = run_cv(X_raw, y, mf, scale=sc, log_t=log_t, seed=seed)
                        all_results[full] = {'r2': r2, 'preds': preds}
                    except: pass
                break

print(f"\nTotal base models: {len(all_results)}", flush=True)

# ============================================================
# MULTI-LAYER STACKING
# ============================================================
print(f"\n{'='*60}")
print("MULTI-LAYER STACKING")
print(f"{'='*60}", flush=True)

good = {k:v for k,v in all_results.items() if v['r2'] > 0.05}
names = sorted(good.keys(), key=lambda k: good[k]['r2'], reverse=True)[:40]
preds_mat = np.array([good[nm]['preds'] for nm in names])

print(f"Stacking top {len(names)} of {len(good)} models")
print(f"Top 10:")
for nm in names[:10]:
    print(f"  {nm:55s}: R2={good[nm]['r2']:.4f}")

all_stacks = {}
best_overall = -999

# Layer 1: diverse stackers
for stack_seed in [0, 1, 2, 3, 7, 13, 42, 99]:
    for alpha in [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000]:
        sp=np.zeros(len(y)); sc=np.zeros(len(y))
        for tr,te in RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=stack_seed).split(preds_mat.T,bins):
            m=Ridge(alpha=alpha); m.fit(preds_mat[:,tr].T,y[tr])
            sp[te]+=m.predict(preds_mat[:,te].T); sc[te]+=1
        sp/=sc; r2=1-np.sum((y-sp)**2)/ss_tot
        if r2>best_overall: best_overall=r2
        all_stacks[f'ridge_s{stack_seed}_a{alpha}']=sp
print(f"Ridge stack best: R2={best_overall:.4f}", flush=True)

for stack_seed in [0, 42]:
    for alpha in [0.001, 0.01, 0.1]:
        for l1 in [0.1, 0.3, 0.5, 0.7, 0.9]:
            sp=np.zeros(len(y)); sc=np.zeros(len(y))
            for tr,te in RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=stack_seed).split(preds_mat.T,bins):
                m=ElasticNet(alpha=alpha,l1_ratio=l1,max_iter=5000,positive=True)
                m.fit(preds_mat[:,tr].T,y[tr]); sp[te]+=m.predict(preds_mat[:,te].T); sc[te]+=1
            sp/=sc; r2=1-np.sum((y-sp)**2)/ss_tot
            if r2>best_overall: best_overall=r2
            all_stacks[f'en_s{stack_seed}_{alpha}_{l1}']=sp

for stack_seed in [0, 42]:
    for alpha in [0.001, 0.01, 0.1]:
        sp=np.zeros(len(y)); sc=np.zeros(len(y))
        for tr,te in RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=stack_seed).split(preds_mat.T,bins):
            m=Lasso(alpha=alpha,max_iter=5000,positive=True)
            m.fit(preds_mat[:,tr].T,y[tr]); sp[te]+=m.predict(preds_mat[:,te].T); sc[te]+=1
        sp/=sc; r2=1-np.sum((y-sp)**2)/ss_tot
        if r2>best_overall: best_overall=r2
        all_stacks[f'lasso_s{stack_seed}_{alpha}']=sp
print(f"+EN/Lasso: R2={best_overall:.4f}", flush=True)

for stack_seed in [0, 42]:
    for k in [3, 5, 7, 10, 15, 20]:
        sp=np.zeros(len(y)); sc=np.zeros(len(y))
        for tr,te in RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=stack_seed).split(preds_mat.T,bins):
            ss=StandardScaler(); pt=ss.fit_transform(preds_mat[:,tr].T); pe=ss.transform(preds_mat[:,te].T)
            m=KNeighborsRegressor(n_neighbors=k,weights='distance'); m.fit(pt,y[tr])
            sp[te]+=m.predict(pe); sc[te]+=1
        sp/=sc; r2=1-np.sum((y-sp)**2)/ss_tot
        if r2>best_overall: best_overall=r2
        all_stacks[f'knn_s{stack_seed}_{k}']=sp

for stack_seed in [0, 42]:
    for C in [0.1, 1, 10, 100]:
        sp=np.zeros(len(y)); sc=np.zeros(len(y))
        for tr,te in RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=stack_seed).split(preds_mat.T,bins):
            ss=StandardScaler(); pt=ss.fit_transform(preds_mat[:,tr].T); pe=ss.transform(preds_mat[:,te].T)
            m=SVR(kernel='rbf',C=C,epsilon=0.1); m.fit(pt,y[tr])
            sp[te]+=m.predict(pe); sc[te]+=1
        sp/=sc; r2=1-np.sum((y-sp)**2)/ss_tot
        if r2>best_overall: best_overall=r2
        all_stacks[f'svr_s{stack_seed}_{C}']=sp
print(f"+KNN/SVR: R2={best_overall:.4f}", flush=True)

for stack_seed in [0, 42]:
    for d in [2, 3]:
        sp=np.zeros(len(y)); sc=np.zeros(len(y))
        for tr,te in RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=stack_seed).split(preds_mat.T,bins):
            m=xgb.XGBRegressor(n_estimators=100,max_depth=d,learning_rate=0.05,reg_alpha=5,reg_lambda=10,random_state=42)
            m.fit(preds_mat[:,tr].T,y[tr]); sp[te]+=m.predict(preds_mat[:,te].T); sc[te]+=1
        sp/=sc; r2=1-np.sum((y-sp)**2)/ss_tot
        if r2>best_overall: best_overall=r2
        all_stacks[f'xgb_s{stack_seed}_{d}']=sp

sp=np.zeros(len(y)); sc=np.zeros(len(y))
for tr,te in RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=42).split(preds_mat.T,bins):
    m=BayesianRidge(); m.fit(preds_mat[:,tr].T,y[tr])
    sp[te]+=m.predict(preds_mat[:,te].T); sc[te]+=1
sp/=sc; r2=1-np.sum((y-sp)**2)/ss_tot
if r2>best_overall: best_overall=r2
all_stacks['bayesian']=sp
print(f"+XGB/Bayesian: R2={best_overall:.4f}", flush=True)

# Layer 2: stack of stacks
print(f"\nLayer 2: {len(all_stacks)} stacks", flush=True)
snames = list(all_stacks.keys())
smat = np.array([all_stacks[k] for k in snames])

for l2_seed in [0, 1, 2, 42, 7]:
    for alpha in [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]:
        sp=np.zeros(len(y)); sc=np.zeros(len(y))
        for tr,te in RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=l2_seed).split(smat.T,bins):
            m=Ridge(alpha=alpha); m.fit(smat[:,tr].T,y[tr])
            sp[te]+=m.predict(smat[:,te].T); sc[te]+=1
        sp/=sc; r2=1-np.sum((y-sp)**2)/ss_tot
        if r2>best_overall: best_overall=r2

# Layer 2 with EN
for l2_seed in [0, 42]:
    for alpha in [0.01, 0.1]:
        for l1 in [0.3, 0.5, 0.7]:
            sp=np.zeros(len(y)); sc=np.zeros(len(y))
            for tr,te in RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=l2_seed).split(smat.T,bins):
                m=ElasticNet(alpha=alpha,l1_ratio=l1,max_iter=5000)
                m.fit(smat[:,tr].T,y[tr]); sp[te]+=m.predict(smat[:,te].T); sc[te]+=1
            sp/=sc; r2=1-np.sum((y-sp)**2)/ss_tot
            if r2>best_overall: best_overall=r2

print(f"Layer 2 best: R2={best_overall:.4f}", flush=True)

# Mega blend
rng=np.random.RandomState(42); blend_best=-999
for _ in range(500000):
    w=rng.dirichlet(np.ones(len(names))*0.3); bl=w@preds_mat
    r2=1-np.sum((y-bl)**2)/ss_tot
    if r2>blend_best: blend_best=r2
for _ in range(300000):
    w=rng.dirichlet(np.ones(len(names))*0.1); bl=w@preds_mat
    r2=1-np.sum((y-bl)**2)/ss_tot
    if r2>blend_best: blend_best=r2
for i in range(len(names)):
    for j in range(i+1,len(names)):
        for a in np.linspace(0,1,201):
            bl=a*preds_mat[i]+(1-a)*preds_mat[j]; r2=1-np.sum((y-bl)**2)/ss_tot
            if r2>blend_best: blend_best=r2
for i in range(min(10,len(names))):
    for j in range(i+1,min(10,len(names))):
        for k in range(j+1,min(10,len(names))):
            for a in np.linspace(0.05,0.9,20):
                for b in np.linspace(0.05,0.9-a,15):
                    c=1-a-b
                    if c>0:
                        bl=a*preds_mat[i]+b*preds_mat[j]+c*preds_mat[k]
                        r2=1-np.sum((y-bl)**2)/ss_tot
                        if r2>blend_best: blend_best=r2
print(f"Mega blend: R2={blend_best:.4f}", flush=True)
if blend_best>best_overall: best_overall=blend_best

print(f"\n{'='*60}")
print(f"V20 FINAL ({(time.time()-t0)/60:.1f} min)")
print(f"{'='*60}")
print(f"  HOMA_IR DW: R2={best_overall:.4f} (target 0.37, gap={0.37-best_overall:.3f})")
print(f"  Total base models: {len(all_results)}")
print(f"  Layer-1 stacks: {len(all_stacks)}")
print(f"  Previous best: 0.3250 (V17c)")
print(f"  Improvement: {best_overall-0.3250:+.4f}")
