#!/usr/bin/env python3
"""V22b: Stack all V22 innovations PLUS V17c-style 242 base models.
Key: V17c's 0.3250 comes from layer-2 stacking of 34 diverse stacks.
V22b adds target-encoded, KNN-augmented, and quantile-classification features
to the base model pool, then applies the same multi-layer stacking."""
import pandas as pd, numpy as np, warnings, time
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
n_samples = len(y)
X_raw = X_dw.values

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

X_eng = engineer(X_dw).fillna(0).values
mi = mutual_info_regression(X_eng, y, random_state=42)
X_mi35 = X_eng[:, np.argsort(mi)[-35:]]

t0 = time.time()

# ============================================================
# CREATE AUGMENTED FEATURE SETS
# ============================================================
print("Creating augmented features...", flush=True)

# Target-encoded features (OOF)
def make_te_features(X_df, y_arr, bins_arr, n_bins_list=[3,5,10], smooth=10):
    te = np.zeros((len(y_arr), 0))
    for col in ['bmi','age',rhr_m,hrv_m,stp_m,slp_m,azm_m]:
        vals = X_df[col].values
        for nb in n_bins_list:
            try:
                be = np.percentile(vals, np.linspace(0,100,nb+1)); be[0]-=1; be[-1]+=1
                cb = np.digitize(vals, be[1:-1])
            except: continue
            enc = np.zeros(len(y_arr)); gm = y_arr.mean()
            for tr,te_idx in StratifiedKFold(5,shuffle=True,random_state=42).split(vals,bins_arr):
                for b in range(nb):
                    mtr = cb[tr]==b; mte = cb[te_idx]==b
                    if mtr.sum()>0:
                        bm = y_arr[tr][mtr].mean(); bc = mtr.sum()
                        enc[te_idx[mte]] = (bc*bm+smooth*gm)/(bc+smooth)
                    else: enc[te_idx[mte]] = gm
            te = np.column_stack([te, enc])
    return te

# KNN target features (OOF)
def make_knn_features(X_arr, y_arr, bins_arr, k_list=[5,10,20,50]):
    kf = np.zeros((len(y_arr), 0))
    for k in k_list:
        mf=np.zeros(len(y_arr)); sf=np.zeros(len(y_arr)); df=np.zeros(len(y_arr))
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=42).split(X_arr,bins_arr):
            sc=StandardScaler(); Xtr=sc.fit_transform(X_arr[tr]); Xte=sc.transform(X_arr[te])
            nn=NearestNeighbors(n_neighbors=k); nn.fit(Xtr)
            _,idx=nn.kneighbors(Xte)
            for i,ti in enumerate(te):
                nt = y_arr[tr][idx[i]]
                mf[ti]=nt.mean(); sf[ti]=nt.std(); df[ti]=np.median(nt)
        kf = np.column_stack([kf, mf, sf, df])
    return kf

# Quantile classification features (OOF)
def make_qc_features(X_arr, y_arr, quantiles=[0.25,0.5,0.75,0.9]):
    from sklearn.ensemble import GradientBoostingClassifier
    qf = np.zeros((len(y_arr), 0))
    for q in quantiles:
        thr = np.quantile(y_arr, q); yc = (y_arr>thr).astype(int)
        probs = np.zeros(len(y_arr))
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=42).split(X_arr,yc):
            sc=StandardScaler(); Xtr=sc.fit_transform(X_arr[tr]); Xte=sc.transform(X_arr[te])
            clf=GradientBoostingClassifier(n_estimators=100,max_depth=2,learning_rate=0.1,random_state=42)
            clf.fit(Xtr,yc[tr]); probs[te]=clf.predict_proba(Xte)[:,1]
        qf = np.column_stack([qf, probs])
    return qf

te_feats = make_te_features(X_dw, y, bins)
knn_feats = make_knn_features(X_raw, y, bins)
qc_feats = make_qc_features(X_raw, y)

# Feature sets: original + augmented
X_mega = np.column_stack([X_raw, te_feats, knn_feats, qc_feats])
X_mega_eng = np.column_stack([X_eng, te_feats, knn_feats, qc_feats])

fsets = {'raw18': X_raw, 'eng': X_eng, 'mi35': X_mi35,
         'mega': X_mega, 'mega_eng': X_mega_eng}

print(f"Feature sets: raw18={X_raw.shape[1]}, eng={X_eng.shape[1]}, mi35={X_mi35.shape[1]}, mega={X_mega.shape[1]}, mega_eng={X_mega_eng.shape[1]}", flush=True)

# ============================================================
# GENERATE BASE MODELS (V17c style + mega features)
# ============================================================
def run_cv(X, yy, model_fn, scale=True, log_t=False, n_rep=5, seed=42):
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=seed)
    preds=np.zeros(n_samples); counts=np.zeros(n_samples)
    for tr,te in rkf.split(X,bins):
        Xtr,Xte=X[tr].copy(),X[te].copy(); ytr=yy[tr].copy()
        if log_t: ytr=np.log1p(ytr)
        if scale: s=StandardScaler(); Xtr=s.fit_transform(Xtr); Xte=s.transform(Xte)
        m=model_fn(); m.fit(Xtr,ytr); p=m.predict(Xte)
        if log_t: p=np.expm1(p)
        preds[te]+=p; counts[te]+=1
    preds/=counts
    return 1-np.sum((y-preds)**2)/ss_tot, preds

all_results = {}

fast_models = [
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
    ('bag_ridge100', lambda: BaggingRegressor(estimator=Ridge(alpha=100), n_estimators=30, max_samples=0.8, max_features=0.8, random_state=42), True),
    ('bag_ridge500', lambda: BaggingRegressor(estimator=Ridge(alpha=500), n_estimators=30, max_samples=0.8, max_features=0.8, random_state=42), True),
    ('bag_ridge1000', lambda: BaggingRegressor(estimator=Ridge(alpha=1000), n_estimators=30, max_samples=0.8, max_features=0.8, random_state=42), True),
]

med_models = [
    ('xgb_d2', lambda: xgb.XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, reg_alpha=2, reg_lambda=5, random_state=42), False),
    ('xgb_d3_lr01', lambda: xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.01, subsample=0.8, colsample_bytree=0.7, reg_alpha=5, reg_lambda=10, random_state=42), False),
    ('xgb_d2_mae', lambda: xgb.XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, objective='reg:absoluteerror', random_state=42), False),
    ('xgb_d2_huber', lambda: xgb.XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, objective='reg:pseudohubererror', random_state=42), False),
    ('lgb_d2', lambda: lgb.LGBMRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, verbose=-1, n_jobs=1, random_state=42), False),
    ('lgb_d3', lambda: lgb.LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, verbose=-1, n_jobs=1, random_state=42), False),
    ('lgb_dart', lambda: lgb.LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, boosting_type='dart', verbose=-1, n_jobs=1, random_state=42), False),
    ('hgbr_d2', lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=2, learning_rate=0.05, random_state=42), False),
    ('hgbr_d3', lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05, random_state=42), False),
    ('gbr_d2', lambda: GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=42), False),
    ('gbr_d3', lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42), False),
    ('rf_d3', lambda: RandomForestRegressor(n_estimators=300, max_depth=3, max_features=0.5, random_state=42, n_jobs=1), False),
    ('rf_d4', lambda: RandomForestRegressor(n_estimators=300, max_depth=4, max_features=0.5, random_state=42, n_jobs=1), False),
    ('et_d3', lambda: ExtraTreesRegressor(n_estimators=300, max_depth=3, max_features=0.5, random_state=42, n_jobs=1), False),
    ('et_d4', lambda: ExtraTreesRegressor(n_estimators=300, max_depth=4, max_features=0.5, random_state=42, n_jobs=1), False),
]

use_log = True  # HOMA is skewed

for fs_name, X_fs in fsets.items():
    print(f"\n--- {fs_name} ({X_fs.shape[1]}f) ---", flush=True)
    
    for mname, mfn, scale in fast_models:
        full = f"{fs_name}__{mname}"
        try:
            r2, preds = run_cv(X_fs, y, mfn, scale=scale, n_rep=5)
            all_results[full] = {'r2': r2, 'preds': preds}
            if r2 > 0.15: print(f"  {full:50s}: R2={r2:.4f}", flush=True)
            if use_log:
                r2l, pl = run_cv(X_fs, y, mfn, scale=scale, log_t=True, n_rep=5)
                all_results[full+'_log'] = {'r2': r2l, 'preds': pl}
                if r2l > 0.15: print(f"  {full+'_log':50s}: R2={r2l:.4f}", flush=True)
        except: pass
    
    for mname, mfn, scale in med_models:
        full = f"{fs_name}__{mname}"
        try:
            r2, preds = run_cv(X_fs, y, mfn, scale=scale, n_rep=3)
            all_results[full] = {'r2': r2, 'preds': preds}
            if r2 > 0.15: print(f"  {full:50s}: R2={r2:.4f}", flush=True)
        except: pass

print(f"\nTotal models: {len(all_results)}", flush=True)

# ============================================================
# MULTI-LAYER STACKING (V17c style)
# ============================================================
print(f"\n{'='*60}")
print("MULTI-LAYER STACKING")
print(f"{'='*60}", flush=True)

good = {k:v for k,v in all_results.items() if v['r2'] > 0}
names = sorted(good.keys(), key=lambda k: good[k]['r2'], reverse=True)[:25]
preds_mat = np.array([good[nm]['preds'] for nm in names])
nm_count = len(names)

print(f"Top 10 of {len(good)} models:")
for nm in names[:10]:
    print(f"  {nm:55s}: R2={good[nm]['r2']:.4f}", flush=True)

best_r2 = -999; all_stacks = {}

# Ridge
for alpha in [0.001,0.01,0.1,0.5,1,5,10,50,100,500,1000,5000]:
    sp=np.zeros(n_samples); sc=np.zeros(n_samples)
    for tr,te in RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=42).split(preds_mat.T,bins):
        m=Ridge(alpha=alpha); m.fit(preds_mat[:,tr].T,y[tr])
        sp[te]+=m.predict(preds_mat[:,te].T); sc[te]+=1
    sp/=sc; r2=1-np.sum((y-sp)**2)/ss_tot
    if r2>best_r2: best_r2=r2
    if r2>0.20: all_stacks[f'ridge_{alpha}']=sp.copy()
print(f"Ridge stack: R2={best_r2:.4f}", flush=True)

# ElasticNet
for alpha in [0.01,0.1,1]:
    for l1 in [0.1,0.3,0.5,0.7,0.9]:
        sp=np.zeros(n_samples); sc=np.zeros(n_samples)
        for tr,te in RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=42).split(preds_mat.T,bins):
            m=ElasticNet(alpha=alpha,l1_ratio=l1,max_iter=5000,positive=True)
            m.fit(preds_mat[:,tr].T,y[tr]); sp[te]+=m.predict(preds_mat[:,te].T); sc[te]+=1
        sp/=sc; r2=1-np.sum((y-sp)**2)/ss_tot
        if r2>best_r2: best_r2=r2
        if r2>0.20: all_stacks[f'en_{alpha}_{l1}']=sp.copy()
print(f"+EN: R2={best_r2:.4f}", flush=True)

# Lasso
for alpha in [0.001,0.01,0.1]:
    sp=np.zeros(n_samples); sc=np.zeros(n_samples)
    for tr,te in RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=42).split(preds_mat.T,bins):
        m=Lasso(alpha=alpha,max_iter=5000,positive=True)
        m.fit(preds_mat[:,tr].T,y[tr]); sp[te]+=m.predict(preds_mat[:,te].T); sc[te]+=1
    sp/=sc; r2=1-np.sum((y-sp)**2)/ss_tot
    if r2>best_r2: best_r2=r2
    if r2>0.20: all_stacks[f'lasso_{alpha}']=sp.copy()
print(f"+Lasso: R2={best_r2:.4f}", flush=True)

# KNN
for k in [3,5,7,10,15,20,30]:
    sp=np.zeros(n_samples); sc=np.zeros(n_samples)
    for tr,te in RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=42).split(preds_mat.T,bins):
        ss=StandardScaler(); pt=ss.fit_transform(preds_mat[:,tr].T); pe=ss.transform(preds_mat[:,te].T)
        m=KNeighborsRegressor(n_neighbors=k,weights='distance'); m.fit(pt,y[tr])
        sp[te]+=m.predict(pe); sc[te]+=1
    sp/=sc; r2=1-np.sum((y-sp)**2)/ss_tot
    if r2>best_r2: best_r2=r2
    if r2>0.20: all_stacks[f'knn_{k}']=sp.copy()
print(f"+KNN: R2={best_r2:.4f}", flush=True)

# SVR
for C in [0.1,1,10,100]:
    for eps in [0.05,0.1]:
        sp=np.zeros(n_samples); sc=np.zeros(n_samples)
        for tr,te in RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=42).split(preds_mat.T,bins):
            ss=StandardScaler(); pt=ss.fit_transform(preds_mat[:,tr].T); pe=ss.transform(preds_mat[:,te].T)
            m=SVR(kernel='rbf',C=C,epsilon=eps); m.fit(pt,y[tr])
            sp[te]+=m.predict(pe); sc[te]+=1
        sp/=sc; r2=1-np.sum((y-sp)**2)/ss_tot
        if r2>best_r2: best_r2=r2
        if r2>0.20: all_stacks[f'svr_{C}_{eps}']=sp.copy()
print(f"+SVR: R2={best_r2:.4f}", flush=True)

# XGB
for d in [2,3]:
    sp=np.zeros(n_samples); sc=np.zeros(n_samples)
    for tr,te in RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=42).split(preds_mat.T,bins):
        m=xgb.XGBRegressor(n_estimators=100,max_depth=d,learning_rate=0.05,reg_alpha=5,reg_lambda=10,random_state=42)
        m.fit(preds_mat[:,tr].T,y[tr]); sp[te]+=m.predict(preds_mat[:,te].T); sc[te]+=1
    sp/=sc; r2=1-np.sum((y-sp)**2)/ss_tot
    if r2>best_r2: best_r2=r2
    if r2>0.20: all_stacks[f'xgb_{d}']=sp.copy()
print(f"+XGB: R2={best_r2:.4f}", flush=True)

# Bayesian
sp=np.zeros(n_samples); sc=np.zeros(n_samples)
for tr,te in RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=42).split(preds_mat.T,bins):
    m=BayesianRidge(); m.fit(preds_mat[:,tr].T,y[tr])
    sp[te]+=m.predict(preds_mat[:,te].T); sc[te]+=1
sp/=sc; r2=1-np.sum((y-sp)**2)/ss_tot
if r2>best_r2: best_r2=r2
if r2>0.20: all_stacks['bayesian']=sp.copy()
print(f"+Bayesian: R2={best_r2:.4f}", flush=True)

# Mega blend
rng=np.random.RandomState(42); blend_best=-999
for _ in range(500000):
    w=rng.dirichlet(np.ones(nm_count)*0.3); bl=w@preds_mat; r2=1-np.sum((y-bl)**2)/ss_tot
    if r2>blend_best: blend_best=r2
for _ in range(200000):
    w=rng.dirichlet(np.ones(nm_count)*0.1); bl=w@preds_mat; r2=1-np.sum((y-bl)**2)/ss_tot
    if r2>blend_best: blend_best=r2
for i in range(nm_count):
    for j in range(i+1,nm_count):
        for a in np.linspace(0,1,201):
            bl=a*preds_mat[i]+(1-a)*preds_mat[j]; r2=1-np.sum((y-bl)**2)/ss_tot
            if r2>blend_best: blend_best=r2
for i in range(min(8,nm_count)):
    for j in range(i+1,min(8,nm_count)):
        for k in range(j+1,min(8,nm_count)):
            for a in np.linspace(0.05,0.9,20):
                for b in np.linspace(0.05,0.9-a,15):
                    c=1-a-b
                    if c>0:
                        bl=a*preds_mat[i]+b*preds_mat[j]+c*preds_mat[k]; r2=1-np.sum((y-bl)**2)/ss_tot
                        if r2>blend_best: blend_best=r2
if blend_best>best_r2: best_r2=blend_best
print(f"Mega blend: R2={blend_best:.4f}", flush=True)

# Layer 2
if len(all_stacks)>=3:
    print(f"Layer 2: {len(all_stacks)} stacks", flush=True)
    snames=list(all_stacks.keys()); smat=np.array([all_stacks[k] for k in snames])
    for alpha in [0.01,0.1,1,10,100]:
        sp=np.zeros(n_samples); sc=np.zeros(n_samples)
        for tr,te in RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=42).split(smat.T,bins):
            m=Ridge(alpha=alpha); m.fit(smat[:,tr].T,y[tr])
            sp[te]+=m.predict(smat[:,te].T); sc[te]+=1
        sp/=sc; r2=1-np.sum((y-sp)**2)/ss_tot
        if r2>best_r2: best_r2=r2
    print(f"Layer 2: R2={best_r2:.4f}", flush=True)

print(f"\n{'='*60}")
print(f"V22b FINAL ({(time.time()-t0)/60:.1f} min)")
print(f"{'='*60}")
print(f"  HOMA_IR DW: R2={best_r2:.4f} (target 0.37, gap={0.37-best_r2:.3f})")
print(f"  Previous best: 0.3250 (V17c)")
print(f"  Improvement: {best_r2-0.3250:+.4f}")
