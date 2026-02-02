#!/usr/bin/env python3
"""V17c: Fast DW push — get to stacking ASAP.
Learnings: RF log is 20s/model, kills throughput. Use 3-rep CV for slow models.
Skip log-target for RF/ET. Focus on diversity for stacking."""
import pandas as pd, numpy as np, warnings, time, os
warnings.filterwarnings('ignore')

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
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

rhr_m='Resting Heart Rate (mean)'; rhr_md='Resting Heart Rate (median)'; rhr_s='Resting Heart Rate (std)'
hrv_m='HRV (mean)'; hrv_md='HRV (median)'; hrv_s='HRV (std)'
stp_m='STEPS (mean)'; stp_md='STEPS (median)'; stp_s='STEPS (std)'
slp_m='SLEEP Duration (mean)'; slp_md='SLEEP Duration (median)'; slp_s='SLEEP Duration (std)'
azm_m='AZM Weekly (mean)'; azm_md='AZM Weekly (median)'; azm_s='AZM Weekly (std)'

def make_bins(y, nb=5): return pd.qcut(y, nb, labels=False, duplicates='drop')

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

def run_cv(X, y, model_fn, scale=True, log_t=False, n_rep=5):
    bins = make_bins(y)
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=42)
    ns = len(y); preds = np.zeros(ns); counts = np.zeros(ns)
    for tr, te in rkf.split(X, bins):
        Xtr, Xte = X[tr].copy(), X[te].copy(); ytr = y[tr].copy()
        if log_t: ytr = np.log1p(ytr)
        if scale:
            s = StandardScaler(); Xtr = s.fit_transform(Xtr); Xte = s.transform(Xte)
        m = model_fn(); m.fit(Xtr, ytr); p = m.predict(Xte)
        if log_t: p = np.expm1(p)
        preds[te] += p; counts[te] += 1
    preds /= counts
    return 1 - np.sum((y-preds)**2)/np.sum((y-y.mean())**2), preds

def advanced_stack(results, y, label):
    ss_tot = np.sum((y-y.mean())**2)
    names = sorted(results.keys(), key=lambda k: results[k]['r2'], reverse=True)[:25]
    preds_mat = np.array([results[nm]['preds'] for nm in names])
    nm_count = len(names)
    bins = make_bins(y)
    
    print(f"\n  Top 10:", flush=True)
    for nm in names[:10]:
        print(f"    {nm:55s}: R2={results[nm]['r2']:.4f}", flush=True)
    
    best_r2 = -999; best_preds = None
    all_stacks = {}
    
    # Ridge
    for alpha in [0.001,0.01,0.1,0.5,1,5,10,50,100,500,1000,5000]:
        sp=np.zeros(len(y)); sc=np.zeros(len(y))
        for tr,te in RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=42).split(preds_mat.T,bins):
            m=Ridge(alpha=alpha); m.fit(preds_mat[:,tr].T,y[tr])
            sp[te]+=m.predict(preds_mat[:,te].T); sc[te]+=1
        sp/=sc; r2=1-np.sum((y-sp)**2)/ss_tot
        if r2>best_r2: best_r2=r2; best_preds=sp.copy()
        if r2>0.20: all_stacks[f'ridge_{alpha}']=sp.copy()
    print(f"  Ridge stack: R2={best_r2:.4f}", flush=True)
    
    # ElasticNet positive
    for alpha in [0.01,0.1,1]:
        for l1 in [0.1,0.3,0.5,0.7,0.9]:
            sp=np.zeros(len(y)); sc=np.zeros(len(y))
            for tr,te in RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=42).split(preds_mat.T,bins):
                m=ElasticNet(alpha=alpha,l1_ratio=l1,max_iter=5000,positive=True)
                m.fit(preds_mat[:,tr].T,y[tr]); sp[te]+=m.predict(preds_mat[:,te].T); sc[te]+=1
            sp/=sc; r2=1-np.sum((y-sp)**2)/ss_tot
            if r2>best_r2: best_r2=r2; best_preds=sp.copy()
            if r2>0.20: all_stacks[f'en_{alpha}_{l1}']=sp.copy()
    print(f"  +ElasticNet: R2={best_r2:.4f}", flush=True)
    
    # Lasso positive
    for alpha in [0.001,0.01,0.1]:
        sp=np.zeros(len(y)); sc=np.zeros(len(y))
        for tr,te in RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=42).split(preds_mat.T,bins):
            m=Lasso(alpha=alpha,max_iter=5000,positive=True)
            m.fit(preds_mat[:,tr].T,y[tr]); sp[te]+=m.predict(preds_mat[:,te].T); sc[te]+=1
        sp/=sc; r2=1-np.sum((y-sp)**2)/ss_tot
        if r2>best_r2: best_r2=r2; best_preds=sp.copy()
        if r2>0.20: all_stacks[f'lasso_{alpha}']=sp.copy()
    print(f"  +Lasso: R2={best_r2:.4f}", flush=True)
    
    # KNN stack
    for k in [3,5,7,10,15,20,30]:
        sp=np.zeros(len(y)); sc=np.zeros(len(y))
        for tr,te in RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=42).split(preds_mat.T,bins):
            ss=StandardScaler(); pt=ss.fit_transform(preds_mat[:,tr].T); pe=ss.transform(preds_mat[:,te].T)
            m=KNeighborsRegressor(n_neighbors=k,weights='distance'); m.fit(pt,y[tr])
            sp[te]+=m.predict(pe); sc[te]+=1
        sp/=sc; r2=1-np.sum((y-sp)**2)/ss_tot
        if r2>best_r2: best_r2=r2; best_preds=sp.copy()
        if r2>0.20: all_stacks[f'knn_{k}']=sp.copy()
    print(f"  +KNN: R2={best_r2:.4f}", flush=True)
    
    # SVR stack
    for C in [0.1,1,10,100]:
        for eps in [0.05,0.1]:
            sp=np.zeros(len(y)); sc=np.zeros(len(y))
            for tr,te in RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=42).split(preds_mat.T,bins):
                ss=StandardScaler(); pt=ss.fit_transform(preds_mat[:,tr].T); pe=ss.transform(preds_mat[:,te].T)
                m=SVR(kernel='rbf',C=C,epsilon=eps); m.fit(pt,y[tr])
                sp[te]+=m.predict(pe); sc[te]+=1
            sp/=sc; r2=1-np.sum((y-sp)**2)/ss_tot
            if r2>best_r2: best_r2=r2; best_preds=sp.copy()
            if r2>0.20: all_stacks[f'svr_{C}_{eps}']=sp.copy()
    print(f"  +SVR: R2={best_r2:.4f}", flush=True)
    
    # XGB stack
    for d in [2,3]:
        sp=np.zeros(len(y)); sc=np.zeros(len(y))
        for tr,te in RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=42).split(preds_mat.T,bins):
            m=xgb.XGBRegressor(n_estimators=100,max_depth=d,learning_rate=0.05,reg_alpha=5,reg_lambda=10,random_state=42)
            m.fit(preds_mat[:,tr].T,y[tr]); sp[te]+=m.predict(preds_mat[:,te].T); sc[te]+=1
        sp/=sc; r2=1-np.sum((y-sp)**2)/ss_tot
        if r2>best_r2: best_r2=r2; best_preds=sp.copy()
        if r2>0.20: all_stacks[f'xgb_{d}']=sp.copy()
    print(f"  +XGB: R2={best_r2:.4f}", flush=True)
    
    # BayesianRidge stack
    sp=np.zeros(len(y)); sc=np.zeros(len(y))
    for tr,te in RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=42).split(preds_mat.T,bins):
        m=BayesianRidge(); m.fit(preds_mat[:,tr].T,y[tr])
        sp[te]+=m.predict(preds_mat[:,te].T); sc[te]+=1
    sp/=sc; r2=1-np.sum((y-sp)**2)/ss_tot
    if r2>best_r2: best_r2=r2; best_preds=sp.copy()
    if r2>0.20: all_stacks['bayesian']=sp.copy()
    print(f"  +Bayesian: R2={best_r2:.4f}", flush=True)
    
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
                            bl=a*preds_mat[i]+b*preds_mat[j]+c*preds_mat[k]
                            r2=1-np.sum((y-bl)**2)/ss_tot
                            if r2>blend_best: blend_best=r2
    print(f"  Mega blend: R2={blend_best:.4f}", flush=True)
    if blend_best>best_r2: best_r2=blend_best
    
    # Layer 2: stack of stacks
    if len(all_stacks)>=3:
        print(f"  Layer 2: {len(all_stacks)} stacks", flush=True)
        snames=list(all_stacks.keys()); smat=np.array([all_stacks[k] for k in snames])
        for alpha in [0.01,0.1,1,10,100]:
            sp=np.zeros(len(y)); sc=np.zeros(len(y))
            for tr,te in RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=42).split(smat.T,bins):
                m=Ridge(alpha=alpha); m.fit(smat[:,tr].T,y[tr])
                sp[te]+=m.predict(smat[:,te].T); sc[te]+=1
            sp/=sc; r2=1-np.sum((y-sp)**2)/ss_tot
            if r2>best_r2: best_r2=r2
        print(f"  Layer 2: R2={best_r2:.4f}", flush=True)
    
    print(f"\n  >>> {label} FINAL: R2={best_r2:.4f}", flush=True)
    return best_r2

def run_target(target_col, target_name, target_r2):
    y_full=df[target_col].values; mask=~np.isnan(y_full)
    X_dw=X_base[mask].reset_index(drop=True); y=y_full[mask]
    use_log='homa' in target_name.lower()
    
    print(f"\n{'='*60}")
    print(f"{target_name} | n={len(y)} | Target: {target_r2}")
    print(f"{'='*60}", flush=True)
    
    X_raw=X_dw.values; X_eng=engineer(X_dw).fillna(0).values
    mi=mutual_info_regression(X_eng,y,random_state=42)
    X_mi35=X_eng[:,np.argsort(mi)[-35:]]
    
    fsets={'raw18':X_raw,'eng':X_eng,'mi35':X_mi35}
    all_results={}
    
    # Fast models (< 1s each)
    fast_models=[
        ('ridge_10',lambda:Ridge(alpha=10),True),
        ('ridge_50',lambda:Ridge(alpha=50),True),
        ('ridge_100',lambda:Ridge(alpha=100),True),
        ('ridge_500',lambda:Ridge(alpha=500),True),
        ('ridge_1000',lambda:Ridge(alpha=1000),True),
        ('ridge_2000',lambda:Ridge(alpha=2000),True),
        ('lasso_001',lambda:Lasso(alpha=0.01,max_iter=10000),True),
        ('lasso_01',lambda:Lasso(alpha=0.1,max_iter=10000),True),
        ('elastic_01_5',lambda:ElasticNet(alpha=0.1,l1_ratio=0.5,max_iter=10000),True),
        ('elastic_001_5',lambda:ElasticNet(alpha=0.01,l1_ratio=0.5,max_iter=10000),True),
        ('bayesian',lambda:BayesianRidge(),True),
        ('huber',lambda:HuberRegressor(max_iter=1000),True),
        ('kr_rbf_01_001',lambda:KernelRidge(alpha=0.1,kernel='rbf',gamma=0.001),True),
        ('kr_rbf_01_01',lambda:KernelRidge(alpha=0.1,kernel='rbf',gamma=0.01),True),
        ('kr_rbf_1_001',lambda:KernelRidge(alpha=1,kernel='rbf',gamma=0.001),True),
        ('kr_rbf_1_01',lambda:KernelRidge(alpha=1,kernel='rbf',gamma=0.01),True),
        ('kr_rbf_10_01',lambda:KernelRidge(alpha=10,kernel='rbf',gamma=0.01),True),
        ('kr_poly2',lambda:KernelRidge(alpha=1,kernel='poly',degree=2,gamma=0.01),True),
        ('svr_01',lambda:SVR(kernel='rbf',C=0.1,epsilon=0.1),True),
        ('svr_1',lambda:SVR(kernel='rbf',C=1,epsilon=0.1),True),
        ('svr_10',lambda:SVR(kernel='rbf',C=10,epsilon=0.1),True),
        ('svr_100',lambda:SVR(kernel='rbf',C=100,epsilon=0.05),True),
        ('nusvr_05',lambda:NuSVR(kernel='rbf',C=1,nu=0.5),True),
        ('nusvr_07',lambda:NuSVR(kernel='rbf',C=1,nu=0.7),True),
        ('knn_5',lambda:KNeighborsRegressor(n_neighbors=5,weights='distance'),True),
        ('knn_10',lambda:KNeighborsRegressor(n_neighbors=10,weights='distance'),True),
        ('knn_15',lambda:KNeighborsRegressor(n_neighbors=15,weights='distance'),True),
        ('knn_20',lambda:KNeighborsRegressor(n_neighbors=20,weights='distance'),True),
        ('knn_30',lambda:KNeighborsRegressor(n_neighbors=30,weights='distance'),True),
        ('knn_50',lambda:KNeighborsRegressor(n_neighbors=50,weights='distance'),True),
        ('knn_75',lambda:KNeighborsRegressor(n_neighbors=75,weights='distance'),True),
        ('bag_ridge100',lambda:BaggingRegressor(estimator=Ridge(alpha=100),n_estimators=30,max_samples=0.8,max_features=0.8,random_state=42),True),
        ('bag_ridge500',lambda:BaggingRegressor(estimator=Ridge(alpha=500),n_estimators=30,max_samples=0.8,max_features=0.8,random_state=42),True),
        ('bag_ridge1000',lambda:BaggingRegressor(estimator=Ridge(alpha=1000),n_estimators=30,max_samples=0.8,max_features=0.8,random_state=42),True),
    ]
    
    # Medium models (1-5s each) — 3 repeats
    med_models=[
        ('xgb_d2',lambda:xgb.XGBRegressor(n_estimators=200,max_depth=2,learning_rate=0.05,subsample=0.8,colsample_bytree=0.7,reg_alpha=2,reg_lambda=5,random_state=42),False),
        ('xgb_d3_lr01',lambda:xgb.XGBRegressor(n_estimators=300,max_depth=3,learning_rate=0.01,subsample=0.8,colsample_bytree=0.7,reg_alpha=5,reg_lambda=10,random_state=42),False),
        ('xgb_d2_mae',lambda:xgb.XGBRegressor(n_estimators=200,max_depth=2,learning_rate=0.05,objective='reg:absoluteerror',subsample=0.8,colsample_bytree=0.7,random_state=42),False),
        ('xgb_d2_huber',lambda:xgb.XGBRegressor(n_estimators=200,max_depth=2,learning_rate=0.05,objective='reg:pseudohubererror',subsample=0.8,colsample_bytree=0.7,random_state=42),False),
        ('lgb_d2',lambda:lgb.LGBMRegressor(n_estimators=200,max_depth=2,learning_rate=0.05,verbose=-1,n_jobs=1,random_state=42),False),
        ('lgb_d3',lambda:lgb.LGBMRegressor(n_estimators=200,max_depth=3,learning_rate=0.05,verbose=-1,n_jobs=1,random_state=42),False),
        ('lgb_dart',lambda:lgb.LGBMRegressor(n_estimators=200,max_depth=3,learning_rate=0.1,boosting_type='dart',verbose=-1,n_jobs=1,random_state=42),False),
        ('hgbr_d3',lambda:HistGradientBoostingRegressor(max_iter=300,max_depth=3,learning_rate=0.05,random_state=42),False),
        ('hgbr_d2',lambda:HistGradientBoostingRegressor(max_iter=300,max_depth=2,learning_rate=0.05,random_state=42),False),
        ('gbr_d2',lambda:GradientBoostingRegressor(n_estimators=200,max_depth=2,learning_rate=0.05,subsample=0.8,random_state=42),False),
        ('gbr_d3',lambda:GradientBoostingRegressor(n_estimators=200,max_depth=3,learning_rate=0.05,subsample=0.8,random_state=42),False),
        ('rf_d3',lambda:RandomForestRegressor(n_estimators=300,max_depth=3,max_features=0.5,random_state=42,n_jobs=1),False),
        ('rf_d4',lambda:RandomForestRegressor(n_estimators=300,max_depth=4,max_features=0.5,random_state=42,n_jobs=1),False),
        ('et_d3',lambda:ExtraTreesRegressor(n_estimators=300,max_depth=3,max_features=0.5,random_state=42,n_jobs=1),False),
        ('et_d4',lambda:ExtraTreesRegressor(n_estimators=300,max_depth=4,max_features=0.5,random_state=42,n_jobs=1),False),
    ]
    
    for fs_name, X_fs in fsets.items():
        print(f"\n--- {fs_name} ({X_fs.shape[1]}f) ---", flush=True)
        
        # Fast models: 5 repeats
        for mname,mfn,needs_scale in fast_models:
            full=f"{fs_name}__{mname}"
            try:
                r2,preds=run_cv(X_fs,y,mfn,scale=needs_scale,n_rep=5)
                all_results[full]={'r2':r2,'preds':preds}
                if r2>0.15: print(f"  {full:50s}: R2={r2:.4f}", flush=True)
                # Log target too (fast)
                if use_log:
                    r2l,pl=run_cv(X_fs,y,mfn,scale=needs_scale,log_t=True,n_rep=5)
                    all_results[full+'_log']={'r2':r2l,'preds':pl}
                    if r2l>0.15: print(f"  {full+'_log':50s}: R2={r2l:.4f}", flush=True)
            except: pass
        
        # Medium models: 3 repeats (faster)
        for mname,mfn,needs_scale in med_models:
            full=f"{fs_name}__{mname}"
            try:
                r2,preds=run_cv(X_fs,y,mfn,scale=needs_scale,n_rep=3)
                all_results[full]={'r2':r2,'preds':preds}
                if r2>0.15: print(f"  {full:50s}: R2={r2:.4f}", flush=True)
            except: pass
    
    # TabPFN skipped (hangs on model download)
    print(f"\n--- TabPFN skipped (hangs) ---", flush=True)
    
    # Stack
    good={k:v for k,v in all_results.items() if v['r2']>0}
    print(f"\n--- Stacking {len(good)} models ---", flush=True)
    best=advanced_stack(good,y,target_name)
    return best

print("="*60)
print("V17c: FAST DW PUSH")
print("="*60)
t0=time.time()

best_homa=run_target('True_HOMA_IR','HOMA_IR DW',0.37)
best_hba1c=run_target('True_hba1c','hba1c DW',0.70)

print(f"\n{'='*60}")
print(f"V17c FINAL ({(time.time()-t0)/60:.1f} min)")
print(f"{'='*60}")
print(f"  HOMA_IR DW: R2={best_homa:.4f} (target 0.37, gap={0.37-best_homa:.3f})")
print(f"  hba1c DW:   R2={best_hba1c:.4f} (target 0.70, gap={0.70-best_hba1c:.3f})")
print(f"  Previous: HOMA=0.2800, hba1c=0.1668")
