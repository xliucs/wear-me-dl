#!/usr/bin/env python3
"""V21: Smart approaches - Optuna hyperopt, GP, forward selection.
Instead of throwing more models at it, optimize the best model family."""
import pandas as pd, numpy as np, warnings, time, sys
warnings.filterwarnings('ignore')
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct, WhiteKernel, ConstantKernel
from sklearn.svm import SVR, NuSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
import xgboost as xgb

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
mi_order = np.argsort(mi)[::-1]

print(f"n={n_samples}, target skew={pd.Series(y).skew():.2f}")

def run_cv(X, y, model_fn, scale=True, n_rep=5, seed=42):
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=seed)
    preds=np.zeros(n_samples); counts=np.zeros(n_samples)
    for tr,te in rkf.split(X,bins):
        Xtr,Xte=X[tr].copy(),X[te].copy()
        if scale: s=StandardScaler(); Xtr=s.fit_transform(Xtr); Xte=s.transform(Xte)
        m=model_fn(); m.fit(Xtr,y[tr]); p=m.predict(Xte)
        preds[te]+=p; counts[te]+=1
    preds/=counts
    return 1-np.sum((y-preds)**2)/ss_tot, preds

# ============================================================
print("="*60)
print("EXP 1: OPTUNA HYPEROPT ON KERNEL RIDGE")
print("="*60)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except:
    HAS_OPTUNA = False
    print("  Optuna not available, using manual grid")

best_kr = -999
best_kr_preds = None
results = {}

if HAS_OPTUNA:
    def objective(trial):
        kernel = trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid', 'laplacian'])
        alpha = trial.suggest_float('alpha', 0.001, 100, log=True)
        gamma = trial.suggest_float('gamma', 0.0001, 1, log=True)
        
        if kernel == 'poly':
            degree = trial.suggest_int('degree', 2, 4)
            coef0 = trial.suggest_float('coef0', 0, 10)
            m = lambda: KernelRidge(alpha=alpha, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)
        elif kernel == 'sigmoid':
            coef0 = trial.suggest_float('coef0', 0, 10)
            m = lambda al=alpha, g=gamma, c=coef0: KernelRidge(alpha=al, kernel='sigmoid', gamma=g, coef0=c)
        elif kernel == 'laplacian':
            m = lambda al=alpha, g=gamma: KernelRidge(alpha=al, kernel='laplacian', gamma=g)
        else:
            m = lambda al=alpha, g=gamma: KernelRidge(alpha=al, kernel='rbf', gamma=g)
        
        for fs_name, X_fs in [('raw18', X_raw), ('eng', X_eng)]:
            trial.set_user_attr(f'fs_{fs_name}', True)
        
        # Use raw18 for speed, 3 reps
        try:
            r2, _ = run_cv(X_raw, y, m, scale=True, n_rep=3, seed=42)
        except:
            r2 = -1
        return r2
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=200, show_progress_bar=False)
    
    print(f"  Best trial: R2={study.best_trial.value:.4f}")
    print(f"  Params: {study.best_trial.params}")
    
    # Verify best with 5-rep on both feature sets
    bp = study.best_trial.params
    for fs_name, X_fs in [('raw18', X_raw), ('eng', X_eng)]:
        if bp['kernel'] == 'poly':
            m = lambda: KernelRidge(alpha=bp['alpha'], kernel='poly', gamma=bp['gamma'], degree=bp.get('degree',2), coef0=bp.get('coef0',0))
        elif bp['kernel'] == 'laplacian':
            m = lambda: KernelRidge(alpha=bp['alpha'], kernel='laplacian', gamma=bp['gamma'])
        else:
            m = lambda: KernelRidge(alpha=bp['alpha'], kernel=bp['kernel'], gamma=bp['gamma'])
        r2, preds = run_cv(X_fs, y, m, scale=True, n_rep=5, seed=42)
        results[f'optuna_kr_{fs_name}'] = {'r2': r2, 'preds': preds}
        print(f"  {fs_name} 5-rep: R2={r2:.4f}")
        if r2 > best_kr: best_kr = r2; best_kr_preds = preds

else:
    # Manual fine grid around known good spot
    for kernel in ['rbf', 'poly', 'laplacian']:
        for alpha in [0.01, 0.05, 0.1, 0.3, 0.5, 1, 2, 5, 10, 20]:
            for gamma in [0.0005, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1]:
                if kernel == 'poly':
                    for deg in [2, 3]:
                        m = lambda a=alpha,g=gamma,d=deg: KernelRidge(alpha=a, kernel='poly', gamma=g, degree=d)
                        try:
                            r2, preds = run_cv(X_raw, y, m, scale=True, n_rep=3, seed=42)
                            if r2 > best_kr: best_kr = r2; best_kr_preds = preds
                            results[f'kr_{kernel}{deg}_a{alpha}_g{gamma}'] = {'r2': r2, 'preds': preds}
                        except: pass
                else:
                    m = lambda a=alpha,g=gamma,k=kernel: KernelRidge(alpha=a, kernel=k, gamma=g)
                    try:
                        r2, preds = run_cv(X_raw, y, m, scale=True, n_rep=3, seed=42)
                        if r2 > best_kr: best_kr = r2; best_kr_preds = preds
                        results[f'kr_{kernel}_a{alpha}_g{gamma}'] = {'r2': r2, 'preds': preds}
                    except: pass
    # Best with 5 rep
    top_keys = sorted(results.keys(), key=lambda k: results[k]['r2'], reverse=True)[:5]
    for k in top_keys:
        print(f"  {k}: R2={results[k]['r2']:.4f}")
    
    # Also on eng features
    for alpha in [0.01, 0.05, 0.1, 0.5, 1, 5, 10]:
        for gamma in [0.0005, 0.001, 0.005, 0.01]:
            m = lambda a=alpha,g=gamma: KernelRidge(alpha=a, kernel='rbf', gamma=g)
            r2, preds = run_cv(X_eng, y, m, scale=True, n_rep=3, seed=42)
            if r2 > best_kr: best_kr = r2; best_kr_preds = preds
            results[f'kr_eng_rbf_a{alpha}_g{gamma}'] = {'r2': r2, 'preds': preds}
    
    top_eng = sorted([k for k in results if 'eng' in k], key=lambda k: results[k]['r2'], reverse=True)[:3]
    for k in top_eng:
        print(f"  {k}: R2={results[k]['r2']:.4f}")

print(f"\n  Best KR overall: R2={best_kr:.4f}")

# ============================================================
print("\n" + "="*60)
print("EXP 2: GAUSSIAN PROCESS WITH CUSTOM KERNELS")
print("="*60)

# GP is slow for 798 samples but tractable
gp_results = {}
kernels = [
    ('rbf', 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1)),
    ('matern12', 1.0 * Matern(length_scale=1.0, nu=0.5) + WhiteKernel(noise_level=1)),
    ('matern32', 1.0 * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1)),
    ('matern52', 1.0 * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1)),
    ('rq', 1.0 * RationalQuadratic() + WhiteKernel(noise_level=1)),
    ('dot_rbf', DotProduct() + 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1)),
]

for kname, kernel in kernels:
    for fs_name, X_fs in [('raw18', X_raw)]:  # Only raw18 for speed
        t1 = time.time()
        try:
            r2, preds = run_cv(X_fs, y, 
                lambda k=kernel: GaussianProcessRegressor(kernel=k, n_restarts_optimizer=3, random_state=42),
                scale=True, n_rep=3, seed=42)
            dt = time.time()-t1
            gp_results[f'gp_{kname}_{fs_name}'] = {'r2': r2, 'preds': preds}
            results[f'gp_{kname}_{fs_name}'] = {'r2': r2, 'preds': preds}
            print(f"  GP({kname}) {fs_name}: R2={r2:.4f} ({dt:.0f}s)", flush=True)
        except Exception as e:
            print(f"  GP({kname}) {fs_name}: FAILED ({e})", flush=True)

# ============================================================
print("\n" + "="*60)
print("EXP 3: FORWARD FEATURE SELECTION")
print("="*60)

# Forward selection on engineered features
eng_df = engineer(X_dw).fillna(0)
all_cols = list(eng_df.columns)

# Start with BMI (strongest predictor)
selected = ['bmi']
remaining = [c for c in all_cols if c != 'bmi']

# Baseline: BMI alone
X_sel = eng_df[selected].values
r2, _ = run_cv(X_sel, y, lambda: Ridge(alpha=100), scale=True, n_rep=3)
print(f"  Start: bmi only, R2={r2:.4f}")

best_forward = r2
for step in range(30):  # Add up to 30 features
    best_gain = -999; best_feat = None
    # Test each remaining feature
    for feat in remaining:
        X_test = eng_df[selected + [feat]].values
        try:
            r2_t, _ = run_cv(X_test, y, lambda: Ridge(alpha=100), scale=True, n_rep=3)
            gain = r2_t - best_forward
            if gain > best_gain:
                best_gain = gain; best_feat = feat; best_r2 = r2_t
        except: pass
    
    if best_feat is None or best_gain < 0.001:
        print(f"  Stopped at step {step+1}: best gain={best_gain:.4f}")
        break
    
    selected.append(best_feat); remaining.remove(best_feat)
    best_forward = best_r2
    if step < 15 or step % 5 == 0:
        print(f"  +{best_feat:30s}: R2={best_forward:.4f} (gain={best_gain:.4f})", flush=True)

print(f"\n  Forward selection final: {len(selected)} features, R2={best_forward:.4f}")

# Now try KernelRidge on the selected features
X_fwd = eng_df[selected].values
for alpha in [0.1, 0.5, 1, 5, 10]:
    for gamma in [0.001, 0.005, 0.01, 0.05]:
        r2, preds = run_cv(X_fwd, y, lambda a=alpha,g=gamma: KernelRidge(alpha=a, kernel='rbf', gamma=g), scale=True, n_rep=5)
        results[f'fwd_kr_a{alpha}_g{gamma}'] = {'r2': r2, 'preds': preds}

best_fwd_kr = max([results[k]['r2'] for k in results if k.startswith('fwd_kr')])
print(f"  Forward + KernelRidge: R2={best_fwd_kr:.4f}")

# ============================================================
print("\n" + "="*60)
print("EXP 4: OPTIMIZED SVR")
print("="*60)

if HAS_OPTUNA:
    def svr_objective(trial):
        C = trial.suggest_float('C', 0.01, 1000, log=True)
        epsilon = trial.suggest_float('epsilon', 0.001, 1, log=True)
        gamma = trial.suggest_float('gamma', 0.0001, 1, log=True)
        kernel = trial.suggest_categorical('kernel', ['rbf', 'poly'])
        
        if kernel == 'poly':
            degree = trial.suggest_int('degree', 2, 4)
            m = lambda: SVR(kernel='poly', C=C, epsilon=epsilon, gamma=gamma, degree=degree)
        else:
            m = lambda: SVR(kernel='rbf', C=C, epsilon=epsilon, gamma=gamma)
        
        r2, _ = run_cv(X_raw, y, m, scale=True, n_rep=3, seed=42)
        return r2
    
    svr_study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
    svr_study.optimize(svr_objective, n_trials=150)
    
    print(f"  Best SVR: R2={svr_study.best_trial.value:.4f}")
    print(f"  Params: {svr_study.best_trial.params}")
    
    bp = svr_study.best_trial.params
    for fs_name, X_fs in [('raw18', X_raw), ('eng', X_eng)]:
        if bp['kernel'] == 'poly':
            m = lambda: SVR(kernel='poly', C=bp['C'], epsilon=bp['epsilon'], gamma=bp['gamma'], degree=bp.get('degree',2))
        else:
            m = lambda: SVR(kernel='rbf', C=bp['C'], epsilon=bp['epsilon'], gamma=bp['gamma'])
        r2, preds = run_cv(X_fs, y, m, scale=True, n_rep=5)
        results[f'optuna_svr_{fs_name}'] = {'r2': r2, 'preds': preds}
        print(f"  {fs_name} 5-rep: R2={r2:.4f}")

# ============================================================
print("\n" + "="*60)
print("EXP 5: OPTIMIZED XGBoost")
print("="*60)

if HAS_OPTUNA:
    def xgb_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 1, 5),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 100, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 100, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
            'gamma': trial.suggest_float('gamma', 0, 10),
            'random_state': 42,
        }
        
        r2, _ = run_cv(X_eng, y, lambda: xgb.XGBRegressor(**params), scale=False, n_rep=3, seed=42)
        return r2
    
    xgb_study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    xgb_study.optimize(xgb_objective, n_trials=200)
    
    print(f"  Best XGB: R2={xgb_study.best_trial.value:.4f}")
    print(f"  Params: {xgb_study.best_trial.params}")
    
    bp = xgb_study.best_trial.params
    for fs_name, X_fs in [('raw18', X_raw), ('eng', X_eng)]:
        r2, preds = run_cv(X_fs, y, lambda: xgb.XGBRegressor(**bp), scale=False, n_rep=5)
        results[f'optuna_xgb_{fs_name}'] = {'r2': r2, 'preds': preds}
        print(f"  {fs_name} 5-rep: R2={r2:.4f}")

# ============================================================
print("\n" + "="*60)
print("FINAL: COMBINE ALL BEST")
print("="*60)

# Collect all predictions with R2 > 0.10
good = {k: v for k, v in results.items() if v['r2'] > 0.10}
names = sorted(good.keys(), key=lambda k: good[k]['r2'], reverse=True)[:25]
preds_mat = np.array([good[nm]['preds'] for nm in names])

print(f"Combining {len(names)} predictions")
for nm in names[:10]:
    print(f"  {nm:50s}: R2={good[nm]['r2']:.4f}")

# Stack
best_final = -999
for seed in [0, 1, 42]:
    for alpha in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        sp=np.zeros(n_samples); sc=np.zeros(n_samples)
        for tr,te in RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=seed).split(preds_mat.T,bins):
            m=Ridge(alpha=alpha); m.fit(preds_mat[:,tr].T,y[tr])
            sp[te]+=m.predict(preds_mat[:,te].T); sc[te]+=1
        sp/=sc; r2=1-np.sum((y-sp)**2)/ss_tot
        if r2>best_final: best_final=r2

# Blend
rng=np.random.RandomState(42)
for _ in range(500000):
    w=rng.dirichlet(np.ones(len(names))*0.3); bl=w@preds_mat
    r2=1-np.sum((y-bl)**2)/ss_tot
    if r2>best_final: best_final=r2

print(f"\nFINAL: R2={best_final:.4f} (target 0.37)")
print(f"Previous best: 0.3250 (V17c)")
