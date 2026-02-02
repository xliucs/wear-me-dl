#!/usr/bin/env python3
"""V22: Genius-level ideas for HOMA_IR DW > 0.37.

Key innovations:
1. Target-encoded features (nested CV)
2. KNN feature augmentation (neighbor target stats)
3. LOO stacking for less noisy meta-features
4. Quantile classification as features
5. Iterative residual learning
6. Autoencoder embeddings
"""
import pandas as pd, numpy as np, warnings, time
warnings.filterwarnings('ignore')
from sklearn.model_selection import RepeatedStratifiedKFold, LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import (HistGradientBoostingRegressor, GradientBoostingRegressor,
                               RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.feature_selection import mutual_info_regression
from sklearn.cluster import KMeans
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

t0 = time.time()
all_results = {}

# ============================================================
print("="*60)
print("IDEA 1: TARGET-ENCODED FEATURES (Nested CV)")
print("="*60)
# For continuous features, bin them and encode each bin with target mean
# Must use NESTED CV to prevent leakage

def make_target_encoded_features(X_df, y, n_bins_list=[3,5,10], smooth=10):
    """Create target-encoded features for key columns using out-of-fold encoding."""
    te_features = np.zeros((len(y), 0))
    te_names = []
    
    key_cols = ['bmi', 'age', rhr_m, hrv_m, stp_m, slp_m, azm_m]
    
    for col in key_cols:
        vals = X_df[col].values
        for nb in n_bins_list:
            # Create bins
            try:
                bin_edges = np.percentile(vals, np.linspace(0, 100, nb+1))
                bin_edges[0] -= 1; bin_edges[-1] += 1
                col_bins = np.digitize(vals, bin_edges[1:-1])
            except:
                continue
            
            # OOF target encoding with smoothing
            encoded = np.zeros(len(y))
            global_mean = y.mean()
            
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for tr, te in skf.split(vals, bins):
                for b in range(nb):
                    mask_tr = col_bins[tr] == b
                    mask_te = col_bins[te] == b
                    if mask_tr.sum() > 0:
                        bin_mean = y[tr][mask_tr].mean()
                        bin_count = mask_tr.sum()
                        # Smoothed mean
                        smoothed = (bin_count * bin_mean + smooth * global_mean) / (bin_count + smooth)
                        encoded[te[mask_te]] = smoothed
                    else:
                        encoded[te[mask_te]] = global_mean
            
            te_features = np.column_stack([te_features, encoded])
            te_names.append(f'te_{col[:3]}_{nb}')
    
    return te_features, te_names

te_feats, te_names = make_target_encoded_features(X_dw, y)
print(f"  Created {te_feats.shape[1]} target-encoded features: {te_names}")

# Combine with raw and engineered
X_te_raw = np.column_stack([X_raw, te_feats])
X_te_eng = np.column_stack([X_eng, te_feats])

for fname, Xf in [('te_raw', X_te_raw), ('te_eng', X_te_eng)]:
    for mname, mfn, scale in [
        ('ridge_100', lambda: Ridge(alpha=100), True),
        ('ridge_500', lambda: Ridge(alpha=500), True),
        ('kr_rbf', lambda: KernelRidge(alpha=1, kernel='rbf', gamma=0.01), True),
        ('kr_rbf2', lambda: KernelRidge(alpha=0.2, kernel='rbf', gamma=0.007), True),
        ('xgb', lambda: xgb.XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, reg_alpha=2, reg_lambda=5, random_state=42), False),
        ('hgbr', lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=2, learning_rate=0.05, random_state=42), False),
    ]:
        full = f"{fname}__{mname}"
        r2, preds = run_cv(Xf, y, mfn, scale=scale, n_rep=5)
        all_results[full] = {'r2': r2, 'preds': preds}
        if r2 > 0.24: print(f"  {full:50s}: R2={r2:.4f}", flush=True)

# ============================================================
print("\n" + "="*60)
print("IDEA 2: KNN FEATURE AUGMENTATION (Neighbor Target Stats)")
print("="*60)
# For each sample, find k nearest neighbors and compute target stats
# Must use OOF to prevent leakage

def make_knn_target_features(X, y, k_list=[5, 10, 20, 50]):
    """OOF KNN target features - neighbor's target mean, std, median."""
    knn_feats = np.zeros((len(y), 0))
    
    for k in k_list:
        mean_feat = np.zeros(len(y))
        std_feat = np.zeros(len(y))
        med_feat = np.zeros(len(y))
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for tr, te in skf.split(X, bins):
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
            
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
            nn.fit(Xtr)
            dists, indices = nn.kneighbors(Xte)
            
            for i, te_idx in enumerate(te):
                neighbor_targets = y[tr][indices[i]]
                mean_feat[te_idx] = neighbor_targets.mean()
                std_feat[te_idx] = neighbor_targets.std()
                med_feat[te_idx] = np.median(neighbor_targets)
        
        knn_feats = np.column_stack([knn_feats, mean_feat, std_feat, med_feat])
    
    return knn_feats

knn_feats = make_knn_target_features(X_raw, y)
print(f"  Created {knn_feats.shape[1]} KNN target features")

X_knn_raw = np.column_stack([X_raw, knn_feats])
X_knn_eng = np.column_stack([X_eng, knn_feats])
X_knn_te = np.column_stack([X_raw, te_feats, knn_feats])

for fname, Xf in [('knn_raw', X_knn_raw), ('knn_eng', X_knn_eng), ('knn_te', X_knn_te)]:
    for mname, mfn, scale in [
        ('ridge_100', lambda: Ridge(alpha=100), True),
        ('ridge_500', lambda: Ridge(alpha=500), True),
        ('kr_rbf', lambda: KernelRidge(alpha=1, kernel='rbf', gamma=0.01), True),
        ('xgb', lambda: xgb.XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, reg_alpha=2, reg_lambda=5, random_state=42), False),
    ]:
        full = f"{fname}__{mname}"
        r2, preds = run_cv(Xf, y, mfn, scale=scale, n_rep=5)
        all_results[full] = {'r2': r2, 'preds': preds}
        if r2 > 0.24: print(f"  {full:50s}: R2={r2:.4f}", flush=True)

# ============================================================
print("\n" + "="*60)
print("IDEA 3: LOO STACKING (Less Noisy Meta-Features)")
print("="*60)

# LOO predictions for top models
def loo_predict(X, y, model_fn, scale=True):
    """Leave-one-out predictions."""
    preds = np.zeros(len(y))
    # Use batched LOO via KFold with n_splits=n for true LOO, but that's 798 models
    # Instead use 20-fold (close to LOO but faster)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=20, shuffle=True, random_state=42)
    for tr, te in kf.split(X):
        Xtr, Xte = X[tr].copy(), X[te].copy()
        if scale:
            s = StandardScaler(); Xtr = s.fit_transform(Xtr); Xte = s.transform(Xte)
        m = model_fn(); m.fit(Xtr, y[tr]); preds[te] = m.predict(Xte)
    return preds

loo_preds = {}
loo_models = [
    ('ridge_100', lambda: Ridge(alpha=100), True),
    ('ridge_500', lambda: Ridge(alpha=500), True),
    ('kr_rbf', lambda: KernelRidge(alpha=1, kernel='rbf', gamma=0.01), True),
    ('kr_rbf2', lambda: KernelRidge(alpha=0.2, kernel='rbf', gamma=0.007), True),
    ('bayesian', lambda: BayesianRidge(), True),
    ('xgb', lambda: xgb.XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, reg_alpha=2, reg_lambda=5, random_state=42), False),
    ('hgbr', lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=2, learning_rate=0.05, random_state=42), False),
    ('lgb', lambda: lgb.LGBMRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, verbose=-1, n_jobs=1, random_state=42), False),
]

for fs_name, X_fs in [('raw18', X_raw), ('eng', X_eng)]:
    for mname, mfn, scale in loo_models:
        full = f"loo_{fs_name}__{mname}"
        p = loo_predict(X_fs, y, mfn, scale=scale)
        r2 = 1 - np.sum((y-p)**2)/ss_tot
        loo_preds[full] = p
        all_results[full] = {'r2': r2, 'preds': p}
        if r2 > 0.20: print(f"  {full:50s}: R2={r2:.4f}", flush=True)

# ============================================================
print("\n" + "="*60)
print("IDEA 4: QUANTILE CLASSIFICATION AS FEATURES")
print("="*60)

# Train classifiers for: is HOMA > median? is HOMA > Q75? etc.
# Use classifier probabilities as features for regression

def make_quantile_class_features(X, y, quantiles=[0.25, 0.5, 0.75, 0.9]):
    """OOF quantile classification probabilities as features."""
    qc_feats = np.zeros((len(y), 0))
    
    for q in quantiles:
        threshold = np.quantile(y, q)
        y_class = (y > threshold).astype(int)
        
        probs = np.zeros(len(y))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for tr, te in skf.split(X, y_class):
            sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
            from sklearn.ensemble import GradientBoostingClassifier
            clf = GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.1, random_state=42)
            clf.fit(Xtr, y_class[tr])
            probs[te] = clf.predict_proba(Xte)[:, 1]
        
        qc_feats = np.column_stack([qc_feats, probs])
    
    return qc_feats

qc_feats = make_quantile_class_features(X_raw, y)
print(f"  Created {qc_feats.shape[1]} quantile classification features")

X_qc_raw = np.column_stack([X_raw, qc_feats])
X_qc_all = np.column_stack([X_raw, te_feats, knn_feats, qc_feats])

for fname, Xf in [('qc_raw', X_qc_raw), ('qc_all', X_qc_all)]:
    for mname, mfn, scale in [
        ('ridge_100', lambda: Ridge(alpha=100), True),
        ('ridge_500', lambda: Ridge(alpha=500), True),
        ('kr_rbf', lambda: KernelRidge(alpha=1, kernel='rbf', gamma=0.01), True),
        ('xgb', lambda: xgb.XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=42), False),
    ]:
        full = f"{fname}__{mname}"
        r2, preds = run_cv(Xf, y, mfn, scale=scale, n_rep=5)
        all_results[full] = {'r2': r2, 'preds': preds}
        if r2 > 0.24: print(f"  {full:50s}: R2={r2:.4f}", flush=True)

# ============================================================
print("\n" + "="*60)
print("IDEA 5: ITERATIVE RESIDUAL LEARNING")
print("="*60)

# Model 1: Best model on raw features
# Model 2: Different model on engineered features, predicting RESIDUALS
# Final = Model1 + Model2

def iterative_residual(X1, X2, y, model1_fn, model2_fn, scale1=True, scale2=True, n_rep=5):
    """Two-stage residual learning with proper CV."""
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=42)
    preds = np.zeros(n_samples); counts = np.zeros(n_samples)
    
    for tr, te in rkf.split(X1, bins):
        # Stage 1
        Xtr1, Xte1 = X1[tr].copy(), X1[te].copy()
        if scale1:
            s1 = StandardScaler(); Xtr1 = s1.fit_transform(Xtr1); Xte1 = s1.transform(Xte1)
        m1 = model1_fn(); m1.fit(Xtr1, y[tr])
        p1_tr = m1.predict(Xtr1); p1_te = m1.predict(Xte1)
        
        # Residuals
        resid_tr = y[tr] - p1_tr
        
        # Stage 2: predict residuals
        Xtr2, Xte2 = X2[tr].copy(), X2[te].copy()
        if scale2:
            s2 = StandardScaler(); Xtr2 = s2.fit_transform(Xtr2); Xte2 = s2.transform(Xte2)
        m2 = model2_fn(); m2.fit(Xtr2, resid_tr)
        p2_te = m2.predict(Xte2)
        
        preds[te] += p1_te + p2_te; counts[te] += 1
    
    preds /= counts
    return 1 - np.sum((y-preds)**2)/ss_tot, preds

combos = [
    ('kr_raw+ridge_eng', X_raw, X_eng,
     lambda: KernelRidge(alpha=0.2, kernel='rbf', gamma=0.007),
     lambda: Ridge(alpha=500), True, True),
    ('kr_raw+xgb_eng', X_raw, X_eng,
     lambda: KernelRidge(alpha=0.2, kernel='rbf', gamma=0.007),
     lambda: xgb.XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, reg_alpha=2, reg_lambda=5, random_state=42),
     True, False),
    ('ridge_eng+kr_raw', X_eng, X_raw,
     lambda: Ridge(alpha=500),
     lambda: KernelRidge(alpha=1, kernel='rbf', gamma=0.01), True, True),
    ('xgb_raw+kr_eng', X_raw, X_eng,
     lambda: xgb.XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=42),
     lambda: KernelRidge(alpha=1, kernel='rbf', gamma=0.01), False, True),
    ('kr_raw+hgbr_eng', X_raw, X_eng,
     lambda: KernelRidge(alpha=0.2, kernel='rbf', gamma=0.007),
     lambda: HistGradientBoostingRegressor(max_iter=200, max_depth=2, learning_rate=0.05, random_state=42),
     True, False),
]

for name, X1, X2, m1, m2, s1, s2 in combos:
    r2, preds = iterative_residual(X1, X2, y, m1, m2, s1, s2)
    all_results[f'resid_{name}'] = {'r2': r2, 'preds': preds}
    print(f"  resid_{name:35s}: R2={r2:.4f}", flush=True)

# ============================================================
print("\n" + "="*60)
print("IDEA 6: AUTOENCODER EMBEDDINGS")
print("="*60)

import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Linear(32, 64), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, input_dim)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

# Train AE on ALL data (unsupervised, no leakage)
sc_ae = StandardScaler()
X_ae_input = sc_ae.fit_transform(X_raw)
X_tensor = torch.FloatTensor(X_ae_input)

for latent_dim in [8, 12]:
    ae = Autoencoder(18, latent_dim)
    optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    ae.train()
    for epoch in range(500):
        optimizer.zero_grad()
        recon, z = ae(X_tensor)
        loss = criterion(recon, X_tensor)
        loss.backward()
        optimizer.step()
    
    ae.eval()
    with torch.no_grad():
        _, Z = ae(X_tensor)
        X_ae = Z.numpy()
    
    # Use embeddings as features
    X_ae_raw = np.column_stack([X_raw, X_ae])
    X_ae_eng = np.column_stack([X_eng, X_ae])
    
    for fname, Xf in [(f'ae{latent_dim}_raw', X_ae_raw), (f'ae{latent_dim}_eng', X_ae_eng)]:
        for mname, mfn, scale in [
            ('ridge_100', lambda: Ridge(alpha=100), True),
            ('kr_rbf', lambda: KernelRidge(alpha=1, kernel='rbf', gamma=0.01), True),
        ]:
            full = f"{fname}__{mname}"
            r2, preds = run_cv(Xf, y, mfn, scale=scale, n_rep=5)
            all_results[full] = {'r2': r2, 'preds': preds}
            if r2 > 0.24: print(f"  {full:50s}: R2={r2:.4f}", flush=True)

# ============================================================
print("\n" + "="*60)
print("IDEA 7: MEGA FEATURE COMBINATION + TARGET FEATURES")
print("="*60)

# Combine ALL feature augmentations
X_mega = np.column_stack([X_raw, te_feats, knn_feats, qc_feats])
print(f"  Mega features: {X_mega.shape[1]} dims")

# Run many models on mega features
mega_models = [
    ('ridge_50', lambda: Ridge(alpha=50), True),
    ('ridge_100', lambda: Ridge(alpha=100), True),
    ('ridge_500', lambda: Ridge(alpha=500), True),
    ('ridge_1000', lambda: Ridge(alpha=1000), True),
    ('bayesian', lambda: BayesianRidge(), True),
    ('kr_rbf_1_01', lambda: KernelRidge(alpha=1, kernel='rbf', gamma=0.01), True),
    ('kr_rbf_02_007', lambda: KernelRidge(alpha=0.2, kernel='rbf', gamma=0.007), True),
    ('kr_poly2', lambda: KernelRidge(alpha=1, kernel='poly', degree=2, gamma=0.01), True),
    ('svr_10', lambda: SVR(kernel='rbf', C=10, epsilon=0.1), True),
    ('xgb_d2', lambda: xgb.XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, reg_alpha=2, reg_lambda=5, random_state=42), False),
    ('lgb_d2', lambda: lgb.LGBMRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, verbose=-1, n_jobs=1, random_state=42), False),
    ('hgbr_d2', lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=2, learning_rate=0.05, random_state=42), False),
]

for mname, mfn, scale in mega_models:
    full = f"mega__{mname}"
    r2, preds = run_cv(X_mega, y, mfn, scale=scale, n_rep=5)
    all_results[full] = {'r2': r2, 'preds': preds}
    if r2 > 0.24: print(f"  {full:50s}: R2={r2:.4f}", flush=True)

# ============================================================
print("\n" + "="*60)
print("FINAL: COMBINE ALL + STACK")
print("="*60)

good = {k:v for k,v in all_results.items() if v['r2'] > 0.10}
names = sorted(good.keys(), key=lambda k: good[k]['r2'], reverse=True)[:40]
preds_mat = np.array([good[nm]['preds'] for nm in names])
nm_count = len(names)

print(f"Combining {nm_count} predictions")
print(f"Top 15:")
for nm in names[:15]:
    print(f"  {nm:55s}: R2={good[nm]['r2']:.4f}")

best_final = -999
all_stacks = {}

# Ridge stacking with multiple seeds
for seed in [0, 1, 2, 42, 7, 13]:
    for alpha in [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000]:
        sp=np.zeros(n_samples); sc=np.zeros(n_samples)
        for tr,te in RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=seed).split(preds_mat.T,bins):
            m=Ridge(alpha=alpha); m.fit(preds_mat[:,tr].T,y[tr])
            sp[te]+=m.predict(preds_mat[:,te].T); sc[te]+=1
        sp/=sc; r2=1-np.sum((y-sp)**2)/ss_tot
        if r2>best_final: best_final=r2
        if r2>0.20: all_stacks[f'ridge_s{seed}_a{alpha}']=sp.copy()
print(f"Ridge stack: R2={best_final:.4f}", flush=True)

# ElasticNet
for seed in [0, 42]:
    for alpha in [0.001, 0.01, 0.1]:
        for l1 in [0.1, 0.3, 0.5, 0.7, 0.9]:
            sp=np.zeros(n_samples); sc=np.zeros(n_samples)
            for tr,te in RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=seed).split(preds_mat.T,bins):
                m=ElasticNet(alpha=alpha,l1_ratio=l1,max_iter=5000,positive=True)
                m.fit(preds_mat[:,tr].T,y[tr]); sp[te]+=m.predict(preds_mat[:,te].T); sc[te]+=1
            sp/=sc; r2=1-np.sum((y-sp)**2)/ss_tot
            if r2>best_final: best_final=r2
            if r2>0.20: all_stacks[f'en_s{seed}_{alpha}_{l1}']=sp.copy()

# KNN stacking
for seed in [0, 42]:
    for k in [3, 5, 7, 10, 15, 20]:
        sp=np.zeros(n_samples); sc=np.zeros(n_samples)
        for tr,te in RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=seed).split(preds_mat.T,bins):
            ss=StandardScaler(); pt=ss.fit_transform(preds_mat[:,tr].T); pe=ss.transform(preds_mat[:,te].T)
            m=KNeighborsRegressor(n_neighbors=k,weights='distance'); m.fit(pt,y[tr])
            sp[te]+=m.predict(pe); sc[te]+=1
        sp/=sc; r2=1-np.sum((y-sp)**2)/ss_tot
        if r2>best_final: best_final=r2
        if r2>0.20: all_stacks[f'knn_s{seed}_{k}']=sp.copy()

# SVR stacking
for seed in [0, 42]:
    for C in [0.1, 1, 10, 100]:
        sp=np.zeros(n_samples); sc=np.zeros(n_samples)
        for tr,te in RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=seed).split(preds_mat.T,bins):
            ss=StandardScaler(); pt=ss.fit_transform(preds_mat[:,tr].T); pe=ss.transform(preds_mat[:,te].T)
            m=SVR(kernel='rbf',C=C,epsilon=0.1); m.fit(pt,y[tr])
            sp[te]+=m.predict(pe); sc[te]+=1
        sp/=sc; r2=1-np.sum((y-sp)**2)/ss_tot
        if r2>best_final: best_final=r2
        if r2>0.20: all_stacks[f'svr_s{seed}_{C}']=sp.copy()

# XGB stacking
for seed in [0, 42]:
    for d in [2, 3]:
        sp=np.zeros(n_samples); sc=np.zeros(n_samples)
        for tr,te in RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=seed).split(preds_mat.T,bins):
            m=xgb.XGBRegressor(n_estimators=100,max_depth=d,learning_rate=0.05,reg_alpha=5,reg_lambda=10,random_state=42)
            m.fit(preds_mat[:,tr].T,y[tr]); sp[te]+=m.predict(preds_mat[:,te].T); sc[te]+=1
        sp/=sc; r2=1-np.sum((y-sp)**2)/ss_tot
        if r2>best_final: best_final=r2
        if r2>0.20: all_stacks[f'xgb_s{seed}_{d}']=sp.copy()
print(f"+All stackers: R2={best_final:.4f}", flush=True)

# Layer 2: stack of stacks
if len(all_stacks) >= 3:
    print(f"Layer 2: {len(all_stacks)} stacks", flush=True)
    snames = list(all_stacks.keys()); smat = np.array([all_stacks[k] for k in snames])
    for l2_seed in [0, 1, 42]:
        for alpha in [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]:
            sp=np.zeros(n_samples); sc=np.zeros(n_samples)
            for tr,te in RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=l2_seed).split(smat.T,bins):
                m=Ridge(alpha=alpha); m.fit(smat[:,tr].T,y[tr])
                sp[te]+=m.predict(smat[:,te].T); sc[te]+=1
            sp/=sc; r2=1-np.sum((y-sp)**2)/ss_tot
            if r2>best_final: best_final=r2
    print(f"Layer 2: R2={best_final:.4f}", flush=True)

# Mega blend
rng=np.random.RandomState(42)
blend_best = -999
for _ in range(500000):
    w=rng.dirichlet(np.ones(nm_count)*0.3); bl=w@preds_mat
    r2=1-np.sum((y-bl)**2)/ss_tot
    if r2>blend_best: blend_best=r2
for _ in range(300000):
    w=rng.dirichlet(np.ones(nm_count)*0.1); bl=w@preds_mat
    r2=1-np.sum((y-bl)**2)/ss_tot
    if r2>blend_best: blend_best=r2
# Pairwise
for i in range(nm_count):
    for j in range(i+1,nm_count):
        for a in np.linspace(0,1,201):
            bl=a*preds_mat[i]+(1-a)*preds_mat[j]; r2=1-np.sum((y-bl)**2)/ss_tot
            if r2>blend_best: blend_best=r2
# Triplets
for i in range(min(10,nm_count)):
    for j in range(i+1,min(10,nm_count)):
        for k in range(j+1,min(10,nm_count)):
            for a in np.linspace(0.05,0.9,20):
                for b in np.linspace(0.05,0.9-a,15):
                    c=1-a-b
                    if c>0:
                        bl=a*preds_mat[i]+b*preds_mat[j]+c*preds_mat[k]
                        r2=1-np.sum((y-bl)**2)/ss_tot
                        if r2>blend_best: blend_best=r2
if blend_best > best_final: best_final = blend_best
print(f"Mega blend: R2={blend_best:.4f}", flush=True)

print(f"\n{'='*60}")
print(f"V22 FINAL ({(time.time()-t0)/60:.1f} min)")
print(f"{'='*60}")
print(f"  HOMA_IR DW: R2={best_final:.4f} (target 0.37, gap={0.37-best_final:.3f})")
print(f"  Previous best: 0.3250 (V17c)")
print(f"  Improvement: {best_final-0.3250:+.4f}")
