#!/usr/bin/env python3
"""V15: SMOTE augmentation + self-training + PyTorch FeatureGatedBlock."""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import (GradientBoostingRegressor, HistGradientBoostingRegressor,
                               RandomForestRegressor, ExtraTreesRegressor)
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
    X = X.copy()
    X['bmi_sq'] = X['bmi'] ** 2
    X['bmi_cubed'] = X['bmi'] ** 3
    X['age_sq'] = X['age'] ** 2
    X['bmi_age'] = X['bmi'] * X['age']
    X['age_sex'] = X['age'] * X['sex_num']
    X['bmi_sex'] = X['bmi'] * X['sex_num']
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
    rhr = 'Resting Heart Rate (mean)'
    hrv = 'HRV (mean)'
    steps = 'STEPS (mean)'
    slp = 'SLEEP Duration (mean)'
    azm = 'AZM Weekly (mean)'
    X['rhr_bmi'] = X[rhr] * X['bmi']
    X['rhr_bmi_sq'] = X[rhr] * X['bmi'] ** 2
    X['rhr_hrv'] = X[rhr] / X[hrv].clip(lower=1)
    X['steps_sleep'] = X[steps] * X[slp]
    X['bmi_rhr_sq'] = X['bmi'] * X[rhr] ** 2
    X['low_azm_obese'] = (X[azm] < X[azm].median()).astype(int) * X['bmi']
    X['cardio_fitness'] = X[hrv] * X[steps] / X[rhr].clip(lower=1)
    X['sleep_quality'] = X[slp] * X[hrv]
    return X

def make_bins(y, n_bins=5):
    return pd.qcut(y, n_bins, labels=False, duplicates='drop')

def feature_selection(X, y, n_top=35):
    gbr = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42, subsample=0.8)
    gbr.fit(X, y)
    imp = pd.Series(gbr.feature_importances_, index=X.columns).sort_values(ascending=False)
    return imp.head(n_top).index.tolist()

# ============================================================
# 1. SMOTE-like augmentation for regression
# ============================================================
def smote_regression(X_train, y_train, threshold_percentile=75, n_synthetic=100, k=5, rng=None):
    """Generate synthetic samples for high-target-value region using SMOTE-like interpolation."""
    if rng is None:
        rng = np.random.RandomState(42)
    
    threshold = np.percentile(y_train, threshold_percentile)
    high_mask = y_train >= threshold
    X_high = X_train[high_mask]
    y_high = y_train[high_mask]
    
    if len(X_high) < k + 1:
        return X_train, y_train
    
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(X_high)
    
    synthetic_X = []
    synthetic_y = []
    
    for _ in range(n_synthetic):
        idx = rng.randint(len(X_high))
        neighbors = nn.kneighbors(X_high[idx:idx+1], return_distance=False)[0][1:]
        nn_idx = rng.choice(neighbors)
        
        lam = rng.random()
        new_x = X_high[idx] + lam * (X_high[nn_idx] - X_high[idx])
        new_y = y_high[idx] + lam * (y_high[nn_idx] - y_high[idx])
        
        synthetic_X.append(new_x)
        synthetic_y.append(new_y)
    
    X_aug = np.vstack([X_train, np.array(synthetic_X)])
    y_aug = np.concatenate([y_train, np.array(synthetic_y)])
    
    return X_aug, y_aug

# ============================================================
# 2. Self-training (pseudo-labeling)
# ============================================================
def self_train_predict(X_train, y_train, X_test, model_fn, n_iters=3, confidence_percentile=50):
    """Self-training: use confident predictions on test as pseudo-labels."""
    model = model_fn()
    model.fit(X_train, y_train)
    
    for iteration in range(n_iters):
        preds = model.predict(X_test)
        
        # Use all test predictions as pseudo-labels with decreasing weight
        weight = 0.3 / (iteration + 1)
        
        # Combine
        X_combined = np.vstack([X_train, X_test])
        y_combined = np.concatenate([y_train, preds])
        
        # Sample weights: real data = 1.0, pseudo = weight
        sample_weights = np.concatenate([np.ones(len(y_train)), np.full(len(preds), weight)])
        
        model = model_fn()
        model.fit(X_combined, y_combined, sample_weight=sample_weights)
    
    return model.predict(X_test)

# ============================================================
# 3. PyTorch FeatureGatedBlock
# ============================================================
class FeatureGatedBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.transform = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim))
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.act(x + self.drop(self.transform(x * self.gate(x))))

class WearMENet(nn.Module):
    def __init__(self, input_dim, hidden=128, blocks=3, drop=0.15):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(drop))
        self.blocks = nn.ModuleList([FeatureGatedBlock(hidden, drop) for _ in range(blocks)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden//2), nn.GELU(), nn.Dropout(drop*0.5), nn.Linear(hidden//2, 1))
    
    def forward(self, x):
        h = self.proj(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h).squeeze(-1)

def train_pytorch(X_tr, y_tr, X_te, hidden=128, blocks=3, drop=0.15, lr=1e-3, epochs=300, bs=64, patience=30):
    X_t = torch.FloatTensor(X_tr)
    y_t = torch.FloatTensor(y_tr)
    X_v = torch.FloatTensor(X_te)
    
    model = WearMENet(X_tr.shape[1], hidden, blocks, drop)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.01)
    
    best_loss = float('inf')
    best_state = None
    wait = 0
    
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=bs, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        for bx, by in loader:
            opt.zero_grad()
            nn.MSELoss()(model(bx), by).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        
        model.eval()
        with torch.no_grad():
            train_loss = nn.MSELoss()(model(X_t), y_t).item()
        
        if train_loss < best_loss:
            best_loss = train_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(X_v).numpy()

# ============================================================
# CV FRAMEWORK
# ============================================================
def comprehensive_cv(X, y, target_name, target_r2, log_target=False):
    """Run all approaches: base models, SMOTE, self-training, PyTorch."""
    bins = make_bins(y)
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    n = len(y)
    
    approaches = {}
    
    # --- Base models (no augmentation) ---
    base_configs = [
        ('xgb_d5', lambda: xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1, random_state=42)),
        ('xgb_d6', lambda: xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.7, colsample_bytree=0.7, reg_alpha=1, reg_lambda=3, random_state=42)),
        ('hgbr', lambda: HistGradientBoostingRegressor(max_iter=500, max_depth=5, learning_rate=0.05, random_state=42)),
        ('lgb_dart', lambda: lgb.LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.1, boosting_type='dart', verbose=-1, n_jobs=2, random_state=42)),
        ('rf', lambda: RandomForestRegressor(n_estimators=500, max_depth=8, max_features=0.5, random_state=42, n_jobs=2)),
        ('et', lambda: ExtraTreesRegressor(n_estimators=500, max_depth=8, max_features=0.5, random_state=42, n_jobs=2)),
    ]
    
    print(f"\n  --- Base models ---")
    for name, model_fn in base_configs:
        for use_log in ([False, True] if log_target else [False]):
            mname = name + ('_log' if use_log else '')
            t0 = time.time()
            preds = np.zeros(n); counts = np.zeros(n)
            for tr_idx, te_idx in rkf.split(X, bins):
                X_tr, X_te = X[tr_idx], X[te_idx]
                y_tr = y[tr_idx]
                y_use = np.log1p(y_tr) if use_log else y_tr
                m = model_fn(); m.fit(X_tr, y_use)
                p = m.predict(X_te)
                if use_log: p = np.expm1(p)
                preds[te_idx] += p; counts[te_idx] += 1
            preds /= counts
            r2 = 1 - np.sum((y - preds)**2) / np.sum((y - y.mean())**2)
            approaches[mname] = preds
            print(f"    {mname:25s}: R2={r2:.4f} ({time.time()-t0:.1f}s)")
    
    # --- SMOTE augmented models ---
    print(f"\n  --- SMOTE augmented ---")
    for name, model_fn in [('xgb_d5', base_configs[0][1]), ('hgbr', base_configs[2][1])]:
        for use_log in ([False, True] if log_target else [False]):
            for n_synth in [50, 100, 200]:
                mname = f'smote_{name}_{n_synth}' + ('_log' if use_log else '')
                t0 = time.time()
                preds = np.zeros(n); counts = np.zeros(n)
                for tr_idx, te_idx in rkf.split(X, bins):
                    X_tr, X_te = X[tr_idx], X[te_idx]
                    y_tr = y[tr_idx]
                    y_use = np.log1p(y_tr) if use_log else y_tr
                    X_aug, y_aug = smote_regression(X_tr, y_use, threshold_percentile=75, n_synthetic=n_synth)
                    m = model_fn(); m.fit(X_aug, y_aug)
                    p = m.predict(X_te)
                    if use_log: p = np.expm1(p)
                    preds[te_idx] += p; counts[te_idx] += 1
                preds /= counts
                r2 = 1 - np.sum((y - preds)**2) / np.sum((y - y.mean())**2)
                approaches[mname] = preds
                print(f"    {mname:35s}: R2={r2:.4f} ({time.time()-t0:.1f}s)")
    
    # --- Self-training ---
    print(f"\n  --- Self-training ---")
    for name, model_fn in [('xgb_d5', base_configs[0][1])]:
        mname = f'self_{name}'
        t0 = time.time()
        preds = np.zeros(n); counts = np.zeros(n)
        for tr_idx, te_idx in rkf.split(X, bins):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr = y[tr_idx]
            p = self_train_predict(X_tr, y_tr, X_te, model_fn, n_iters=3)
            preds[te_idx] += p; counts[te_idx] += 1
        preds /= counts
        r2 = 1 - np.sum((y - preds)**2) / np.sum((y - y.mean())**2)
        approaches[mname] = preds
        print(f"    {mname:25s}: R2={r2:.4f} ({time.time()-t0:.1f}s)")
    
    # --- PyTorch ---
    print(f"\n  --- PyTorch FeatureGatedBlock ---")
    for hidden, blocks, drop, lr_val in [(128, 3, 0.15, 1e-3), (256, 4, 0.20, 5e-4)]:
        mname = f'pytorch_{hidden}_{blocks}'
        t0 = time.time()
        preds = np.zeros(n); counts = np.zeros(n)
        # Only 1 repeat for speed
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for tr_idx, te_idx in skf.split(X, bins):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr = y[tr_idx]
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            p = train_pytorch(X_tr_s, y_tr, X_te_s, hidden, blocks, drop, lr_val)
            preds[te_idx] += p; counts[te_idx] += 1
        preds /= counts
        r2 = 1 - np.sum((y - preds)**2) / np.sum((y - y.mean())**2)
        approaches[mname] = preds
        print(f"    {mname:25s}: R2={r2:.4f} ({time.time()-t0:.1f}s)")
    
    # --- Mega blend ---
    print(f"\n  --- Mega blend ---")
    ss_tot = np.sum((y - y.mean()) ** 2)
    names = sorted(approaches.keys(), key=lambda k: 1 - np.sum((y - approaches[k])**2)/ss_tot, reverse=True)[:12]
    preds_mat = np.array([approaches[nm] for nm in names])
    
    best_r2 = -999
    best_w = None
    rng = np.random.RandomState(42)
    
    for _ in range(300000):
        w = rng.dirichlet(np.ones(len(names)))
        blend = w @ preds_mat
        r2 = 1 - np.sum((y - blend) ** 2) / ss_tot
        if r2 > best_r2:
            best_r2 = r2
            best_w = w
    
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            for alpha in np.linspace(0, 1, 201):
                blend = alpha * preds_mat[i] + (1-alpha) * preds_mat[j]
                r2 = 1 - np.sum((y - blend) ** 2) / ss_tot
                if r2 > best_r2:
                    best_r2 = r2
                    best_w = np.zeros(len(names))
                    best_w[i] = alpha
                    best_w[j] = 1-alpha
    
    print(f"  Mega blend: R2={best_r2:.4f}")
    for nm, w in zip(names, best_w):
        if w > 0.01:
            print(f"    {nm}: {w:.3f}")
    
    # Ridge stack
    n_s = len(y)
    oof_mat = np.column_stack([approaches[nm] for nm in names])
    stack_preds = np.zeros(n_s); stack_counts = np.zeros(n_s)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, te_idx in skf.split(oof_mat, bins):
        ridge = RidgeCV(alphas=[0.1, 1, 10, 100, 1000])
        ridge.fit(oof_mat[tr_idx], y[tr_idx])
        stack_preds[te_idx] = ridge.predict(oof_mat[te_idx])
        stack_counts[te_idx] += 1
    stack_preds /= stack_counts
    r2_stack = 1 - np.sum((y - stack_preds) ** 2) / ss_tot
    print(f"  Ridge stack: R2={r2_stack:.4f}")
    
    final = max(best_r2, r2_stack)
    print(f"\n  >>> {target_name} BEST: R2={final:.4f} (target {target_r2}, gap={target_r2-final:.3f})")
    return final

# ============================================================
# MAIN
# ============================================================
print("=" * 60)
print("V15: SMOTE + SELF-TRAINING + PYTORCH")
print("=" * 60)

# HOMA_IR ALL
y_homa = df['True_HOMA_IR'].values
mask = ~np.isnan(y_homa)
X_eng = engineer_all(X_raw[mask].reset_index(drop=True))
X_eng = X_eng.fillna(X_eng.median())
y_homa = y_homa[mask]
top35 = feature_selection(X_eng, y_homa, 35)
X_sel = X_eng[top35].values

print("\nHOMA_IR ALL | Target: 0.65")
best_homa_all = comprehensive_cv(X_sel, y_homa, 'HOMA_IR ALL', 0.65, log_target=True)

# hba1c ALL
y_hba1c = df['True_hba1c'].values
mask2 = ~np.isnan(y_hba1c)
X_eng2 = engineer_all(X_raw[mask2].reset_index(drop=True))
X_eng2 = X_eng2.fillna(X_eng2.median())
y_hba1c = y_hba1c[mask2]
top30_h = feature_selection(X_eng2, y_hba1c, 30)
X_sel2 = X_eng2[top30_h].values

print("\nhba1c ALL | Target: 0.85")
best_hba1c_all = comprehensive_cv(X_sel2, y_hba1c, 'hba1c ALL', 0.85, log_target=False)

# Summary
print("\n" + "=" * 60)
print("V15 FINAL SUMMARY")
print("=" * 60)
print(f"  HOMA_IR ALL: R2={best_homa_all:.4f} (target 0.65, gap={0.65-best_homa_all:.3f})")
print(f"  hba1c ALL:   R2={best_hba1c_all:.4f} (target 0.85, gap={0.85-best_hba1c_all:.3f})")
print("\nDone.")
