#!/usr/bin/env python3
"""
Lean trainer - memory efficient, one experiment at a time.
Runs on CPU, small batches, explicit garbage collection.
"""
import gc, warnings, json, time, math, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from copy import deepcopy

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)
DEVICE = torch.device('cpu')

def load_data():
    df = pd.read_csv('data.csv', skiprows=[0])
    sex_map = {'Female': 0, 'Male': 1}
    df['sex_num'] = df['sex'].map(lambda x: sex_map.get(x, 0.5))
    df['is_male'] = (df['sex'] == 'Male').astype(float)
    
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    
    # Core engineered features
    df['trig_hdl'] = df['triglycerides'] / (df['hdl'] + 0.1)
    df['glucose_bmi'] = df['glucose'] * df['bmi']
    df['glucose_trig'] = df['glucose'] * df['triglycerides'] / 1000
    df['ldl_hdl'] = df['ldl'] / (df['hdl'] + 0.1)
    df['glucose_sq'] = df['glucose'] ** 2 / 1000
    df['bmi_sq'] = df['bmi'] ** 2
    df['log_trig'] = np.log1p(df['triglycerides'])
    df['log_glucose'] = np.log1p(df['glucose'])
    df['glucose_proxy'] = df['glucose'] * df['trig_hdl']
    df['bmi_age'] = df['bmi'] * df['age']
    df['rhr_hrv'] = df['Resting Heart Rate (mean)'] / (df['HRV (mean)'] + 0.1)
    df['ast_alt'] = df['ast'] / (df['alt'] + 0.1)
    df['bun_creat'] = df['bun'] / (df['creatinine'] + 0.01)
    df['nlr'] = df['absolute_neutrophils'] / (df['absolute_lymphocytes'] + 0.1)
    df['log_crp'] = np.log1p(df['crp'])
    df['non_hdl_hdl'] = df['non hdl'] / (df['hdl'] + 0.1)
    df['ggt_alt'] = df['ggt'] / (df['alt'] + 0.1)
    df['hr_cv'] = df['Resting Heart Rate (std)'] / (df['Resting Heart Rate (mean)'] + 0.1)
    df['hrv_cv'] = df['HRV (std)'] / (df['HRV (mean)'] + 0.1)
    df['steps_cv'] = df['STEPS (std)'] / (df['STEPS (mean)'] + 1)
    df['sleep_cv'] = df['SLEEP Duration (std)'] / (df['SLEEP Duration (mean)'] + 1)
    df['activity'] = df['STEPS (mean)'] * df['AZM Weekly (mean)'] / 1e6
    df['steps_sleep'] = df['STEPS (mean)'] / (df['SLEEP Duration (mean)'] + 1)
    
    # Additional metabolic features
    df['met_syndrome_score'] = (
        (df['bmi'] > 30).astype(float) + 
        (df['triglycerides'] > 150).astype(float) + 
        (df['glucose'] > 100).astype(float) + 
        (df['hdl'] < 40).astype(float)
    )
    df['glucose_hdl'] = df['glucose'] / (df['hdl'] + 0.1)
    df['trig_glucose_idx'] = np.log(df['triglycerides'] * df['glucose'] / 2)  # TyG index
    df['bmi_trig'] = df['bmi'] * df['triglycerides'] / 100
    df['age_glucose'] = df['age'] * df['glucose'] / 100
    df['crp_bmi'] = df['crp'] * df['bmi']
    df['albumin_crp'] = df['albumin'] / (df['crp'] + 0.1)
    df['wbc_crp'] = df['white_blood_cell'] * df['crp']
    
    demo = ['age', 'bmi', 'sex_num', 'is_male']
    fitbit = ['Resting Heart Rate (mean)', 'Resting Heart Rate (median)', 'Resting Heart Rate (std)',
              'HRV (mean)', 'HRV (median)', 'HRV (std)',
              'STEPS (mean)', 'STEPS (median)', 'STEPS (std)',
              'SLEEP Duration (mean)', 'SLEEP Duration (median)', 'SLEEP Duration (std)',
              'AZM Weekly (mean)', 'AZM Weekly (median)', 'AZM Weekly (std)']
    
    label_cols = ['True_hba1c', 'True_HOMA_IR', 'True_IR_Class', 'True_Diabetes_2_Class', 
                  'True_Normoglycemic_2_Class', 'True_Diabetes_3_Class']
    
    blood = [c for c in df.columns if c not in demo + fitbit + label_cols + ['Participant_id', 'sex', 'sex_num', 'is_male']
             and c in df.select_dtypes(include=[np.number]).columns
             and c not in ['trig_hdl','glucose_bmi','glucose_trig','ldl_hdl','glucose_sq','bmi_sq',
                          'log_trig','log_glucose','glucose_proxy','bmi_age','rhr_hrv','ast_alt',
                          'bun_creat','nlr','log_crp','non_hdl_hdl','ggt_alt','hr_cv','hrv_cv',
                          'steps_cv','sleep_cv','activity','steps_sleep','met_syndrome_score',
                          'glucose_hdl','trig_glucose_idx','bmi_trig','age_glucose','crp_bmi',
                          'albumin_crp','wbc_crp']]
    
    eng_all = ['trig_hdl','glucose_bmi','glucose_trig','ldl_hdl','glucose_sq','bmi_sq',
               'log_trig','log_glucose','glucose_proxy','bmi_age','rhr_hrv','ast_alt',
               'bun_creat','nlr','log_crp','non_hdl_hdl','ggt_alt','hr_cv','hrv_cv',
               'steps_cv','sleep_cv','activity','steps_sleep','met_syndrome_score',
               'glucose_hdl','trig_glucose_idx','bmi_trig','age_glucose','crp_bmi',
               'albumin_crp','wbc_crp']
    
    eng_dw = ['bmi_sq', 'bmi_age', 'rhr_hrv', 'hr_cv', 'hrv_cv', 'steps_cv', 'sleep_cv',
              'activity', 'steps_sleep']
    
    all_cols = list(dict.fromkeys(demo + fitbit + blood + eng_all))
    dw_cols = list(dict.fromkeys(demo + fitbit + eng_dw))
    
    all_cols = [c for c in all_cols if c in df.columns]
    dw_cols = [c for c in dw_cols if c in df.columns]
    
    mask = df[['True_HOMA_IR', 'True_hba1c']].notna().all(axis=1)
    df = df[mask].reset_index(drop=True)
    
    for col in set(all_cols + dw_cols):
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(df[col].median())
    
    return df, all_cols, dw_cols


# ============ MODELS ============

class ResBlock(nn.Module):
    def __init__(self, d, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d), nn.GELU(), nn.Linear(d, d*2), nn.GELU(),
            nn.Dropout(drop), nn.Linear(d*2, d), nn.Dropout(drop/2))
    def forward(self, x): return x + self.net(x)

class ResNet(nn.Module):
    def __init__(self, ind, h=256, nb=6, dr=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ind, h),
            *[ResBlock(h, dr) for _ in range(nb)],
            nn.LayerNorm(h), nn.GELU(), nn.Linear(h, 1))
    def forward(self, x): return self.net(x)

class SNN(nn.Module):
    def __init__(self, ind, h=256, nl=6, dr=0.05):
        super().__init__()
        layers = [nn.Linear(ind, h), nn.SELU(), nn.AlphaDropout(dr)]
        for _ in range(nl-1):
            layers += [nn.Linear(h, h), nn.SELU(), nn.AlphaDropout(dr)]
        layers.append(nn.Linear(h, 1))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
    def forward(self, x): return self.net(x)

class DCN(nn.Module):
    def __init__(self, ind, h=256, nc=3, nd=3, dr=0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(ind)
        self.cw = nn.ParameterList([nn.Parameter(torch.randn(ind,ind)*0.01) for _ in range(nc)])
        self.cb = nn.ParameterList([nn.Parameter(torch.zeros(ind)) for _ in range(nc)])
        dl = []
        d = ind
        for _ in range(nd):
            dl += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dr)]
            d = h
        self.deep = nn.Sequential(*dl)
        self.head = nn.Sequential(nn.Linear(ind+h, h), nn.GELU(), nn.Linear(h, 1))
    def forward(self, x):
        x = self.bn(x); x0 = x
        for w, b in zip(self.cw, self.cb):
            x = x0 * (x @ w) + b + x
        return self.head(torch.cat([x, self.deep(x0)], 1))


# ============ TRAINING ============

def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    idx = torch.randperm(x.size(0))
    return lam * x + (1-lam) * x[idx], lam * y + (1-lam) * y[idx]

def train_one(model, Xtr, ytr, Xva, yva, epochs=500, lr=5e-4, wd=1e-3, bs=64, patience=60, use_mixup=True):
    model = model.to(DEVICE)
    ds = TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(ytr))
    dl = DataLoader(ds, batch_size=bs, shuffle=True)
    vx = torch.FloatTensor(Xva); vy = torch.FloatTensor(yva)
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    warmup = 15
    def lr_fn(ep):
        if ep < warmup: return (ep+1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (ep-warmup) / max(epochs-warmup, 1)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    
    best = 1e9; best_st = None; wait = 0
    for ep in range(epochs):
        model.train()
        for bx, by in dl:
            if use_mixup and np.random.random() < 0.5:
                bx, by = mixup_data(bx, by, 0.2)
            p = model(bx).squeeze(-1)
            loss = F.huber_loss(p, by, delta=1.0)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        
        model.eval()
        with torch.no_grad():
            vl = F.mse_loss(model(vx).squeeze(-1), vy).item()
        if vl < best - 1e-7:
            best = vl; best_st = deepcopy(model.state_dict()); wait = 0
        else:
            wait += 1
            if wait >= patience: break
    
    if best_st: model.load_state_dict(best_st)
    del ds, dl, vx, vy, opt, sched; gc.collect()
    return model


def evaluate(model, X, y_raw, transform, qt_or_sc=None):
    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(X)).squeeze(-1).numpy()
    if transform == 'log': pred = np.expm1(pred)
    elif transform == 'quantile': pred = qt_or_sc.inverse_transform(pred.reshape(-1,1)).flatten()
    elif transform == 'standard': pred = qt_or_sc.inverse_transform(pred.reshape(-1,1)).flatten()
    return r2_score(y_raw, pred), pred


def run_one_config(X, y_raw, model_fn, transform, lr, wd, bs, epochs, patience, n_folds=5, use_mixup=True, label=''):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_r2 = []
    
    for fold, (ti, vi) in enumerate(kf.split(X)):
        Xtr, Xva = X[ti], X[vi]
        ytr_raw, yva_raw = y_raw[ti], y_raw[vi]
        
        sx = StandardScaler()
        Xtr_s = sx.fit_transform(Xtr).astype(np.float32)
        Xva_s = sx.transform(Xva).astype(np.float32)
        
        qt_or_sc = None
        if transform == 'log':
            ytr = np.log1p(ytr_raw).astype(np.float32)
        elif transform == 'quantile':
            qt_or_sc = QuantileTransformer(output_distribution='normal', n_quantiles=200, random_state=42)
            ytr = qt_or_sc.fit_transform(ytr_raw.reshape(-1,1)).flatten().astype(np.float32)
        elif transform == 'standard':
            qt_or_sc = StandardScaler()
            ytr = qt_or_sc.fit_transform(ytr_raw.reshape(-1,1)).flatten().astype(np.float32)
        else:
            ytr = ytr_raw.astype(np.float32)
        
        # Also transform val target for training loss
        if transform == 'log': yva_t = np.log1p(yva_raw).astype(np.float32)
        elif transform == 'quantile': yva_t = qt_or_sc.transform(yva_raw.reshape(-1,1)).flatten().astype(np.float32)
        elif transform == 'standard': yva_t = qt_or_sc.transform(yva_raw.reshape(-1,1)).flatten().astype(np.float32)
        else: yva_t = yva_raw.astype(np.float32)
        
        model = model_fn(Xtr_s.shape[1])
        model = train_one(model, Xtr_s, ytr, Xva_s, yva_t, epochs, lr, wd, bs, patience, use_mixup)
        
        r2, _ = evaluate(model, Xva_s, yva_raw, transform, qt_or_sc)
        fold_r2.append(r2)
        
        del model; gc.collect()
    
    mean_r2 = np.mean(fold_r2)
    std_r2 = np.std(fold_r2)
    print(f"  {label:35s} | R²={mean_r2:.4f}±{std_r2:.4f} | folds={[f'{r:.3f}' for r in fold_r2]}", flush=True)
    return mean_r2, std_r2, fold_r2


def deep_ensemble_cv(X, y_raw, model_fns, transform, lr, wd, bs, epochs, patience, n_folds=5, label=''):
    """Train multiple models per fold, ensemble their predictions."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_r2 = []
    
    for fold, (ti, vi) in enumerate(kf.split(X)):
        Xtr, Xva = X[ti], X[vi]
        ytr_raw, yva_raw = y_raw[ti], y_raw[vi]
        
        sx = StandardScaler()
        Xtr_s = sx.fit_transform(Xtr).astype(np.float32)
        Xva_s = sx.transform(Xva).astype(np.float32)
        
        qt_or_sc = None
        if transform == 'log':
            ytr = np.log1p(ytr_raw).astype(np.float32)
            yva_t = np.log1p(yva_raw).astype(np.float32)
        elif transform == 'quantile':
            qt_or_sc = QuantileTransformer(output_distribution='normal', n_quantiles=200, random_state=42)
            ytr = qt_or_sc.fit_transform(ytr_raw.reshape(-1,1)).flatten().astype(np.float32)
            yva_t = qt_or_sc.transform(yva_raw.reshape(-1,1)).flatten().astype(np.float32)
        else:
            ytr = ytr_raw.astype(np.float32)
            yva_t = yva_raw.astype(np.float32)
        
        all_preds = []
        for mfn in model_fns:
            # Different seed per model
            torch.manual_seed(42 + len(all_preds))
            model = mfn(Xtr_s.shape[1])
            model = train_one(model, Xtr_s, ytr, Xva_s, yva_t, epochs, lr, wd, bs, patience, True)
            _, pred = evaluate(model, Xva_s, yva_raw, transform, qt_or_sc)
            all_preds.append(pred)
            del model; gc.collect()
        
        # Ensemble average
        ensemble_pred = np.mean(all_preds, axis=0)
        r2 = r2_score(yva_raw, ensemble_pred)
        fold_r2.append(r2)
    
    mean_r2 = np.mean(fold_r2)
    std_r2 = np.std(fold_r2)
    print(f"  {label:35s} | R²={mean_r2:.4f}±{std_r2:.4f} | folds={[f'{r:.3f}' for r in fold_r2]}", flush=True)
    return mean_r2, std_r2, fold_r2


# ============ MAIN ============

def main():
    t0 = time.time()
    df, all_cols, dw_cols = load_data()
    
    X_all = df[all_cols].values.astype(np.float32)
    X_dw = df[dw_cols].values.astype(np.float32)
    y_homa = df['True_HOMA_IR'].values.astype(np.float32)
    y_hba1c = df['True_hba1c'].values.astype(np.float32)
    
    print(f"Features: all={X_all.shape[1]}, dw={X_dw.shape[1]}, N={len(df)}", flush=True)
    
    results = {}
    
    # ==== EXP 1: HOMA_IR all features ====
    print(f"\n{'='*70}\nEXP 1: True_HOMA_IR | ALL features ({X_all.shape[1]}) | Goal: 0.85\n{'='*70}", flush=True)
    
    best1 = -1
    configs1 = [
        # Single models
        (lambda d: ResNet(d, 256, 6, 0.1), 'log', 5e-4, 1e-3, 64, 500, 60, 'ResNet256x6_log'),
        (lambda d: ResNet(d, 256, 8, 0.1), 'log', 3e-4, 5e-4, 64, 600, 60, 'ResNet256x8_log'),
        (lambda d: ResNet(d, 256, 6, 0.1), 'quantile', 5e-4, 1e-3, 64, 500, 60, 'ResNet256x6_qt'),
        (lambda d: SNN(d, 256, 6, 0.05), 'log', 5e-4, 1e-3, 64, 500, 60, 'SNN256x6_log'),
        (lambda d: SNN(d, 512, 8, 0.03), 'log', 3e-4, 5e-4, 64, 500, 60, 'SNN512x8_log'),
        (lambda d: SNN(d, 256, 6, 0.05), 'quantile', 5e-4, 1e-3, 64, 500, 60, 'SNN256x6_qt'),
        (lambda d: DCN(d, 256, 3, 4, 0.1), 'log', 5e-4, 1e-3, 64, 500, 60, 'DCN256_log'),
        (lambda d: DCN(d, 256, 3, 4, 0.1), 'quantile', 5e-4, 1e-3, 64, 500, 60, 'DCN256_qt'),
    ]
    for mfn, tfm, lr, wd, bs, ep, pat, lbl in configs1:
        r2, _, _ = run_one_config(X_all, y_homa, mfn, tfm, lr, wd, bs, ep, pat, label=lbl)
        if r2 > best1: best1 = r2; results['HOMA_all'] = {'r2': r2, 'model': lbl}
        gc.collect()
    
    # Ensemble
    print("  --- Ensembles ---", flush=True)
    ensemble_fns = [
        lambda d: ResNet(d, 256, 6, 0.1),
        lambda d: ResNet(d, 256, 8, 0.15),
        lambda d: SNN(d, 256, 6, 0.05),
        lambda d: DCN(d, 256, 3, 4, 0.1),
    ]
    r2, _, _ = deep_ensemble_cv(X_all, y_homa, ensemble_fns, 'log', 5e-4, 1e-3, 64, 500, 60, label='Ensemble4_log')
    if r2 > best1: best1 = r2; results['HOMA_all'] = {'r2': r2, 'model': 'Ensemble4_log'}
    
    r2, _, _ = deep_ensemble_cv(X_all, y_homa, ensemble_fns, 'quantile', 5e-4, 1e-3, 64, 500, 60, label='Ensemble4_qt')
    if r2 > best1: best1 = r2; results['HOMA_all'] = {'r2': r2, 'model': 'Ensemble4_qt'}
    
    print(f"\n  >>> BEST HOMA_all: R²={best1:.4f}", flush=True)
    
    # ==== EXP 2: hba1c all features ====
    print(f"\n{'='*70}\nEXP 2: True_hba1c | ALL features ({X_all.shape[1]}) | Goal: 0.85\n{'='*70}", flush=True)
    
    best2 = -1
    configs2 = [
        (lambda d: ResNet(d, 256, 6, 0.1), 'log', 5e-4, 1e-3, 64, 500, 60, 'ResNet256x6_log'),
        (lambda d: ResNet(d, 256, 8, 0.1), 'log', 3e-4, 5e-4, 64, 600, 60, 'ResNet256x8_log'),
        (lambda d: ResNet(d, 256, 6, 0.1), 'quantile', 5e-4, 1e-3, 64, 500, 60, 'ResNet256x6_qt'),
        (lambda d: SNN(d, 256, 6, 0.05), 'log', 5e-4, 1e-3, 64, 500, 60, 'SNN256x6_log'),
        (lambda d: SNN(d, 512, 8, 0.03), 'log', 3e-4, 5e-4, 64, 500, 60, 'SNN512x8_log'),
        (lambda d: DCN(d, 256, 3, 4, 0.1), 'log', 5e-4, 1e-3, 64, 500, 60, 'DCN256_log'),
        (lambda d: DCN(d, 256, 3, 4, 0.1), 'quantile', 5e-4, 1e-3, 64, 500, 60, 'DCN256_qt'),
    ]
    for mfn, tfm, lr, wd, bs, ep, pat, lbl in configs2:
        r2, _, _ = run_one_config(X_all, y_hba1c, mfn, tfm, lr, wd, bs, ep, pat, label=lbl)
        if r2 > best2: best2 = r2; results['hba1c_all'] = {'r2': r2, 'model': lbl}
        gc.collect()
    
    r2, _, _ = deep_ensemble_cv(X_all, y_hba1c, ensemble_fns, 'log', 5e-4, 1e-3, 64, 500, 60, label='Ensemble4_log')
    if r2 > best2: best2 = r2; results['hba1c_all'] = {'r2': r2, 'model': 'Ensemble4_log'}
    
    print(f"\n  >>> BEST hba1c_all: R²={best2:.4f}", flush=True)
    
    # ==== EXP 3: HOMA_IR demo+wearable ====
    print(f"\n{'='*70}\nEXP 3: True_HOMA_IR | Demo+Wearable ({X_dw.shape[1]}) | Goal: 0.70\n{'='*70}", flush=True)
    
    best3 = -1
    configs3 = [
        (lambda d: ResNet(d, 128, 4, 0.1), 'log', 5e-4, 1e-3, 64, 500, 60, 'ResNet128x4_log'),
        (lambda d: ResNet(d, 256, 6, 0.1), 'log', 5e-4, 1e-3, 64, 500, 60, 'ResNet256x6_log'),
        (lambda d: SNN(d, 256, 6, 0.05), 'log', 5e-4, 1e-3, 64, 500, 60, 'SNN256x6_log'),
        (lambda d: DCN(d, 128, 3, 3, 0.1), 'log', 5e-4, 1e-3, 64, 500, 60, 'DCN128_log'),
        (lambda d: DCN(d, 256, 4, 4, 0.1), 'log', 5e-4, 1e-3, 64, 500, 60, 'DCN256_log'),
    ]
    for mfn, tfm, lr, wd, bs, ep, pat, lbl in configs3:
        r2, _, _ = run_one_config(X_dw, y_homa, mfn, tfm, lr, wd, bs, ep, pat, label=lbl)
        if r2 > best3: best3 = r2; results['HOMA_dw'] = {'r2': r2, 'model': lbl}
        gc.collect()
    
    dw_ensemble = [lambda d: ResNet(d, 256, 6, 0.1), lambda d: SNN(d, 256, 6, 0.05), lambda d: DCN(d, 256, 4, 4, 0.1)]
    r2, _, _ = deep_ensemble_cv(X_dw, y_homa, dw_ensemble, 'log', 5e-4, 1e-3, 64, 500, 60, label='Ensemble3_log')
    if r2 > best3: best3 = r2; results['HOMA_dw'] = {'r2': r2, 'model': 'Ensemble3_log'}
    
    print(f"\n  >>> BEST HOMA_dw: R²={best3:.4f}", flush=True)
    
    # ==== EXP 4: hba1c demo+wearable ====
    print(f"\n{'='*70}\nEXP 4: True_hba1c | Demo+Wearable ({X_dw.shape[1]}) | Goal: 0.70\n{'='*70}", flush=True)
    
    best4 = -1
    configs4 = [
        (lambda d: ResNet(d, 128, 4, 0.1), 'log', 5e-4, 1e-3, 64, 500, 60, 'ResNet128x4_log'),
        (lambda d: ResNet(d, 256, 6, 0.1), 'log', 5e-4, 1e-3, 64, 500, 60, 'ResNet256x6_log'),
        (lambda d: SNN(d, 256, 6, 0.05), 'log', 5e-4, 1e-3, 64, 500, 60, 'SNN256x6_log'),
        (lambda d: DCN(d, 128, 3, 3, 0.1), 'log', 5e-4, 1e-3, 64, 500, 60, 'DCN128_log'),
        (lambda d: DCN(d, 256, 4, 4, 0.1), 'log', 5e-4, 1e-3, 64, 500, 60, 'DCN256_log'),
    ]
    for mfn, tfm, lr, wd, bs, ep, pat, lbl in configs4:
        r2, _, _ = run_one_config(X_dw, y_hba1c, mfn, tfm, lr, wd, bs, ep, pat, label=lbl)
        if r2 > best4: best4 = r2; results['hba1c_dw'] = {'r2': r2, 'model': lbl}
        gc.collect()
    
    r2, _, _ = deep_ensemble_cv(X_dw, y_hba1c, dw_ensemble, 'log', 5e-4, 1e-3, 64, 500, 60, label='Ensemble3_log')
    if r2 > best4: best4 = r2; results['hba1c_dw'] = {'r2': r2, 'model': 'Ensemble3_log'}
    
    print(f"\n  >>> BEST hba1c_dw: R²={best4:.4f}", flush=True)
    
    # ==== FINAL ====
    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*70}\nFINAL SUMMARY ({elapsed:.1f} min)\n{'='*70}", flush=True)
    goals = {'HOMA_all': 0.85, 'hba1c_all': 0.85, 'HOMA_dw': 0.70, 'hba1c_dw': 0.70}
    for k in ['HOMA_all', 'hba1c_all', 'HOMA_dw', 'hba1c_dw']:
        if k in results:
            g = goals[k]
            f = "✅" if results[k]['r2'] >= g else "❌"
            print(f"  {k:15s}: R²={results[k]['r2']:.4f} (goal={g}) {f} [{results[k]['model']}]", flush=True)
    
    with open('results_lean.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


if __name__ == '__main__':
    main()
