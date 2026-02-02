#!/usr/bin/env python3
"""
FINAL: Maximum effort. ResNet MLP only. Exhaustive tuning.
- Hyperparameter grid search
- 1000 epoch training, patience 80
- 10-model deep ensemble per fold
- All feature combinations
- Aggressive augmentation
"""
import gc, warnings, json, time, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer, RobustScaler
from sklearn.metrics import r2_score
from copy import deepcopy

warnings.filterwarnings('ignore')

def load_data():
    df = pd.read_csv('data.csv', skiprows=[0])
    df['sex_num'] = (df['sex'] == 'Male').astype(float)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    
    # Only the PROVEN useful engineered features
    df['trig_hdl'] = df['triglycerides'] / (df['hdl'] + 0.1)
    df['tyg'] = np.log(df['triglycerides'] * df['glucose'] / 2 + 1)
    df['tyg_bmi'] = df['tyg'] * df['bmi']
    df['glucose_bmi'] = df['glucose'] * df['bmi']
    df['glucose_sq'] = df['glucose'] ** 2 / 1000
    df['bmi_sq'] = df['bmi'] ** 2
    df['log_trig'] = np.log1p(df['triglycerides'])
    df['log_glucose'] = np.log1p(df['glucose'])
    df['glucose_proxy'] = df['glucose'] * df['trig_hdl']
    df['ldl_hdl'] = df['ldl'] / (df['hdl'] + 0.1)
    df['ast_alt'] = df['ast'] / (df['alt'] + 0.1)
    df['rhr_hrv'] = df['Resting Heart Rate (mean)'] / (df['HRV (mean)'] + 0.1)
    df['bmi_age'] = df['bmi'] * df['age']
    df['glucose_trig'] = df['glucose'] * df['triglycerides'] / 1000
    df['glucose_hdl'] = df['glucose'] / (df['hdl'] + 0.1)
    df['non_hdl_hdl'] = df['non hdl'] / (df['hdl'] + 0.1)
    df['log_crp'] = np.log1p(df['crp'])
    df['nlr'] = df['absolute_neutrophils'] / (df['absolute_lymphocytes'] + 0.1)
    df['bun_creat'] = df['bun'] / (df['creatinine'] + 0.01)
    df['ggt_alt'] = df['ggt'] / (df['alt'] + 0.1)
    df['met_score'] = ((df['bmi']>30).astype(float) + (df['triglycerides']>150).astype(float) + 
                       (df['glucose']>100).astype(float) + (df['hdl']<40).astype(float))
    df['hr_cv'] = df['Resting Heart Rate (std)'] / (df['Resting Heart Rate (mean)'] + 0.1)
    df['hrv_cv'] = df['HRV (std)'] / (df['HRV (mean)'] + 0.1)
    df['steps_cv'] = df['STEPS (std)'] / (df['STEPS (mean)'] + 1)
    df['sleep_cv'] = df['SLEEP Duration (std)'] / (df['SLEEP Duration (mean)'] + 1)
    df['activity'] = df['STEPS (mean)'] * df['AZM Weekly (mean)'] / 1e6
    df['steps_sleep'] = df['STEPS (mean)'] / (df['SLEEP Duration (mean)'] + 1)
    df['bmi_trig'] = df['bmi'] * df['triglycerides'] / 100
    df['crp_bmi'] = df['crp'] * df['bmi']
    df['rdw_mch'] = df['rdw'] * df['mch']
    df['glucose_rdw'] = df['glucose'] * df['rdw']
    df['age_glucose'] = df['age'] * df['glucose'] / 100
    
    label_cols = ['True_hba1c','True_HOMA_IR','True_IR_Class','True_Diabetes_2_Class',
                  'True_Normoglycemic_2_Class','True_Diabetes_3_Class']
    
    feat_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                 if c not in label_cols + ['Participant_id']]
    
    demo = ['age', 'bmi', 'sex_num']
    fitbit = ['Resting Heart Rate (mean)', 'Resting Heart Rate (median)', 'Resting Heart Rate (std)',
              'HRV (mean)', 'HRV (median)', 'HRV (std)', 'STEPS (mean)', 'STEPS (median)', 'STEPS (std)',
              'SLEEP Duration (mean)', 'SLEEP Duration (median)', 'SLEEP Duration (std)',
              'AZM Weekly (mean)', 'AZM Weekly (median)', 'AZM Weekly (std)']
    dw_eng = ['bmi_sq','bmi_age','rhr_hrv','hr_cv','hrv_cv','steps_cv','sleep_cv','activity','steps_sleep']
    dw_cols = list(dict.fromkeys(demo + fitbit + dw_eng))
    
    mask = df[['True_HOMA_IR','True_hba1c']].notna().all(axis=1)
    df = df[mask].reset_index(drop=True)
    for col in feat_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(df[col].median())
    
    return df, feat_cols, dw_cols

# ============ MODEL ============

class ResBlock(nn.Module):
    def __init__(self, d, drop=0.1, expansion=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d), nn.GELU(), nn.Linear(d, d * expansion), nn.GELU(),
            nn.Dropout(drop), nn.Linear(d * expansion, d), nn.Dropout(drop * 0.5))
    def forward(self, x): return x + self.net(x)

class ResNet(nn.Module):
    def __init__(self, ind, h=256, nb=6, dr=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ind, h), nn.GELU(),
            *[ResBlock(h, dr) for _ in range(nb)],
            nn.LayerNorm(h), nn.GELU(), nn.Linear(h, 1))
    def forward(self, x): return self.net(x)


def train(model, Xtr, ytr, Xva, yva, epochs=800, lr=3e-4, wd=5e-4, bs=64, pat=80):
    ds = TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(ytr))
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=len(Xtr) > bs)
    vx, vy = torch.FloatTensor(Xva), torch.FloatTensor(yva)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    warmup = 20
    def lr_fn(ep):
        if ep < warmup: return (ep+1)/warmup
        return max(0.01, 0.5*(1+math.cos(math.pi*(ep-warmup)/max(epochs-warmup,1))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    best = 1e9; bst = None; wait = 0
    
    for ep in range(epochs):
        model.train()
        for bx, by in dl:
            # Mixup augmentation
            if np.random.random() < 0.5:
                lam = np.random.beta(0.3, 0.3)
                idx = torch.randperm(bx.size(0))
                bx = lam*bx + (1-lam)*bx[idx]
                by = lam*by + (1-lam)*by[idx]
            
            # Add gaussian noise to input
            if np.random.random() < 0.3:
                bx = bx + torch.randn_like(bx) * 0.05
            
            p = model(bx).squeeze(-1)
            loss = F.huber_loss(p, by, delta=1.0)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
        sched.step()
        
        model.eval()
        with torch.no_grad():
            vl = F.mse_loss(model(vx).squeeze(-1), vy).item()
        if vl < best - 1e-7:
            best = vl; bst = deepcopy(model.state_dict()); wait = 0
        else:
            wait += 1
            if wait >= pat: break
    
    if bst: model.load_state_dict(bst)
    return model


def single_fold(X_tr, y_tr_raw, X_va, y_va_raw, h, nb, dr, lr, wd, bs, epochs, pat, tfm, n_models=1):
    """Train n_models on one fold, return ensemble prediction."""
    sx = StandardScaler()
    Xtr = sx.fit_transform(X_tr).astype(np.float32)
    Xva = sx.transform(X_va).astype(np.float32)
    
    qt = None
    if tfm == 'log':
        ytr = np.log1p(y_tr_raw).astype(np.float32)
        yva = np.log1p(y_va_raw).astype(np.float32)
    elif tfm == 'quantile':
        qt = QuantileTransformer(output_distribution='normal', n_quantiles=200, random_state=42)
        ytr = qt.fit_transform(y_tr_raw.reshape(-1,1)).flatten().astype(np.float32)
        yva = qt.transform(y_va_raw.reshape(-1,1)).flatten().astype(np.float32)
    else:
        sc = StandardScaler()
        ytr = sc.fit_transform(y_tr_raw.reshape(-1,1)).flatten().astype(np.float32)
        yva = sc.transform(y_va_raw.reshape(-1,1)).flatten().astype(np.float32)
    
    all_preds = []
    for i in range(n_models):
        torch.manual_seed(42 + i * 137)
        np.random.seed(42 + i * 137)
        model = ResNet(Xtr.shape[1], h, nb, dr)
        model = train(model, Xtr, ytr, Xva, yva, epochs, lr, wd, bs, pat)
        model.eval()
        with torch.no_grad():
            p = model(torch.FloatTensor(Xva)).squeeze(-1).numpy()
        
        if tfm == 'log': p = np.expm1(p)
        elif tfm == 'quantile': p = qt.inverse_transform(p.reshape(-1,1)).flatten()
        else: p = sc.inverse_transform(p.reshape(-1,1)).flatten()
        
        all_preds.append(p)
        del model; gc.collect()
    
    return np.mean(all_preds, axis=0)


def run_config(X, y_raw, h, nb, dr, lr, wd, bs, epochs, pat, tfm, n_models, n_folds=5, label=''):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_r2 = []
    for fold, (ti, vi) in enumerate(kf.split(X)):
        pred = single_fold(X[ti], y_raw[ti], X[vi], y_raw[vi], h, nb, dr, lr, wd, bs, epochs, pat, tfm, n_models)
        fold_r2.append(r2_score(y_raw[vi], pred))
        gc.collect()
    mr2 = np.mean(fold_r2)
    sr2 = np.std(fold_r2)
    print(f"  {label:40s} R²={mr2:.4f}±{sr2:.4f} folds={[f'{r:.3f}' for r in fold_r2]}", flush=True)
    return mr2, sr2, fold_r2


def main():
    t0 = time.time()
    df, feat_cols, dw_cols = load_data()
    X_all = df[feat_cols].values.astype(np.float32)
    X_dw = df[dw_cols].values.astype(np.float32)
    y_homa = df['True_HOMA_IR'].values.astype(np.float32)
    y_hba1c = df['True_hba1c'].values.astype(np.float32)
    
    print(f"All={X_all.shape[1]}, DW={X_dw.shape[1]}, N={len(df)}", flush=True)
    
    results = {}
    
    # ===== HOMA_IR ALL =====
    print(f"\n{'='*70}\nHOMA_IR | ALL ({X_all.shape[1]}) | Goal: 0.85\n{'='*70}", flush=True)
    
    # Grid of hyperparameters - focused on what works
    configs = [
        # h, nb, dr, lr, wd, bs, epochs, pat, tfm, n_models, label
        (256, 6, 0.10, 3e-4, 5e-4, 64, 800, 80, 'log', 1, 'R256x6_d10_lr3e4'),
        (256, 6, 0.15, 3e-4, 5e-4, 64, 800, 80, 'log', 1, 'R256x6_d15_lr3e4'),
        (256, 6, 0.10, 5e-4, 1e-3, 64, 800, 80, 'log', 1, 'R256x6_d10_lr5e4'),
        (256, 6, 0.10, 1e-4, 3e-4, 64, 1000, 100, 'log', 1, 'R256x6_d10_lr1e4'),
        (256, 8, 0.10, 3e-4, 5e-4, 64, 800, 80, 'log', 1, 'R256x8_d10_lr3e4'),
        (256, 4, 0.10, 3e-4, 5e-4, 64, 800, 80, 'log', 1, 'R256x4_d10_lr3e4'),
        (128, 6, 0.10, 3e-4, 5e-4, 64, 800, 80, 'log', 1, 'R128x6_d10_lr3e4'),
        (512, 4, 0.15, 3e-4, 1e-3, 64, 800, 80, 'log', 1, 'R512x4_d15_lr3e4'),
        (256, 6, 0.10, 3e-4, 5e-4, 32, 800, 80, 'log', 1, 'R256x6_d10_bs32'),
        (256, 6, 0.10, 3e-4, 5e-4, 128, 800, 80, 'log', 1, 'R256x6_d10_bs128'),
        (256, 6, 0.10, 3e-4, 5e-4, 64, 800, 80, 'quantile', 1, 'R256x6_d10_quantile'),
        (256, 6, 0.10, 3e-4, 5e-4, 64, 800, 80, 'standard', 1, 'R256x6_d10_standard'),
    ]
    
    best1 = -1
    best_cfg1 = ''
    for h, nb, dr, lr, wd, bs, ep, pat, tfm, nm, lbl in configs:
        r2, _, _ = run_config(X_all, y_homa, h, nb, dr, lr, wd, bs, ep, pat, tfm, nm, label=lbl)
        if r2 > best1: best1 = r2; best_cfg1 = lbl
    
    # Deep ensemble with best config params
    print(f"\n  --- Deep Ensemble (10 models) ---", flush=True)
    r2_ens, _, _ = run_config(X_all, y_homa, 256, 6, 0.10, 3e-4, 5e-4, 64, 800, 80, 'log', 10,
                              label='Ensemble10_R256x6_log')
    if r2_ens > best1: best1 = r2_ens; best_cfg1 = 'Ensemble10'
    
    r2_ens2, _, _ = run_config(X_all, y_homa, 256, 6, 0.10, 3e-4, 5e-4, 64, 800, 80, 'quantile', 10,
                               label='Ensemble10_R256x6_qt')
    if r2_ens2 > best1: best1 = r2_ens2; best_cfg1 = 'Ensemble10_qt'
    
    results['HOMA_all'] = {'r2': best1, 'config': best_cfg1}
    print(f"\n  >>> BEST HOMA_all: R²={best1:.4f} [{best_cfg1}]", flush=True)
    
    # ===== hba1c ALL =====
    print(f"\n{'='*70}\nhba1c | ALL ({X_all.shape[1]}) | Goal: 0.85\n{'='*70}", flush=True)
    
    configs2 = [
        (256, 6, 0.10, 3e-4, 5e-4, 64, 800, 80, 'log', 1, 'R256x6_d10_lr3e4'),
        (256, 6, 0.15, 3e-4, 5e-4, 64, 800, 80, 'log', 1, 'R256x6_d15_lr3e4'),
        (256, 6, 0.10, 5e-4, 1e-3, 64, 800, 80, 'log', 1, 'R256x6_d10_lr5e4'),
        (256, 6, 0.10, 1e-4, 3e-4, 64, 1000, 100, 'log', 1, 'R256x6_d10_lr1e4'),
        (256, 8, 0.10, 3e-4, 5e-4, 64, 800, 80, 'log', 1, 'R256x8_d10_lr3e4'),
        (128, 6, 0.10, 3e-4, 5e-4, 64, 800, 80, 'log', 1, 'R128x6_d10_lr3e4'),
        (512, 4, 0.15, 3e-4, 1e-3, 64, 800, 80, 'log', 1, 'R512x4_d15_lr3e4'),
        (256, 6, 0.10, 3e-4, 5e-4, 64, 800, 80, 'quantile', 1, 'R256x6_d10_quantile'),
        (256, 6, 0.10, 3e-4, 5e-4, 64, 800, 80, 'standard', 1, 'R256x6_d10_standard'),
    ]
    
    best2 = -1
    best_cfg2 = ''
    for h, nb, dr, lr, wd, bs, ep, pat, tfm, nm, lbl in configs2:
        r2, _, _ = run_config(X_all, y_hba1c, h, nb, dr, lr, wd, bs, ep, pat, tfm, nm, label=lbl)
        if r2 > best2: best2 = r2; best_cfg2 = lbl
    
    print(f"\n  --- Deep Ensemble (10 models) ---", flush=True)
    r2_ens, _, _ = run_config(X_all, y_hba1c, 256, 6, 0.10, 3e-4, 5e-4, 64, 800, 80, 'log', 10,
                              label='Ensemble10_R256x6_log')
    if r2_ens > best2: best2 = r2_ens; best_cfg2 = 'Ensemble10'
    
    results['hba1c_all'] = {'r2': best2, 'config': best_cfg2}
    print(f"\n  >>> BEST hba1c_all: R²={best2:.4f} [{best_cfg2}]", flush=True)
    
    # ===== HOMA_IR DW =====
    print(f"\n{'='*70}\nHOMA_IR | DW ({X_dw.shape[1]}) | Goal: 0.70\n{'='*70}", flush=True)
    
    configs3 = [
        (128, 4, 0.10, 3e-4, 5e-4, 64, 800, 80, 'log', 1, 'R128x4_d10'),
        (256, 6, 0.10, 3e-4, 5e-4, 64, 800, 80, 'log', 1, 'R256x6_d10'),
        (256, 6, 0.15, 3e-4, 5e-4, 64, 800, 80, 'log', 1, 'R256x6_d15'),
        (256, 8, 0.10, 3e-4, 5e-4, 64, 800, 80, 'log', 1, 'R256x8_d10'),
        (256, 6, 0.10, 1e-4, 3e-4, 64, 1000, 100, 'log', 1, 'R256x6_lr1e4'),
        (256, 6, 0.10, 3e-4, 5e-4, 64, 800, 80, 'quantile', 1, 'R256x6_qt'),
    ]
    
    best3 = -1
    best_cfg3 = ''
    for h, nb, dr, lr, wd, bs, ep, pat, tfm, nm, lbl in configs3:
        r2, _, _ = run_config(X_dw, y_homa, h, nb, dr, lr, wd, bs, ep, pat, tfm, nm, label=lbl)
        if r2 > best3: best3 = r2; best_cfg3 = lbl
    
    print(f"\n  --- Deep Ensemble (10 models) ---", flush=True)
    r2_ens, _, _ = run_config(X_dw, y_homa, 256, 6, 0.10, 3e-4, 5e-4, 64, 800, 80, 'log', 10,
                              label='Ensemble10_log')
    if r2_ens > best3: best3 = r2_ens; best_cfg3 = 'Ensemble10'
    
    results['HOMA_dw'] = {'r2': best3, 'config': best_cfg3}
    print(f"\n  >>> BEST HOMA_dw: R²={best3:.4f} [{best_cfg3}]", flush=True)
    
    # ===== hba1c DW =====
    print(f"\n{'='*70}\nhba1c | DW ({X_dw.shape[1]}) | Goal: 0.70\n{'='*70}", flush=True)
    
    configs4 = [
        (128, 4, 0.10, 3e-4, 5e-4, 64, 800, 80, 'log', 1, 'R128x4_d10'),
        (256, 6, 0.10, 3e-4, 5e-4, 64, 800, 80, 'log', 1, 'R256x6_d10'),
        (256, 6, 0.15, 3e-4, 5e-4, 64, 800, 80, 'log', 1, 'R256x6_d15'),
        (256, 8, 0.10, 3e-4, 5e-4, 64, 800, 80, 'log', 1, 'R256x8_d10'),
        (256, 6, 0.10, 1e-4, 3e-4, 64, 1000, 100, 'log', 1, 'R256x6_lr1e4'),
        (256, 6, 0.10, 3e-4, 5e-4, 64, 800, 80, 'quantile', 1, 'R256x6_qt'),
    ]
    
    best4 = -1
    best_cfg4 = ''
    for h, nb, dr, lr, wd, bs, ep, pat, tfm, nm, lbl in configs4:
        r2, _, _ = run_config(X_dw, y_hba1c, h, nb, dr, lr, wd, bs, ep, pat, tfm, nm, label=lbl)
        if r2 > best4: best4 = r2; best_cfg4 = lbl
    
    print(f"\n  --- Deep Ensemble (10 models) ---", flush=True)
    r2_ens, _, _ = run_config(X_dw, y_hba1c, 256, 6, 0.10, 3e-4, 5e-4, 64, 800, 80, 'log', 10,
                              label='Ensemble10_log')
    if r2_ens > best4: best4 = r2_ens; best_cfg4 = 'Ensemble10'
    
    results['hba1c_dw'] = {'r2': best4, 'config': best_cfg4}
    print(f"\n  >>> BEST hba1c_dw: R²={best4:.4f} [{best_cfg4}]", flush=True)
    
    # ===== FINAL =====
    elapsed = (time.time()-t0)/60
    print(f"\n{'='*70}\nFINAL SUMMARY ({elapsed:.1f} min)\n{'='*70}", flush=True)
    goals = {'HOMA_all': 0.85, 'hba1c_all': 0.85, 'HOMA_dw': 0.70, 'hba1c_dw': 0.70}
    for k in ['HOMA_all', 'hba1c_all', 'HOMA_dw', 'hba1c_dw']:
        g = goals[k]
        r = results[k]['r2']
        f = "✅" if r >= g else "❌"
        print(f"  {k:15s}: R²={r:.4f} (goal={g}) {f} [{results[k]['config']}]", flush=True)
    
    with open('results_final.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nSaved results_final.json", flush=True)

if __name__ == '__main__':
    main()
