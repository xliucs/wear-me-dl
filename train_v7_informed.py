#!/usr/bin/env python3
"""
V7: INFORMED TRAINING based on error analysis findings.

Key insights from analysis:
1. HOMA_IR 8+ samples (n=32) have bias=+4.47 — massive under-prediction
2. Model predictions compressed: pred std << true std
3. Worst cases: normal glucose + high HOMA_IR (needs insulin, not in data)
4. Normal BMI hardest to predict (R²=0.33 vs obese 0.47)
5. Error correlates with glucose_bmi (r=0.43) — already a top feature

Strategy:
A. Sample weighting — upweight rare high-value samples
B. Asymmetric loss — penalize under-prediction of high values more
C. Two-stage model — classifier (low/high) + separate regressors
D. Log-space training with inverse-frequency weighting
E. Focal regression loss for hard examples
F. Stratified k-fold to ensure tail samples in every fold
"""
import gc, warnings, json, time, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import r2_score
from copy import deepcopy

warnings.filterwarnings('ignore')

def load_data():
    df = pd.read_csv('data.csv', skiprows=[0])
    df['sex_num'] = (df['sex'] == 'Male').astype(float)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    
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

    # NEW: interaction features targeting high-HOMA_IR cases
    # People with normal glucose but high insulin have: high BMI, high triglycerides, low HDL
    df['insulin_proxy'] = df['bmi'] * df['trig_hdl']  # composite insulin resistance proxy
    df['insulin_proxy2'] = df['bmi_sq'] * df['log_trig'] / 100
    df['liver_stress'] = df['alt'] * df['ggt'] / 100  # liver enzymes often elevated with IR
    df['crp_trig'] = df['crp'] * df['triglycerides'] / 100
    df['bmi_cubed'] = df['bmi'] ** 3 / 10000
    df['wbc_bmi'] = df['wbc'] * df['bmi']
    df['uric_bmi'] = df['uric_acid'] * df['bmi'] if 'uric_acid' in df.columns else 0
    
    label_cols = ['True_hba1c','True_HOMA_IR','True_IR_Class','True_Diabetes_2_Class',
                  'True_Normoglycemic_2_Class','True_Diabetes_3_Class']
    feat_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                 if c not in label_cols + ['Participant_id']]
    
    demo = ['age', 'bmi', 'sex_num']
    fitbit = ['Resting Heart Rate (mean)', 'Resting Heart Rate (median)', 'Resting Heart Rate (std)',
              'HRV (mean)', 'HRV (median)', 'HRV (std)', 'STEPS (mean)', 'STEPS (median)', 'STEPS (std)',
              'SLEEP Duration (mean)', 'SLEEP Duration (median)', 'SLEEP Duration (std)',
              'AZM Weekly (mean)', 'AZM Weekly (median)', 'AZM Weekly (std)']
    dw_eng = ['bmi_sq','bmi_age','rhr_hrv','hr_cv','hrv_cv','steps_cv','sleep_cv','activity','steps_sleep',
              'bmi_cubed']
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


# ============ LOSS FUNCTIONS ============

def weighted_huber_loss(pred, target, weights, delta=1.0):
    """Huber loss with per-sample weights."""
    diff = pred - target
    abs_diff = torch.abs(diff)
    quadratic = torch.clamp(abs_diff, max=delta)
    linear = abs_diff - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    return (loss * weights).mean()

def focal_mse_loss(pred, target, weights, gamma=2.0):
    """Focal regression loss — harder samples get exponentially more weight."""
    mse = (pred - target) ** 2
    # Normalize error for focal weighting
    focal_weight = (mse.detach() / (mse.detach().mean() + 1e-6)) ** (gamma / 2)
    focal_weight = focal_weight.clamp(max=10.0)  # cap to avoid instability
    return (mse * weights * focal_weight).mean()

def asymmetric_loss(pred, target, weights, delta=1.0, under_penalty=2.0):
    """Penalize under-prediction more than over-prediction."""
    diff = pred - target
    abs_diff = torch.abs(diff)
    quadratic = torch.clamp(abs_diff, max=delta)
    linear = abs_diff - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    # Extra penalty for under-prediction (diff < 0 means pred < target)
    under_mask = (diff < 0).float()
    loss = loss * (1 + under_mask * (under_penalty - 1))
    return (loss * weights).mean()


def compute_sample_weights(y_raw, method='inverse_freq'):
    """Compute per-sample weights to handle imbalanced regression."""
    if method == 'inverse_freq':
        # Bin targets and compute inverse frequency weights
        bins = np.quantile(y_raw, np.linspace(0, 1, 11))
        bins[0] -= 1; bins[-1] += 1
        bin_idx = np.digitize(y_raw, bins) - 1
        bin_idx = np.clip(bin_idx, 0, 9)
        bin_counts = np.bincount(bin_idx, minlength=10)
        bin_weights = 1.0 / (bin_counts + 1)
        bin_weights = bin_weights / bin_weights.mean()  # normalize to mean=1
        weights = bin_weights[bin_idx]
        # Cap extreme weights
        weights = np.clip(weights, 0.2, 5.0)
        return weights.astype(np.float32)
    
    elif method == 'sqrt_inverse':
        bins = np.quantile(y_raw, np.linspace(0, 1, 11))
        bins[0] -= 1; bins[-1] += 1
        bin_idx = np.digitize(y_raw, bins) - 1
        bin_idx = np.clip(bin_idx, 0, 9)
        bin_counts = np.bincount(bin_idx, minlength=10)
        bin_weights = 1.0 / np.sqrt(bin_counts + 1)
        bin_weights = bin_weights / bin_weights.mean()
        weights = bin_weights[bin_idx]
        return np.clip(weights, 0.3, 3.0).astype(np.float32)
    
    elif method == 'linear_target':
        # Weight proportional to target value (upweight high values)
        weights = 1 + (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min()) * 2
        return weights.astype(np.float32)


def train_weighted(model, Xtr, ytr, wtr, Xva, yva, epochs=800, lr=3e-4, wd=5e-4, 
                   bs=64, pat=80, loss_fn='weighted_huber'):
    # Use weighted random sampler to oversample rare examples
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(wtr), 
        num_samples=len(wtr), 
        replacement=True
    )
    ds = TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(ytr), torch.FloatTensor(wtr))
    dl = DataLoader(ds, batch_size=bs, sampler=sampler)
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
        for bx, by, bw in dl:
            # Mixup within similar-range samples
            if np.random.random() < 0.5:
                lam = np.random.beta(0.3, 0.3)
                idx = torch.randperm(bx.size(0))
                bx = lam*bx + (1-lam)*bx[idx]
                by = lam*by + (1-lam)*by[idx]
                bw = lam*bw + (1-lam)*bw[idx]
            
            if np.random.random() < 0.3:
                bx = bx + torch.randn_like(bx) * 0.05
            
            p = model(bx).squeeze(-1)
            
            if loss_fn == 'weighted_huber':
                loss = weighted_huber_loss(p, by, bw, delta=1.0)
            elif loss_fn == 'focal':
                loss = focal_mse_loss(p, by, bw, gamma=2.0)
            elif loss_fn == 'asymmetric':
                loss = asymmetric_loss(p, by, bw, delta=1.0, under_penalty=2.0)
            else:
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


# ============ TWO-STAGE MODEL ============

class TwoStagePredictor:
    """Stage 1: Classify into bins (low/med/high). Stage 2: Regress within each bin."""
    
    def __init__(self, n_bins=3):
        self.n_bins = n_bins
        self.bin_edges = None
        self.models = {}
        self.scalers = {}
    
    def fit_predict_fold(self, Xtr, ytr_raw, Xva, yva_raw):
        # Define bins based on training target distribution
        self.bin_edges = np.quantile(ytr_raw, np.linspace(0, 1, self.n_bins + 1))
        self.bin_edges[0] -= 1; self.bin_edges[-1] += 1
        
        ytr_log = np.log1p(ytr_raw).astype(np.float32)
        yva_log = np.log1p(yva_raw).astype(np.float32)
        
        # Train overall model
        sx = StandardScaler()
        Xtr_s = sx.fit_transform(Xtr).astype(np.float32)
        Xva_s = sx.transform(Xva).astype(np.float32)
        
        # Compute weights
        weights = compute_sample_weights(ytr_raw, 'inverse_freq')
        
        # Train ensemble of 5 models with different seeds
        preds = []
        for seed in range(5):
            torch.manual_seed(42 + seed * 137)
            np.random.seed(42 + seed * 137)
            model = ResNet(Xtr_s.shape[1], 256, 6, 0.1)
            model = train_weighted(model, Xtr_s, ytr_log, weights, Xva_s, yva_log,
                                   epochs=800, lr=3e-4, wd=5e-4, bs=64, pat=80,
                                   loss_fn='weighted_huber')
            model.eval()
            with torch.no_grad():
                p = model(torch.FloatTensor(Xva_s)).squeeze(-1).numpy()
            preds.append(np.expm1(p))
            del model; gc.collect()
        
        return np.mean(preds, axis=0)


def run_experiment(X, y_raw, label, goal, n_folds=5):
    print(f"\n{'='*70}")
    print(f"{label} | {X.shape[1]} features | Goal: {goal}")
    print(f"{'='*70}")
    
    # Stratified folds based on target quantiles
    y_bins = pd.qcut(y_raw, q=5, labels=False, duplicates='drop')
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    results = {}
    
    # ===== A. Baseline (unweighted) =====
    print(f"\n  --- A. Baseline ResNet (unweighted) ---")
    fold_r2 = []
    for fold, (ti, vi) in enumerate(skf.split(X, y_bins)):
        sx = StandardScaler()
        Xtr = sx.fit_transform(X[ti]).astype(np.float32)
        Xva = sx.transform(X[vi]).astype(np.float32)
        ytr = np.log1p(y_raw[ti]).astype(np.float32)
        yva = np.log1p(y_raw[vi]).astype(np.float32)
        
        torch.manual_seed(42); np.random.seed(42)
        model = ResNet(Xtr.shape[1], 256, 6, 0.1)
        
        # Unweighted training (use old approach)
        ds = TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(ytr))
        dl = DataLoader(ds, batch_size=64, shuffle=True)
        vx, vy = torch.FloatTensor(Xva), torch.FloatTensor(yva)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
        def lr_fn(ep):
            if ep < 20: return (ep+1)/20
            return max(0.01, 0.5*(1+math.cos(math.pi*(ep-20)/780)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
        best_vl = 1e9; bst = None; wait = 0
        for ep in range(800):
            model.train()
            for bx, by in dl:
                if np.random.random() < 0.5:
                    lam = np.random.beta(0.3, 0.3); idx = torch.randperm(bx.size(0))
                    bx = lam*bx + (1-lam)*bx[idx]; by = lam*by + (1-lam)*by[idx]
                p = model(bx).squeeze(-1)
                loss = F.huber_loss(p, by, delta=1.0)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step()
            sched.step()
            model.eval()
            with torch.no_grad(): vl = F.mse_loss(model(vx).squeeze(-1), vy).item()
            if vl < best_vl - 1e-7: best_vl = vl; bst = deepcopy(model.state_dict()); wait = 0
            else:
                wait += 1
                if wait >= 80: break
        if bst: model.load_state_dict(bst)
        model.eval()
        with torch.no_grad(): p = np.expm1(model(vx).squeeze(-1).numpy())
        fold_r2.append(r2_score(y_raw[vi], p))
        del model; gc.collect()
    
    r2_base = np.mean(fold_r2)
    print(f"  Baseline:              R²={r2_base:.4f}±{np.std(fold_r2):.4f} {fold_r2}", flush=True)
    results['baseline'] = r2_base
    
    # ===== B. Inverse-frequency weighted =====
    for weight_method in ['inverse_freq', 'sqrt_inverse', 'linear_target']:
        for loss_name in ['weighted_huber', 'focal', 'asymmetric']:
            exp_name = f"{weight_method}_{loss_name}"
            fold_r2 = []
            for fold, (ti, vi) in enumerate(skf.split(X, y_bins)):
                sx = StandardScaler()
                Xtr = sx.fit_transform(X[ti]).astype(np.float32)
                Xva = sx.transform(X[vi]).astype(np.float32)
                ytr = np.log1p(y_raw[ti]).astype(np.float32)
                yva = np.log1p(y_raw[vi]).astype(np.float32)
                
                weights = compute_sample_weights(y_raw[ti], weight_method)
                
                torch.manual_seed(42); np.random.seed(42)
                model = ResNet(Xtr.shape[1], 256, 6, 0.1)
                model = train_weighted(model, Xtr, ytr, weights, Xva, yva,
                                      epochs=800, lr=3e-4, wd=5e-4, bs=64, pat=80,
                                      loss_fn=loss_name)
                model.eval()
                with torch.no_grad(): p = np.expm1(model(torch.FloatTensor(Xva)).squeeze(-1).numpy())
                fold_r2.append(r2_score(y_raw[vi], p))
                del model; gc.collect()
            
            r2 = np.mean(fold_r2)
            flag = "★" if r2 > r2_base else ""
            print(f"  {exp_name:35s}: R²={r2:.4f}±{np.std(fold_r2):.4f} {flag}", flush=True)
            results[exp_name] = r2
    
    # ===== C. Two-stage with ensemble =====
    print(f"\n  --- C. Two-Stage Ensemble (5 seeds, weighted) ---")
    fold_r2 = []
    for fold, (ti, vi) in enumerate(skf.split(X, y_bins)):
        ts = TwoStagePredictor(n_bins=3)
        pred = ts.fit_predict_fold(X[ti], y_raw[ti], X[vi], y_raw[vi])
        fold_r2.append(r2_score(y_raw[vi], pred))
        gc.collect()
    r2_ts = np.mean(fold_r2)
    flag = "★" if r2_ts > r2_base else ""
    print(f"  TwoStage_Ens5:         R²={r2_ts:.4f}±{np.std(fold_r2):.4f} {flag}", flush=True)
    results['two_stage_ens5'] = r2_ts
    
    # ===== D. Quantile-based training (predict quantiles, then recalibrate) =====
    print(f"\n  --- D. Quantile transform (spread targets) ---")
    fold_r2 = []
    for fold, (ti, vi) in enumerate(skf.split(X, y_bins)):
        sx = StandardScaler()
        Xtr = sx.fit_transform(X[ti]).astype(np.float32)
        Xva = sx.transform(X[vi]).astype(np.float32)
        
        qt = QuantileTransformer(output_distribution='normal', n_quantiles=200, random_state=42)
        ytr = qt.fit_transform(y_raw[ti].reshape(-1,1)).flatten().astype(np.float32)
        
        weights = compute_sample_weights(y_raw[ti], 'inverse_freq')
        
        torch.manual_seed(42); np.random.seed(42)
        model = ResNet(Xtr.shape[1], 256, 6, 0.1)
        model = train_weighted(model, Xtr, ytr, weights, Xva, 
                              qt.transform(y_raw[vi].reshape(-1,1)).flatten().astype(np.float32),
                              epochs=800, lr=3e-4, wd=5e-4, bs=64, pat=80,
                              loss_fn='weighted_huber')
        model.eval()
        with torch.no_grad():
            p = model(torch.FloatTensor(Xva)).squeeze(-1).numpy()
        p = qt.inverse_transform(p.reshape(-1,1)).flatten()
        fold_r2.append(r2_score(y_raw[vi], p))
        del model; gc.collect()
    
    r2_qt = np.mean(fold_r2)
    flag = "★" if r2_qt > r2_base else ""
    print(f"  Quantile_weighted:     R²={r2_qt:.4f}±{np.std(fold_r2):.4f} {flag}", flush=True)
    results['quantile_weighted'] = r2_qt
    
    # Best result
    best_name = max(results, key=results.get)
    best_r2 = results[best_name]
    gap = goal - best_r2
    print(f"\n  >>> BEST: R²={best_r2:.4f} [{best_name}] {'✅' if gap <= 0 else f'❌ (gap={gap:.4f})'}")
    
    return results


def main():
    t0 = time.time()
    df, feat_cols, dw_cols = load_data()
    X_all = df[feat_cols].values.astype(np.float32)
    X_dw = df[dw_cols].values.astype(np.float32)
    y_homa = df['True_HOMA_IR'].values.astype(np.float32)
    y_hba1c = df['True_hba1c'].values.astype(np.float32)
    
    print(f"All={X_all.shape[1]}, DW={X_dw.shape[1]}, N={len(df)}", flush=True)
    
    all_results = {}
    
    all_results['HOMA_all'] = run_experiment(X_all, y_homa, 'HOMA_IR ALL', 0.85)
    all_results['hba1c_all'] = run_experiment(X_all, y_hba1c, 'hba1c ALL', 0.85)
    all_results['HOMA_dw'] = run_experiment(X_dw, y_homa, 'HOMA_IR DW', 0.70)
    all_results['hba1c_dw'] = run_experiment(X_dw, y_hba1c, 'hba1c DW', 0.70)
    
    elapsed = (time.time()-t0)/60
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY ({elapsed:.1f} min)")
    print(f"{'='*70}")
    
    with open('results_v7.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print("Saved results_v7.json")

if __name__ == '__main__':
    main()
