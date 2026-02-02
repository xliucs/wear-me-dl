#!/usr/bin/env python3
"""
V8: RESEARCH-INFORMED TRAINING

Hypotheses from literature review:

H1: GBR STACKING — The WEAR-ME paper (2505.03784) used XGBoost as primary model.
    Combining GBR out-of-fold predictions as an extra feature for NN should capture
    tree-based patterns the NN misses. This is a proven technique for tabular data.

H2: METS-IR FEATURE — Non-insulin based insulin resistance surrogate:
    METS-IR = ln((2*glucose + trig) / HDL). Published as a strong IR predictor
    without needing fasting insulin.

H3: OUTLIER CAPPING — The WEAR-ME paper excluded HOMA-IR >= 15.
    Capping extreme values reduces noise from physiological outliers.

H4: MAE + SMOOTH L1 LOSS — The paper used MAE + Smooth L1, not MSE.
    More robust to outliers, better for skewed targets.

H5: SMOGN-STYLE SYNTHETIC OVERSAMPLING — Generate synthetic minority samples
    for high HOMA-IR range using interpolation + Gaussian noise.

H6: 10-SEED ENSEMBLE — 5-seed gave best result. Scaling to 10 seeds should
    further reduce variance.

H7: LOG-COSH LOSS — Smoother alternative to Huber, differentiable everywhere.
    log(cosh(x)) ≈ 0.5*x² for small x, |x| - log(2) for large x.

References:
- WEAR-ME paper: arxiv.org/abs/2505.03784 (R²=0.50 with all features)
- METS-IR: journals.sagepub.com/doi/10.1177/03000605241285550
- SMOGN: proceedings.mlr.press/v74/branco17a.html
"""
import gc, warnings, json, time, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from copy import deepcopy

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

# ================================================================
# DATA LOADING
# ================================================================

def load_data():
    df = pd.read_csv('data.csv', skiprows=[0])
    df['sex_num'] = (df['sex'] == 'Male').astype(float)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    
    # === ENGINEERED FEATURES ===
    # Standard metabolic ratios
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
    df['insulin_proxy'] = df['bmi'] * df['trig_hdl']
    df['insulin_proxy2'] = df['bmi_sq'] * df['log_trig'] / 100
    df['liver_stress'] = df['alt'] * df['ggt'] / 100
    df['crp_trig'] = df['crp'] * df['triglycerides'] / 100
    df['bmi_cubed'] = df['bmi'] ** 3 / 10000
    df['wbc_bmi'] = df['white_blood_cell'] * df['bmi'] if 'white_blood_cell' in df.columns else 0
    df['uric_bmi'] = df['uric_acid'] * df['bmi'] if 'uric_acid' in df.columns else 0
    
    # === NEW H2: METS-IR (published non-insulin surrogate for IR) ===
    df['mets_ir'] = np.log((2 * df['glucose'] + df['triglycerides']) / (df['hdl'] + 0.1))
    df['mets_ir_bmi'] = df['mets_ir'] * df['bmi']
    
    # === NEW: Additional physiologically-informed features ===
    df['vat_proxy'] = df['bmi'] * df['triglycerides'] / (df['hdl'] + 0.1)  # visceral adiposity proxy
    df['atherogenic_idx'] = np.log10(df['triglycerides'] / (df['hdl'] + 0.1))  # atherogenic index of plasma
    df['crp_glucose'] = df['crp'] * df['glucose'] / 100  # inflammation × glycemia interaction
    
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


# ================================================================
# MODELS
# ================================================================

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


# ================================================================
# LOSS FUNCTIONS
# ================================================================

def smooth_l1_loss(pred, target, weights, beta=1.0):
    """Smooth L1 loss (same as paper) with per-sample weights."""
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return (loss * weights).mean()

def log_cosh_loss(pred, target, weights):
    """Log-cosh loss — smooth approximation between L1 and L2."""
    diff = pred - target
    loss = torch.log(torch.cosh(diff + 1e-12))
    return (loss * weights).mean()

def mae_smooth_l1_loss(pred, target, weights, alpha=0.5, beta=1.0):
    """Combined MAE + Smooth L1 loss (inspired by paper's approach)."""
    mae = torch.abs(pred - target)
    diff = torch.abs(pred - target)
    sl1 = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    loss = alpha * mae + (1 - alpha) * sl1
    return (loss * weights).mean()

def focal_mse_loss(pred, target, weights, gamma=2.0):
    """Focal regression loss."""
    mse = (pred - target) ** 2
    focal_weight = (mse.detach() / (mse.detach().mean() + 1e-6)) ** (gamma / 2)
    focal_weight = focal_weight.clamp(max=10.0)
    return (mse * weights * focal_weight).mean()

def asymmetric_loss(pred, target, weights, delta=1.0, under_penalty=2.0):
    """Penalize under-prediction of high values more."""
    diff = pred - target
    abs_diff = torch.abs(diff)
    quadratic = torch.clamp(abs_diff, max=delta)
    linear = abs_diff - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    under_mask = (diff < 0).float()
    loss = loss * (1 + under_mask * (under_penalty - 1))
    return (loss * weights).mean()


# ================================================================
# H5: SMOGN-STYLE SYNTHETIC OVERSAMPLING
# ================================================================

def smogn_oversample(X, y, threshold_percentile=75, k=5, n_synthetic=100):
    """Generate synthetic minority samples for high-target regions.
    
    Uses SMOTE-like interpolation + Gaussian noise for regression.
    Targets above threshold_percentile are considered 'minority'.
    """
    threshold = np.percentile(y, threshold_percentile)
    minority_mask = y >= threshold
    minority_X = X[minority_mask]
    minority_y = y[minority_mask]
    
    if len(minority_X) < k + 1:
        return X, y
    
    # KNN within minority samples
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k+1).fit(minority_X)
    distances, indices = nn.kneighbors(minority_X)
    
    synthetic_X = []
    synthetic_y = []
    
    for _ in range(n_synthetic):
        # Pick random minority sample
        i = np.random.randint(len(minority_X))
        # Pick random neighbor (skip self at index 0)
        j = indices[i, np.random.randint(1, k+1)]
        
        # Interpolate
        lam = np.random.uniform(0, 1)
        new_x = minority_X[i] + lam * (minority_X[j] - minority_X[i])
        new_y = minority_y[i] + lam * (minority_y[j] - minority_y[i])
        
        # Add small Gaussian noise
        noise = np.random.normal(0, 0.05, size=new_x.shape)
        new_x = new_x + noise
        
        synthetic_X.append(new_x)
        synthetic_y.append(new_y)
    
    aug_X = np.vstack([X, np.array(synthetic_X)])
    aug_y = np.concatenate([y, np.array(synthetic_y)])
    
    return aug_X, aug_y


# ================================================================
# H1: GBR STACKING (out-of-fold predictions)
# ================================================================

def get_gbr_oof_predictions(X, y, y_bins, n_splits=5, target_name='HOMA_IR'):
    """Generate out-of-fold GBR predictions to use as meta-feature."""
    print(f"  Computing GBR OOF predictions for {target_name}...")
    
    oof_preds = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_bins)):
        gbr = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, max_depth=5,
            min_samples_leaf=10, l2_regularization=0.1,
            random_state=42
        )
        gbr.fit(X[train_idx], y[train_idx])
        oof_preds[val_idx] = gbr.predict(X[val_idx])
    
    oof_r2 = r2_score(y, oof_preds)
    print(f"  GBR OOF R²: {oof_r2:.4f}")
    return oof_preds, oof_r2


# ================================================================
# TRAINING
# ================================================================

def compute_weights(y_raw, method='sqrt_inverse'):
    if method == 'inverse_freq':
        bins = np.quantile(y_raw, np.linspace(0, 1, 11))
        bins[0] -= 1; bins[-1] += 1
        bin_idx = np.clip(np.digitize(y_raw, bins) - 1, 0, 9)
        bin_counts = np.bincount(bin_idx, minlength=10)
        bin_weights = 1.0 / (bin_counts + 1)
        bin_weights = bin_weights / bin_weights.mean()
        return np.clip(bin_weights[bin_idx], 0.2, 5.0).astype(np.float32)
    elif method == 'sqrt_inverse':
        bins = np.quantile(y_raw, np.linspace(0, 1, 11))
        bins[0] -= 1; bins[-1] += 1
        bin_idx = np.clip(np.digitize(y_raw, bins) - 1, 0, 9)
        bin_counts = np.bincount(bin_idx, minlength=10)
        bin_weights = 1.0 / np.sqrt(bin_counts + 1)
        bin_weights = bin_weights / bin_weights.mean()
        return np.clip(bin_weights[bin_idx], 0.3, 3.0).astype(np.float32)
    else:
        return np.ones(len(y_raw), dtype=np.float32)


def train_model(model, Xtr, ytr, wtr, Xva, yva, epochs=800, lr=3e-4, wd=5e-4, 
                bs=64, pat=80, loss_fn='mae_smooth_l1'):
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(wtr), num_samples=len(wtr), replacement=True)
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
            if np.random.random() < 0.5:
                lam = np.random.beta(0.3, 0.3)
                idx = torch.randperm(bx.size(0))
                bx = lam*bx + (1-lam)*bx[idx]
                by = lam*by + (1-lam)*by[idx]
                bw = lam*bw + (1-lam)*bw[idx]
            if np.random.random() < 0.3:
                bx = bx + torch.randn_like(bx) * 0.05
            
            p = model(bx).squeeze(-1)
            
            if loss_fn == 'mae_smooth_l1':
                loss = mae_smooth_l1_loss(p, by, bw)
            elif loss_fn == 'log_cosh':
                loss = log_cosh_loss(p, by, bw)
            elif loss_fn == 'smooth_l1':
                loss = smooth_l1_loss(p, by, bw)
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
            vl = F.l1_loss(model(vx).squeeze(-1), vy).item()  # Use MAE for early stopping (like paper)
        if vl < best - 1e-7:
            best = vl; bst = deepcopy(model.state_dict()); wait = 0
        else:
            wait += 1
            if wait >= pat: break
    
    if bst: model.load_state_dict(bst)
    return model


def two_stage_ensemble(X_tr, y_tr, w_tr, X_va, y_va, n_feats, n_seeds=10, loss_fn='mae_smooth_l1'):
    """H6: 10-seed ensemble with diversity from different seeds."""
    preds = []
    for seed in range(n_seeds):
        torch.manual_seed(seed * 137 + 42)
        np.random.seed(seed * 137 + 42)
        
        # Stage 1: Train base model
        m1 = ResNet(n_feats, h=256, nb=6, dr=0.12)
        m1 = train_model(m1, X_tr, y_tr, w_tr, X_va, y_va, 
                        epochs=600, lr=3e-4, wd=5e-4, pat=60, loss_fn=loss_fn)
        
        # Stage 2: Fine-tune on hard examples
        m1.eval()
        with torch.no_grad():
            p1 = m1(torch.FloatTensor(X_tr)).squeeze(-1).numpy()
        errors = np.abs(p1 - y_tr)
        hard_weight = 1 + 2 * (errors / (errors.max() + 1e-8))
        combined_w = (w_tr * hard_weight).astype(np.float32)
        combined_w = combined_w / combined_w.mean()
        
        m2 = deepcopy(m1)
        m2 = train_model(m2, X_tr, y_tr, combined_w, X_va, y_va, 
                        epochs=300, lr=5e-5, wd=1e-3, pat=40, loss_fn=loss_fn)
        
        m2.eval()
        with torch.no_grad():
            pred = m2(torch.FloatTensor(X_va)).squeeze(-1).numpy()
        preds.append(pred)
        del m1, m2; gc.collect()
    
    # Trim mean (remove highest and lowest predictions, average rest)
    preds = np.array(preds)
    if n_seeds >= 5:
        sorted_preds = np.sort(preds, axis=0)
        trimmed = sorted_preds[1:-1]  # remove best and worst
        ensemble_pred = trimmed.mean(axis=0)
    else:
        ensemble_pred = preds.mean(axis=0)
    
    return ensemble_pred


# ================================================================
# MAIN EXPERIMENT RUNNER
# ================================================================

def run_experiment(df, feat_cols, target, goal, label, use_stacking=True, use_smogn=True):
    """Run full experiment with all hypotheses for one target+feature set."""
    print(f"\n{'='*70}")
    print(f"{label} | {len(feat_cols)} features | Goal: {goal}")
    print(f"{'='*70}")
    
    X_raw = df[feat_cols].values.astype(np.float32)
    
    # H3: Cap HOMA-IR at 15 (like the paper)
    if 'HOMA_IR' in target:
        y_raw = df[target].values.copy()
        n_capped = (y_raw >= 15).sum()
        y_raw = np.clip(y_raw, None, 14.99)
        print(f"  H3: Capped {n_capped} samples at HOMA-IR=15")
    else:
        y_raw = df[target].values.copy()
    
    # Log transform for HOMA_IR
    if 'HOMA_IR' in target:
        y_transformed = np.log1p(y_raw)
    else:
        y_transformed = y_raw.copy()
    
    # Stratification bins
    y_bins = pd.qcut(y_raw, q=5, labels=False, duplicates='drop')
    
    # H1: GBR stacking — get OOF predictions
    if use_stacking:
        gbr_oof, gbr_r2 = get_gbr_oof_predictions(X_raw, y_raw, y_bins, target_name=label)
    
    results = {}
    
    # === EXPERIMENT CONFIGS ===
    configs = [
        # (name, loss_fn, weight_method, use_smogn, use_stacking, n_seeds)
        ('baseline_mae_sl1', 'mae_smooth_l1', 'sqrt_inverse', False, False, 1),
        ('log_cosh', 'log_cosh', 'sqrt_inverse', False, False, 1),
        ('smooth_l1', 'smooth_l1', 'sqrt_inverse', False, False, 1),
        ('focal', 'focal', 'sqrt_inverse', False, False, 1),
        ('stacking_mae_sl1', 'mae_smooth_l1', 'sqrt_inverse', False, True, 1),
        ('stacking_log_cosh', 'log_cosh', 'sqrt_inverse', False, True, 1),
        ('smogn_mae_sl1', 'mae_smooth_l1', 'sqrt_inverse', True, False, 1),
        ('smogn_stacking', 'mae_smooth_l1', 'sqrt_inverse', True, True, 1),
        # Ensembles (best configs × more seeds)
        ('ens10_mae_sl1', 'mae_smooth_l1', 'sqrt_inverse', False, False, 10),
        ('ens10_stacking', 'mae_smooth_l1', 'sqrt_inverse', False, True, 10),
        ('ens10_smogn_stacking', 'mae_smooth_l1', 'sqrt_inverse', True, True, 10),
    ]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for cfg_name, loss_fn, weight_method, cfg_smogn, cfg_stack, n_seeds in configs:
        fold_r2s = []
        fold_maes = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y_bins)):
            Xtr, Xva = X_raw[train_idx], X_raw[val_idx]
            ytr_raw, yva_raw = y_raw[train_idx], y_raw[val_idx]
            
            # Scale features
            scaler = StandardScaler()
            Xtr_s = scaler.fit_transform(Xtr)
            Xva_s = scaler.transform(Xva)
            
            # H1: Add GBR stacking features
            if cfg_stack and use_stacking:
                gbr_tr = gbr_oof[train_idx].reshape(-1, 1)
                gbr_va = gbr_oof[val_idx].reshape(-1, 1)
                # Scale the GBR feature too
                gbr_scaler = StandardScaler()
                gbr_tr = gbr_scaler.fit_transform(gbr_tr)
                gbr_va = gbr_scaler.transform(gbr_va)
                Xtr_s = np.hstack([Xtr_s, gbr_tr])
                Xva_s = np.hstack([Xva_s, gbr_va])
            
            # Transform target
            if 'HOMA_IR' in target:
                ytr = np.log1p(ytr_raw)
                yva = np.log1p(yva_raw)
            else:
                ytr = ytr_raw.copy()
                yva = yva_raw.copy()
            
            # H5: SMOGN oversampling
            if cfg_smogn and use_smogn:
                Xtr_s, ytr = smogn_oversample(Xtr_s, ytr, threshold_percentile=75, 
                                               k=5, n_synthetic=int(len(ytr)*0.3))
            
            # Compute weights
            if 'HOMA_IR' in target:
                wtr = compute_weights(np.expm1(ytr), method=weight_method)
            else:
                wtr = compute_weights(ytr, method=weight_method)
            
            n_feats = Xtr_s.shape[1]
            
            if n_seeds > 1:
                # H6: Multi-seed ensemble
                pred_va = two_stage_ensemble(Xtr_s, ytr, wtr, Xva_s, yva, 
                                            n_feats, n_seeds=n_seeds, loss_fn=loss_fn)
            else:
                # Single model
                torch.manual_seed(42)
                model = ResNet(n_feats, h=256, nb=6, dr=0.12)
                model = train_model(model, Xtr_s, ytr, wtr, Xva_s, yva,
                                   epochs=800, lr=3e-4, wd=5e-4, pat=80, loss_fn=loss_fn)
                model.eval()
                with torch.no_grad():
                    pred_va = model(torch.FloatTensor(Xva_s)).squeeze(-1).numpy()
                del model; gc.collect()
            
            # Inverse transform
            if 'HOMA_IR' in target:
                pred_orig = np.expm1(pred_va)
            else:
                pred_orig = pred_va
            
            r2 = r2_score(yva_raw, pred_orig)
            mae = mean_absolute_error(yva_raw, pred_orig)
            fold_r2s.append(r2)
            fold_maes.append(mae)
        
        mean_r2 = np.mean(fold_r2s)
        std_r2 = np.std(fold_r2s)
        mean_mae = np.mean(fold_maes)
        
        star = '★' if not results or mean_r2 > max(r['r2'] for r in results.values()) else ''
        print(f"  {cfg_name:40s}: R²={mean_r2:.4f}±{std_r2:.4f} MAE={mean_mae:.3f} {star}")
        
        results[cfg_name] = {
            'r2': mean_r2, 'std': std_r2, 'mae': mean_mae,
            'folds': fold_r2s
        }
    
    best_name = max(results, key=lambda k: results[k]['r2'])
    best_r2 = results[best_name]['r2']
    print(f"\n  >>> BEST: R²={best_r2:.4f} [{best_name}] {'✅' if best_r2 >= goal else '❌'} (gap={goal-best_r2:.4f})")
    
    return results


# ================================================================
# MAIN
# ================================================================

if __name__ == '__main__':
    print("V8: RESEARCH-INFORMED TRAINING")
    print("=" * 70)
    print("Hypotheses being tested:")
    print("  H1: GBR Stacking (tree-based OOF predictions as meta-feature)")
    print("  H2: METS-IR feature (published non-insulin IR surrogate)")
    print("  H3: Outlier capping at HOMA-IR=15 (following WEAR-ME paper)")
    print("  H4: MAE + Smooth L1 loss (as used in the paper)")
    print("  H5: SMOGN-style synthetic oversampling for tail")
    print("  H6: 10-seed ensemble (scaling from 5-seed)")
    print("  H7: Log-cosh loss (smooth L1/L2 hybrid)")
    print()
    
    df, feat_cols, dw_cols = load_data()
    print(f"Data: {len(df)} samples, {len(feat_cols)} ALL features, {len(dw_cols)} DW features")
    
    # Run ALL experiments
    t0 = time.time()
    
    # 1. HOMA_IR ALL features
    results_homa_all = run_experiment(df, feat_cols, 'True_HOMA_IR', 0.85, 'HOMA_IR ALL')
    
    # 2. hba1c ALL features
    results_hba1c_all = run_experiment(df, feat_cols, 'True_hba1c', 0.85, 'hba1c ALL')
    
    # 3. HOMA_IR DW only
    results_homa_dw = run_experiment(df, dw_cols, 'True_HOMA_IR', 0.70, 'HOMA_IR DW',
                                    use_stacking=True, use_smogn=True)
    
    # 4. hba1c DW only
    results_hba1c_dw = run_experiment(df, dw_cols, 'True_hba1c', 0.70, 'hba1c DW',
                                     use_stacking=True, use_smogn=True)
    
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"TOTAL TIME: {elapsed/3600:.1f}h")
    print(f"{'='*70}")
    
    # Summary
    print("\n=== FINAL SUMMARY ===")
    for name, results in [('HOMA_IR ALL', results_homa_all), ('hba1c ALL', results_hba1c_all),
                          ('HOMA_IR DW', results_homa_dw), ('hba1c DW', results_hba1c_dw)]:
        best = max(results, key=lambda k: results[k]['r2'])
        print(f"  {name:15s}: R²={results[best]['r2']:.4f}±{results[best]['std']:.4f} [{best}]")
