#!/usr/bin/env python3
"""
ERROR ANALYSIS: Where does the model fail and why?
- Residual analysis by target range
- t-SNE visualization of easy vs hard examples
- Feature importance for high-error samples
- Subgroup analysis (BMI, age, sex, glucose ranges)
- Correlation of residuals with features
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, math, gc, warnings
warnings.filterwarnings('ignore')

def load_data():
    df = pd.read_csv('data.csv', skiprows=[0])
    df['sex_num'] = (df['sex'] == 'Male').astype(float)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    
    # Engineered features
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
    
    mask = df[['True_HOMA_IR','True_hba1c']].notna().all(axis=1)
    df = df[mask].reset_index(drop=True)
    for col in feat_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(df[col].median())
    
    return df, feat_cols

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

def train_model(Xtr, ytr, Xva, yva, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ResNet(Xtr.shape[1], 256, 6, 0.1)
    ds = TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(ytr))
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    vx, vy = torch.FloatTensor(Xva), torch.FloatTensor(yva)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
    warmup = 20; epochs = 800
    def lr_fn(ep):
        if ep < warmup: return (ep+1)/warmup
        return max(0.01, 0.5*(1+math.cos(math.pi*(ep-warmup)/max(epochs-warmup,1))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    best = 1e9; bst = None; wait = 0
    for ep in range(epochs):
        model.train()
        for bx, by in dl:
            if np.random.random() < 0.5:
                lam = np.random.beta(0.3, 0.3)
                idx = torch.randperm(bx.size(0))
                bx = lam*bx + (1-lam)*bx[idx]; by = lam*by + (1-lam)*by[idx]
            p = model(bx).squeeze(-1)
            loss = F.huber_loss(p, by, delta=1.0)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad(): vl = F.mse_loss(model(vx).squeeze(-1), vy).item()
        if vl < best - 1e-7: best = vl; bst = deepcopy(model.state_dict()); wait = 0
        else:
            wait += 1
            if wait >= 80: break
    if bst: model.load_state_dict(bst)
    return model

def main():
    print("="*70)
    print("ERROR ANALYSIS - Diagnosing where the model fails")
    print("="*70)
    
    df, feat_cols = load_data()
    X = df[feat_cols].values.astype(np.float32)
    
    for target in ['True_HOMA_IR', 'True_hba1c']:
        print(f"\n{'='*70}")
        print(f"TARGET: {target}")
        print(f"{'='*70}")
        
        y_raw = df[target].values.astype(np.float32)
        y_log = np.log1p(y_raw).astype(np.float32)
        
        # ===== 1. TARGET DISTRIBUTION ANALYSIS =====
        print(f"\n--- Target Distribution ---")
        print(f"  Range: [{y_raw.min():.2f}, {y_raw.max():.2f}]")
        print(f"  Mean: {y_raw.mean():.2f}, Median: {np.median(y_raw):.2f}, Std: {y_raw.std():.2f}")
        print(f"  Skewness: {pd.Series(y_raw).skew():.2f}")
        pcts = np.percentile(y_raw, [5, 10, 25, 50, 75, 90, 95])
        print(f"  Percentiles [5,10,25,50,75,90,95]: {[f'{p:.2f}' for p in pcts]}")
        
        # Count samples in different ranges
        if target == 'True_HOMA_IR':
            bins = [0, 1, 2, 3, 5, 8, 15]
            labels_bin = ['<1', '1-2', '2-3', '3-5', '5-8', '8+']
        else:
            bins = [0, 5.0, 5.4, 5.7, 6.0, 6.5, 7.0, 10]
            labels_bin = ['<5.0', '5.0-5.4', '5.4-5.7', '5.7-6.0', '6.0-6.5', '6.5-7.0', '7.0+']
        
        bin_counts = pd.cut(y_raw, bins=bins, labels=labels_bin).value_counts().sort_index()
        print(f"\n  Sample distribution by range:")
        for b, c in bin_counts.items():
            print(f"    {b}: {c} samples ({100*c/len(y_raw):.1f}%)")
        
        # ===== 2. COLLECT OOF PREDICTIONS =====
        print(f"\n--- Collecting OOF predictions (5-fold CV) ---")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_pred = np.zeros(len(y_raw))
        oof_fold = np.zeros(len(y_raw), dtype=int)
        
        for fold, (ti, vi) in enumerate(kf.split(X)):
            sx = StandardScaler()
            Xtr = sx.fit_transform(X[ti]).astype(np.float32)
            Xva = sx.transform(X[vi]).astype(np.float32)
            ytr = np.log1p(y_raw[ti])
            
            model = train_model(Xtr, ytr, Xva, np.log1p(y_raw[vi]), seed=42)
            model.eval()
            with torch.no_grad():
                p = model(torch.FloatTensor(Xva)).squeeze(-1).numpy()
            oof_pred[vi] = np.expm1(p)
            oof_fold[vi] = fold
            r2 = r2_score(y_raw[vi], oof_pred[vi])
            print(f"  Fold {fold}: R²={r2:.4f}, n={len(vi)}")
            del model; gc.collect()
        
        overall_r2 = r2_score(y_raw, oof_pred)
        overall_mae = mean_absolute_error(y_raw, oof_pred)
        print(f"\n  Overall OOF: R²={overall_r2:.4f}, MAE={overall_mae:.4f}")
        
        # ===== 3. RESIDUAL ANALYSIS =====
        residuals = y_raw - oof_pred
        abs_residuals = np.abs(residuals)
        rel_error = abs_residuals / (np.abs(y_raw) + 0.01)
        
        print(f"\n--- Residual Analysis ---")
        print(f"  Mean residual: {residuals.mean():.4f} (bias)")
        print(f"  Std residual: {residuals.std():.4f}")
        print(f"  Mean abs error: {abs_residuals.mean():.4f}")
        print(f"  Median abs error: {np.median(abs_residuals):.4f}")
        print(f"  Mean rel error: {rel_error.mean():.2%}")
        
        # Error by target range
        print(f"\n--- Error by Target Range ---")
        df_analysis = pd.DataFrame({
            'y_true': y_raw, 'y_pred': oof_pred, 'residual': residuals,
            'abs_error': abs_residuals, 'rel_error': rel_error, 'fold': oof_fold
        })
        df_analysis['bin'] = pd.cut(y_raw, bins=bins, labels=labels_bin)
        
        for b in labels_bin:
            mask = df_analysis['bin'] == b
            if mask.sum() == 0: continue
            sub = df_analysis[mask]
            r2_sub = r2_score(sub['y_true'], sub['y_pred']) if len(sub) > 1 else float('nan')
            print(f"  {b:10s}: n={len(sub):4d}, MAE={sub['abs_error'].mean():.3f}, "
                  f"RelErr={sub['rel_error'].mean():.1%}, R²={r2_sub:.3f}, "
                  f"Bias={sub['residual'].mean():+.3f}")
        
        # ===== 4. WORST PREDICTIONS =====
        print(f"\n--- Top 20 Worst Predictions ---")
        worst_idx = np.argsort(-abs_residuals)[:20]
        print(f"  {'True':>8s} {'Pred':>8s} {'Error':>8s} {'BMI':>6s} {'Age':>4s} {'Gluc':>6s} {'TG':>6s} {'Sex':>4s}")
        for i in worst_idx:
            print(f"  {y_raw[i]:8.2f} {oof_pred[i]:8.2f} {residuals[i]:+8.2f} "
                  f"{df.iloc[i]['bmi']:6.1f} {df.iloc[i]['age']:4.0f} "
                  f"{df.iloc[i]['glucose']:6.1f} {df.iloc[i]['triglycerides']:6.1f} "
                  f"{'M' if df.iloc[i]['sex_num']==1 else 'F':>4s}")
        
        # ===== 5. CORRELATION OF RESIDUALS WITH FEATURES =====
        print(f"\n--- Feature correlations with absolute error ---")
        corrs = []
        for col in feat_cols:
            c = np.corrcoef(df[col].values, abs_residuals)[0,1]
            if not np.isnan(c): corrs.append((col, c))
        corrs.sort(key=lambda x: abs(x[1]), reverse=True)
        for col, c in corrs[:20]:
            print(f"  {col:35s}: r={c:+.3f}")
        
        # ===== 6. SUBGROUP ANALYSIS =====
        print(f"\n--- Subgroup Analysis ---")
        # By BMI
        for lo, hi, label in [(0,25,'Normal BMI'), (25,30,'Overweight'), (30,100,'Obese')]:
            mask = (df['bmi'] >= lo) & (df['bmi'] < hi)
            if mask.sum() < 10: continue
            r2 = r2_score(y_raw[mask], oof_pred[mask]) if mask.sum() > 1 else float('nan')
            print(f"  {label:20s}: n={mask.sum():4d}, R²={r2:.3f}, MAE={abs_residuals[mask].mean():.3f}")
        
        # By sex
        for s, label in [(1, 'Male'), (0, 'Female')]:
            mask = df['sex_num'] == s
            r2 = r2_score(y_raw[mask], oof_pred[mask])
            print(f"  {label:20s}: n={mask.sum():4d}, R²={r2:.3f}, MAE={abs_residuals[mask].mean():.3f}")
        
        # By age
        for lo, hi, label in [(0,40,'Young'), (40,55,'Middle'), (55,100,'Senior')]:
            mask = (df['age'] >= lo) & (df['age'] < hi)
            if mask.sum() < 10: continue
            r2 = r2_score(y_raw[mask], oof_pred[mask]) if mask.sum() > 1 else float('nan')
            print(f"  {label:20s}: n={mask.sum():4d}, R²={r2:.3f}, MAE={abs_residuals[mask].mean():.3f}")
        
        # By glucose
        for lo, hi, label in [(0,90,'Low glucose'), (90,100,'Normal glucose'), (100,126,'Pre-diabetic'), (126,500,'Diabetic')]:
            mask = (df['glucose'] >= lo) & (df['glucose'] < hi)
            if mask.sum() < 5: continue
            r2 = r2_score(y_raw[mask], oof_pred[mask]) if mask.sum() > 1 else float('nan')
            print(f"  {label:20s}: n={mask.sum():4d}, R²={r2:.3f}, MAE={abs_residuals[mask].mean():.3f}")
        
        # ===== 7. t-SNE VISUALIZATION =====
        print(f"\n--- t-SNE visualization ---")
        sx = StandardScaler()
        X_scaled = sx.fit_transform(X)
        
        # PCA to 30 dims first (faster t-SNE)
        pca = PCA(n_components=min(30, X.shape[1]))
        X_pca = pca.fit_transform(X_scaled)
        print(f"  PCA explained variance (30 comps): {pca.explained_variance_ratio_.sum():.2%}")
        
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
        X_tsne = tsne.fit_transform(X_pca)
        
        # Plot 1: t-SNE colored by target value
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        sc = axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_raw, cmap='viridis', s=8, alpha=0.7)
        plt.colorbar(sc, ax=axes[0])
        axes[0].set_title(f'{target} - True Values')
        
        # Plot 2: t-SNE colored by absolute error
        sc = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=abs_residuals, cmap='Reds', s=8, alpha=0.7)
        plt.colorbar(sc, ax=axes[1])
        axes[1].set_title(f'{target} - Absolute Error (red=high)')
        
        # Plot 3: Predicted vs actual
        axes[2].scatter(y_raw, oof_pred, s=8, alpha=0.5)
        mn, mx = y_raw.min(), y_raw.max()
        axes[2].plot([mn, mx], [mn, mx], 'r--', lw=2)
        axes[2].set_xlabel('True'); axes[2].set_ylabel('Predicted')
        axes[2].set_title(f'{target} - Pred vs True (R²={overall_r2:.3f})')
        
        plt.tight_layout()
        fname = f'analysis_{target}.png'
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"  Saved {fname}")
        
        # ===== 8. CLUSTER ANALYSIS =====
        print(f"\n--- Cluster: high error vs low error samples ---")
        high_err_mask = abs_residuals > np.percentile(abs_residuals, 75)
        low_err_mask = abs_residuals < np.percentile(abs_residuals, 25)
        
        print(f"  High error samples (top 25%): n={high_err_mask.sum()}")
        print(f"  Low error samples (bottom 25%): n={low_err_mask.sum()}")
        
        # Compare feature distributions
        print(f"\n  Feature differences (high_err - low_err means):")
        diffs = []
        for col in feat_cols:
            hi_mean = df.loc[high_err_mask, col].mean()
            lo_mean = df.loc[low_err_mask, col].mean()
            overall_std = df[col].std()
            if overall_std > 0:
                effect = (hi_mean - lo_mean) / overall_std
                diffs.append((col, effect, hi_mean, lo_mean))
        diffs.sort(key=lambda x: abs(x[1]), reverse=True)
        for col, eff, hi, lo in diffs[:15]:
            print(f"    {col:35s}: effect={eff:+.3f} (high_err={hi:.2f}, low_err={lo:.2f})")
        
        # ===== 9. PREDICTION COMPRESSION CHECK =====
        print(f"\n--- Prediction Compression ---")
        print(f"  True std: {y_raw.std():.4f}")
        print(f"  Pred std: {oof_pred.std():.4f}")
        print(f"  Compression ratio: {oof_pred.std()/y_raw.std():.3f}")
        print(f"  True range: [{y_raw.min():.2f}, {y_raw.max():.2f}]")
        print(f"  Pred range: [{oof_pred.min():.2f}, {oof_pred.max():.2f}]")
        
        # Check if model is just predicting mean for hard cases
        high_true = y_raw > np.percentile(y_raw, 90)
        low_true = y_raw < np.percentile(y_raw, 10)
        print(f"\n  Top 10% true values: mean_true={y_raw[high_true].mean():.2f}, mean_pred={oof_pred[high_true].mean():.2f}")
        print(f"  Bottom 10% true values: mean_true={y_raw[low_true].mean():.2f}, mean_pred={oof_pred[low_true].mean():.2f}")
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == '__main__':
    main()
