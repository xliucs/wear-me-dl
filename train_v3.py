#!/usr/bin/env python3
"""
V3: Lean, CPU-based, focused on results.
Key insights from SoTA tabular DL research:
- For <1000 samples, simpler models often beat transformers
- Feature engineering + proper preprocessing > model complexity
- Quantile/log target transforms critical for skewed targets
- Huber loss for robustness
- Ensemble predictions across folds
"""

import warnings, json, time, math, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import r2_score, mean_squared_error
from copy import deepcopy

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device('cpu')
print(f"Device: {DEVICE}", flush=True)

# ============================================================
# DATA
# ============================================================

def load_data():
    df = pd.read_csv('data.csv', skiprows=[0])
    
    # Sex encoding
    sex_map = {'Female': 0, 'Male': 1}
    df['sex_num'] = df['sex'].map(lambda x: sex_map.get(x, 0.5))
    df['is_male'] = (df['sex'] == 'Male').astype(float)
    
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    
    # Feature engineering
    eng = {}
    eng['trig_hdl'] = df['triglycerides'] / (df['hdl'] + 0.1)
    eng['glucose_bmi'] = df['glucose'] * df['bmi']
    eng['glucose_trig'] = df['glucose'] * df['triglycerides'] / 1000
    eng['bmi_age'] = df['bmi'] * df['age']
    eng['ldl_hdl'] = df['ldl'] / (df['hdl'] + 0.1)
    eng['ast_alt'] = df['ast'] / (df['alt'] + 0.1)
    eng['bun_creat'] = df['bun'] / (df['creatinine'] + 0.01)
    eng['nlr'] = df['absolute_neutrophils'] / (df['absolute_lymphocytes'] + 0.1)
    eng['rhr_hrv'] = df['Resting Heart Rate (mean)'] / (df['HRV (mean)'] + 0.1)
    eng['steps_sleep'] = df['STEPS (mean)'] / (df['SLEEP Duration (mean)'] + 1)
    eng['activity'] = df['STEPS (mean)'] * df['AZM Weekly (mean)'] / 1e6
    eng['glucose_sq'] = df['glucose'] ** 2 / 1000
    eng['bmi_sq'] = df['bmi'] ** 2
    eng['log_trig'] = np.log1p(df['triglycerides'])
    eng['log_glucose'] = np.log1p(df['glucose'])
    eng['log_crp'] = np.log1p(df['crp'])
    eng['ggt_alt'] = df['ggt'] / (df['alt'] + 0.1)
    eng['hb_hct'] = df['hb'] / (df['hematocrit'] + 0.1)
    eng['hr_cv'] = df['Resting Heart Rate (std)'] / (df['Resting Heart Rate (mean)'] + 0.1)
    eng['hrv_cv'] = df['HRV (std)'] / (df['HRV (mean)'] + 0.1)
    eng['steps_cv'] = df['STEPS (std)'] / (df['STEPS (mean)'] + 1)
    eng['sleep_cv'] = df['SLEEP Duration (std)'] / (df['SLEEP Duration (mean)'] + 1)
    eng['chol_trig'] = df['total cholesterol'] / (df['triglycerides'] + 0.1)
    eng['non_hdl_hdl'] = df['non hdl'] / (df['hdl'] + 0.1)
    eng['glucose_insulin_proxy'] = df['glucose'] * eng['trig_hdl']  # proxy for insulin resistance
    
    for k, v in eng.items():
        df[k] = v
    
    # Define feature groups
    demo = ['age', 'bmi', 'sex_num', 'is_male']
    fitbit = ['Resting Heart Rate (mean)', 'Resting Heart Rate (median)', 'Resting Heart Rate (std)',
              'HRV (mean)', 'HRV (median)', 'HRV (std)',
              'STEPS (mean)', 'STEPS (median)', 'STEPS (std)',
              'SLEEP Duration (mean)', 'SLEEP Duration (median)', 'SLEEP Duration (std)',
              'AZM Weekly (mean)', 'AZM Weekly (median)', 'AZM Weekly (std)']
    wearable_eng = ['rhr_hrv', 'steps_sleep', 'activity', 'hr_cv', 'hrv_cv', 'steps_cv', 'sleep_cv',
                    'bmi_sq', 'bmi_age']
    
    blood = ['total cholesterol', 'hdl', 'triglycerides', 'ldl', 'chol/hdl', 'non hdl',
             'glucose', 'bun', 'creatinine', 'egfr', 'sodium', 'potassium', 'chloride',
             'co2', 'calcium', 'total protein', 'albumin', 'globulin', 'albumin/globulin',
             'total bilirubin', 'alp', 'ast', 'alt', 'crp', 'white_blood_cell', 'red_blood_cell',
             'hb', 'hematocrit', 'mcv', 'mch', 'mchc', 'rdw', 'platelet', 'mpv',
             'absolute_neutrophils', 'absolute_lymphocytes', 'absolute_monocytes',
             'absolute_eosinophils', 'absolute_basophils', 'neutrophils', 'lymphocytes',
             'monocytes', 'eosinophils', 'basophils', 'ggt', 'total testosterone']
    blood_eng = ['trig_hdl', 'glucose_bmi', 'glucose_trig', 'ldl_hdl', 'ast_alt',
                 'bun_creat', 'nlr', 'glucose_sq', 'log_trig', 'log_glucose', 'log_crp',
                 'ggt_alt', 'hb_hct', 'chol_trig', 'non_hdl_hdl', 'glucose_insulin_proxy']
    
    targets = ['True_HOMA_IR', 'True_hba1c']
    label_cols = targets + ['True_IR_Class', 'True_Diabetes_2_Class', 'True_Normoglycemic_2_Class', 'True_Diabetes_3_Class']
    
    all_cols = demo + fitbit + blood + list(eng.keys())
    dw_cols = demo + fitbit + wearable_eng
    
    # Clean
    mask = df[targets].notna().all(axis=1)
    df = df[mask].reset_index(drop=True)
    for col in all_cols + dw_cols:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(df[col].median())
    
    # Deduplicate
    all_cols = list(dict.fromkeys(all_cols))
    dw_cols = list(dict.fromkeys(dw_cols))
    
    # Verify all cols exist
    all_cols = [c for c in all_cols if c in df.columns]
    dw_cols = [c for c in dw_cols if c in df.columns]
    
    return df, all_cols, dw_cols, targets


# ============================================================
# MODELS
# ============================================================

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1, expansion=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout / 2),
        )
    def forward(self, x):
        return x + self.net(x)

class ResNetMLP(nn.Module):
    def __init__(self, in_dim, hidden=256, blocks=6, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(blocks)])
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, 1))
    def forward(self, x):
        return self.head(self.blocks(self.proj(x)))

class FeatureTokenizer(nn.Module):
    def __init__(self, n_feat, d):
        super().__init__()
        self.w = nn.Parameter(torch.empty(n_feat, d))
        self.b = nn.Parameter(torch.zeros(n_feat, d))
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))
    def forward(self, x):
        return x.unsqueeze(-1) * self.w.unsqueeze(0) + self.b.unsqueeze(0)

class FTTransformer(nn.Module):
    def __init__(self, n_feat, d=48, heads=4, layers=2, drop=0.1):
        super().__init__()
        self.tok = FeatureTokenizer(n_feat, d)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, d*4, drop, batch_first=True, activation='gelu', norm_first=True)
        self.tf = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
    def forward(self, x):
        t = torch.cat([self.cls.expand(x.size(0),-1,-1), self.tok(x)], 1)
        return self.head(self.tf(t)[:,0])

class DCNv2(nn.Module):
    def __init__(self, in_dim, hidden=256, n_cross=3, n_deep=3, drop=0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        self.cross_w = nn.ParameterList([nn.Parameter(torch.randn(in_dim, in_dim)*0.01) for _ in range(n_cross)])
        self.cross_b = nn.ParameterList([nn.Parameter(torch.zeros(in_dim)) for _ in range(n_cross)])
        layers = []
        d = in_dim
        for _ in range(n_deep):
            layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(drop)]
            d = hidden
        self.deep = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(in_dim + hidden, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, 1))
    
    def forward(self, x):
        x = self.bn(x)
        x0 = x
        xc = x
        for w, b in zip(self.cross_w, self.cross_b):
            xc = x0 * (xc @ w) + b + xc
        xd = self.deep(x)
        return self.head(torch.cat([xc, xd], 1))


class SNN(nn.Module):
    """Self-Normalizing Neural Network (SELU + AlphaDropout)."""
    def __init__(self, in_dim, hidden=256, layers=6, drop=0.05):
        super().__init__()
        mods = [nn.Linear(in_dim, hidden), nn.SELU(), nn.AlphaDropout(drop)]
        for _ in range(layers - 1):
            mods += [nn.Linear(hidden, hidden), nn.SELU(), nn.AlphaDropout(drop)]
        mods.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*mods)
        # Init weights for SELU
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.net(x)


# ============================================================
# TRAINING
# ============================================================

def train_model(model, X_tr, y_tr, X_va, y_va, epochs=500, lr=5e-4, wd=1e-3, 
                bs=64, patience=50, huber_delta=1.0):
    model = model.to(DEVICE)
    ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
    dl = DataLoader(ds, batch_size=bs, shuffle=True)
    
    vx = torch.FloatTensor(X_va).to(DEVICE)
    vy = torch.FloatTensor(y_va).to(DEVICE)
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    
    warmup = 15
    def lr_fn(ep):
        if ep < warmup: return ep / warmup
        return 0.5 * (1 + math.cos(math.pi * (ep - warmup) / (epochs - warmup)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    
    best_loss = 1e9
    best_state = None
    wait = 0
    
    for ep in range(epochs):
        model.train()
        for bx, by in dl:
            p = model(bx).squeeze()
            loss = F.huber_loss(p, by, delta=huber_delta)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        
        model.eval()
        with torch.no_grad():
            vp = model(vx).squeeze()
            vl = F.mse_loss(vp, vy).item()
        
        if vl < best_loss - 1e-7:
            best_loss = vl
            best_state = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    
    if best_state:
        model.load_state_dict(best_state)
    return model


def run_experiment(name, X, y_raw, target_idx, target_name, goal, model_configs, n_folds=5):
    print(f"\n{'='*70}", flush=True)
    print(f"{name}: {target_name} ({X.shape[1]} features) | Goal: R²≥{goal}", flush=True)
    print(f"{'='*70}", flush=True)
    
    # Try different target transforms
    transforms = ['log', 'quantile', 'standard']
    
    best_overall = -999
    best_info = None
    
    for tfm in transforms:
        for cfg_name, model_fn, lr, wd, bs, epochs, pat in model_configs:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            fold_r2 = []
            all_pred = []
            all_true = []
            
            for fold, (ti, vi) in enumerate(kf.split(X)):
                Xtr, Xva = X[ti], X[vi]
                ytr_raw, yva_raw = y_raw[ti], y_raw[vi]
                
                # Scale features
                sx = StandardScaler()
                Xtr_s = sx.fit_transform(Xtr).astype(np.float32)
                Xva_s = sx.transform(Xva).astype(np.float32)
                
                # Transform target
                if tfm == 'log':
                    ytr = np.log1p(ytr_raw).astype(np.float32)
                    yva = np.log1p(yva_raw).astype(np.float32)
                elif tfm == 'quantile':
                    qt = QuantileTransformer(output_distribution='normal', n_quantiles=min(len(ytr_raw), 200), random_state=42)
                    ytr = qt.fit_transform(ytr_raw.reshape(-1,1)).flatten().astype(np.float32)
                    yva = qt.transform(yva_raw.reshape(-1,1)).flatten().astype(np.float32)
                else:  # standard
                    sy = StandardScaler()
                    ytr = sy.fit_transform(ytr_raw.reshape(-1,1)).flatten().astype(np.float32)
                    yva = sy.transform(yva_raw.reshape(-1,1)).flatten().astype(np.float32)
                
                model = model_fn(Xtr_s.shape[1])
                model = train_model(model, Xtr_s, ytr, Xva_s, yva, 
                                   epochs=epochs, lr=lr, wd=wd, bs=bs, patience=pat)
                
                model.eval()
                with torch.no_grad():
                    pred = model(torch.FloatTensor(Xva_s)).squeeze().numpy()
                
                # Inverse transform
                if tfm == 'log':
                    pred_raw = np.expm1(pred)
                elif tfm == 'quantile':
                    pred_raw = qt.inverse_transform(pred.reshape(-1,1)).flatten()
                else:
                    pred_raw = sy.inverse_transform(pred.reshape(-1,1)).flatten()
                
                r2 = r2_score(yva_raw, pred_raw)
                fold_r2.append(r2)
                all_pred.extend(pred_raw)
                all_true.extend(yva_raw)
            
            mean_r2 = np.mean(fold_r2)
            std_r2 = np.std(fold_r2)
            overall_r2 = r2_score(all_true, all_pred)
            rmse = np.sqrt(mean_squared_error(all_true, all_pred))
            
            marker = " ★" if mean_r2 > best_overall else ""
            print(f"  {cfg_name:25s} {tfm:10s} | R²={mean_r2:.4f}±{std_r2:.4f} | ov={overall_r2:.4f} | RMSE={rmse:.4f}{marker}", flush=True)
            
            if mean_r2 > best_overall:
                best_overall = mean_r2
                best_info = {
                    'name': cfg_name, 'transform': tfm, 'mean_r2': mean_r2, 
                    'std_r2': std_r2, 'overall_r2': overall_r2, 'rmse': rmse,
                    'fold_r2s': fold_r2
                }
    
    flag = "✅" if best_overall >= goal else f"❌ (gap: {goal - best_overall:.4f})"
    print(f"\n  >>> BEST: {best_info['name']} ({best_info['transform']}) R²={best_overall:.4f} {flag}", flush=True)
    return best_info


def main():
    df, all_cols, dw_cols, targets = load_data()
    
    X_all = df[all_cols].values.astype(np.float32)
    X_dw = df[dw_cols].values.astype(np.float32)
    y_homa = df['True_HOMA_IR'].values.astype(np.float32)
    y_hba1c = df['True_hba1c'].values.astype(np.float32)
    
    print(f"All features: {X_all.shape[1]}, DW features: {X_dw.shape[1]}, Samples: {len(df)}", flush=True)
    
    # Model configs: (name, model_fn, lr, wd, batch_size, epochs, patience)
    all_feat_configs = [
        ('ResNet128x4', lambda d: ResNetMLP(d, 128, 4, 0.1), 5e-4, 1e-3, 64, 500, 50),
        ('ResNet256x6', lambda d: ResNetMLP(d, 256, 6, 0.1), 5e-4, 1e-3, 64, 500, 50),
        ('ResNet256x6_lr3', lambda d: ResNetMLP(d, 256, 6, 0.1), 3e-4, 5e-4, 64, 500, 50),
        ('ResNet512x8', lambda d: ResNetMLP(d, 512, 8, 0.15), 3e-4, 1e-3, 64, 500, 50),
        ('FTTrans48x2', lambda d: FTTransformer(d, 48, 4, 2, 0.1), 1e-4, 1e-4, 64, 500, 50),
        ('FTTrans64x3', lambda d: FTTransformer(d, 64, 4, 3, 0.15), 1e-4, 1e-4, 64, 500, 50),
        ('DCN256x3', lambda d: DCNv2(d, 256, 3, 3, 0.1), 5e-4, 1e-3, 64, 500, 50),
        ('DCN256x4', lambda d: DCNv2(d, 256, 4, 4, 0.15), 3e-4, 1e-3, 64, 500, 50),
        ('SNN256x6', lambda d: SNN(d, 256, 6, 0.05), 5e-4, 1e-3, 64, 500, 50),
        ('SNN512x8', lambda d: SNN(d, 512, 8, 0.05), 3e-4, 5e-4, 64, 500, 50),
    ]
    
    dw_configs = [
        ('ResNet128x4', lambda d: ResNetMLP(d, 128, 4, 0.1), 5e-4, 1e-3, 64, 500, 50),
        ('ResNet256x6', lambda d: ResNetMLP(d, 256, 6, 0.1), 5e-4, 1e-3, 64, 500, 50),
        ('ResNet256x8', lambda d: ResNetMLP(d, 256, 8, 0.15), 3e-4, 1e-3, 64, 500, 60),
        ('FTTrans32x2', lambda d: FTTransformer(d, 32, 4, 2, 0.1), 1e-4, 1e-4, 64, 500, 50),
        ('FTTrans48x3', lambda d: FTTransformer(d, 48, 4, 3, 0.15), 1e-4, 1e-4, 64, 500, 50),
        ('DCN128x3', lambda d: DCNv2(d, 128, 3, 3, 0.1), 5e-4, 1e-3, 64, 500, 50),
        ('DCN256x4', lambda d: DCNv2(d, 256, 4, 4, 0.15), 3e-4, 1e-3, 64, 500, 50),
        ('SNN256x6', lambda d: SNN(d, 256, 6, 0.05), 5e-4, 1e-3, 64, 500, 50),
    ]
    
    results = {}
    
    results['HOMA_all'] = run_experiment("EXP 1", X_all, y_homa, 0, 'True_HOMA_IR', 0.85, all_feat_configs)
    results['hba1c_all'] = run_experiment("EXP 2", X_all, y_hba1c, 1, 'True_hba1c', 0.85, all_feat_configs)
    results['HOMA_dw'] = run_experiment("EXP 3", X_dw, y_homa, 0, 'True_HOMA_IR', 0.70, dw_configs)
    results['hba1c_dw'] = run_experiment("EXP 4", X_dw, y_hba1c, 1, 'True_hba1c', 0.70, dw_configs)
    
    print(f"\n{'='*70}", flush=True)
    print("FINAL SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    goals = {'HOMA_all': 0.85, 'hba1c_all': 0.85, 'HOMA_dw': 0.70, 'hba1c_dw': 0.70}
    for k, v in results.items():
        g = goals[k]
        f = "✅" if v['mean_r2'] >= g else "❌"
        print(f"  {k:15s}: R²={v['mean_r2']:.4f}±{v['std_r2']:.4f} (goal={g}) {f} [{v['name']}/{v['transform']}]", flush=True)
    
    with open('results_v3.json', 'w') as f:
        json.dump({k: {kk: (vv if not isinstance(vv, np.floating) else float(vv)) 
                       for kk, vv in v.items()} for k, v in results.items()}, f, indent=2, default=str)
    
    return results


if __name__ == '__main__':
    t0 = time.time()
    results = main()
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min", flush=True)
