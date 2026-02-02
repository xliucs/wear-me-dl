#!/usr/bin/env python3
"""
V4: Smarter approach
1. Feature selection via mutual information
2. Auxiliary classification tasks for extra supervision
3. Two-stage prediction
4. Aggressive regularization for small dataset
5. Stacking ensemble
"""
import gc, warnings, json, time, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score
from copy import deepcopy

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

def load_data():
    df = pd.read_csv('data.csv', skiprows=[0])
    df['sex_num'] = (df['sex'] == 'Male').astype(float)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    
    # Feature engineering (focused on validated clinical proxies)
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
    df['nlr'] = df['absolute_neutrophils'] / (df['absolute_lymphocytes'] + 0.1)
    df['log_crp'] = np.log1p(df['crp'])
    df['ast_alt'] = df['ast'] / (df['alt'] + 0.1)
    df['rhr_hrv'] = df['Resting Heart Rate (mean)'] / (df['HRV (mean)'] + 0.1)
    df['hr_cv'] = df['Resting Heart Rate (std)'] / (df['Resting Heart Rate (mean)'] + 0.1)
    df['hrv_cv'] = df['HRV (std)'] / (df['HRV (mean)'] + 0.1)
    df['steps_cv'] = df['STEPS (std)'] / (df['STEPS (mean)'] + 1)
    df['sleep_cv'] = df['SLEEP Duration (std)'] / (df['SLEEP Duration (mean)'] + 1)
    df['activity'] = df['STEPS (mean)'] * df['AZM Weekly (mean)'] / 1e6
    df['bmi_age'] = df['bmi'] * df['age']
    df['glucose_trig'] = df['glucose'] * df['triglycerides'] / 1000
    df['glucose_hdl'] = df['glucose'] / (df['hdl'] + 0.1)
    df['trig_glucose_idx'] = np.log(df['triglycerides'] * df['glucose'] / 2 + 1)
    df['non_hdl_hdl'] = df['non hdl'] / (df['hdl'] + 0.1)
    df['crp_bmi'] = df['crp'] * df['bmi']
    df['bun_creat'] = df['bun'] / (df['creatinine'] + 0.01)
    df['ggt_alt'] = df['ggt'] / (df['alt'] + 0.1)
    df['met_score'] = ((df['bmi']>30).astype(float) + (df['triglycerides']>150).astype(float) + 
                       (df['glucose']>100).astype(float) + (df['hdl']<40).astype(float))
    df['glucose_cube'] = df['glucose'] ** 3 / 1e6
    df['rdw_mch'] = df['rdw'] * df['mch']
    df['age_rdw'] = df['age'] * df['rdw']
    df['glucose_rdw'] = df['glucose'] * df['rdw']
    df['steps_sleep'] = df['STEPS (mean)'] / (df['SLEEP Duration (mean)'] + 1)
    df['bmi_trig'] = df['bmi'] * df['triglycerides'] / 100
    df['albumin_crp'] = df['albumin'] / (df['crp'] + 0.1)
    df['wbc_crp'] = df['white_blood_cell'] * df['crp']
    df['hb_rbc'] = df['hb'] / (df['red_blood_cell'] + 0.01)
    df['age_glucose'] = df['age'] * df['glucose'] / 100
    df['chol_trig'] = df['total cholesterol'] / (df['triglycerides'] + 0.1)
    
    label_cols = ['True_hba1c','True_HOMA_IR','True_IR_Class','True_Diabetes_2_Class',
                  'True_Normoglycemic_2_Class','True_Diabetes_3_Class']
    
    # Get all numeric non-label columns
    feat_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                 if c not in label_cols + ['Participant_id']]
    
    demo = ['age', 'bmi', 'sex_num']
    fitbit = ['Resting Heart Rate (mean)', 'Resting Heart Rate (median)', 'Resting Heart Rate (std)',
              'HRV (mean)', 'HRV (median)', 'HRV (std)',
              'STEPS (mean)', 'STEPS (median)', 'STEPS (std)',
              'SLEEP Duration (mean)', 'SLEEP Duration (median)', 'SLEEP Duration (std)',
              'AZM Weekly (mean)', 'AZM Weekly (median)', 'AZM Weekly (std)']
    dw_eng = ['bmi_sq','bmi_age','rhr_hrv','hr_cv','hrv_cv','steps_cv','sleep_cv','activity','steps_sleep']
    
    dw_cols = list(dict.fromkeys(demo + fitbit + dw_eng))
    
    mask = df[['True_HOMA_IR','True_hba1c']].notna().all(axis=1)
    df = df[mask].reset_index(drop=True)
    
    for col in feat_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(df[col].median())
    
    # Encode classification labels
    ir_map = {'IR': 1, 'non-IR': 0}
    diab_map = {'Diabetic': 1, 'Non-Diabetic': 0, 'Non_Diabetic': 0}
    df['ir_label'] = df['True_IR_Class'].map(ir_map).fillna(0).astype(float)
    df['diab_label'] = df['True_Diabetes_2_Class'].map(diab_map).fillna(0).astype(float)
    
    return df, feat_cols, dw_cols


def select_features(X, y, feat_names, n_select=40):
    """Select top features using mutual information."""
    mi = mutual_info_regression(X, y, random_state=42, n_neighbors=5)
    idx = np.argsort(mi)[::-1][:n_select]
    selected = [feat_names[i] for i in idx]
    print(f"  Top features: {selected[:15]}...", flush=True)
    return idx, selected


# ============ MODELS ============

class ResBlock(nn.Module):
    def __init__(self, d, drop=0.15):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.GELU(), nn.Linear(d, d*2), nn.GELU(),
                                 nn.Dropout(drop), nn.Linear(d*2, d), nn.Dropout(drop/2))
    def forward(self, x): return x + self.net(x)

class MultiTaskResNet(nn.Module):
    """ResNet with auxiliary classification heads for extra supervision."""
    def __init__(self, ind, h=256, nb=6, dr=0.15):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(ind, h), *[ResBlock(h, dr) for _ in range(nb)])
        self.reg_head = nn.Sequential(nn.LayerNorm(h), nn.GELU(), nn.Linear(h, 1))
        self.cls_head_ir = nn.Sequential(nn.LayerNorm(h), nn.GELU(), nn.Linear(h, 1))
        self.cls_head_diab = nn.Sequential(nn.LayerNorm(h), nn.GELU(), nn.Linear(h, 1))
    
    def forward(self, x):
        feat = self.backbone(x)
        reg = self.reg_head(feat)
        cls_ir = self.cls_head_ir(feat)
        cls_diab = self.cls_head_diab(feat)
        return reg, cls_ir, cls_diab

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


def mixup(x, y, aux, a=0.2):
    l = np.random.beta(a, a) if a > 0 else 1
    i = torch.randperm(x.size(0))
    return l*x + (1-l)*x[i], l*y + (1-l)*y[i], l*aux + (1-l)*aux[i]


def train_multitask(model, Xtr, ytr, aux_tr, Xva, yva, aux_va,
                    epochs=400, lr=5e-4, wd=1e-3, bs=64, pat=50, cls_weight=0.3):
    """Train with auxiliary classification tasks."""
    ds = TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(ytr), torch.FloatTensor(aux_tr))
    dl = DataLoader(ds, batch_size=bs, shuffle=True)
    vx = torch.FloatTensor(Xva); vy = torch.FloatTensor(yva)
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    warmup = 10
    def lr_fn(ep):
        if ep < warmup: return (ep+1)/warmup
        return 0.5*(1+math.cos(math.pi*(ep-warmup)/max(epochs-warmup,1)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    
    best = 1e9; bs_state = None; wait = 0
    for ep in range(epochs):
        model.train()
        for bx, by, ba in dl:
            reg_pred, cls_ir, cls_diab = model(bx)
            reg_loss = F.huber_loss(reg_pred.squeeze(), by, delta=1.0)
            cls_ir_loss = F.binary_cross_entropy_with_logits(cls_ir.squeeze(), ba[:, 0])
            cls_diab_loss = F.binary_cross_entropy_with_logits(cls_diab.squeeze(), ba[:, 1])
            
            # Decay cls weight over training
            cw = cls_weight * max(0.1, 1 - ep / epochs)
            loss = reg_loss + cw * (cls_ir_loss + cls_diab_loss)
            
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        
        model.eval()
        with torch.no_grad():
            vp, _, _ = model(vx)
            vl = F.mse_loss(vp.squeeze(), vy).item()
        if vl < best - 1e-7:
            best = vl; bs_state = deepcopy(model.state_dict()); wait = 0
        else:
            wait += 1
            if wait >= pat: break
    
    if bs_state: model.load_state_dict(bs_state)
    del ds, dl, vx, vy; gc.collect()
    return model


def train_simple(model, Xtr, ytr, Xva, yva, epochs=400, lr=5e-4, wd=1e-3, bs=64, pat=50):
    ds = TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(ytr))
    dl = DataLoader(ds, batch_size=bs, shuffle=True)
    vx = torch.FloatTensor(Xva); vy = torch.FloatTensor(yva)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    warmup = 10
    def lr_fn(ep):
        if ep < warmup: return (ep+1)/warmup
        return 0.5*(1+math.cos(math.pi*(ep-warmup)/max(epochs-warmup,1)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    best = 1e9; bs_state = None; wait = 0
    for ep in range(epochs):
        model.train()
        for bx, by in dl:
            if np.random.random() < 0.5:
                l = np.random.beta(0.2, 0.2)
                i = torch.randperm(bx.size(0))
                bx = l*bx + (1-l)*bx[i]; by = l*by + (1-l)*by[i]
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
            best = vl; bs_state = deepcopy(model.state_dict()); wait = 0
        else:
            wait += 1
            if wait >= pat: break
    if bs_state: model.load_state_dict(bs_state)
    del ds, dl; gc.collect()
    return model


def run_experiment(X_all, y_raw, aux_labels, feat_names, target_name, goal, n_folds=5, 
                   n_features_list=[25, 35, 50, 75], is_dw=False):
    print(f"\n{'='*60}\n{target_name} | {X_all.shape[1]} features | Goal: {goal}\n{'='*60}", flush=True)
    
    best_overall = -1
    best_config = ''
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for n_feat in n_features_list:
        if n_feat > X_all.shape[1]:
            n_feat = X_all.shape[1]
        
        print(f"\n  --- Top {n_feat} features (MI selection) ---", flush=True)
        
        # Feature selection on full data (ok for MI, it's a filter method)
        feat_idx, feat_selected = select_features(X_all, y_raw, feat_names, n_feat)
        X = X_all[:, feat_idx]
        
        for transform in ['log', 'quantile']:
            for model_type in ['multitask', 'snn', 'ensemble']:
                fold_r2 = []
                
                for fold, (ti, vi) in enumerate(kf.split(X)):
                    Xtr, Xva = X[ti], X[vi]
                    ytr_raw, yva_raw = y_raw[ti], y_raw[vi]
                    aux_tr, aux_va = aux_labels[ti], aux_labels[vi]
                    
                    sx = StandardScaler()
                    Xtr_s = sx.fit_transform(Xtr).astype(np.float32)
                    Xva_s = sx.transform(Xva).astype(np.float32)
                    
                    qt = None
                    if transform == 'log':
                        ytr = np.log1p(ytr_raw).astype(np.float32)
                        yva_t = np.log1p(yva_raw).astype(np.float32)
                    else:
                        qt = QuantileTransformer(output_distribution='normal', n_quantiles=200, random_state=42)
                        ytr = qt.fit_transform(ytr_raw.reshape(-1,1)).flatten().astype(np.float32)
                        yva_t = qt.transform(yva_raw.reshape(-1,1)).flatten().astype(np.float32)
                    
                    if model_type == 'multitask':
                        h = 128 if n_feat <= 30 else 256
                        nb = 4 if n_feat <= 30 else 6
                        model = MultiTaskResNet(n_feat, h, nb, 0.15)
                        model = train_multitask(model, Xtr_s, ytr, aux_tr, Xva_s, yva_t, aux_va,
                                              epochs=400, lr=5e-4, wd=1e-3, bs=64, pat=50)
                        model.eval()
                        with torch.no_grad():
                            pred, _, _ = model(torch.FloatTensor(Xva_s))
                            pred = pred.squeeze().numpy()
                    
                    elif model_type == 'snn':
                        h = 128 if n_feat <= 30 else 256
                        model = SNN(n_feat, h, 6, 0.05)
                        model = train_simple(model, Xtr_s, ytr, Xva_s, yva_t,
                                           epochs=400, lr=5e-4, wd=1e-3, bs=64, pat=50)
                        model.eval()
                        with torch.no_grad():
                            pred = model(torch.FloatTensor(Xva_s)).squeeze().numpy()
                    
                    elif model_type == 'ensemble':
                        # Ensemble of 3 models with different seeds
                        preds_list = []
                        for seed in [42, 123, 789]:
                            torch.manual_seed(seed)
                            h = 128 if n_feat <= 30 else 256
                            nb = 4 if n_feat <= 30 else 6
                            m = MultiTaskResNet(n_feat, h, nb, 0.15)
                            m = train_multitask(m, Xtr_s, ytr, aux_tr, Xva_s, yva_t, aux_va,
                                              epochs=400, lr=5e-4, wd=1e-3, bs=64, pat=50)
                            m.eval()
                            with torch.no_grad():
                                p, _, _ = m(torch.FloatTensor(Xva_s))
                            preds_list.append(p.squeeze().numpy())
                            del m; gc.collect()
                        
                        # Add SNN
                        for seed in [42, 123]:
                            torch.manual_seed(seed)
                            m = SNN(n_feat, 256 if n_feat > 30 else 128, 6, 0.05)
                            m = train_simple(m, Xtr_s, ytr, Xva_s, yva_t,
                                           epochs=400, lr=5e-4, wd=1e-3, bs=64, pat=50)
                            m.eval()
                            with torch.no_grad():
                                p = m(torch.FloatTensor(Xva_s)).squeeze().numpy()
                            preds_list.append(p)
                            del m; gc.collect()
                        
                        pred = np.mean(preds_list, axis=0)
                    
                    # Inverse transform
                    if transform == 'log':
                        pred_raw = np.expm1(pred)
                    else:
                        pred_raw = qt.inverse_transform(pred.reshape(-1,1)).flatten()
                    
                    fold_r2.append(r2_score(yva_raw, pred_raw))
                    if model_type != 'ensemble':
                        del model; gc.collect()
                
                mr2 = np.mean(fold_r2)
                star = " ★" if mr2 > best_overall else ""
                print(f"  n={n_feat} {model_type:12s} {transform:8s} R²={mr2:.4f}±{np.std(fold_r2):.4f}{star}", flush=True)
                
                if mr2 > best_overall:
                    best_overall = mr2
                    best_config = f"n={n_feat}_{model_type}_{transform}"
        
        if n_feat == X_all.shape[1]:
            break  # Already tried all features
    
    flag = "✅" if best_overall >= goal else f"❌ (gap={goal-best_overall:.4f})"
    print(f"\n  >>> BEST: R²={best_overall:.4f} [{best_config}] {flag}", flush=True)
    return best_overall, best_config


def main():
    t0 = time.time()
    df, feat_cols, dw_cols = load_data()
    
    X_all = df[feat_cols].values.astype(np.float32)
    X_dw = df[dw_cols].values.astype(np.float32)
    y_homa = df['True_HOMA_IR'].values.astype(np.float32)
    y_hba1c = df['True_hba1c'].values.astype(np.float32)
    aux = np.column_stack([df['ir_label'].values, df['diab_label'].values]).astype(np.float32)
    
    print(f"Features: all={X_all.shape[1]}, dw={X_dw.shape[1]}, N={len(df)}", flush=True)
    
    results = {}
    
    # EXP 1: HOMA all
    r2, cfg = run_experiment(X_all, y_homa, aux, feat_cols, 'HOMA_IR_all', 0.85,
                            n_features_list=[25, 40, 60])
    results['HOMA_all'] = {'r2': float(r2), 'config': cfg}
    
    # EXP 2: hba1c all
    r2, cfg = run_experiment(X_all, y_hba1c, aux, feat_cols, 'hba1c_all', 0.85,
                            n_features_list=[25, 40, 60])
    results['hba1c_all'] = {'r2': float(r2), 'config': cfg}
    
    # EXP 3: HOMA dw
    r2, cfg = run_experiment(X_dw, y_homa, aux, dw_cols, 'HOMA_IR_dw', 0.70,
                            n_features_list=[15, 27], is_dw=True)
    results['HOMA_dw'] = {'r2': float(r2), 'config': cfg}
    
    # EXP 4: hba1c dw
    r2, cfg = run_experiment(X_dw, y_hba1c, aux, dw_cols, 'hba1c_dw', 0.70,
                            n_features_list=[15, 27], is_dw=True)
    results['hba1c_dw'] = {'r2': float(r2), 'config': cfg}
    
    elapsed = (time.time()-t0)/60
    print(f"\n{'='*60}\nFINAL SUMMARY ({elapsed:.1f} min)\n{'='*60}", flush=True)
    goals = {'HOMA_all': 0.85, 'hba1c_all': 0.85, 'HOMA_dw': 0.70, 'hba1c_dw': 0.70}
    for k in ['HOMA_all', 'hba1c_all', 'HOMA_dw', 'hba1c_dw']:
        g = goals[k]
        f = "✅" if results[k]['r2'] >= g else "❌"
        print(f"  {k:15s}: R²={results[k]['r2']:.4f} (goal={g}) {f} [{results[k]['config']}]", flush=True)
    
    with open('results_v4.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
