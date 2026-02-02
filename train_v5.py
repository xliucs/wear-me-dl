#!/usr/bin/env python3
"""
V5: TabPFN + Neural stacking + MPS-safe DL
- TabPFN as base predictions (pre-trained tabular foundation model)
- Stack TabPFN predictions with DL models
- Use MPS with gc.collect() between folds
- More training epochs, lower LR
"""
import gc, warnings, json, time, math, os
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
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

np.random.seed(42)
torch.manual_seed(42)

# Use MPS but with manual memory management
DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f"Device: {DEVICE}", flush=True)


def load_data():
    df = pd.read_csv('data.csv', skiprows=[0])
    df['sex_num'] = (df['sex'] == 'Male').astype(float)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    
    # Feature engineering
    eng = {
        'trig_hdl': df['triglycerides'] / (df['hdl'] + 0.1),
        'tyg': np.log(df['triglycerides'] * df['glucose'] / 2 + 1),
        'glucose_bmi': df['glucose'] * df['bmi'],
        'glucose_sq': df['glucose'] ** 2 / 1000,
        'bmi_sq': df['bmi'] ** 2,
        'log_trig': np.log1p(df['triglycerides']),
        'log_glucose': np.log1p(df['glucose']),
        'ldl_hdl': df['ldl'] / (df['hdl'] + 0.1),
        'nlr': df['absolute_neutrophils'] / (df['absolute_lymphocytes'] + 0.1),
        'log_crp': np.log1p(df['crp']),
        'ast_alt': df['ast'] / (df['alt'] + 0.1),
        'rhr_hrv': df['Resting Heart Rate (mean)'] / (df['HRV (mean)'] + 0.1),
        'bmi_age': df['bmi'] * df['age'],
        'glucose_trig': df['glucose'] * df['triglycerides'] / 1000,
        'glucose_hdl': df['glucose'] / (df['hdl'] + 0.1),
        'non_hdl_hdl': df['non hdl'] / (df['hdl'] + 0.1),
        'crp_bmi': df['crp'] * df['bmi'],
        'bun_creat': df['bun'] / (df['creatinine'] + 0.01),
        'ggt_alt': df['ggt'] / (df['alt'] + 0.1),
        'tyg_bmi': np.log(df['triglycerides'] * df['glucose'] / 2 + 1) * df['bmi'],
        'glucose_proxy': df['glucose'] * df['triglycerides'] / (df['hdl'] + 0.1),
        'met_score': ((df['bmi']>30).astype(float) + (df['triglycerides']>150).astype(float) + 
                      (df['glucose']>100).astype(float) + (df['hdl']<40).astype(float)),
        'hr_cv': df['Resting Heart Rate (std)'] / (df['Resting Heart Rate (mean)'] + 0.1),
        'hrv_cv': df['HRV (std)'] / (df['HRV (mean)'] + 0.1),
        'steps_cv': df['STEPS (std)'] / (df['STEPS (mean)'] + 1),
        'sleep_cv': df['SLEEP Duration (std)'] / (df['SLEEP Duration (mean)'] + 1),
        'activity': df['STEPS (mean)'] * df['AZM Weekly (mean)'] / 1e6,
        'steps_sleep': df['STEPS (mean)'] / (df['SLEEP Duration (mean)'] + 1),
        'bmi_trig': df['bmi'] * df['triglycerides'] / 100,
        'rdw_mch': df['rdw'] * df['mch'],
        'glucose_rdw': df['glucose'] * df['rdw'],
        'age_glucose': df['age'] * df['glucose'] / 100,
    }
    for k, v in eng.items():
        df[k] = v
    
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
    
    ir_map = {'IR': 1, 'non-IR': 0}
    diab_map = {'Diabetic': 1, 'Non-Diabetic': 0, 'Non_Diabetic': 0}
    df['ir_label'] = df['True_IR_Class'].map(ir_map).fillna(0).astype(float)
    df['diab_label'] = df['True_Diabetes_2_Class'].map(diab_map).fillna(0).astype(float)
    
    return df, feat_cols, dw_cols


# ============ TABPFN ============

def get_tabpfn_predictions(X_train, y_train, X_val):
    """Get TabPFN predictions as meta-features."""
    try:
        from tabpfn import TabPFNRegressor
        model = TabPFNRegressor(device='cpu', n_estimators=8)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        return pred
    except Exception as e:
        print(f"  TabPFN failed: {e}", flush=True)
        return None


# ============ DL MODELS ============

class ResBlock(nn.Module):
    def __init__(self, d, drop=0.15):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.GELU(), nn.Linear(d, d*2), nn.GELU(),
                                 nn.Dropout(drop), nn.Linear(d*2, d), nn.Dropout(drop/2))
    def forward(self, x): return x + self.net(x)

class ResNet(nn.Module):
    def __init__(self, ind, h=256, nb=6, dr=0.15):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(ind, h), *[ResBlock(h, dr) for _ in range(nb)],
                                 nn.LayerNorm(h), nn.GELU(), nn.Linear(h, 1))
    def forward(self, x): return self.net(x)

class MultiTaskResNet(nn.Module):
    def __init__(self, ind, h=256, nb=6, dr=0.15):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(ind, h), *[ResBlock(h, dr) for _ in range(nb)])
        self.reg_head = nn.Sequential(nn.LayerNorm(h), nn.GELU(), nn.Linear(h, 1))
        self.cls_ir = nn.Sequential(nn.LayerNorm(h), nn.GELU(), nn.Linear(h, 1))
        self.cls_diab = nn.Sequential(nn.LayerNorm(h), nn.GELU(), nn.Linear(h, 1))
    def forward(self, x):
        f = self.backbone(x)
        return self.reg_head(f), self.cls_ir(f), self.cls_diab(f)


def train_multitask(model, Xtr, ytr, aux_tr, Xva, yva, epochs=500, lr=3e-4, wd=1e-3, bs=64, pat=60):
    model = model.to(DEVICE)
    ds = TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(ytr), torch.FloatTensor(aux_tr))
    dl = DataLoader(ds, batch_size=bs, shuffle=True)
    vx = torch.FloatTensor(Xva).to(DEVICE)
    vy = torch.FloatTensor(yva).to(DEVICE)
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    warmup = 15
    def lr_fn(ep):
        if ep < warmup: return (ep+1)/warmup
        return 0.5*(1+math.cos(math.pi*(ep-warmup)/max(epochs-warmup,1)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    
    best = 1e9; bs_state = None; wait = 0
    for ep in range(epochs):
        model.train()
        for bx, by, ba in dl:
            bx, by, ba = bx.to(DEVICE), by.to(DEVICE), ba.to(DEVICE)
            
            # Mixup
            if np.random.random() < 0.5:
                lam = np.random.beta(0.2, 0.2)
                idx = torch.randperm(bx.size(0))
                bx = lam*bx + (1-lam)*bx[idx]
                by = lam*by + (1-lam)*by[idx]
                ba = lam*ba + (1-lam)*ba[idx]
            
            reg, cir, cdi = model(bx)
            loss = F.huber_loss(reg.squeeze(), by, delta=1.0)
            cw = 0.3 * max(0.1, 1 - ep/epochs)
            loss += cw * (F.binary_cross_entropy_with_logits(cir.squeeze(), ba[:,0]) +
                         F.binary_cross_entropy_with_logits(cdi.squeeze(), ba[:,1]))
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
    model = model.cpu()
    if DEVICE.type == 'mps':
        torch.mps.empty_cache()
    gc.collect()
    return model


def train_simple(model, Xtr, ytr, Xva, yva, epochs=500, lr=3e-4, wd=1e-3, bs=64, pat=60):
    model = model.to(DEVICE)
    ds = TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(ytr))
    dl = DataLoader(ds, batch_size=bs, shuffle=True)
    vx = torch.FloatTensor(Xva).to(DEVICE)
    vy = torch.FloatTensor(yva).to(DEVICE)
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    warmup = 15
    def lr_fn(ep):
        if ep < warmup: return (ep+1)/warmup
        return 0.5*(1+math.cos(math.pi*(ep-warmup)/max(epochs-warmup,1)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    
    best = 1e9; bs_state = None; wait = 0
    for ep in range(epochs):
        model.train()
        for bx, by in dl:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            if np.random.random() < 0.5:
                lam = np.random.beta(0.2, 0.2)
                idx = torch.randperm(bx.size(0))
                bx = lam*bx + (1-lam)*bx[idx]; by = lam*by + (1-lam)*by[idx]
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
    model = model.cpu()
    if DEVICE.type == 'mps':
        torch.mps.empty_cache()
    gc.collect()
    return model


def run_full_experiment(X, y_raw, aux, feat_names, target_name, goal, is_dw=False, n_folds=5):
    print(f"\n{'='*60}\n{target_name} | {X.shape[1]} features | Goal: {goal}\n{'='*60}", flush=True)
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Feature selection
    n_sel = min(40, X.shape[1])
    mi = mutual_info_regression(X, y_raw, random_state=42, n_neighbors=5)
    feat_idx = np.argsort(mi)[::-1][:n_sel]
    X_sel = X[:, feat_idx]
    sel_names = [feat_names[i] for i in feat_idx]
    print(f"  Selected {n_sel} features: {sel_names[:10]}...", flush=True)
    
    best_r2 = -1
    best_method = ''
    
    # ---- Method 1: TabPFN ----
    print("\n  [TabPFN]", flush=True)
    tabpfn_fold_r2 = []
    tabpfn_oof = np.zeros(len(y_raw))
    
    for fold, (ti, vi) in enumerate(kf.split(X_sel)):
        sx = StandardScaler()
        Xtr = sx.fit_transform(X_sel[ti]).astype(np.float32)
        Xva = sx.transform(X_sel[vi]).astype(np.float32)
        
        pred = get_tabpfn_predictions(Xtr, y_raw[ti], Xva)
        if pred is not None:
            r2 = r2_score(y_raw[vi], pred)
            tabpfn_fold_r2.append(r2)
            tabpfn_oof[vi] = pred
        else:
            tabpfn_fold_r2.append(-1)
    
    if all(r > -1 for r in tabpfn_fold_r2):
        mr2 = np.mean(tabpfn_fold_r2)
        print(f"  TabPFN                       R²={mr2:.4f}±{np.std(tabpfn_fold_r2):.4f}", flush=True)
        if mr2 > best_r2:
            best_r2 = mr2; best_method = 'TabPFN'
    
    # ---- Method 2: TabPFN on log target ----
    print("\n  [TabPFN + log target]", flush=True)
    tabpfn_log_fold_r2 = []
    tabpfn_log_oof = np.zeros(len(y_raw))
    
    for fold, (ti, vi) in enumerate(kf.split(X_sel)):
        sx = StandardScaler()
        Xtr = sx.fit_transform(X_sel[ti]).astype(np.float32)
        Xva = sx.transform(X_sel[vi]).astype(np.float32)
        
        pred = get_tabpfn_predictions(Xtr, np.log1p(y_raw[ti]), Xva)
        if pred is not None:
            pred_raw = np.expm1(pred)
            r2 = r2_score(y_raw[vi], pred_raw)
            tabpfn_log_fold_r2.append(r2)
            tabpfn_log_oof[vi] = pred_raw
        else:
            tabpfn_log_fold_r2.append(-1)
    
    if all(r > -1 for r in tabpfn_log_fold_r2):
        mr2 = np.mean(tabpfn_log_fold_r2)
        print(f"  TabPFN_log                   R²={mr2:.4f}±{np.std(tabpfn_log_fold_r2):.4f}", flush=True)
        if mr2 > best_r2:
            best_r2 = mr2; best_method = 'TabPFN_log'
    
    # ---- Method 3: DL models ----
    for tfm in ['log', 'quantile']:
        for model_cfg in [
            ('MT_ResNet256x6', lambda d: MultiTaskResNet(d, 256, 6, 0.15), True),
            ('MT_ResNet128x4', lambda d: MultiTaskResNet(d, 128, 4, 0.15), True),
            ('ResNet256x6', lambda d: ResNet(d, 256, 6, 0.15), False),
        ]:
            name, mfn, is_mt = model_cfg
            fold_r2 = []
            
            for fold, (ti, vi) in enumerate(kf.split(X_sel)):
                sx = StandardScaler()
                Xtr = sx.fit_transform(X_sel[ti]).astype(np.float32)
                Xva = sx.transform(X_sel[vi]).astype(np.float32)
                
                qt = None
                if tfm == 'log':
                    ytr = np.log1p(y_raw[ti]).astype(np.float32)
                    yva = np.log1p(y_raw[vi]).astype(np.float32)
                else:
                    qt = QuantileTransformer(output_distribution='normal', n_quantiles=200, random_state=42)
                    ytr = qt.fit_transform(y_raw[ti].reshape(-1,1)).flatten().astype(np.float32)
                    yva = qt.transform(y_raw[vi].reshape(-1,1)).flatten().astype(np.float32)
                
                model = mfn(n_sel)
                if is_mt:
                    model = train_multitask(model, Xtr, ytr, aux[ti], Xva, yva,
                                          epochs=500, lr=3e-4, wd=1e-3, bs=64, pat=60)
                    model.eval()
                    with torch.no_grad():
                        pred, _, _ = model(torch.FloatTensor(Xva))
                        pred = pred.squeeze().numpy()
                else:
                    model = train_simple(model, Xtr, ytr, Xva, yva,
                                       epochs=500, lr=3e-4, wd=1e-3, bs=64, pat=60)
                    model.eval()
                    with torch.no_grad():
                        pred = model(torch.FloatTensor(Xva)).squeeze().numpy()
                
                if tfm == 'log': pred_raw = np.expm1(pred)
                else: pred_raw = qt.inverse_transform(pred.reshape(-1,1)).flatten()
                
                fold_r2.append(r2_score(y_raw[vi], pred_raw))
                del model; gc.collect()
            
            mr2 = np.mean(fold_r2)
            star = " ★" if mr2 > best_r2 else ""
            print(f"  {name:20s} {tfm:8s} R²={mr2:.4f}±{np.std(fold_r2):.4f}{star}", flush=True)
            if mr2 > best_r2:
                best_r2 = mr2; best_method = f"{name}_{tfm}"
    
    # ---- Method 4: Stacking (TabPFN + DL ensemble) ----
    if all(r > -1 for r in tabpfn_fold_r2):
        print("\n  [Stacking: TabPFN + DL]", flush=True)
        # Use TabPFN OOF as extra feature
        X_stacked = np.column_stack([X_sel, tabpfn_oof.reshape(-1,1)])
        if all(r > -1 for r in tabpfn_log_fold_r2):
            X_stacked = np.column_stack([X_stacked, tabpfn_log_oof.reshape(-1,1)])
        
        fold_r2_stack = []
        for fold, (ti, vi) in enumerate(kf.split(X_stacked)):
            sx = StandardScaler()
            Xtr = sx.fit_transform(X_stacked[ti]).astype(np.float32)
            Xva = sx.transform(X_stacked[vi]).astype(np.float32)
            ytr = np.log1p(y_raw[ti]).astype(np.float32)
            yva = np.log1p(y_raw[vi]).astype(np.float32)
            
            model = MultiTaskResNet(Xtr.shape[1], 128, 4, 0.15)
            model = train_multitask(model, Xtr, ytr, aux[ti], Xva, yva,
                                  epochs=500, lr=3e-4, wd=1e-3, bs=64, pat=60)
            model.eval()
            with torch.no_grad():
                pred, _, _ = model(torch.FloatTensor(Xva))
            pred_raw = np.expm1(pred.squeeze().numpy())
            fold_r2_stack.append(r2_score(y_raw[vi], pred_raw))
            del model; gc.collect()
        
        mr2 = np.mean(fold_r2_stack)
        star = " ★" if mr2 > best_r2 else ""
        print(f"  Stacked_MT128x4   log      R²={mr2:.4f}±{np.std(fold_r2_stack):.4f}{star}", flush=True)
        if mr2 > best_r2:
            best_r2 = mr2; best_method = 'Stacked_MT128x4_log'
    
    flag = "✅" if best_r2 >= goal else f"❌ (gap={goal-best_r2:.4f})"
    print(f"\n  >>> BEST: R²={best_r2:.4f} [{best_method}] {flag}", flush=True)
    return best_r2, best_method


def main():
    t0 = time.time()
    df, feat_cols, dw_cols = load_data()
    
    X_all = df[feat_cols].values.astype(np.float32)
    X_dw = df[dw_cols].values.astype(np.float32)
    y_homa = df['True_HOMA_IR'].values.astype(np.float32)
    y_hba1c = df['True_hba1c'].values.astype(np.float32)
    aux = np.column_stack([df['ir_label'].values, df['diab_label'].values]).astype(np.float32)
    
    print(f"All={X_all.shape[1]}, DW={X_dw.shape[1]}, N={len(df)}", flush=True)
    
    results = {}
    
    r2, m = run_full_experiment(X_all, y_homa, aux, feat_cols, 'HOMA_IR_all', 0.85)
    results['HOMA_all'] = {'r2': float(r2), 'method': m}
    
    r2, m = run_full_experiment(X_all, y_hba1c, aux, feat_cols, 'hba1c_all', 0.85)
    results['hba1c_all'] = {'r2': float(r2), 'method': m}
    
    r2, m = run_full_experiment(X_dw, y_homa, aux, dw_cols, 'HOMA_IR_dw', 0.70, is_dw=True)
    results['HOMA_dw'] = {'r2': float(r2), 'method': m}
    
    r2, m = run_full_experiment(X_dw, y_hba1c, aux, dw_cols, 'hba1c_dw', 0.70, is_dw=True)
    results['hba1c_dw'] = {'r2': float(r2), 'method': m}
    
    elapsed = (time.time()-t0)/60
    print(f"\n{'='*60}\nFINAL SUMMARY ({elapsed:.1f} min)\n{'='*60}", flush=True)
    goals = {'HOMA_all': 0.85, 'hba1c_all': 0.85, 'HOMA_dw': 0.70, 'hba1c_dw': 0.70}
    for k in ['HOMA_all', 'hba1c_all', 'HOMA_dw', 'hba1c_dw']:
        g = goals[k]
        f = "✅" if results[k]['r2'] >= g else "❌"
        print(f"  {k:15s}: R²={results[k]['r2']:.4f} (goal={g}) {f} [{results[k]['method']}]", flush=True)
    
    with open('results_v5.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
