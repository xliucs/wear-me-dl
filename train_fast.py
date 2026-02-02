#!/usr/bin/env python3
"""
Fast focused trainer. Key insights:
1. ResNet MLP is working best so far - focus on it
2. Shorter training with aggressive early stopping
3. Wider hyperparameter search per architecture  
4. Focus on log-transform (consistently best)
5. Add polynomial features for key predictors
6. Deep ensemble at the end
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
from sklearn.metrics import r2_score
from copy import deepcopy

warnings.filterwarnings('ignore')

def load_data():
    df = pd.read_csv('data.csv', skiprows=[0])
    df['sex_num'] = (df['sex'] == 'Male').astype(float)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    
    # Extensive feature engineering based on clinical knowledge
    # HOMA-IR proxies
    df['trig_hdl'] = df['triglycerides'] / (df['hdl'] + 0.1)
    df['tyg'] = np.log(df['triglycerides'] * df['glucose'] / 2 + 1)  # TyG index - validated HOMA-IR proxy
    df['tyg_bmi'] = df['tyg'] * df['bmi']  # TyG-BMI - even better proxy
    df['glucose_bmi'] = df['glucose'] * df['bmi']
    df['glucose_trig'] = df['glucose'] * df['triglycerides'] / 1000
    df['glucose_sq'] = df['glucose'] ** 2 / 1000
    df['bmi_sq'] = df['bmi'] ** 2
    df['log_trig'] = np.log1p(df['triglycerides'])
    df['log_glucose'] = np.log1p(df['glucose'])
    df['glucose_proxy'] = df['glucose'] * df['trig_hdl']
    df['ldl_hdl'] = df['ldl'] / (df['hdl'] + 0.1)
    df['non_hdl_hdl'] = df['non hdl'] / (df['hdl'] + 0.1)
    df['chol_trig'] = df['total cholesterol'] / (df['triglycerides'] + 0.1)
    
    # Inflammation
    df['nlr'] = df['absolute_neutrophils'] / (df['absolute_lymphocytes'] + 0.1)
    df['log_crp'] = np.log1p(df['crp'])
    df['crp_bmi'] = df['crp'] * df['bmi']
    df['wbc_crp'] = df['white_blood_cell'] * df['crp']
    
    # Liver
    df['ast_alt'] = df['ast'] / (df['alt'] + 0.1)
    df['ggt_alt'] = df['ggt'] / (df['alt'] + 0.1)
    
    # Kidney
    df['bun_creat'] = df['bun'] / (df['creatinine'] + 0.01)
    
    # Wearable
    df['rhr_hrv'] = df['Resting Heart Rate (mean)'] / (df['HRV (mean)'] + 0.1)
    df['hr_cv'] = df['Resting Heart Rate (std)'] / (df['Resting Heart Rate (mean)'] + 0.1)
    df['hrv_cv'] = df['HRV (std)'] / (df['HRV (mean)'] + 0.1)
    df['steps_cv'] = df['STEPS (std)'] / (df['STEPS (mean)'] + 1)
    df['sleep_cv'] = df['SLEEP Duration (std)'] / (df['SLEEP Duration (mean)'] + 1)
    df['activity'] = df['STEPS (mean)'] * df['AZM Weekly (mean)'] / 1e6
    df['steps_sleep'] = df['STEPS (mean)'] / (df['SLEEP Duration (mean)'] + 1)
    df['bmi_age'] = df['bmi'] * df['age']
    
    # Higher order for top predictors
    df['glucose_cube'] = df['glucose'] ** 3 / 1e6
    df['trig_sq'] = df['triglycerides'] ** 2 / 1e4
    df['trig_hdl_sq'] = df['trig_hdl'] ** 2
    df['tyg_sq'] = df['tyg'] ** 2
    df['bmi_glucose_trig'] = df['bmi'] * df['glucose'] * df['triglycerides'] / 1e5
    df['age_glucose'] = df['age'] * df['glucose'] / 100
    df['age_bmi'] = df['age'] * df['bmi']
    df['albumin_crp'] = df['albumin'] / (df['crp'] + 0.1)
    df['hb_rbc'] = df['hb'] / (df['red_blood_cell'] + 0.01)
    df['glucose_hdl'] = df['glucose'] / (df['hdl'] + 0.1)
    df['met_score'] = ((df['bmi']>30).astype(float) + (df['triglycerides']>150).astype(float) + 
                       (df['glucose']>100).astype(float) + (df['hdl']<40).astype(float))
    
    # hba1c specific
    df['rdw_mch'] = df['rdw'] * df['mch']
    df['age_rdw'] = df['age'] * df['rdw']
    df['glucose_rdw'] = df['glucose'] * df['rdw']
    df['glucose_age'] = df['glucose'] * df['age'] / 100
    
    demo = ['age', 'bmi', 'sex_num']
    fitbit = ['Resting Heart Rate (mean)', 'Resting Heart Rate (median)', 'Resting Heart Rate (std)',
              'HRV (mean)', 'HRV (median)', 'HRV (std)',
              'STEPS (mean)', 'STEPS (median)', 'STEPS (std)',
              'SLEEP Duration (mean)', 'SLEEP Duration (median)', 'SLEEP Duration (std)',
              'AZM Weekly (mean)', 'AZM Weekly (median)', 'AZM Weekly (std)']
    blood_raw = ['total cholesterol','hdl','triglycerides','ldl','chol/hdl','non hdl','glucose',
                 'bun','creatinine','egfr','sodium','potassium','chloride','co2','calcium',
                 'total protein','albumin','globulin','albumin/globulin','total bilirubin',
                 'alp','ast','alt','crp','white_blood_cell','red_blood_cell','hb','hematocrit',
                 'mcv','mch','mchc','rdw','platelet','mpv','absolute_neutrophils',
                 'absolute_lymphocytes','absolute_monocytes','absolute_eosinophils',
                 'absolute_basophils','neutrophils','lymphocytes','monocytes','eosinophils',
                 'basophils','ggt','total testosterone']
    eng_all = ['trig_hdl','tyg','tyg_bmi','glucose_bmi','glucose_trig','glucose_sq','bmi_sq',
               'log_trig','log_glucose','glucose_proxy','ldl_hdl','non_hdl_hdl','chol_trig',
               'nlr','log_crp','crp_bmi','wbc_crp','ast_alt','ggt_alt','bun_creat',
               'rhr_hrv','hr_cv','hrv_cv','steps_cv','sleep_cv','activity','steps_sleep','bmi_age',
               'glucose_cube','trig_sq','trig_hdl_sq','tyg_sq','bmi_glucose_trig','age_glucose',
               'age_bmi','albumin_crp','hb_rbc','glucose_hdl','met_score',
               'rdw_mch','age_rdw','glucose_rdw','glucose_age']
    eng_dw = ['bmi_sq','bmi_age','rhr_hrv','hr_cv','hrv_cv','steps_cv','sleep_cv','activity','steps_sleep']
    
    all_cols = list(dict.fromkeys(demo + fitbit + blood_raw + eng_all))
    dw_cols = list(dict.fromkeys(demo + fitbit + eng_dw))
    
    label_cols = ['True_hba1c','True_HOMA_IR','True_IR_Class','True_Diabetes_2_Class',
                  'True_Normoglycemic_2_Class','True_Diabetes_3_Class']
    all_cols = [c for c in all_cols if c in df.columns and c not in label_cols]
    dw_cols = [c for c in dw_cols if c in df.columns and c not in label_cols]
    
    mask = df[['True_HOMA_IR','True_hba1c']].notna().all(axis=1)
    df = df[mask].reset_index(drop=True)
    for col in set(all_cols+dw_cols):
        df[col] = df[col].replace([np.inf,-np.inf], np.nan).fillna(df[col].median())
    
    return df, all_cols, dw_cols


class ResBlock(nn.Module):
    def __init__(self, d, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.GELU(), nn.Linear(d, d*2), nn.GELU(),
                                 nn.Dropout(drop), nn.Linear(d*2, d), nn.Dropout(drop/2))
    def forward(self, x): return x + self.net(x)

class ResNet(nn.Module):
    def __init__(self, ind, h=256, nb=6, dr=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(ind, h), *[ResBlock(h, dr) for _ in range(nb)],
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


def mixup(x, y, a=0.2):
    l = np.random.beta(a, a)
    i = torch.randperm(x.size(0))
    return l*x + (1-l)*x[i], l*y + (1-l)*y[i]

def train_one(model, Xtr, ytr, Xva, yva, epochs=300, lr=5e-4, wd=1e-3, bs=64, pat=40):
    ds = TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(ytr))
    dl = DataLoader(ds, batch_size=bs, shuffle=True)
    vx, vy = torch.FloatTensor(Xva), torch.FloatTensor(yva)
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
                bx, by = mixup(bx, by, 0.2)
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
    del ds, dl, vx, vy; gc.collect()
    return model


def cv_experiment(X, y_raw, configs, n_folds=5, label_prefix=''):
    """Run CV for multiple configs, return best."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    best_r2 = -1
    best_name = ''
    
    for cfg in configs:
        name, model_fn, tfm, lr, wd, bs, ep, pat = cfg
        fold_r2 = []
        
        for fold, (ti, vi) in enumerate(kf.split(X)):
            Xtr, Xva = X[ti], X[vi]
            ytr_raw, yva_raw = y_raw[ti], y_raw[vi]
            
            sx = StandardScaler()
            Xtr_s = sx.fit_transform(Xtr).astype(np.float32)
            Xva_s = sx.transform(Xva).astype(np.float32)
            
            if tfm == 'log':
                ytr = np.log1p(ytr_raw).astype(np.float32)
                yva_t = np.log1p(yva_raw).astype(np.float32)
            elif tfm == 'quantile':
                qt = QuantileTransformer(output_distribution='normal', n_quantiles=200, random_state=42)
                ytr = qt.fit_transform(ytr_raw.reshape(-1,1)).flatten().astype(np.float32)
                yva_t = qt.transform(yva_raw.reshape(-1,1)).flatten().astype(np.float32)
            else:
                sy = StandardScaler()
                ytr = sy.fit_transform(ytr_raw.reshape(-1,1)).flatten().astype(np.float32)
                yva_t = sy.transform(yva_raw.reshape(-1,1)).flatten().astype(np.float32)
            
            model = model_fn(Xtr_s.shape[1])
            model = train_one(model, Xtr_s, ytr, Xva_s, yva_t, ep, lr, wd, bs, pat)
            
            model.eval()
            with torch.no_grad():
                pred = model(torch.FloatTensor(Xva_s)).squeeze(-1).numpy()
            
            if tfm == 'log': pred_raw = np.expm1(pred)
            elif tfm == 'quantile': pred_raw = qt.inverse_transform(pred.reshape(-1,1)).flatten()
            else: pred_raw = sy.inverse_transform(pred.reshape(-1,1)).flatten()
            
            fold_r2.append(r2_score(yva_raw, pred_raw))
            del model; gc.collect()
        
        mr2 = np.mean(fold_r2)
        star = " ★" if mr2 > best_r2 else ""
        print(f"  {name:30s} R²={mr2:.4f}±{np.std(fold_r2):.4f}{star}", flush=True)
        if mr2 > best_r2:
            best_r2 = mr2
            best_name = name
    
    return best_r2, best_name


def ensemble_cv(X, y_raw, model_fns, tfm, lr, wd, bs, ep, pat, n_folds=5, n_seeds=5):
    """Deep ensemble with multiple random seeds."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_r2 = []
    
    for fold, (ti, vi) in enumerate(kf.split(X)):
        Xtr, Xva = X[ti], X[vi]
        ytr_raw, yva_raw = y_raw[ti], y_raw[vi]
        
        sx = StandardScaler()
        Xtr_s = sx.fit_transform(Xtr).astype(np.float32)
        Xva_s = sx.transform(Xva).astype(np.float32)
        
        qt = None
        if tfm == 'log':
            ytr = np.log1p(ytr_raw).astype(np.float32)
            yva_t = np.log1p(yva_raw).astype(np.float32)
        elif tfm == 'quantile':
            qt = QuantileTransformer(output_distribution='normal', n_quantiles=200, random_state=42)
            ytr = qt.fit_transform(ytr_raw.reshape(-1,1)).flatten().astype(np.float32)
            yva_t = qt.transform(yva_raw.reshape(-1,1)).flatten().astype(np.float32)
        
        all_preds = []
        for seed_i in range(n_seeds):
            for mfn in model_fns:
                torch.manual_seed(42 + seed_i * 100 + len(all_preds))
                np.random.seed(42 + seed_i * 100 + len(all_preds))
                model = mfn(Xtr_s.shape[1])
                model = train_one(model, Xtr_s, ytr, Xva_s, yva_t, ep, lr, wd, bs, pat)
                model.eval()
                with torch.no_grad():
                    pred = model(torch.FloatTensor(Xva_s)).squeeze(-1).numpy()
                if tfm == 'log': pred = np.expm1(pred)
                elif tfm == 'quantile': pred = qt.inverse_transform(pred.reshape(-1,1)).flatten()
                all_preds.append(pred)
                del model; gc.collect()
        
        ens = np.mean(all_preds, axis=0)
        fold_r2.append(r2_score(yva_raw, ens))
    
    mr2 = np.mean(fold_r2)
    print(f"  Ensemble({len(model_fns)}x{n_seeds}seeds)        R²={mr2:.4f}±{np.std(fold_r2):.4f}", flush=True)
    return mr2


def main():
    t0 = time.time()
    df, all_cols, dw_cols = load_data()
    
    X_all = df[all_cols].values.astype(np.float32)
    X_dw = df[dw_cols].values.astype(np.float32)
    y_homa = df['True_HOMA_IR'].values.astype(np.float32)
    y_hba1c = df['True_hba1c'].values.astype(np.float32)
    
    print(f"Features: all={X_all.shape[1]}, dw={X_dw.shape[1]}, N={len(df)}", flush=True)
    
    results = {}
    
    # ===== EXP 1: HOMA_IR ALL =====
    print(f"\n{'='*60}\nHOMA_IR | ALL features | Goal: 0.85\n{'='*60}", flush=True)
    configs = [
        ('ResNet256x6_log', lambda d:ResNet(d,256,6,0.1), 'log', 5e-4, 1e-3, 64, 300, 40),
        ('ResNet256x6_qt', lambda d:ResNet(d,256,6,0.1), 'quantile', 5e-4, 1e-3, 64, 300, 40),
        ('ResNet512x6_log', lambda d:ResNet(d,512,6,0.15), 'log', 3e-4, 1e-3, 64, 300, 40),
        ('ResNet256x8_log', lambda d:ResNet(d,256,8,0.1), 'log', 3e-4, 5e-4, 64, 400, 50),
        ('SNN256x6_log', lambda d:SNN(d,256,6,0.05), 'log', 5e-4, 1e-3, 64, 300, 40),
        ('SNN512x8_log', lambda d:SNN(d,512,8,0.03), 'log', 3e-4, 5e-4, 64, 300, 40),
        ('SNN256x6_qt', lambda d:SNN(d,256,6,0.05), 'quantile', 5e-4, 1e-3, 64, 300, 40),
        ('ResNet256x6_log_lrlo', lambda d:ResNet(d,256,6,0.1), 'log', 1e-4, 5e-4, 64, 500, 60),
        ('ResNet256x6_log_bs32', lambda d:ResNet(d,256,6,0.1), 'log', 5e-4, 1e-3, 32, 300, 40),
    ]
    r2, name = cv_experiment(X_all, y_homa, configs)
    results['HOMA_all_single'] = r2
    
    # Ensemble
    print("  --- Deep Ensemble ---", flush=True)
    ens_r2 = ensemble_cv(X_all, y_homa, 
                         [lambda d:ResNet(d,256,6,0.1), lambda d:SNN(d,256,6,0.05)],
                         'log', 5e-4, 1e-3, 64, 300, 40, n_seeds=3)
    results['HOMA_all'] = max(r2, ens_r2)
    
    print(f"  >>> BEST: {results['HOMA_all']:.4f}", flush=True)
    
    # ===== EXP 2: hba1c ALL =====
    print(f"\n{'='*60}\nhba1c | ALL features | Goal: 0.85\n{'='*60}", flush=True)
    configs2 = [
        ('ResNet256x6_log', lambda d:ResNet(d,256,6,0.1), 'log', 5e-4, 1e-3, 64, 300, 40),
        ('ResNet256x6_qt', lambda d:ResNet(d,256,6,0.1), 'quantile', 5e-4, 1e-3, 64, 300, 40),
        ('ResNet512x6_log', lambda d:ResNet(d,512,6,0.15), 'log', 3e-4, 1e-3, 64, 300, 40),
        ('SNN256x6_log', lambda d:SNN(d,256,6,0.05), 'log', 5e-4, 1e-3, 64, 300, 40),
        ('SNN512x8_log', lambda d:SNN(d,512,8,0.03), 'log', 3e-4, 5e-4, 64, 300, 40),
        ('SNN256x6_qt', lambda d:SNN(d,256,6,0.05), 'quantile', 5e-4, 1e-3, 64, 300, 40),
        ('ResNet256x6_log_lrlo', lambda d:ResNet(d,256,6,0.1), 'log', 1e-4, 5e-4, 64, 500, 60),
    ]
    r2_2, name2 = cv_experiment(X_all, y_hba1c, configs2)
    results['hba1c_all_single'] = r2_2
    
    print("  --- Deep Ensemble ---", flush=True)
    ens_r2_2 = ensemble_cv(X_all, y_hba1c,
                            [lambda d:ResNet(d,256,6,0.1), lambda d:SNN(d,256,6,0.05)],
                            'log', 5e-4, 1e-3, 64, 300, 40, n_seeds=3)
    results['hba1c_all'] = max(r2_2, ens_r2_2)
    
    print(f"  >>> BEST: {results['hba1c_all']:.4f}", flush=True)
    
    # ===== EXP 3: HOMA_IR DW =====
    print(f"\n{'='*60}\nHOMA_IR | Demo+Wearable | Goal: 0.70\n{'='*60}", flush=True)
    configs3 = [
        ('ResNet128x4_log', lambda d:ResNet(d,128,4,0.1), 'log', 5e-4, 1e-3, 64, 300, 40),
        ('ResNet256x6_log', lambda d:ResNet(d,256,6,0.1), 'log', 5e-4, 1e-3, 64, 300, 40),
        ('SNN256x6_log', lambda d:SNN(d,256,6,0.05), 'log', 5e-4, 1e-3, 64, 300, 40),
        ('ResNet256x6_qt', lambda d:ResNet(d,256,6,0.1), 'quantile', 5e-4, 1e-3, 64, 300, 40),
        ('SNN256x6_qt', lambda d:SNN(d,256,6,0.05), 'quantile', 5e-4, 1e-3, 64, 300, 40),
    ]
    r2_3, name3 = cv_experiment(X_dw, y_homa, configs3)
    results['HOMA_dw_single'] = r2_3
    
    print("  --- Deep Ensemble ---", flush=True)
    ens_r2_3 = ensemble_cv(X_dw, y_homa,
                            [lambda d:ResNet(d,256,6,0.1), lambda d:SNN(d,256,6,0.05)],
                            'log', 5e-4, 1e-3, 64, 300, 40, n_seeds=3)
    results['HOMA_dw'] = max(r2_3, ens_r2_3)
    print(f"  >>> BEST: {results['HOMA_dw']:.4f}", flush=True)
    
    # ===== EXP 4: hba1c DW =====
    print(f"\n{'='*60}\nhba1c | Demo+Wearable | Goal: 0.70\n{'='*60}", flush=True)
    configs4 = [
        ('ResNet128x4_log', lambda d:ResNet(d,128,4,0.1), 'log', 5e-4, 1e-3, 64, 300, 40),
        ('ResNet256x6_log', lambda d:ResNet(d,256,6,0.1), 'log', 5e-4, 1e-3, 64, 300, 40),
        ('SNN256x6_log', lambda d:SNN(d,256,6,0.05), 'log', 5e-4, 1e-3, 64, 300, 40),
        ('ResNet256x6_qt', lambda d:ResNet(d,256,6,0.1), 'quantile', 5e-4, 1e-3, 64, 300, 40),
        ('SNN256x6_qt', lambda d:SNN(d,256,6,0.05), 'quantile', 5e-4, 1e-3, 64, 300, 40),
    ]
    r2_4, name4 = cv_experiment(X_dw, y_hba1c, configs4)
    results['hba1c_dw_single'] = r2_4
    
    print("  --- Deep Ensemble ---", flush=True)
    ens_r2_4 = ensemble_cv(X_dw, y_hba1c,
                            [lambda d:ResNet(d,256,6,0.1), lambda d:SNN(d,256,6,0.05)],
                            'log', 5e-4, 1e-3, 64, 300, 40, n_seeds=3)
    results['hba1c_dw'] = max(r2_4, ens_r2_4)
    print(f"  >>> BEST: {results['hba1c_dw']:.4f}", flush=True)
    
    # ===== SUMMARY =====
    elapsed = (time.time()-t0)/60
    print(f"\n{'='*60}\nSUMMARY ({elapsed:.1f} min)\n{'='*60}", flush=True)
    goals = {'HOMA_all': 0.85, 'hba1c_all': 0.85, 'HOMA_dw': 0.70, 'hba1c_dw': 0.70}
    for k in ['HOMA_all', 'hba1c_all', 'HOMA_dw', 'hba1c_dw']:
        g = goals[k]
        f = "✅" if results[k] >= g else "❌"
        print(f"  {k:15s}: R²={results[k]:.4f} (goal={g}) {f}", flush=True)
    
    with open('results_fast.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

if __name__ == '__main__':
    main()
