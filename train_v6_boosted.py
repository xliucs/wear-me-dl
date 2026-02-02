#!/usr/bin/env python3
"""
V6: Gradient-boosted neural networks + wide shallow nets + KNN-NN hybrid
Key ideas:
1. Boosted NN: Train sequence of small NNs on residuals (like XGBoost but neural)
2. Wide shallow: 1024-2048 neurons, 1-2 hidden layers + heavy regularization
3. KNN-augmented: Use KNN predictions as extra features
4. Autoencoder pretraining: Learn representations first, then finetune
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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from copy import deepcopy

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device('cpu')  # CPU to avoid OOM


def load_data():
    df = pd.read_csv('data.csv', skiprows=[0])
    df['sex_num'] = (df['sex'] == 'Male').astype(float)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    
    eng = {
        'trig_hdl': df['triglycerides'] / (df['hdl'] + 0.1),
        'tyg': np.log(df['triglycerides'] * df['glucose'] / 2 + 1),
        'tyg_bmi': np.log(df['triglycerides'] * df['glucose'] / 2 + 1) * df['bmi'],
        'glucose_bmi': df['glucose'] * df['bmi'],
        'glucose_sq': df['glucose'] ** 2 / 1000,
        'bmi_sq': df['bmi'] ** 2,
        'log_trig': np.log1p(df['triglycerides']),
        'log_glucose': np.log1p(df['glucose']),
        'glucose_proxy': df['glucose'] * df['triglycerides'] / (df['hdl'] + 0.1),
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


# ============ MODELS ============

class WideShallowNet(nn.Module):
    """Very wide, shallow network - good for small datasets."""
    def __init__(self, ind, h=1024, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ind, h),
            nn.BatchNorm1d(h),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(h, h // 2),
            nn.BatchNorm1d(h // 2),
            nn.GELU(),
            nn.Dropout(drop / 2),
            nn.Linear(h // 2, 1)
        )
    def forward(self, x): return self.net(x)

class ResBlock(nn.Module):
    def __init__(self, d, drop=0.15):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.GELU(), nn.Linear(d, d*2), nn.GELU(),
                                 nn.Dropout(drop), nn.Linear(d*2, d), nn.Dropout(drop/2))
    def forward(self, x): return x + self.net(x)

class SmallResNet(nn.Module):
    """Small model for boosting."""
    def __init__(self, ind, h=64, nb=2, dr=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(ind, h), *[ResBlock(h, dr) for _ in range(nb)],
                                 nn.LayerNorm(h), nn.GELU(), nn.Linear(h, 1))
    def forward(self, x): return self.net(x)

class Autoencoder(nn.Module):
    """Denoising autoencoder for representation learning."""
    def __init__(self, ind, h=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(ind, h*2), nn.GELU(), nn.Linear(h*2, h))
        self.decoder = nn.Sequential(nn.Linear(h, h*2), nn.GELU(), nn.Linear(h*2, ind))
    def encode(self, x): return self.encoder(x)
    def forward(self, x): return self.decoder(self.encoder(x))

class AERegressor(nn.Module):
    """Autoencoder + regression head."""
    def __init__(self, ind, ae_h=64, reg_h=128, nb=3, dr=0.15):
        super().__init__()
        self.ae = Autoencoder(ind, ae_h)
        self.head = nn.Sequential(
            nn.Linear(ae_h, reg_h), nn.GELU(), nn.Dropout(dr),
            *[nn.Sequential(nn.Linear(reg_h, reg_h), nn.GELU(), nn.Dropout(dr)) for _ in range(nb-1)],
            nn.Linear(reg_h, 1)
        )
    def forward(self, x): return self.head(self.ae.encode(x))
    def reconstruct(self, x): return self.ae(x)


def train_simple(model, Xtr, ytr, Xva, yva, epochs=300, lr=5e-4, wd=1e-3, bs=64, pat=40):
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
                l = np.random.beta(0.2, 0.2); i = torch.randperm(bx.size(0))
                bx = l*bx + (1-l)*bx[i]; by = l*by + (1-l)*by[i]
            loss = F.huber_loss(model(bx).squeeze(-1), by, delta=1.0)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = F.mse_loss(model(vx).squeeze(-1), vy).item()
        if vl < best - 1e-7: best = vl; bs_state = deepcopy(model.state_dict()); wait = 0
        else:
            wait += 1
            if wait >= pat: break
    if bs_state: model.load_state_dict(bs_state)
    return model


def train_ae_regressor(model, Xtr, ytr, Xva, yva, ae_epochs=100, reg_epochs=300, lr=5e-4, wd=1e-3, bs=64, pat=40):
    """Two-phase training: pretrain AE, then finetune with regression."""
    # Phase 1: Pretrain AE
    ds = TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(Xtr))
    dl = DataLoader(ds, batch_size=bs, shuffle=True)
    opt = torch.optim.Adam(model.ae.parameters(), lr=1e-3)
    for ep in range(ae_epochs):
        model.train()
        for bx, _ in dl:
            noise = torch.randn_like(bx) * 0.1
            recon = model.reconstruct(bx + noise)
            loss = F.mse_loss(recon, bx)
            opt.zero_grad(); loss.backward(); opt.step()
    
    # Phase 2: Train full model
    return train_simple(model, Xtr, ytr, Xva, yva, reg_epochs, lr, wd, bs, pat)


def boosted_nn_predict(X_train, y_train, X_val, n_rounds=10, lr_shrink=0.3, h=64, nb=2):
    """Gradient-boosted neural networks."""
    residual = y_train.copy()
    train_pred = np.zeros(len(y_train))
    val_pred = np.zeros(len(X_val))
    
    for r in range(n_rounds):
        model = SmallResNet(X_train.shape[1], h, nb, 0.1)
        model = train_simple(model, X_train, residual, X_val, np.zeros(len(X_val)),
                            epochs=200, lr=5e-4, wd=1e-3, bs=64, pat=30)
        model.eval()
        with torch.no_grad():
            tr_p = model(torch.FloatTensor(X_train)).squeeze(-1).numpy()
            va_p = model(torch.FloatTensor(X_val)).squeeze(-1).numpy()
        
        train_pred += lr_shrink * tr_p
        val_pred += lr_shrink * va_p
        residual = y_train - train_pred  # Next round targets residuals
        del model; gc.collect()
    
    return val_pred


def knn_augmented_features(X_train, y_train, X_val, k_list=[3, 5, 10, 20]):
    """Add KNN predictions as extra features."""
    extra_tr = []
    extra_va = []
    for k in k_list:
        knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
        knn.fit(X_train, y_train)
        extra_tr.append(knn.predict(X_train).reshape(-1, 1))
        extra_va.append(knn.predict(X_val).reshape(-1, 1))
    
    X_train_aug = np.hstack([X_train] + extra_tr)
    X_val_aug = np.hstack([X_val] + extra_va)
    return X_train_aug, X_val_aug


def run_experiment(X, y_raw, feat_names, target_name, goal, is_dw=False, n_folds=5):
    print(f"\n{'='*60}\n{target_name} | {X.shape[1]} features | Goal: {goal}\n{'='*60}", flush=True)
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Feature selection
    n_sel = min(40, X.shape[1])
    mi = mutual_info_regression(X, y_raw, random_state=42)
    feat_idx = np.argsort(mi)[::-1][:n_sel]
    X_sel = X[:, feat_idx]
    
    best_r2 = -1
    best_method = ''
    
    methods = [
        ('WideShallow_1024', 'log'),
        ('WideShallow_2048', 'log'),
        ('BoostedNN_10r', 'log'),
        ('BoostedNN_20r', 'log'),
        ('KNN_augmented', 'log'),
        ('AE_Regressor', 'log'),
        ('WideShallow_1024', 'quantile'),
        ('BoostedNN_10r', 'quantile'),
    ]
    
    for method_name, tfm in methods:
        fold_r2 = []
        
        for fold, (ti, vi) in enumerate(kf.split(X_sel)):
            Xtr, Xva = X_sel[ti], X_sel[vi]
            ytr_raw, yva_raw = y_raw[ti], y_raw[vi]
            
            sx = StandardScaler()
            Xtr_s = sx.fit_transform(Xtr).astype(np.float32)
            Xva_s = sx.transform(Xva).astype(np.float32)
            
            qt = None
            if tfm == 'log':
                ytr = np.log1p(ytr_raw).astype(np.float32)
            else:
                qt = QuantileTransformer(output_distribution='normal', n_quantiles=200, random_state=42)
                ytr = qt.fit_transform(ytr_raw.reshape(-1,1)).flatten().astype(np.float32)
            yva = np.log1p(yva_raw).astype(np.float32) if tfm == 'log' else qt.transform(yva_raw.reshape(-1,1)).flatten().astype(np.float32)
            
            if method_name == 'WideShallow_1024':
                model = WideShallowNet(n_sel, 1024, 0.3)
                model = train_simple(model, Xtr_s, ytr, Xva_s, yva, 400, 5e-4, 1e-3, 64, 50)
                model.eval()
                with torch.no_grad():
                    pred = model(torch.FloatTensor(Xva_s)).squeeze(-1).numpy()
                del model
            
            elif method_name == 'WideShallow_2048':
                model = WideShallowNet(n_sel, 2048, 0.4)
                model = train_simple(model, Xtr_s, ytr, Xva_s, yva, 400, 3e-4, 1e-3, 64, 50)
                model.eval()
                with torch.no_grad():
                    pred = model(torch.FloatTensor(Xva_s)).squeeze(-1).numpy()
                del model
            
            elif method_name.startswith('BoostedNN'):
                n_rounds = int(method_name.split('_')[1].replace('r',''))
                pred = boosted_nn_predict(Xtr_s, ytr, Xva_s, n_rounds=n_rounds)
            
            elif method_name == 'KNN_augmented':
                Xtr_aug, Xva_aug = knn_augmented_features(Xtr_s, ytr, Xva_s)
                model = WideShallowNet(Xtr_aug.shape[1], 1024, 0.3)
                model = train_simple(model, Xtr_aug.astype(np.float32), ytr, 
                                    Xva_aug.astype(np.float32), yva, 400, 5e-4, 1e-3, 64, 50)
                model.eval()
                with torch.no_grad():
                    pred = model(torch.FloatTensor(Xva_aug.astype(np.float32))).squeeze(-1).numpy()
                del model
            
            elif method_name == 'AE_Regressor':
                model = AERegressor(n_sel, ae_h=32, reg_h=128, nb=3, dr=0.15)
                model = train_ae_regressor(model, Xtr_s, ytr, Xva_s, yva, 
                                          ae_epochs=100, reg_epochs=300, pat=40)
                model.eval()
                with torch.no_grad():
                    pred = model(torch.FloatTensor(Xva_s)).squeeze(-1).numpy()
                del model
            
            # Inverse transform
            if tfm == 'log': pred_raw = np.expm1(pred)
            else: pred_raw = qt.inverse_transform(pred.reshape(-1,1)).flatten()
            
            fold_r2.append(r2_score(yva_raw, pred_raw))
            gc.collect()
        
        mr2 = np.mean(fold_r2)
        star = " ★" if mr2 > best_r2 else ""
        print(f"  {method_name:25s} {tfm:8s} R²={mr2:.4f}±{np.std(fold_r2):.4f}{star}", flush=True)
        if mr2 > best_r2:
            best_r2 = mr2; best_method = f"{method_name}_{tfm}"
    
    # ---- MEGA ENSEMBLE: combine all methods ----
    print(f"\n  --- MEGA ENSEMBLE ---", flush=True)
    fold_r2_mega = []
    for fold, (ti, vi) in enumerate(kf.split(X_sel)):
        Xtr, Xva = X_sel[ti], X_sel[vi]
        ytr_raw, yva_raw = y_raw[ti], y_raw[vi]
        sx = StandardScaler()
        Xtr_s = sx.fit_transform(Xtr).astype(np.float32)
        Xva_s = sx.transform(Xva).astype(np.float32)
        ytr = np.log1p(ytr_raw).astype(np.float32)
        
        all_preds = []
        
        # Wide shallow
        for seed, h in [(42, 1024), (123, 1024), (789, 2048)]:
            torch.manual_seed(seed)
            m = WideShallowNet(n_sel, h, 0.3)
            m = train_simple(m, Xtr_s, ytr, Xva_s, np.log1p(yva_raw).astype(np.float32), 400, 5e-4, 1e-3, 64, 50)
            m.eval()
            with torch.no_grad():
                p = np.expm1(m(torch.FloatTensor(Xva_s)).squeeze(-1).numpy())
            all_preds.append(p)
            del m; gc.collect()
        
        # Boosted
        torch.manual_seed(42)
        p = np.expm1(boosted_nn_predict(Xtr_s, ytr, Xva_s, 10))
        all_preds.append(p)
        
        # KNN augmented
        torch.manual_seed(42)
        Xtr_aug, Xva_aug = knn_augmented_features(Xtr_s, ytr, Xva_s)
        m = WideShallowNet(Xtr_aug.shape[1], 1024, 0.3)
        m = train_simple(m, Xtr_aug.astype(np.float32), ytr, 
                        Xva_aug.astype(np.float32), np.log1p(yva_raw).astype(np.float32), 400, 5e-4, 1e-3, 64, 50)
        m.eval()
        with torch.no_grad():
            p = np.expm1(m(torch.FloatTensor(Xva_aug.astype(np.float32))).squeeze(-1).numpy())
        all_preds.append(p)
        del m; gc.collect()
        
        ens = np.mean(all_preds, axis=0)
        fold_r2_mega.append(r2_score(yva_raw, ens))
    
    mr2 = np.mean(fold_r2_mega)
    star = " ★" if mr2 > best_r2 else ""
    print(f"  MegaEnsemble(5)            R²={mr2:.4f}±{np.std(fold_r2_mega):.4f}{star}", flush=True)
    if mr2 > best_r2:
        best_r2 = mr2; best_method = 'MegaEnsemble'
    
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
    
    print(f"All={X_all.shape[1]}, DW={X_dw.shape[1]}, N={len(df)}", flush=True)
    
    results = {}
    r2, m = run_experiment(X_all, y_homa, feat_cols, 'HOMA_IR_all', 0.85)
    results['HOMA_all'] = {'r2': float(r2), 'method': m}
    
    r2, m = run_experiment(X_all, y_hba1c, feat_cols, 'hba1c_all', 0.85)
    results['hba1c_all'] = {'r2': float(r2), 'method': m}
    
    r2, m = run_experiment(X_dw, y_homa, dw_cols, 'HOMA_IR_dw', 0.70, is_dw=True)
    results['HOMA_dw'] = {'r2': float(r2), 'method': m}
    
    r2, m = run_experiment(X_dw, y_hba1c, dw_cols, 'hba1c_dw', 0.70, is_dw=True)
    results['hba1c_dw'] = {'r2': float(r2), 'method': m}
    
    elapsed = (time.time()-t0)/60
    print(f"\n{'='*60}\nFINAL SUMMARY ({elapsed:.1f} min)\n{'='*60}", flush=True)
    goals = {'HOMA_all': 0.85, 'hba1c_all': 0.85, 'HOMA_dw': 0.70, 'hba1c_dw': 0.70}
    for k in ['HOMA_all', 'hba1c_all', 'HOMA_dw', 'hba1c_dw']:
        g = goals[k]
        f = "✅" if results[k]['r2'] >= g else "❌"
        print(f"  {k:15s}: R²={results[k]['r2']:.4f} (goal={g}) {f} [{results[k]['method']}]", flush=True)
    
    with open('results_v6.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
