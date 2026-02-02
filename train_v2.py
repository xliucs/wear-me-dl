#!/usr/bin/env python3
"""
V2: Smarter training with log-transforms, feature engineering, 
target-specific tuning, and multi-target learning.
"""

import os, warnings, json, time, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, QuantileTransformer, PowerTransformer
from sklearn.metrics import r2_score, mean_squared_error
from copy import deepcopy

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f"Device: {DEVICE}")

# ============================================================
# DATA LOADING & FEATURE ENGINEERING
# ============================================================

def load_and_engineer():
    df = pd.read_csv('data.csv', skiprows=[0])
    
    # Encode sex with ordinal (could try one-hot too)
    sex_map = {'Female': 0, 'Male': 1}
    df['sex_binary'] = df['sex'].map(lambda x: sex_map.get(x, 0.5))
    df['is_male'] = (df['sex'] == 'Male').astype(float)
    df['is_female'] = (df['sex'] == 'Female').astype(float)
    
    # Fill missing with median
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    
    # ---- FEATURE ENGINEERING ----
    # Metabolic ratios (domain knowledge)
    df['trig_hdl_ratio'] = df['triglycerides'] / (df['hdl'] + 1e-6)
    df['glucose_bmi'] = df['glucose'] * df['bmi']
    df['glucose_trig'] = df['glucose'] * df['triglycerides']
    df['bmi_age'] = df['bmi'] * df['age']
    df['chol_trig_ratio'] = df['total cholesterol'] / (df['triglycerides'] + 1e-6)
    df['ldl_hdl_ratio'] = df['ldl'] / (df['hdl'] + 1e-6)
    df['ast_alt_ratio'] = df['ast'] / (df['alt'] + 1e-6)
    df['bun_creat_ratio'] = df['bun'] / (df['creatinine'] + 1e-6)
    df['neutrophil_lymphocyte'] = df['absolute_neutrophils'] / (df['absolute_lymphocytes'] + 1e-6)
    df['rhr_hrv_ratio'] = df['Resting Heart Rate (mean)'] / (df['HRV (mean)'] + 1e-6)
    df['steps_sleep_ratio'] = df['STEPS (mean)'] / (df['SLEEP Duration (mean)'] + 1e-6)
    df['activity_index'] = df['STEPS (mean)'] * df['AZM Weekly (mean)'] / 1e6
    df['glucose_sq'] = df['glucose'] ** 2
    df['bmi_sq'] = df['bmi'] ** 2
    df['log_triglycerides'] = np.log1p(df['triglycerides'])
    df['log_glucose'] = np.log1p(df['glucose'])
    df['log_crp'] = np.log1p(df['crp'])
    df['ggt_alt_ratio'] = df['ggt'] / (df['alt'] + 1e-6)
    df['globulin_albumin_diff'] = df['globulin'] - df['albumin']
    df['hb_hematocrit_ratio'] = df['hb'] / (df['hematocrit'] + 1e-6)
    
    # Wearable variability features
    df['hr_cv'] = df['Resting Heart Rate (std)'] / (df['Resting Heart Rate (mean)'] + 1e-6)
    df['hrv_cv'] = df['HRV (std)'] / (df['HRV (mean)'] + 1e-6)
    df['steps_cv'] = df['STEPS (std)'] / (df['STEPS (mean)'] + 1e-6)
    df['sleep_cv'] = df['SLEEP Duration (std)'] / (df['SLEEP Duration (mean)'] + 1e-6)
    
    # Define feature groups
    demo_cols = ['age', 'bmi', 'sex_binary', 'is_male', 'is_female']
    fitbit_cols = [c for c in df.columns if any(x in c.lower() for x in ['heart rate', 'hrv', 'step', 'sleep', 'azm'])]
    
    # Engineered features for wearable group
    wearable_eng = ['rhr_hrv_ratio', 'steps_sleep_ratio', 'activity_index', 
                    'hr_cv', 'hrv_cv', 'steps_cv', 'sleep_cv',
                    'bmi_sq', 'bmi_age']
    
    # All engineered
    all_eng = ['trig_hdl_ratio', 'glucose_bmi', 'glucose_trig', 'bmi_age',
               'chol_trig_ratio', 'ldl_hdl_ratio', 'ast_alt_ratio', 'bun_creat_ratio',
               'neutrophil_lymphocyte', 'rhr_hrv_ratio', 'steps_sleep_ratio',
               'activity_index', 'glucose_sq', 'bmi_sq', 'log_triglycerides',
               'log_glucose', 'log_crp', 'ggt_alt_ratio', 'globulin_albumin_diff',
               'hb_hematocrit_ratio', 'hr_cv', 'hrv_cv', 'steps_cv', 'sleep_cv']
    
    label_cols = ['True_hba1c', 'True_HOMA_IR', 'True_IR_Class', 'True_Diabetes_2_Class', 
                  'True_Normoglycemic_2_Class', 'True_Diabetes_3_Class']
    blood_cols = [c for c in df.columns if c not in demo_cols + fitbit_cols + label_cols + 
                  ['Participant_id', 'sex'] + all_eng + ['sex_binary', 'is_male', 'is_female']]
    
    all_feature_cols = demo_cols + fitbit_cols + blood_cols + all_eng
    demo_wearable_cols = demo_cols + fitbit_cols + wearable_eng
    
    targets = ['True_HOMA_IR', 'True_hba1c']
    
    # Drop rows with NaN targets
    mask = df[targets].notna().all(axis=1)
    df = df[mask].reset_index(drop=True)
    
    # Replace inf
    for col in all_feature_cols + demo_wearable_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(df[col].median())
    
    print(f"All features: {len(all_feature_cols)}")
    print(f"Demo+wearable features: {len(demo_wearable_cols)}")
    print(f"Samples: {len(df)}")
    
    return df, all_feature_cols, demo_wearable_cols, targets


# ============================================================
# MODEL ARCHITECTURES (Refined)
# ============================================================

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return x + self.net(x)


class ResNetMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, n_blocks=6, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.head(self.blocks(self.input_proj(x)))


class FeatureTokenizer(nn.Module):
    def __init__(self, n_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, x):
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class FTTransformer(nn.Module):
    def __init__(self, n_features, d_token=64, n_heads=4, n_layers=3, dropout=0.1,
                 ffn_factor=4, prenorm=True):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, 
            dim_feedforward=d_token * ffn_factor,
            dropout=dropout, batch_first=True, activation='gelu',
            norm_first=prenorm
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, 1)
        )
    
    def forward(self, x):
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.transformer(tokens)
        return self.head(tokens[:, 0])


class DeepCrossNetwork(nn.Module):
    """DCN v2 - explicit feature crossing + deep network."""
    def __init__(self, input_dim, hidden_dim=256, n_cross=3, n_deep=3, dropout=0.1):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        # Cross layers
        self.cross_weights = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, input_dim) * 0.01) for _ in range(n_cross)
        ])
        self.cross_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim)) for _ in range(n_cross)
        ])
        
        # Deep layers
        deep_layers = []
        in_dim = input_dim
        for _ in range(n_deep):
            deep_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        self.deep = nn.Sequential(*deep_layers)
        
        self.head = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        x = self.input_bn(x)
        x0 = x
        
        # Cross network
        xc = x
        for w, b in zip(self.cross_weights, self.cross_biases):
            xc = x0 * (xc @ w) + b + xc
        
        # Deep network
        xd = self.deep(x)
        
        # Combine
        combined = torch.cat([xc, xd], dim=1)
        return self.head(combined)


class MultiTargetWrapper(nn.Module):
    """Train on both targets with shared backbone."""
    def __init__(self, backbone_fn, input_dim, hidden_dim=128):
        super().__init__()
        self.backbone = backbone_fn(input_dim)
        # Replace head
        self.backbone.head = nn.Identity()
        
        # Detect backbone output dim by forward pass
        with torch.no_grad():
            dummy = torch.randn(2, input_dim)
            out = self.backbone(dummy)
            feat_dim = out.shape[-1]
        
        self.head_homa = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.GELU(), nn.Linear(feat_dim, 1)
        )
        self.head_hba1c = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.GELU(), nn.Linear(feat_dim, 1)
        )
    
    def forward(self, x):
        feat = self.backbone(x)
        return torch.cat([self.head_homa(feat), self.head_hba1c(feat)], dim=1)


# ============================================================
# TRAINING
# ============================================================

def train_single(model, X_train, y_train, X_val, y_val, target_idx,
                 epochs=600, lr=1e-3, wd=1e-3, batch_size=64,
                 patience=50, use_quantile=False):
    """Train with advanced scheduling and early stopping."""
    model = model.to(DEVICE)
    
    # Data
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    
    val_X_t = torch.FloatTensor(X_val).to(DEVICE)
    val_y_t = torch.FloatTensor(y_val[:, target_idx:target_idx+1]).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    
    # Warmup + cosine decay
    warmup_epochs = 20
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    best_val_loss = float('inf')
    best_state = None
    no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by[:, target_idx:target_idx+1].to(DEVICE)
            pred = model(bx)
            if pred.dim() == 2 and pred.size(1) > 1:
                pred = pred[:, target_idx:target_idx+1]
            
            # Huber loss for robustness to outliers
            loss = F.huber_loss(pred, by, delta=1.0)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        
        # Val
        model.eval()
        with torch.no_grad():
            vp = model(val_X_t)
            if vp.dim() == 2 and vp.size(1) > 1:
                vp = vp[:, target_idx:target_idx+1]
            val_loss = F.mse_loss(vp, val_y_t).item()
        
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    
    if best_state:
        model.load_state_dict(best_state)
    return model


def run_cv(model_fn, X, y, target_idx, n_folds=5, transform_target='log',
           lr=1e-3, wd=1e-3, batch_size=64, epochs=600, patience=50, label=''):
    """Run k-fold CV with proper target transformation."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_r2s = []
    fold_preds = []
    fold_trues = []
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        
        # Feature scaling
        scX = StandardScaler()
        X_tr_s = scX.fit_transform(X_tr)
        X_va_s = scX.transform(X_va)
        
        # Target transformation
        if transform_target == 'log':
            y_tr_t = y_tr.copy()
            y_va_t = y_va.copy()
            y_tr_t[:, target_idx] = np.log1p(y_tr[:, target_idx])
            y_va_t[:, target_idx] = np.log1p(y_va[:, target_idx])
        elif transform_target == 'quantile':
            qt = QuantileTransformer(output_distribution='normal', n_quantiles=min(len(y_tr), 100))
            y_tr_t = y_tr.copy()
            y_va_t = y_va.copy()
            y_tr_t[:, target_idx:target_idx+1] = qt.fit_transform(y_tr[:, target_idx:target_idx+1])
            y_va_t[:, target_idx:target_idx+1] = qt.transform(y_va[:, target_idx:target_idx+1])
        elif transform_target == 'standard':
            scy = StandardScaler()
            y_tr_t = y_tr.copy()
            y_va_t = y_va.copy()
            y_tr_t[:, target_idx:target_idx+1] = scy.fit_transform(y_tr[:, target_idx:target_idx+1])
            y_va_t[:, target_idx:target_idx+1] = scy.transform(y_va[:, target_idx:target_idx+1])
        else:
            y_tr_t, y_va_t = y_tr.copy(), y_va.copy()
        
        model = model_fn(X_tr_s.shape[1])
        model = train_single(model, X_tr_s, y_tr_t, X_va_s, y_va_t, target_idx,
                            epochs=epochs, lr=lr, wd=wd, batch_size=batch_size, patience=patience)
        
        # Predict
        model.eval()
        with torch.no_grad():
            pred = model(torch.FloatTensor(X_va_s).to(DEVICE))
            if pred.dim() == 2 and pred.size(1) > 1:
                pred = pred[:, target_idx:target_idx+1]
            pred = pred.cpu().numpy().flatten()
        
        # Inverse transform
        true = y_va[:, target_idx]
        if transform_target == 'log':
            pred = np.expm1(pred)
        elif transform_target == 'quantile':
            pred = qt.inverse_transform(pred.reshape(-1, 1)).flatten()
        elif transform_target == 'standard':
            pred = scy.inverse_transform(pred.reshape(-1, 1)).flatten()
        
        r2 = r2_score(true, pred)
        fold_r2s.append(r2)
        fold_preds.extend(pred)
        fold_trues.extend(true)
    
    mean_r2 = np.mean(fold_r2s)
    std_r2 = np.std(fold_r2s)
    overall_r2 = r2_score(fold_trues, fold_preds)
    rmse = np.sqrt(mean_squared_error(fold_trues, fold_preds))
    
    print(f"  {label:40s} | R²={mean_r2:.4f}±{std_r2:.4f} | overall={overall_r2:.4f} | RMSE={rmse:.4f}")
    
    return {'label': label, 'mean_r2': mean_r2, 'std_r2': std_r2, 'overall_r2': overall_r2, 
            'rmse': rmse, 'fold_r2s': fold_r2s}


# ============================================================
# MAIN
# ============================================================

def main():
    df, all_cols, dw_cols, targets = load_and_engineer()
    
    X_all = df[all_cols].values.astype(np.float32)
    X_dw = df[dw_cols].values.astype(np.float32)
    y = df[targets].values.astype(np.float32)
    
    all_results = {}
    
    # ---- EXPERIMENT 1: True_HOMA_IR with ALL features ----
    print(f"\n{'='*80}")
    print(f"EXPERIMENT 1: True_HOMA_IR | ALL features ({X_all.shape[1]})")
    print(f"{'='*80}")
    
    configs = [
        # (model_fn, transform, lr, wd, batch, epochs, patience, label)
        (lambda d: ResNetMLP(d, 256, 6, 0.1), 'log', 5e-4, 1e-3, 64, 600, 60, 'ResNet256x6_log'),
        (lambda d: ResNetMLP(d, 512, 8, 0.15), 'log', 3e-4, 1e-3, 64, 600, 60, 'ResNet512x8_log'),
        (lambda d: ResNetMLP(d, 256, 6, 0.1), 'quantile', 5e-4, 1e-3, 64, 600, 60, 'ResNet256x6_quantile'),
        (lambda d: FTTransformer(d, 64, 4, 3, 0.1), 'log', 1e-4, 1e-4, 64, 600, 60, 'FTTrans64x3_log'),
        (lambda d: FTTransformer(d, 128, 8, 4, 0.15), 'log', 1e-4, 1e-4, 64, 600, 60, 'FTTrans128x4_log'),
        (lambda d: FTTransformer(d, 64, 4, 3, 0.1), 'quantile', 1e-4, 1e-4, 64, 600, 60, 'FTTrans64x3_quantile'),
        (lambda d: DeepCrossNetwork(d, 256, 3, 4, 0.1), 'log', 5e-4, 1e-3, 64, 600, 60, 'DCN256_log'),
        (lambda d: DeepCrossNetwork(d, 512, 4, 5, 0.15), 'log', 3e-4, 1e-3, 64, 600, 60, 'DCN512_log'),
        (lambda d: DeepCrossNetwork(d, 256, 3, 4, 0.1), 'quantile', 5e-4, 1e-3, 64, 600, 60, 'DCN256_quantile'),
    ]
    
    best_r2_1 = -1
    for model_fn, transform, lr, wd, bs, ep, pat, label in configs:
        res = run_cv(model_fn, X_all, y, 0, transform_target=transform, lr=lr, wd=wd, 
                     batch_size=bs, epochs=ep, patience=pat, label=label)
        if res['mean_r2'] > best_r2_1:
            best_r2_1 = res['mean_r2']
            all_results['HOMA_all'] = res
    print(f"  >>> BEST: R²={best_r2_1:.4f}")
    
    # ---- EXPERIMENT 2: True_hba1c with ALL features ----
    print(f"\n{'='*80}")
    print(f"EXPERIMENT 2: True_hba1c | ALL features ({X_all.shape[1]})")
    print(f"{'='*80}")
    
    configs2 = [
        (lambda d: ResNetMLP(d, 256, 6, 0.1), 'log', 5e-4, 1e-3, 64, 600, 60, 'ResNet256x6_log'),
        (lambda d: ResNetMLP(d, 512, 8, 0.15), 'log', 3e-4, 1e-3, 64, 600, 60, 'ResNet512x8_log'),
        (lambda d: ResNetMLP(d, 256, 6, 0.1), 'quantile', 5e-4, 1e-3, 64, 600, 60, 'ResNet256x6_quantile'),
        (lambda d: FTTransformer(d, 64, 4, 3, 0.1), 'log', 1e-4, 1e-4, 64, 600, 60, 'FTTrans64x3_log'),
        (lambda d: FTTransformer(d, 128, 8, 4, 0.15), 'log', 1e-4, 1e-4, 64, 600, 60, 'FTTrans128x4_log'),
        (lambda d: DeepCrossNetwork(d, 256, 3, 4, 0.1), 'log', 5e-4, 1e-3, 64, 600, 60, 'DCN256_log'),
        (lambda d: DeepCrossNetwork(d, 512, 4, 5, 0.15), 'log', 3e-4, 1e-3, 64, 600, 60, 'DCN512_log'),
        (lambda d: DeepCrossNetwork(d, 256, 3, 4, 0.1), 'quantile', 5e-4, 1e-3, 64, 600, 60, 'DCN256_quantile'),
    ]
    
    best_r2_2 = -1
    for model_fn, transform, lr, wd, bs, ep, pat, label in configs2:
        res = run_cv(model_fn, X_all, y, 1, transform_target=transform, lr=lr, wd=wd,
                     batch_size=bs, epochs=ep, patience=pat, label=label)
        if res['mean_r2'] > best_r2_2:
            best_r2_2 = res['mean_r2']
            all_results['hba1c_all'] = res
    print(f"  >>> BEST: R²={best_r2_2:.4f}")
    
    # ---- EXPERIMENT 3: True_HOMA_IR with Demo+Wearable ----
    print(f"\n{'='*80}")
    print(f"EXPERIMENT 3: True_HOMA_IR | Demo+Wearable ({X_dw.shape[1]})")
    print(f"{'='*80}")
    
    configs3 = [
        (lambda d: ResNetMLP(d, 128, 4, 0.1), 'log', 5e-4, 1e-3, 64, 600, 60, 'ResNet128x4_log'),
        (lambda d: ResNetMLP(d, 256, 6, 0.15), 'log', 5e-4, 1e-3, 64, 600, 60, 'ResNet256x6_log'),
        (lambda d: FTTransformer(d, 32, 4, 2, 0.1), 'log', 1e-4, 1e-4, 64, 600, 60, 'FTTrans32x2_log'),
        (lambda d: FTTransformer(d, 64, 4, 3, 0.1), 'log', 1e-4, 1e-4, 64, 600, 60, 'FTTrans64x3_log'),
        (lambda d: DeepCrossNetwork(d, 128, 3, 3, 0.1), 'log', 5e-4, 1e-3, 64, 600, 60, 'DCN128_log'),
        (lambda d: DeepCrossNetwork(d, 256, 4, 4, 0.1), 'log', 5e-4, 1e-3, 64, 600, 60, 'DCN256_log'),
    ]
    
    best_r2_3 = -1
    for model_fn, transform, lr, wd, bs, ep, pat, label in configs3:
        res = run_cv(model_fn, X_dw, y, 0, transform_target=transform, lr=lr, wd=wd,
                     batch_size=bs, epochs=ep, patience=pat, label=label)
        if res['mean_r2'] > best_r2_3:
            best_r2_3 = res['mean_r2']
            all_results['HOMA_dw'] = res
    print(f"  >>> BEST: R²={best_r2_3:.4f}")
    
    # ---- EXPERIMENT 4: True_hba1c with Demo+Wearable ----
    print(f"\n{'='*80}")
    print(f"EXPERIMENT 4: True_hba1c | Demo+Wearable ({X_dw.shape[1]})")
    print(f"{'='*80}")
    
    configs4 = [
        (lambda d: ResNetMLP(d, 128, 4, 0.1), 'log', 5e-4, 1e-3, 64, 600, 60, 'ResNet128x4_log'),
        (lambda d: ResNetMLP(d, 256, 6, 0.15), 'log', 5e-4, 1e-3, 64, 600, 60, 'ResNet256x6_log'),
        (lambda d: FTTransformer(d, 32, 4, 2, 0.1), 'log', 1e-4, 1e-4, 64, 600, 60, 'FTTrans32x2_log'),
        (lambda d: FTTransformer(d, 64, 4, 3, 0.1), 'log', 1e-4, 1e-4, 64, 600, 60, 'FTTrans64x3_log'),
        (lambda d: DeepCrossNetwork(d, 128, 3, 3, 0.1), 'log', 5e-4, 1e-3, 64, 600, 60, 'DCN128_log'),
        (lambda d: DeepCrossNetwork(d, 256, 4, 4, 0.1), 'log', 5e-4, 1e-3, 64, 600, 60, 'DCN256_log'),
    ]
    
    best_r2_4 = -1
    for model_fn, transform, lr, wd, bs, ep, pat, label in configs4:
        res = run_cv(model_fn, X_dw, y, 1, transform_target=transform, lr=lr, wd=wd,
                     batch_size=bs, epochs=ep, patience=pat, label=label)
        if res['mean_r2'] > best_r2_4:
            best_r2_4 = res['mean_r2']
            all_results['hba1c_dw'] = res
    print(f"  >>> BEST: R²={best_r2_4:.4f}")
    
    # ---- SUMMARY ----
    print(f"\n{'='*80}")
    print(f"V2 SUMMARY")
    print(f"{'='*80}")
    goals = {'HOMA_all': 0.85, 'hba1c_all': 0.85, 'HOMA_dw': 0.70, 'hba1c_dw': 0.70}
    for key in ['HOMA_all', 'hba1c_all', 'HOMA_dw', 'hba1c_dw']:
        if key in all_results:
            r = all_results[key]
            g = goals[key]
            flag = "✅" if r['mean_r2'] >= g else "❌"
            print(f"  {key:20s}: R²={r['mean_r2']:.4f}±{r['std_r2']:.4f} (goal={g}) {flag} [{r['label']}]")
    
    with open('results_v2.json', 'w') as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'fold_r2s'} for k, v in all_results.items()}, f, indent=2)
    
    print("\nSaved results_v2.json")
    return all_results


if __name__ == '__main__':
    t0 = time.time()
    results = main()
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
