#!/usr/bin/env python3
"""
Deep Learning for Insulin Resistance & HbA1c Prediction
========================================================
SoTA tabular DL approaches:
1. ResNet-like MLP (deep residual blocks)
2. FT-Transformer (Feature Tokenizer + Transformer)
3. TabNet-style attention network
4. Ensemble of the above

Uses 5-fold CV, MPS acceleration, and extensive hyperparameter search.
"""

import os
import warnings
import json
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from copy import deepcopy

warnings.filterwarnings('ignore')

# Device setup
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print("Using Apple MPS acceleration")
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print("Using CUDA")
else:
    DEVICE = torch.device('cpu')
    print("Using CPU")

# ============================================================
# DATA LOADING & PREPROCESSING
# ============================================================

def load_data():
    df = pd.read_csv('data.csv', skiprows=[0])
    
    # Define feature groups
    demo_cols = ['age', 'sex', 'bmi']
    fitbit_cols = [c for c in df.columns if any(x in c.lower() for x in ['heart rate', 'hrv', 'step', 'sleep', 'azm'])]
    label_cols = ['True_hba1c', 'True_HOMA_IR', 'True_IR_Class', 'True_Diabetes_2_Class', 'True_Normoglycemic_2_Class', 'True_Diabetes_3_Class']
    blood_cols = [c for c in df.columns if c not in demo_cols + fitbit_cols + label_cols + ['Participant_id']]
    
    # Encode sex
    le = LabelEncoder()
    df['sex'] = le.fit_transform(df['sex'].fillna('Unknown'))
    
    # Fill missing values with median
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
    
    # All features (excluding labels and participant_id)
    all_feature_cols = demo_cols + fitbit_cols + blood_cols
    demo_wearable_cols = demo_cols + fitbit_cols
    
    targets = ['True_HOMA_IR', 'True_hba1c']
    
    # Drop rows where targets are NaN
    mask = df[targets].notna().all(axis=1)
    df = df[mask].reset_index(drop=True)
    
    return df, all_feature_cols, demo_wearable_cols, targets


# ============================================================
# MODEL ARCHITECTURES
# ============================================================

class ResBlock(nn.Module):
    """Residual block with dropout and batch norm."""
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
    
    def forward(self, x):
        return x + self.net(x)


class ResNetMLP(nn.Module):
    """Deep residual MLP for tabular data."""
    def __init__(self, input_dim, hidden_dim=256, n_blocks=4, dropout=0.3, n_targets=1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_targets)
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class FeatureTokenizer(nn.Module):
    """Tokenize each feature into an embedding."""
    def __init__(self, n_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_token))
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x):
        # x: (batch, n_features) -> (batch, n_features, d_token)
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class FTTransformer(nn.Module):
    """Feature Tokenizer Transformer - SoTA for tabular data."""
    def __init__(self, n_features, d_token=64, n_heads=4, n_layers=3, dropout=0.2, n_targets=1):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_token*4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Linear(d_token, n_targets)
        )
    
    def forward(self, x):
        tokens = self.tokenizer(x)  # (B, N, D)
        cls = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, D)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, N+1, D)
        tokens = self.transformer(tokens)
        cls_out = tokens[:, 0]  # CLS token
        return self.head(cls_out)


class GatedLinearUnit(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)
        self.bn = nn.BatchNorm1d(output_dim)
    
    def forward(self, x):
        out = self.fc(x)
        a, b = out.chunk(2, dim=-1)
        return self.bn(a * torch.sigmoid(b))


class AttentiveTabNet(nn.Module):
    """Simplified TabNet-style with attentive feature selection."""
    def __init__(self, input_dim, hidden_dim=128, n_steps=3, dropout=0.2, n_targets=1):
        super().__init__()
        self.n_steps = n_steps
        self.bn_input = nn.BatchNorm1d(input_dim)
        
        self.shared_fc = nn.Linear(input_dim, hidden_dim)
        self.step_attns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, input_dim),
                nn.BatchNorm1d(input_dim),
            ) for _ in range(n_steps)
        ])
        self.step_fcs = nn.ModuleList([
            GatedLinearUnit(input_dim, hidden_dim) for _ in range(n_steps)
        ])
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_targets)
        )
    
    def forward(self, x):
        x = self.bn_input(x)
        prior_scales = torch.ones(x.size(0), x.size(1), device=x.device)
        aggregated = torch.zeros(x.size(0), self.step_fcs[0].fc.out_features // 2, device=x.device)
        
        h = self.shared_fc(x)
        
        for i in range(self.n_steps):
            attn = self.step_attns[i](h)
            attn = attn * prior_scales
            attn = F.softmax(attn, dim=-1)
            prior_scales = prior_scales * (1.0 - attn + 1e-8)
            
            masked_x = attn * x
            step_out = self.step_fcs[i](masked_x)
            aggregated = aggregated + step_out
        
        return self.head(aggregated)


class MixedExpertMLP(nn.Module):
    """Mixture of expert MLPs - different experts for different feature subsets."""
    def __init__(self, input_dim, hidden_dim=128, n_experts=4, dropout=0.3, n_targets=1):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ) for _ in range(n_experts)
        ])
        self.gate = nn.Sequential(
            nn.Linear(input_dim, n_experts),
            nn.Softmax(dim=-1)
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_targets)
        )
    
    def forward(self, x):
        x = self.bn(x)
        weights = self.gate(x)  # (B, n_experts)
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)  # (B, n_experts, H)
        mixed = (weights.unsqueeze(-1) * expert_outs).sum(dim=1)  # (B, H)
        return self.head(mixed)


# ============================================================
# TRAINING UTILITIES
# ============================================================

class EarlyStopping:
    def __init__(self, patience=30, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_state = None
        self.stopped = False
    
    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped = True
        return self.stopped


def train_model(model, train_loader, val_X, val_y, target_idx, epochs=500, lr=1e-3, weight_decay=1e-4):
    """Train a single model with early stopping."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    es = EarlyStopping(patience=40)
    
    model.to(DEVICE)
    val_X_t = torch.FloatTensor(val_X).to(DEVICE)
    val_y_t = torch.FloatTensor(val_y[:, target_idx:target_idx+1]).to(DEVICE)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y[:, target_idx:target_idx+1].to(DEVICE)
            
            pred = model(batch_X)
            loss = F.mse_loss(pred, batch_y) + F.l1_loss(pred, batch_y) * 0.1  # Combined loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(val_X_t)
            val_loss = F.mse_loss(val_pred, val_y_t).item()
        
        if es.step(val_loss, model):
            break
    
    # Restore best
    if es.best_state is not None:
        model.load_state_dict(es.best_state)
    
    return model


def evaluate_fold(model, X_val, y_val, target_idx, scaler_y=None):
    """Evaluate model on validation fold."""
    model.eval()
    X_t = torch.FloatTensor(X_val).to(DEVICE)
    
    with torch.no_grad():
        pred = model(X_t).cpu().numpy().flatten()
    
    true = y_val[:, target_idx]
    
    # Inverse transform if scaler provided
    if scaler_y is not None:
        pred_full = np.zeros((len(pred), 2))
        true_full = np.zeros((len(true), 2))
        pred_full[:, target_idx] = pred
        true_full[:, target_idx] = true
        pred = scaler_y.inverse_transform(pred_full)[:, target_idx]
        true = scaler_y.inverse_transform(true_full)[:, target_idx]
    
    r2 = r2_score(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    
    return r2, rmse, mae, pred, true


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def run_experiment(feature_mode='all', target_name='True_HOMA_IR', n_folds=5):
    """Run full cross-validation experiment."""
    df, all_feature_cols, demo_wearable_cols, targets = load_data()
    
    target_idx = 0 if target_name == 'True_HOMA_IR' else 1
    
    if feature_mode == 'all':
        feature_cols = all_feature_cols
    else:
        feature_cols = demo_wearable_cols
    
    X = df[feature_cols].values.astype(np.float32)
    y = df[targets].values.astype(np.float32)
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {target_name} | Features: {feature_mode} ({len(feature_cols)} cols)")
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"{'='*60}")
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Model configs to try
    model_configs = {
        'ResNetMLP_small': lambda n_feat: ResNetMLP(n_feat, hidden_dim=128, n_blocks=3, dropout=0.2, n_targets=1),
        'ResNetMLP_large': lambda n_feat: ResNetMLP(n_feat, hidden_dim=256, n_blocks=5, dropout=0.3, n_targets=1),
        'ResNetMLP_wide': lambda n_feat: ResNetMLP(n_feat, hidden_dim=512, n_blocks=3, dropout=0.4, n_targets=1),
        'FTTransformer_small': lambda n_feat: FTTransformer(n_feat, d_token=32, n_heads=4, n_layers=2, dropout=0.2, n_targets=1),
        'FTTransformer_med': lambda n_feat: FTTransformer(n_feat, d_token=64, n_heads=4, n_layers=3, dropout=0.2, n_targets=1),
        'FTTransformer_large': lambda n_feat: FTTransformer(n_feat, d_token=128, n_heads=8, n_layers=4, dropout=0.3, n_targets=1),
        'AttentiveTabNet': lambda n_feat: AttentiveTabNet(n_feat, hidden_dim=128, n_steps=3, dropout=0.2, n_targets=1),
        'AttentiveTabNet_large': lambda n_feat: AttentiveTabNet(n_feat, hidden_dim=256, n_steps=5, dropout=0.3, n_targets=1),
        'MoE_small': lambda n_feat: MixedExpertMLP(n_feat, hidden_dim=128, n_experts=4, dropout=0.3, n_targets=1),
        'MoE_large': lambda n_feat: MixedExpertMLP(n_feat, hidden_dim=256, n_experts=8, dropout=0.3, n_targets=1),
    }
    
    lr_configs = [1e-3, 5e-4, 1e-4]
    
    best_r2 = -float('inf')
    best_config = None
    results_log = []
    
    for model_name, model_fn in model_configs.items():
        for lr in lr_configs:
            fold_r2s = []
            fold_rmses = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Scale features
                scaler_X = StandardScaler()
                X_train_s = scaler_X.fit_transform(X_train)
                X_val_s = scaler_X.transform(X_val)
                
                # Scale targets
                scaler_y = StandardScaler()
                y_train_s = scaler_y.fit_transform(y_train)
                y_val_s = scaler_y.transform(y_val)
                
                # DataLoader
                train_ds = TensorDataset(
                    torch.FloatTensor(X_train_s),
                    torch.FloatTensor(y_train_s)
                )
                train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
                
                # Create model
                model = model_fn(X_train_s.shape[1])
                
                # Train
                model = train_model(model, train_loader, X_val_s, y_val_s, target_idx,
                                   epochs=400, lr=lr, weight_decay=1e-4)
                
                # Evaluate
                r2, rmse, mae, _, _ = evaluate_fold(model, X_val_s, y_val_s, target_idx, scaler_y)
                fold_r2s.append(r2)
                fold_rmses.append(rmse)
            
            mean_r2 = np.mean(fold_r2s)
            std_r2 = np.std(fold_r2s)
            mean_rmse = np.mean(fold_rmses)
            
            result = {
                'model': model_name, 'lr': lr, 'target': target_name,
                'features': feature_mode, 'mean_r2': mean_r2, 'std_r2': std_r2,
                'mean_rmse': mean_rmse, 'fold_r2s': fold_r2s
            }
            results_log.append(result)
            
            marker = " ★★★" if mean_r2 > best_r2 else ""
            print(f"  {model_name} lr={lr:.0e} | R²={mean_r2:.4f}±{std_r2:.4f} | RMSE={mean_rmse:.4f}{marker}")
            
            if mean_r2 > best_r2:
                best_r2 = mean_r2
                best_config = result
    
    print(f"\n{'─'*60}")
    print(f"BEST for {target_name} ({feature_mode}): {best_config['model']} lr={best_config['lr']:.0e}")
    print(f"  R² = {best_config['mean_r2']:.4f} ± {best_config['std_r2']:.4f}")
    print(f"  RMSE = {best_config['mean_rmse']:.4f}")
    print(f"  Folds: {[f'{r:.4f}' for r in best_config['fold_r2s']]}")
    print(f"{'─'*60}")
    
    return best_config, results_log


if __name__ == '__main__':
    start = time.time()
    all_results = []
    
    experiments = [
        ('all', 'True_HOMA_IR'),
        ('all', 'True_hba1c'),
        ('demo_wearable', 'True_HOMA_IR'),
        ('demo_wearable', 'True_hba1c'),
    ]
    
    summary = {}
    for feat_mode, target in experiments:
        best, results = run_experiment(feat_mode, target)
        all_results.extend(results)
        summary[f"{target}_{feat_mode}"] = best
    
    # Final summary
    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY (elapsed: {elapsed/60:.1f} min)")
    print(f"{'='*60}")
    
    targets_goals = {
        'True_HOMA_IR_all': 0.85,
        'True_hba1c_all': 0.85,
        'True_HOMA_IR_demo_wearable': 0.70,
        'True_hba1c_demo_wearable': 0.70,
    }
    
    for key, best in summary.items():
        goal = targets_goals[key]
        achieved = "✅" if best['mean_r2'] >= goal else "❌"
        print(f"  {key}: R²={best['mean_r2']:.4f} (goal={goal}) {achieved} [{best['model']}]")
    
    # Save results
    with open('results_round1.json', 'w') as f:
        json.dump({k: {kk: (vv if not isinstance(vv, list) else vv) for kk, vv in v.items()} 
                   for k, v in summary.items()}, f, indent=2, default=str)
    
    print(f"\nResults saved to results_round1.json")
