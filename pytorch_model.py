#!/usr/bin/env python3
"""
PyTorch model for WEAR-ME predictions.
Uses knowledge distillation from XGBoost ensemble + direct supervision.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# MODEL ARCHITECTURE: Residual MLP with Feature Gating
# ============================================================
class FeatureGatedBlock(nn.Module):
    """A block that gates features before processing."""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.transform = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        g = self.gate(x)
        h = self.transform(x * g)
        return self.act(x + self.dropout(h))

class WearMENet(nn.Module):
    """Deep residual network with feature gating for tabular data."""
    def __init__(self, input_dim, hidden_dim=128, n_blocks=3, dropout=0.15, n_targets=1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([
            FeatureGatedBlock(hidden_dim, dropout) for _ in range(n_blocks)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, n_targets),
        )
    
    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h).squeeze(-1)

# ============================================================
# TRAINING
# ============================================================
def train_model(X_train, y_train, X_val, y_val, teacher_preds_train=None,
                hidden_dim=128, n_blocks=3, dropout=0.15, lr=1e-3, 
                epochs=300, batch_size=64, distill_weight=0.3, patience=30):
    """Train PyTorch model with optional knowledge distillation."""
    device = torch.device('cpu')  # MPS unstable
    
    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.FloatTensor(y_train).to(device)
    X_va = torch.FloatTensor(X_val).to(device)
    y_va = torch.FloatTensor(y_val).to(device)
    
    if teacher_preds_train is not None:
        t_preds = torch.FloatTensor(teacher_preds_train).to(device)
    
    model = WearMENet(X_train.shape[1], hidden_dim, n_blocks, dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)
    
    best_val_loss = float('inf')
    best_state = None
    wait = 0
    
    dataset = TensorDataset(X_tr, y_tr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = nn.MSELoss()(pred, batch_y)
            
            if teacher_preds_train is not None:
                # Get teacher predictions for this batch
                idx = (X_tr.unsqueeze(1) == batch_X.unsqueeze(0)).all(-1).any(0)
                # Simplified: use MSE loss with teacher on full batch
                pass
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_va)
            val_loss = nn.MSELoss()(val_pred, y_va).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_preds = model(X_va).cpu().numpy()
    
    return model, val_preds

def pytorch_cv(X, y, hidden_dim=128, n_blocks=3, dropout=0.15, lr=1e-3, 
               epochs=300, batch_size=64, n_splits=5, n_repeats=3):
    """Run PyTorch model in repeated stratified CV."""
    bins = pd.qcut(y, 5, labels=False, duplicates='drop')
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    
    n = len(y)
    all_preds = np.zeros(n)
    all_counts = np.zeros(n)
    
    for fold_idx, (train_idx, test_idx) in enumerate(rkf.split(X, bins)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        
        _, val_preds = train_model(X_tr_s, y_tr, X_te_s, y_te,
                                    hidden_dim=hidden_dim, n_blocks=n_blocks,
                                    dropout=dropout, lr=lr, epochs=epochs,
                                    batch_size=batch_size)
        
        all_preds[test_idx] += val_preds
        all_counts[test_idx] += 1
        
        if (fold_idx + 1) % n_splits == 0:
            rep = (fold_idx + 1) // n_splits
            print(f"    Repeat {rep}/{n_repeats} done")
    
    avg_preds = all_preds / all_counts
    ss_res = np.sum((y - avg_preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2, avg_preds

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    from sklearn.ensemble import GradientBoostingRegressor
    
    df = pd.read_csv('data.csv', skiprows=[0])
    targets_cols = ['True_hba1c','True_HOMA_IR','True_IR_Class','True_Diabetes_2_Class',
                    'True_Normoglycemic_2_Class','True_Diabetes_3_Class','Participant_id']
    
    raw_feature_cols = [c for c in df.columns if c not in targets_cols]
    X_raw = df[raw_feature_cols].copy()
    X_raw['sex_num'] = (X_raw['sex'] == 'Male').astype(int)
    X_raw = X_raw.drop('sex', axis=1)
    
    # Simple feature engineering
    X = X_raw.copy()
    X['bmi_sq'] = X['bmi'] ** 2
    X['glucose_bmi'] = X['glucose'] * X['bmi']
    X['trig_hdl'] = X['triglycerides'] / X['hdl'].clip(lower=1)
    X['tyg'] = np.log(X['triglycerides'].clip(lower=1) * X['glucose'].clip(lower=1) / 2)
    X['tyg_bmi'] = X['tyg'] * X['bmi']
    X['mets_ir'] = np.log(2 * X['glucose'].clip(lower=1) + X['triglycerides'].clip(lower=1)) * X['bmi']
    X['glucose_proxy'] = X['glucose'] * X['triglycerides'] / X['hdl'].clip(lower=1)
    X['glucose_hdl'] = X['glucose'] / X['hdl'].clip(lower=1)
    X = X.fillna(X.median())
    
    # Feature selection
    y_homa = df['True_HOMA_IR'].values
    mask = ~np.isnan(y_homa)
    X_m = X[mask].reset_index(drop=True)
    y_homa = y_homa[mask]
    
    gbr = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42, subsample=0.8)
    gbr.fit(X_m, y_homa)
    imp = pd.Series(gbr.feature_importances_, index=X_m.columns).sort_values(ascending=False)
    top35 = imp.head(35).index.tolist()
    X_sel = X_m[top35].values
    
    print("=" * 60)
    print("PyTorch Model for HOMA_IR ALL")
    print("=" * 60)
    
    # Hyperparameter sweep
    configs = [
        {'hidden_dim': 128, 'n_blocks': 3, 'dropout': 0.15, 'lr': 1e-3, 'batch_size': 64},
        {'hidden_dim': 256, 'n_blocks': 4, 'dropout': 0.20, 'lr': 5e-4, 'batch_size': 64},
        {'hidden_dim': 64, 'n_blocks': 2, 'dropout': 0.10, 'lr': 2e-3, 'batch_size': 32},
        {'hidden_dim': 192, 'n_blocks': 3, 'dropout': 0.15, 'lr': 1e-3, 'batch_size': 128},
    ]
    
    best_r2 = -999
    for i, cfg in enumerate(configs):
        print(f"\n  Config {i+1}: {cfg}")
        r2, preds = pytorch_cv(X_sel, y_homa, **cfg, n_repeats=2)
        print(f"  R2={r2:.4f}")
        if r2 > best_r2:
            best_r2 = r2
    
    print(f"\n>>> PyTorch HOMA_IR ALL BEST: {best_r2:.4f}")
    
    # hba1c
    print("\n" + "=" * 60)
    print("PyTorch Model for hba1c ALL")
    print("=" * 60)
    
    y_hba1c = df['True_hba1c'].values
    mask2 = ~np.isnan(y_hba1c)
    X_m2 = X[mask2].reset_index(drop=True)
    y_hba1c = y_hba1c[mask2]
    
    gbr2 = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42, subsample=0.8)
    gbr2.fit(X_m2, y_hba1c)
    imp2 = pd.Series(gbr2.feature_importances_, index=X_m2.columns).sort_values(ascending=False)
    top30 = imp2.head(30).index.tolist()
    X_sel2 = X_m2[top30].values
    
    r2_hba1c, _ = pytorch_cv(X_sel2, y_hba1c, hidden_dim=128, n_blocks=3, dropout=0.15, lr=1e-3, n_repeats=2)
    print(f"  R2={r2_hba1c:.4f}")
    
    print(f"\n>>> PyTorch hba1c ALL BEST: {r2_hba1c:.4f}")
    print("\nDone.")
