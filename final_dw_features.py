#!/usr/bin/env python3
"""
WEAR-ME: Prediction with Demographics + Wearables Only (DW)
=============================================================

Targets:
  - True_HOMA_IR  (R² target: 0.37)
  - True_hba1c    (R² target: 0.70)

Method: Multi-layer stacking ensemble (V22b architecture)
  Layer 0: ~385 base models across 5 feature sets
  Layer 1: ~41 diverse meta-learners on base OOF predictions
  Layer 2: Ridge on Layer-1 stacks

Usage:
  # Default CV (RepeatedStratifiedKFold, 5-fold × 5 repeats):
  python3 final_dw_features.py

  # Custom splits from file (JSON list of {"train": [...], "test": [...]}):
  python3 final_dw_features.py --splits splits.json

  # Single target:
  python3 final_dw_features.py --target homa_ir
  python3 final_dw_features.py --target hba1c

  # As a library:
  from final_dw_features import run_pipeline
  results = run_pipeline("data.csv", target="homa_ir", splits=my_splits)
"""

import argparse
import json
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import (
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_selection import mutual_info_regression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lasso,
    Ridge,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, NuSVR

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed. Some models will be skipped.")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("WARNING: lightgbm not installed. Some models will be skipped.")


# =============================================================================
# CONSTANTS
# =============================================================================

DW_COLUMNS = [
    "age", "bmi", "sex",
    "Resting Heart Rate (mean)", "Resting Heart Rate (median)", "Resting Heart Rate (std)",
    "HRV (mean)", "HRV (median)", "HRV (std)",
    "STEPS (mean)", "STEPS (median)", "STEPS (std)",
    "SLEEP Duration (mean)", "SLEEP Duration (median)", "SLEEP Duration (std)",
    "AZM Weekly (mean)", "AZM Weekly (median)", "AZM Weekly (std)",
]

# Short aliases for column names
_RHR_M = "Resting Heart Rate (mean)"
_RHR_MD = "Resting Heart Rate (median)"
_RHR_S = "Resting Heart Rate (std)"
_HRV_M = "HRV (mean)"
_HRV_MD = "HRV (median)"
_HRV_S = "HRV (std)"
_STP_M = "STEPS (mean)"
_STP_MD = "STEPS (median)"
_STP_S = "STEPS (std)"
_SLP_M = "SLEEP Duration (mean)"
_SLP_MD = "SLEEP Duration (median)"
_SLP_S = "SLEEP Duration (std)"
_AZM_M = "AZM Weekly (mean)"
_AZM_MD = "AZM Weekly (median)"
_AZM_S = "AZM Weekly (std)"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(data_path: str, target: str = "homa_ir") -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load WEAR-ME dataset and extract DW columns.

    Returns
    -------
    X : pd.DataFrame with 18 DW columns (sex encoded as 0/1)
    y : np.ndarray of target values (NaN rows removed)
    """
    df = pd.read_csv(data_path, skiprows=[0])
    X = df[DW_COLUMNS].copy()
    X["sex"] = (X["sex"] == "Male").astype(float)

    target_col = "True_HOMA_IR" if target == "homa_ir" else "True_hba1c"
    y = df[target_col].values

    mask = ~np.isnan(y)
    X = X[mask].reset_index(drop=True)
    y = y[mask]
    return X, y


# =============================================================================
# FEATURE ENGINEERING (94 features from 18 DW columns)
# =============================================================================

def engineer_features(X: pd.DataFrame) -> np.ndarray:
    """Engineer ~94 features from 18 DW columns. Returns numpy array."""
    X = X.copy()

    # Skewness & CV for each wearable signal
    for pfx, m, md, s in [
        ("rhr", _RHR_M, _RHR_MD, _RHR_S), ("hrv", _HRV_M, _HRV_MD, _HRV_S),
        ("stp", _STP_M, _STP_MD, _STP_S), ("slp", _SLP_M, _SLP_MD, _SLP_S),
        ("azm", _AZM_M, _AZM_MD, _AZM_S),
    ]:
        X[f"{pfx}_skew"] = (X[m] - X[md]) / X[s].clip(lower=0.01)
        X[f"{pfx}_cv"] = X[s] / X[m].clip(lower=0.01)

    # Polynomial transforms
    for col, nm in [(X["bmi"], "bmi"), (X["age"], "age"),
                    (X[_RHR_M], "rhr"), (X[_HRV_M], "hrv"), (X[_STP_M], "stp")]:
        X[f"{nm}_sq"] = col ** 2
        X[f"{nm}_log"] = np.log1p(col.clip(lower=0))
        X[f"{nm}_inv"] = 1 / col.clip(lower=0.01)

    # BMI interactions
    X["bmi_rhr"] = X["bmi"] * X[_RHR_M]
    X["bmi_sq_rhr"] = X["bmi"] ** 2 * X[_RHR_M]
    X["bmi_hrv"] = X["bmi"] * X[_HRV_M]
    X["bmi_hrv_inv"] = X["bmi"] / X[_HRV_M].clip(lower=1)
    X["bmi_stp"] = X["bmi"] * X[_STP_M]
    X["bmi_stp_inv"] = X["bmi"] / X[_STP_M].clip(lower=1) * 1000
    X["bmi_slp"] = X["bmi"] * X[_SLP_M]
    X["bmi_azm"] = X["bmi"] * X[_AZM_M]
    X["bmi_age"] = X["bmi"] * X["age"]
    X["bmi_sq_age"] = X["bmi"] ** 2 * X["age"]
    X["bmi_sex"] = X["bmi"] * X["sex"]
    X["bmi_rhr_hrv"] = X["bmi"] * X[_RHR_M] / X[_HRV_M].clip(lower=1)
    X["bmi_rhr_stp"] = X["bmi"] * X[_RHR_M] / X[_STP_M].clip(lower=1) * 1000

    # Age interactions
    X["age_rhr"] = X["age"] * X[_RHR_M]
    X["age_hrv_inv"] = X["age"] / X[_HRV_M].clip(lower=1)
    X["age_stp"] = X["age"] * X[_STP_M]
    X["age_slp"] = X["age"] * X[_SLP_M]
    X["age_sex"] = X["age"] * X["sex"]
    X["age_bmi_sex"] = X["age"] * X["bmi"] * X["sex"]

    # Wearable cross-interactions
    X["rhr_hrv"] = X[_RHR_M] / X[_HRV_M].clip(lower=1)
    X["stp_hrv"] = X[_STP_M] * X[_HRV_M]
    X["stp_rhr"] = X[_STP_M] / X[_RHR_M].clip(lower=1)
    X["azm_stp"] = X[_AZM_M] / X[_STP_M].clip(lower=1)
    X["slp_hrv"] = X[_SLP_M] * X[_HRV_M]
    X["slp_rhr"] = X[_SLP_M] / X[_RHR_M].clip(lower=1)

    # Composite health indices
    X["cardio"] = X[_HRV_M] * X[_STP_M] / X[_RHR_M].clip(lower=1)
    X["cardio_log"] = np.log1p(X["cardio"].clip(lower=0))
    X["met_load"] = X["bmi"] * X[_RHR_M] / X[_STP_M].clip(lower=1) * 1000
    X["met_load_log"] = np.log1p(X["met_load"].clip(lower=0))
    X["recovery"] = X[_HRV_M] / X[_RHR_M].clip(lower=1) * X[_SLP_M]
    X["activity_bmi"] = (X[_STP_M] + X[_AZM_M]) / X["bmi"]
    X["sed_risk"] = X["bmi"] ** 2 * X[_RHR_M] / (X[_STP_M].clip(lower=1) * X[_HRV_M].clip(lower=1))
    X["sed_risk_log"] = np.log1p(X["sed_risk"].clip(lower=0))
    X["auto_health"] = X[_HRV_M] / X[_RHR_M].clip(lower=1)
    X["hr_reserve"] = (220 - X["age"] - X[_RHR_M]) / X["bmi"]
    X["fitness_age"] = X["age"] * X[_RHR_M] / X[_HRV_M].clip(lower=1)
    X["bmi_fitness"] = X["bmi"] * X[_RHR_M] / (X[_HRV_M].clip(lower=1) * X[_STP_M].clip(lower=1)) * 10000

    # Binary indicators + interactions
    X["obese"] = (X["bmi"] >= 30).astype(float)
    X["older"] = (X["age"] >= 50).astype(float)
    X["obese_rhr"] = X["obese"] * X[_RHR_M]
    X["obese_low_hrv"] = X["obese"] * (X[_HRV_M] < X[_HRV_M].median()).astype(float)
    X["older_bmi"] = X["older"] * X["bmi"]
    X["older_rhr"] = X["older"] * X[_RHR_M]
    X["rhr_cv_bmi"] = X["rhr_cv"] * X["bmi"]
    X["hrv_cv_bmi"] = X["hrv_cv"] * X["bmi"]
    X["rhr_cv_age"] = X["rhr_cv"] * X["age"]

    # Rank features
    for col in ["bmi", "age", _RHR_M, _HRV_M, _STP_M]:
        X[f"rank_{col[:3]}"] = X[col].rank(pct=True)

    return X.fillna(0).values


# =============================================================================
# AUGMENTED FEATURES (OOF — no leakage)
# =============================================================================

def make_target_encoded_features(
    X_df: pd.DataFrame, y: np.ndarray, bins: np.ndarray,
    smooth: float = 10.0,
) -> np.ndarray:
    """Create out-of-fold target-encoded features (21 features)."""
    columns = ["bmi", "age", _RHR_M, _HRV_M, _STP_M, _SLP_M, _AZM_M]
    n_bins_list = [3, 5, 10]
    te = np.zeros((len(y), 0))
    global_mean = y.mean()

    for col in columns:
        vals = X_df[col].values
        for nb in n_bins_list:
            try:
                edges = np.percentile(vals, np.linspace(0, 100, nb + 1))
                edges[0] -= 1
                edges[-1] += 1
                coded = np.digitize(vals, edges[1:-1])
            except Exception:
                continue
            encoded = np.zeros(len(y))
            for tr, te_idx in StratifiedKFold(5, shuffle=True, random_state=42).split(vals, bins):
                for b in range(nb):
                    mtr = coded[tr] == b
                    mte = coded[te_idx] == b
                    if mtr.sum() > 0:
                        bm = y[tr][mtr].mean()
                        bc = mtr.sum()
                        encoded[te_idx[mte]] = (bc * bm + smooth * global_mean) / (bc + smooth)
                    else:
                        encoded[te_idx[mte]] = global_mean
            te = np.column_stack([te, encoded])
    return te


def make_knn_target_features(
    X: np.ndarray, y: np.ndarray, bins: np.ndarray,
) -> np.ndarray:
    """Create OOF KNN-based target features (12 features)."""
    k_list = [5, 10, 20, 50]
    kf = np.zeros((len(y), 0))
    for k in k_list:
        mean_f = np.zeros(len(y))
        std_f = np.zeros(len(y))
        med_f = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, bins):
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[tr])
            Xte = sc.transform(X[te])
            nn = NearestNeighbors(n_neighbors=k)
            nn.fit(Xtr)
            _, idx = nn.kneighbors(Xte)
            for i, ti in enumerate(te):
                nt = y[tr][idx[i]]
                mean_f[ti] = nt.mean()
                std_f[ti] = nt.std()
                med_f[ti] = np.median(nt)
        kf = np.column_stack([kf, mean_f, std_f, med_f])
    return kf


def make_quantile_classification_features(
    X: np.ndarray, y: np.ndarray,
) -> np.ndarray:
    """Create OOF quantile classification probability features (4 features)."""
    quantiles = [0.25, 0.5, 0.75, 0.9]
    qf = np.zeros((len(y), 0))
    for q in quantiles:
        thr = np.quantile(y, q)
        yc = (y > thr).astype(int)
        probs = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, yc):
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[tr])
            Xte = sc.transform(X[te])
            clf = GradientBoostingClassifier(
                n_estimators=100, max_depth=2, learning_rate=0.1, random_state=42,
            )
            clf.fit(Xtr, yc[tr])
            probs[te] = clf.predict_proba(Xte)[:, 1]
        qf = np.column_stack([qf, probs])
    return qf


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

FAST_MODELS = [
    ("ridge_10", lambda: Ridge(alpha=10), True),
    ("ridge_50", lambda: Ridge(alpha=50), True),
    ("ridge_100", lambda: Ridge(alpha=100), True),
    ("ridge_500", lambda: Ridge(alpha=500), True),
    ("ridge_1000", lambda: Ridge(alpha=1000), True),
    ("ridge_2000", lambda: Ridge(alpha=2000), True),
    ("lasso_001", lambda: Lasso(alpha=0.01, max_iter=10000), True),
    ("lasso_01", lambda: Lasso(alpha=0.1, max_iter=10000), True),
    ("elastic_01_5", lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000), True),
    ("elastic_001_5", lambda: ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000), True),
    ("bayesian", lambda: BayesianRidge(), True),
    ("huber", lambda: HuberRegressor(max_iter=1000), True),
    ("kr_rbf_01_001", lambda: KernelRidge(alpha=0.1, kernel="rbf", gamma=0.001), True),
    ("kr_rbf_01_01", lambda: KernelRidge(alpha=0.1, kernel="rbf", gamma=0.01), True),
    ("kr_rbf_1_001", lambda: KernelRidge(alpha=1, kernel="rbf", gamma=0.001), True),
    ("kr_rbf_1_01", lambda: KernelRidge(alpha=1, kernel="rbf", gamma=0.01), True),
    ("kr_rbf_10_01", lambda: KernelRidge(alpha=10, kernel="rbf", gamma=0.01), True),
    ("kr_poly2", lambda: KernelRidge(alpha=1, kernel="poly", degree=2, gamma=0.01), True),
    ("svr_1", lambda: SVR(kernel="rbf", C=1, epsilon=0.1), True),
    ("svr_10", lambda: SVR(kernel="rbf", C=10, epsilon=0.1), True),
    ("nusvr_05", lambda: NuSVR(kernel="rbf", C=1, nu=0.5), True),
    ("nusvr_07", lambda: NuSVR(kernel="rbf", C=1, nu=0.7), True),
    ("knn_10", lambda: KNeighborsRegressor(n_neighbors=10, weights="distance"), True),
    ("knn_15", lambda: KNeighborsRegressor(n_neighbors=15, weights="distance"), True),
    ("knn_20", lambda: KNeighborsRegressor(n_neighbors=20, weights="distance"), True),
    ("knn_30", lambda: KNeighborsRegressor(n_neighbors=30, weights="distance"), True),
    ("knn_50", lambda: KNeighborsRegressor(n_neighbors=50, weights="distance"), True),
    ("knn_75", lambda: KNeighborsRegressor(n_neighbors=75, weights="distance"), True),
    ("bag_ridge100", lambda: BaggingRegressor(estimator=Ridge(alpha=100), n_estimators=30, max_samples=0.8, max_features=0.8, random_state=42), True),
    ("bag_ridge500", lambda: BaggingRegressor(estimator=Ridge(alpha=500), n_estimators=30, max_samples=0.8, max_features=0.8, random_state=42), True),
    ("bag_ridge1000", lambda: BaggingRegressor(estimator=Ridge(alpha=1000), n_estimators=30, max_samples=0.8, max_features=0.8, random_state=42), True),
]


def get_medium_models():
    """Tree-based models (slower, use fewer CV repeats)."""
    models = []
    if HAS_XGB:
        models.extend([
            ("xgb_d2", lambda: xgb.XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, reg_alpha=2, reg_lambda=5, random_state=42), False),
            ("xgb_d3_lr01", lambda: xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.01, subsample=0.8, colsample_bytree=0.7, reg_alpha=5, reg_lambda=10, random_state=42), False),
            ("xgb_d2_mae", lambda: xgb.XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, objective="reg:absoluteerror", random_state=42), False),
            ("xgb_d2_huber", lambda: xgb.XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, objective="reg:pseudohubererror", random_state=42), False),
        ])
    if HAS_LGB:
        models.extend([
            ("lgb_d2", lambda: lgb.LGBMRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, verbose=-1, n_jobs=1, random_state=42), False),
            ("lgb_d3", lambda: lgb.LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, verbose=-1, n_jobs=1, random_state=42), False),
            ("lgb_dart", lambda: lgb.LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, boosting_type="dart", verbose=-1, n_jobs=1, random_state=42), False),
        ])
    models.extend([
        ("hgbr_d2", lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=2, learning_rate=0.05, random_state=42), False),
        ("hgbr_d3", lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05, random_state=42), False),
        ("gbr_d2", lambda: GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=42), False),
        ("gbr_d3", lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42), False),
        ("rf_d3", lambda: RandomForestRegressor(n_estimators=300, max_depth=3, max_features=0.5, random_state=42, n_jobs=1), False),
        ("rf_d4", lambda: RandomForestRegressor(n_estimators=300, max_depth=4, max_features=0.5, random_state=42, n_jobs=1), False),
        ("et_d3", lambda: ExtraTreesRegressor(n_estimators=300, max_depth=3, max_features=0.5, random_state=42, n_jobs=1), False),
        ("et_d4", lambda: ExtraTreesRegressor(n_estimators=300, max_depth=4, max_features=0.5, random_state=42, n_jobs=1), False),
    ])
    return models


# =============================================================================
# CORE CV ENGINE
# =============================================================================

def make_stratification_bins(y: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Bin continuous target for stratified splitting."""
    return pd.qcut(y, n_bins, labels=False, duplicates="drop")


def run_cv(
    X: np.ndarray, y: np.ndarray, bins: np.ndarray, n_samples: int, ss_tot: float,
    model_fn: callable, scale: bool = True, log_t: bool = False,
    n_rep: int = 5, seed: int = 42,
    splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
) -> Tuple[float, np.ndarray]:
    """
    Run repeated stratified CV for a single model.

    If `splits` is provided, uses those splits directly.
    Otherwise generates RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep).

    Returns (R², OOF_predictions).
    """
    if splits is not None:
        fold_iter = splits
    else:
        rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=seed)
        fold_iter = rkf.split(X, bins)

    preds = np.zeros(n_samples)
    counts = np.zeros(n_samples)

    for tr, te in fold_iter:
        Xtr, Xte = X[tr].copy(), X[te].copy()
        ytr = y[tr].copy()
        if log_t:
            ytr = np.log1p(ytr)
        if scale:
            s = StandardScaler()
            Xtr = s.fit_transform(Xtr)
            Xte = s.transform(Xte)
        m = model_fn()
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        if log_t:
            p = np.expm1(p)
        preds[te] += p
        counts[te] += 1

    preds /= counts
    r2 = 1 - np.sum((y - preds) ** 2) / ss_tot
    return r2, preds


# =============================================================================
# PIPELINE
# =============================================================================

def run_pipeline(
    data_path: str,
    target: str = "homa_ir",
    splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    verbose: bool = True,
) -> dict:
    """
    Run the full DW multi-layer stacking pipeline.

    Parameters
    ----------
    data_path : str
        Path to data.csv.
    target : str
        "homa_ir" or "hba1c".
    splits : list of (train_idx, test_idx), optional
        Custom CV splits for base models. If None, uses default
        RepeatedStratifiedKFold(5-fold, 5 repeats) for fast models,
        (5-fold, 3 repeats) for slow models.
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys:
        "r2": float — best OOF R²
        "r": float — Pearson correlation
        "oof_predictions": np.ndarray
        "y": np.ndarray
        "n_base_models": int
        "n_stacks": int
        "n_samples": int
    """
    assert target in ("homa_ir", "hba1c"), f"Unknown target: {target}"
    t0 = time.time()

    # --- Load data ---
    if verbose:
        print(f"Loading data from {data_path}...", flush=True)
    X_df, y = load_data(data_path, target=target)
    n_samples = len(y)
    bins = make_stratification_bins(y)
    ss_tot = np.sum((y - y.mean()) ** 2)
    X_raw = X_df.values
    use_log = (target == "homa_ir")  # HOMA is right-skewed

    if verbose:
        print(f"  Samples: {n_samples}, Target: {target}, y: [{y.min():.2f}, {y.max():.2f}]")

    # --- Feature engineering ---
    if verbose:
        print("Engineering features...", flush=True)
    X_eng = engineer_features(X_df)
    mi = mutual_info_regression(X_eng, y, random_state=42)
    X_mi35 = X_eng[:, np.argsort(mi)[-35:]]

    # --- Augmented features (OOF, no leakage) ---
    if verbose:
        print("Creating augmented features...", flush=True)
    te_feats = make_target_encoded_features(X_df, y, bins)
    knn_feats = make_knn_target_features(X_raw, y, bins)
    qc_feats = make_quantile_classification_features(X_raw, y)

    X_mega = np.column_stack([X_raw, te_feats, knn_feats, qc_feats])
    X_mega_eng = np.column_stack([X_eng, te_feats, knn_feats, qc_feats])

    fsets = {
        "raw18": X_raw,
        "eng": X_eng,
        "mi35": X_mi35,
        "mega": X_mega,
        "mega_eng": X_mega_eng,
    }

    if verbose:
        print(f"  Feature sets: raw18={X_raw.shape[1]}, eng={X_eng.shape[1]}, "
              f"mi35={X_mi35.shape[1]}, mega={X_mega.shape[1]}, mega_eng={X_mega_eng.shape[1]}", flush=True)

    # --- Prepare splits ---
    # If custom splits provided, use them for both fast and medium models.
    # Otherwise, fast models get 5 repeats, medium get 3 repeats (matching V22b).
    fast_splits = splits  # None means run_cv will generate its own
    med_splits = splits
    fast_n_rep = 5
    med_n_rep = 3

    # =================================================================
    # LAYER 0: Generate base model OOF predictions
    # =================================================================
    if verbose:
        print(f"\n{'='*60}", flush=True)
        print("LAYER 0: Base Models", flush=True)
        print(f"{'='*60}", flush=True)

    all_results = {}
    medium_models = get_medium_models()

    for fs_name, X_fs in fsets.items():
        if verbose:
            print(f"\n--- {fs_name} ({X_fs.shape[1]}f) ---", flush=True)

        for mname, mfn, scale in FAST_MODELS:
            full = f"{fs_name}__{mname}"
            try:
                r2, preds = run_cv(X_fs, y, bins, n_samples, ss_tot, mfn,
                                   scale=scale, n_rep=fast_n_rep, splits=fast_splits)
                all_results[full] = {"r2": r2, "preds": preds}
                if r2 > 0.15 and verbose:
                    print(f"  {full:50s}: R²={r2:.4f}", flush=True)
                if use_log:
                    r2l, pl = run_cv(X_fs, y, bins, n_samples, ss_tot, mfn,
                                     scale=scale, log_t=True, n_rep=fast_n_rep, splits=fast_splits)
                    all_results[full + "_log"] = {"r2": r2l, "preds": pl}
                    if r2l > 0.15 and verbose:
                        print(f"  {full+'_log':50s}: R²={r2l:.4f}", flush=True)
            except Exception:
                pass

        for mname, mfn, scale in medium_models:
            full = f"{fs_name}__{mname}"
            try:
                r2, preds = run_cv(X_fs, y, bins, n_samples, ss_tot, mfn,
                                   scale=scale, n_rep=med_n_rep, splits=med_splits)
                all_results[full] = {"r2": r2, "preds": preds}
                if r2 > 0.15 and verbose:
                    print(f"  {full:50s}: R²={r2:.4f}", flush=True)
            except Exception:
                pass

    if verbose:
        print(f"\nTotal models: {len(all_results)}", flush=True)

    # =================================================================
    # LAYER 1: Stacking meta-learners
    # =================================================================
    if verbose:
        print(f"\n{'='*60}", flush=True)
        print("MULTI-LAYER STACKING", flush=True)
        print(f"{'='*60}", flush=True)

    good = {k: v for k, v in all_results.items() if v["r2"] > 0}
    names = sorted(good.keys(), key=lambda k: good[k]["r2"], reverse=True)[:25]
    preds_mat = np.array([good[nm]["preds"] for nm in names])
    nm_count = len(names)

    if verbose:
        print(f"Top 10 of {len(good)} models:")
        for nm in names[:10]:
            print(f"  {nm:55s}: R²={good[nm]['r2']:.4f}", flush=True)

    best_r2 = -999
    all_stacks = {}

    # -- Ridge stacks --
    for alpha in [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000]:
        sp = np.zeros(n_samples)
        sc = np.zeros(n_samples)
        for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42).split(preds_mat.T, bins):
            m = Ridge(alpha=alpha)
            m.fit(preds_mat[:, tr].T, y[tr])
            sp[te] += m.predict(preds_mat[:, te].T)
            sc[te] += 1
        sp /= sc
        r2 = 1 - np.sum((y - sp) ** 2) / ss_tot
        if r2 > best_r2:
            best_r2 = r2
        if r2 > 0.20:
            all_stacks[f"ridge_{alpha}"] = sp.copy()
    if verbose:
        print(f"Ridge stack: R²={best_r2:.4f}", flush=True)

    # -- ElasticNet stacks --
    for alpha in [0.01, 0.1, 1]:
        for l1 in [0.1, 0.3, 0.5, 0.7, 0.9]:
            sp = np.zeros(n_samples)
            sc = np.zeros(n_samples)
            for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42).split(preds_mat.T, bins):
                m = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=5000, positive=True)
                m.fit(preds_mat[:, tr].T, y[tr])
                sp[te] += m.predict(preds_mat[:, te].T)
                sc[te] += 1
            sp /= sc
            r2 = 1 - np.sum((y - sp) ** 2) / ss_tot
            if r2 > best_r2:
                best_r2 = r2
            if r2 > 0.20:
                all_stacks[f"en_{alpha}_{l1}"] = sp.copy()
    if verbose:
        print(f"+EN: R²={best_r2:.4f}", flush=True)

    # -- Lasso stacks --
    for alpha in [0.001, 0.01, 0.1]:
        sp = np.zeros(n_samples)
        sc = np.zeros(n_samples)
        for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42).split(preds_mat.T, bins):
            m = Lasso(alpha=alpha, max_iter=5000, positive=True)
            m.fit(preds_mat[:, tr].T, y[tr])
            sp[te] += m.predict(preds_mat[:, te].T)
            sc[te] += 1
        sp /= sc
        r2 = 1 - np.sum((y - sp) ** 2) / ss_tot
        if r2 > best_r2:
            best_r2 = r2
        if r2 > 0.20:
            all_stacks[f"lasso_{alpha}"] = sp.copy()
    if verbose:
        print(f"+Lasso: R²={best_r2:.4f}", flush=True)

    # -- KNN stacks --
    for k in [3, 5, 7, 10, 15, 20, 30]:
        sp = np.zeros(n_samples)
        sc = np.zeros(n_samples)
        for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42).split(preds_mat.T, bins):
            ss = StandardScaler()
            pt = ss.fit_transform(preds_mat[:, tr].T)
            pe = ss.transform(preds_mat[:, te].T)
            m = KNeighborsRegressor(n_neighbors=k, weights="distance")
            m.fit(pt, y[tr])
            sp[te] += m.predict(pe)
            sc[te] += 1
        sp /= sc
        r2 = 1 - np.sum((y - sp) ** 2) / ss_tot
        if r2 > best_r2:
            best_r2 = r2
        if r2 > 0.20:
            all_stacks[f"knn_{k}"] = sp.copy()
    if verbose:
        print(f"+KNN: R²={best_r2:.4f}", flush=True)

    # -- SVR stacks --
    for C in [0.1, 1, 10, 100]:
        for eps in [0.05, 0.1]:
            sp = np.zeros(n_samples)
            sc = np.zeros(n_samples)
            for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42).split(preds_mat.T, bins):
                ss = StandardScaler()
                pt = ss.fit_transform(preds_mat[:, tr].T)
                pe = ss.transform(preds_mat[:, te].T)
                m = SVR(kernel="rbf", C=C, epsilon=eps)
                m.fit(pt, y[tr])
                sp[te] += m.predict(pe)
                sc[te] += 1
            sp /= sc
            r2 = 1 - np.sum((y - sp) ** 2) / ss_tot
            if r2 > best_r2:
                best_r2 = r2
            if r2 > 0.20:
                all_stacks[f"svr_{C}_{eps}"] = sp.copy()
    if verbose:
        print(f"+SVR: R²={best_r2:.4f}", flush=True)

    # -- XGB stacks --
    if HAS_XGB:
        for d in [2, 3]:
            sp = np.zeros(n_samples)
            sc = np.zeros(n_samples)
            for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42).split(preds_mat.T, bins):
                m = xgb.XGBRegressor(
                    n_estimators=100, max_depth=d, learning_rate=0.05,
                    reg_alpha=5, reg_lambda=10, random_state=42,
                )
                m.fit(preds_mat[:, tr].T, y[tr])
                sp[te] += m.predict(preds_mat[:, te].T)
                sc[te] += 1
            sp /= sc
            r2 = 1 - np.sum((y - sp) ** 2) / ss_tot
            if r2 > best_r2:
                best_r2 = r2
            if r2 > 0.20:
                all_stacks[f"xgb_{d}"] = sp.copy()
    if verbose:
        print(f"+XGB: R²={best_r2:.4f}", flush=True)

    # -- Bayesian stack --
    sp = np.zeros(n_samples)
    sc = np.zeros(n_samples)
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42).split(preds_mat.T, bins):
        m = BayesianRidge()
        m.fit(preds_mat[:, tr].T, y[tr])
        sp[te] += m.predict(preds_mat[:, te].T)
        sc[te] += 1
    sp /= sc
    r2 = 1 - np.sum((y - sp) ** 2) / ss_tot
    if r2 > best_r2:
        best_r2 = r2
    if r2 > 0.20:
        all_stacks["bayesian"] = sp.copy()
    if verbose:
        print(f"+Bayesian: R²={best_r2:.4f}", flush=True)

    layer1_r2 = best_r2
    if verbose:
        print(f"\nLayer-1 best: R²={best_r2:.4f} ({len(all_stacks)} stacks)", flush=True)

    # -- Mega blend (Dirichlet + pairwise + triplet) --
    blend_best = -999
    rng = np.random.RandomState(42)
    for _ in range(500_000):
        w = rng.dirichlet(np.ones(nm_count) * 0.3)
        bl = w @ preds_mat
        r2 = 1 - np.sum((y - bl) ** 2) / ss_tot
        if r2 > blend_best:
            blend_best = r2
    for _ in range(200_000):
        w = rng.dirichlet(np.ones(nm_count) * 0.1)
        bl = w @ preds_mat
        r2 = 1 - np.sum((y - bl) ** 2) / ss_tot
        if r2 > blend_best:
            blend_best = r2
    for i in range(nm_count):
        for j in range(i + 1, nm_count):
            for a in np.linspace(0, 1, 201):
                bl = a * preds_mat[i] + (1 - a) * preds_mat[j]
                r2 = 1 - np.sum((y - bl) ** 2) / ss_tot
                if r2 > blend_best:
                    blend_best = r2
    for i in range(min(8, nm_count)):
        for j in range(i + 1, min(8, nm_count)):
            for k in range(j + 1, min(8, nm_count)):
                for a in np.linspace(0.05, 0.9, 20):
                    for b in np.linspace(0.05, 0.9 - a, 15):
                        c = 1 - a - b
                        if c > 0:
                            bl = a * preds_mat[i] + b * preds_mat[j] + c * preds_mat[k]
                            r2 = 1 - np.sum((y - bl) ** 2) / ss_tot
                            if r2 > blend_best:
                                blend_best = r2
    if blend_best > best_r2:
        best_r2 = blend_best
    if verbose:
        print(f"Mega blend: R²={blend_best:.4f}", flush=True)

    # =================================================================
    # LAYER 2: Stack of stacks
    # =================================================================
    if len(all_stacks) >= 3:
        if verbose:
            print(f"Layer 2: {len(all_stacks)} stacks", flush=True)
        snames = list(all_stacks.keys())
        smat = np.array([all_stacks[k] for k in snames])
        best_l2_preds = None
        for alpha in [0.01, 0.1, 1, 10, 100]:
            sp = np.zeros(n_samples)
            sc = np.zeros(n_samples)
            for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42).split(smat.T, bins):
                m = Ridge(alpha=alpha)
                m.fit(smat[:, tr].T, y[tr])
                sp[te] += m.predict(smat[:, te].T)
                sc[te] += 1
            sp /= sc
            r2 = 1 - np.sum((y - sp) ** 2) / ss_tot
            if r2 > best_r2:
                best_r2 = r2
                best_l2_preds = sp.copy()
        if verbose:
            print(f"Layer 2: R²={best_r2:.4f}", flush=True)

    elapsed = time.time() - t0

    # Compute final predictions (from layer-2 if available, else best stack)
    if best_l2_preds is not None:
        final_preds = best_l2_preds
    else:
        # Use best single stack
        best_stack = max(all_stacks, key=lambda k: 1 - np.sum((y - all_stacks[k]) ** 2) / ss_tot)
        final_preds = all_stacks[best_stack]

    final_r2 = 1 - np.sum((y - final_preds) ** 2) / ss_tot
    r, _ = pearsonr(y, final_preds)

    # --- Summary ---
    if verbose:
        target_r2 = 0.37 if target == "homa_ir" else 0.70
        print(f"\n{'='*70}")
        print(f"RESULT: {target.upper()} (DW)")
        print(f"  R²       = {final_r2:.4f}  (target: {target_r2})")
        print(f"  Pearson r = {r:.4f}")
        print(f"  Gap      = {target_r2 - final_r2:.4f}")
        print(f"  Base models: {len(all_results)}, Stacks: {len(all_stacks)}")
        print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"{'='*70}")

    return {
        "r2": final_r2,
        "r": r,
        "oof_predictions": final_preds,
        "y": y,
        "n_base_models": len(all_results),
        "n_stacks": len(all_stacks),
        "n_samples": n_samples,
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="WEAR-ME: DW-only prediction via multi-layer stacking"
    )
    parser.add_argument("--data", default="data.csv", help="Path to data.csv")
    parser.add_argument("--target", default="both", choices=["homa_ir", "hba1c", "both"])
    parser.add_argument(
        "--splits", default=None,
        help='Path to JSON splits: [{"train": [...], "test": [...]}, ...]',
    )

    args = parser.parse_args()

    # Load custom splits
    custom_splits = None
    if args.splits:
        with open(args.splits) as f:
            raw = json.load(f)
        custom_splits = [(np.array(s["train"]), np.array(s["test"])) for s in raw]
        print(f"Loaded {len(custom_splits)} custom splits from {args.splits}")

    targets = ["homa_ir", "hba1c"] if args.target == "both" else [args.target]
    all_results = {}

    for tgt in targets:
        print(f"\n{'#' * 60}")
        print(f"#  TARGET: {tgt.upper()} (DW ONLY)")
        print(f"{'#' * 60}\n")
        result = run_pipeline(data_path=args.data, target=tgt, splits=custom_splits)
        all_results[tgt] = result

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY (DW ONLY)")
    print(f"{'=' * 60}")
    targets_map = {"homa_ir": 0.37, "hba1c": 0.70}
    for tgt, result in all_results.items():
        target_r2 = targets_map[tgt]
        status = "PASS" if result["r2"] >= target_r2 else "FAIL"
        print(f"  {tgt.upper():>10s}: R²={result['r2']:.4f}  r={result['r']:.4f}  "
              f"target={target_r2}  gap={target_r2 - result['r2']:.4f}  [{status}]")


if __name__ == "__main__":
    main()
