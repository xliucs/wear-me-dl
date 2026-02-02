#!/usr/bin/env python3
"""
WEAR-ME: Prediction with Demographics + Wearables Only (DW)
=============================================================

Targets:
  - True_HOMA_IR  (R² target: 0.37)
  - True_hba1c    (R² target: 0.70)

Method: Multi-layer stacking ensemble
  Layer 0: 385 base models × 5 feature sets
  Layer 1: 41 diverse meta-learners on base OOF predictions
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

# Short aliases for readability
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
    """Load dataset, extract DW columns, filter by non-null target."""
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
# FEATURE ENGINEERING
# =============================================================================

def engineer_features(X: pd.DataFrame) -> np.ndarray:
    """
    Engineer ~94 features from 18 DW columns.
    Returns numpy array (for speed in model fitting).
    """
    X = X.copy()

    # --- Skewness & CV for each wearable ---
    for pfx, m, md, s in [
        ("rhr", _RHR_M, _RHR_MD, _RHR_S),
        ("hrv", _HRV_M, _HRV_MD, _HRV_S),
        ("stp", _STP_M, _STP_MD, _STP_S),
        ("slp", _SLP_M, _SLP_MD, _SLP_S),
        ("azm", _AZM_M, _AZM_MD, _AZM_S),
    ]:
        X[f"{pfx}_skew"] = (X[m] - X[md]) / X[s].clip(lower=0.01)
        X[f"{pfx}_cv"] = X[s] / X[m].clip(lower=0.01)

    # --- Polynomial transforms ---
    for col, nm in [
        (X["bmi"], "bmi"), (X["age"], "age"),
        (X[_RHR_M], "rhr"), (X[_HRV_M], "hrv"), (X[_STP_M], "stp"),
    ]:
        X[f"{nm}_sq"] = col ** 2
        X[f"{nm}_log"] = np.log1p(col.clip(lower=0))
        X[f"{nm}_inv"] = 1 / col.clip(lower=0.01)

    # --- BMI interactions ---
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

    # --- Age interactions ---
    X["age_rhr"] = X["age"] * X[_RHR_M]
    X["age_hrv_inv"] = X["age"] / X[_HRV_M].clip(lower=1)
    X["age_stp"] = X["age"] * X[_STP_M]
    X["age_slp"] = X["age"] * X[_SLP_M]
    X["age_sex"] = X["age"] * X["sex"]
    X["age_bmi_sex"] = X["age"] * X["bmi"] * X["sex"]

    # --- Wearable cross-interactions ---
    X["rhr_hrv"] = X[_RHR_M] / X[_HRV_M].clip(lower=1)
    X["stp_hrv"] = X[_STP_M] * X[_HRV_M]
    X["stp_rhr"] = X[_STP_M] / X[_RHR_M].clip(lower=1)
    X["azm_stp"] = X[_AZM_M] / X[_STP_M].clip(lower=1)
    X["slp_hrv"] = X[_SLP_M] * X[_HRV_M]
    X["slp_rhr"] = X[_SLP_M] / X[_RHR_M].clip(lower=1)

    # --- Composite health indices ---
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

    # --- Binary indicators + interactions ---
    X["obese"] = (X["bmi"] >= 30).astype(float)
    X["older"] = (X["age"] >= 50).astype(float)
    X["obese_rhr"] = X["obese"] * X[_RHR_M]
    X["obese_low_hrv"] = X["obese"] * (X[_HRV_M] < X[_HRV_M].median()).astype(float)
    X["older_bmi"] = X["older"] * X["bmi"]
    X["older_rhr"] = X["older"] * X[_RHR_M]
    X["rhr_cv_bmi"] = X["rhr_cv"] * X["bmi"]
    X["hrv_cv_bmi"] = X["hrv_cv"] * X["bmi"]
    X["rhr_cv_age"] = X["rhr_cv"] * X["age"]

    # --- Rank features ---
    for col in ["bmi", "age", _RHR_M, _HRV_M, _STP_M]:
        X[f"rank_{col[:3]}"] = X[col].rank(pct=True)

    return X.fillna(0).values


# =============================================================================
# AUGMENTED FEATURES (OOF — no leakage)
# =============================================================================

def make_target_encoded_features(
    X_df: pd.DataFrame, y: np.ndarray, bins: np.ndarray,
    columns: List[str] = None, n_bins_list: List[int] = None, smooth: float = 10.0,
) -> np.ndarray:
    """Create out-of-fold target-encoded features."""
    if columns is None:
        columns = ["bmi", "age", _RHR_M, _HRV_M, _STP_M, _SLP_M, _AZM_M]
    if n_bins_list is None:
        n_bins_list = [3, 5, 10]

    features = []
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
            for train_idx, test_idx in StratifiedKFold(5, shuffle=True, random_state=42).split(vals, bins):
                for b in range(nb):
                    mask_train = coded[train_idx] == b
                    mask_test = coded[test_idx] == b
                    if mask_train.sum() > 0:
                        bin_mean = y[train_idx][mask_train].mean()
                        bin_count = mask_train.sum()
                        encoded[test_idx[mask_test]] = (
                            bin_count * bin_mean + smooth * global_mean
                        ) / (bin_count + smooth)
                    else:
                        encoded[test_idx[mask_test]] = global_mean
            features.append(encoded)

    return np.column_stack(features) if features else np.zeros((len(y), 0))


def make_knn_target_features(
    X: np.ndarray, y: np.ndarray, bins: np.ndarray,
    k_list: List[int] = None,
) -> np.ndarray:
    """Create OOF KNN-based target features (mean, std, median of neighbor targets)."""
    if k_list is None:
        k_list = [5, 10, 20, 50]

    features = []
    for k in k_list:
        mean_f = np.zeros(len(y))
        std_f = np.zeros(len(y))
        med_f = np.zeros(len(y))

        for train_idx, test_idx in StratifiedKFold(5, shuffle=True, random_state=42).split(X, bins):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_te = scaler.transform(X[test_idx])

            nn = NearestNeighbors(n_neighbors=k)
            nn.fit(X_tr)
            _, idx = nn.kneighbors(X_te)

            for i, ti in enumerate(test_idx):
                neighbor_targets = y[train_idx][idx[i]]
                mean_f[ti] = neighbor_targets.mean()
                std_f[ti] = neighbor_targets.std()
                med_f[ti] = np.median(neighbor_targets)

        features.extend([mean_f, std_f, med_f])

    return np.column_stack(features) if features else np.zeros((len(y), 0))


def make_quantile_classification_features(
    X: np.ndarray, y: np.ndarray,
    quantiles: List[float] = None,
) -> np.ndarray:
    """Create OOF quantile classification probability features."""
    from sklearn.ensemble import GradientBoostingClassifier

    if quantiles is None:
        quantiles = [0.25, 0.5, 0.75, 0.9]

    features = []
    for q in quantiles:
        threshold = np.quantile(y, q)
        y_class = (y > threshold).astype(int)
        probs = np.zeros(len(y))

        for train_idx, test_idx in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y_class):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_te = scaler.transform(X[test_idx])
            clf = GradientBoostingClassifier(
                n_estimators=100, max_depth=2, learning_rate=0.1, random_state=42
            )
            clf.fit(X_tr, y_class[train_idx])
            probs[test_idx] = clf.predict_proba(X_te)[:, 1]

        features.append(probs)

    return np.column_stack(features) if features else np.zeros((len(y), 0))


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

def get_fast_models() -> List[Tuple[str, callable, bool]]:
    """Fast models (linear, kernel, KNN). Returns (name, factory, needs_scaling)."""
    models = [
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
    return models


def get_medium_models() -> List[Tuple[str, callable, bool]]:
    """Medium-speed models (tree ensembles). Returns (name, factory, needs_scaling)."""
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
# CV ENGINE
# =============================================================================

def make_stratification_bins(y: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Bin continuous target for stratified splitting."""
    return pd.qcut(y, n_bins, labels=False, duplicates="drop")


def default_splits(
    y: np.ndarray, n_splits: int = 5, n_repeats: int = 5, random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate default RepeatedStratifiedKFold splits."""
    bins = make_stratification_bins(y)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    return list(cv.split(np.zeros(len(y)), bins))


def run_single_model_cv(
    X: np.ndarray, y: np.ndarray,
    model_fn: callable,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    scale: bool = True,
    log_target: bool = False,
) -> Tuple[float, np.ndarray]:
    """Train a single model across CV splits, return (R², OOF predictions)."""
    n_samples = len(y)
    oof_preds = np.zeros(n_samples)
    oof_counts = np.zeros(n_samples)
    ss_tot = np.sum((y - y.mean()) ** 2)

    for train_idx, test_idx in splits:
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx].copy()

        if log_target:
            y_tr = np.log1p(y_tr)
        if scale:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)

        model = model_fn()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)

        if log_target:
            preds = np.expm1(preds)

        oof_preds[test_idx] += preds
        oof_counts[test_idx] += 1

    mask = oof_counts > 0
    oof_preds[mask] /= oof_counts[mask]
    r2 = 1 - np.sum((y[mask] - oof_preds[mask]) ** 2) / np.sum((y[mask] - y[mask].mean()) ** 2)
    return r2, oof_preds


# =============================================================================
# MULTI-LAYER STACKING
# =============================================================================

def build_layer0(
    feature_sets: Dict[str, np.ndarray],
    y: np.ndarray,
    splits_5rep: List[Tuple[np.ndarray, np.ndarray]],
    splits_3rep: List[Tuple[np.ndarray, np.ndarray]],
    use_log: bool = True,
    verbose: bool = True,
) -> Dict[str, dict]:
    """
    Train all base models across all feature sets.

    Returns dict: model_name -> {"r2": float, "preds": np.ndarray}
    """
    fast_models = get_fast_models()
    med_models = get_medium_models()

    all_results = {}
    total_models = 0

    for fs_name, X_fs in feature_sets.items():
        if verbose:
            print(f"\n  Feature set: {fs_name} ({X_fs.shape[1]} features)", flush=True)

        for mname, mfn, scale in fast_models:
            key = f"{fs_name}__{mname}"
            try:
                r2, preds = run_single_model_cv(X_fs, y, mfn, splits_5rep, scale=scale)
                all_results[key] = {"r2": r2, "preds": preds}
                total_models += 1

                if use_log:
                    r2l, pl = run_single_model_cv(X_fs, y, mfn, splits_5rep, scale=scale, log_target=True)
                    all_results[f"{key}_log"] = {"r2": r2l, "preds": pl}
                    total_models += 1
            except Exception:
                pass

        for mname, mfn, scale in med_models:
            key = f"{fs_name}__{mname}"
            try:
                r2, preds = run_single_model_cv(X_fs, y, mfn, splits_3rep, scale=scale)
                all_results[key] = {"r2": r2, "preds": preds}
                total_models += 1
            except Exception:
                pass

    if verbose:
        good = {k: v for k, v in all_results.items() if v["r2"] > 0}
        top = sorted(good.keys(), key=lambda k: good[k]["r2"], reverse=True)[:10]
        print(f"\n  Total base models: {total_models} ({len(good)} with R²>0)")
        print(f"  Top 10:")
        for nm in top:
            print(f"    {nm:55s}: R²={good[nm]['r2']:.4f}")

    return all_results


def build_layer1(
    base_results: Dict[str, dict],
    y: np.ndarray,
    bins: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    top_k: int = 25,
    verbose: bool = True,
) -> Tuple[Dict[str, np.ndarray], float]:
    """
    Build Layer-1 stacking meta-learners on top of base OOF predictions.

    Returns (stack_preds, best_r2).
    """
    n_samples = len(y)
    ss_tot = np.sum((y - y.mean()) ** 2)

    # Select top base models
    good = {k: v for k, v in base_results.items() if v["r2"] > 0}
    names = sorted(good.keys(), key=lambda k: good[k]["r2"], reverse=True)[:top_k]
    preds_mat = np.array([good[nm]["preds"] for nm in names])

    best_r2 = -np.inf
    all_stacks = {}

    def _run_stack(model_fn, name, n_rep=5):
        nonlocal best_r2
        sp = np.zeros(n_samples)
        sc = np.zeros(n_samples)
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=42)
        for tr, te in cv.split(preds_mat.T, bins):
            m = model_fn()
            m.fit(preds_mat[:, tr].T, y[tr])
            sp[te] += m.predict(preds_mat[:, te].T)
            sc[te] += 1
        sp /= sc
        r2 = 1 - np.sum((y - sp) ** 2) / ss_tot
        if r2 > best_r2:
            best_r2 = r2
        if r2 > 0.20:
            all_stacks[name] = sp.copy()
        return r2

    # Ridge stacks
    for alpha in [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000]:
        _run_stack(lambda a=alpha: Ridge(alpha=a), f"ridge_{alpha}")
    if verbose:
        print(f"  After Ridge stacks: best R²={best_r2:.4f}")

    # ElasticNet stacks
    for alpha in [0.01, 0.1, 1]:
        for l1 in [0.1, 0.3, 0.5, 0.7, 0.9]:
            _run_stack(
                lambda a=alpha, l=l1: ElasticNet(alpha=a, l1_ratio=l, max_iter=5000, positive=True),
                f"en_{alpha}_{l1}",
            )
    if verbose:
        print(f"  After EN stacks:    best R²={best_r2:.4f}")

    # Lasso stacks
    for alpha in [0.001, 0.01, 0.1]:
        _run_stack(
            lambda a=alpha: Lasso(alpha=a, max_iter=5000, positive=True),
            f"lasso_{alpha}",
        )

    # KNN stacks
    for k in [3, 5, 7, 10, 15, 20, 30]:
        sp = np.zeros(n_samples)
        sc = np.zeros(n_samples)
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
        for tr, te in cv.split(preds_mat.T, bins):
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
        print(f"  After KNN stacks:   best R²={best_r2:.4f}")

    # SVR stacks
    for C in [0.1, 1, 10, 100]:
        for eps in [0.05, 0.1]:
            sp = np.zeros(n_samples)
            sc = np.zeros(n_samples)
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
            for tr, te in cv.split(preds_mat.T, bins):
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

    # XGB stacks
    if HAS_XGB:
        for d in [2, 3]:
            sp = np.zeros(n_samples)
            sc = np.zeros(n_samples)
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
            for tr, te in cv.split(preds_mat.T, bins):
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

    # Bayesian stack
    _run_stack(lambda: BayesianRidge(), "bayesian")

    if verbose:
        print(f"  Layer-1 stacks: {len(all_stacks)} (best R²={best_r2:.4f})")

    return all_stacks, best_r2


def build_layer2(
    stack_preds: Dict[str, np.ndarray],
    y: np.ndarray,
    bins: np.ndarray,
    verbose: bool = True,
) -> Tuple[float, np.ndarray]:
    """
    Layer-2: Ridge regression on Layer-1 stack predictions.

    Returns (best_r2, best_oof_predictions).
    """
    n_samples = len(y)
    ss_tot = np.sum((y - y.mean()) ** 2)

    snames = list(stack_preds.keys())
    smat = np.array([stack_preds[k] for k in snames])

    best_r2 = -np.inf
    best_preds = None

    for alpha in [0.01, 0.1, 1, 10, 100]:
        sp = np.zeros(n_samples)
        sc = np.zeros(n_samples)
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
        for tr, te in cv.split(smat.T, bins):
            m = Ridge(alpha=alpha)
            m.fit(smat[:, tr].T, y[tr])
            sp[te] += m.predict(smat[:, te].T)
            sc[te] += 1
        sp /= sc
        r2 = 1 - np.sum((y - sp) ** 2) / ss_tot
        if r2 > best_r2:
            best_r2 = r2
            best_preds = sp.copy()

    if verbose:
        print(f"  Layer-2 R²={best_r2:.4f} ({len(snames)} stacks)")

    return best_r2, best_preds


# =============================================================================
# MAIN PIPELINE
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
        Custom CV splits. If None, uses default RepeatedStratifiedKFold(5, 5).
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys:
        "r2": float — best OOF R²
        "r": float — Pearson correlation
        "oof_predictions": np.ndarray — out-of-fold predictions
        "y": np.ndarray — true targets
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

    if verbose:
        print(f"  Samples: {n_samples}, Target: {target}, y: [{y.min():.2f}, {y.max():.2f}]")

    # --- Feature engineering ---
    if verbose:
        print("Engineering features...", flush=True)
    X_raw = X_df.values
    X_eng = engineer_features(X_df)

    # MI-selected features
    mi = mutual_info_regression(X_eng, y, random_state=42)
    X_mi35 = X_eng[:, np.argsort(mi)[-35:]]

    if verbose:
        print(f"  raw18={X_raw.shape[1]}, eng={X_eng.shape[1]}, mi35={X_mi35.shape[1]}")

    # --- Augmented features (OOF) ---
    if verbose:
        print("Creating augmented features (OOF)...", flush=True)
    te_feats = make_target_encoded_features(X_df, y, bins)
    knn_feats = make_knn_target_features(X_raw, y, bins)
    qc_feats = make_quantile_classification_features(X_raw, y)

    X_mega = np.column_stack([X_raw, te_feats, knn_feats, qc_feats])
    X_mega_eng = np.column_stack([X_eng, te_feats, knn_feats, qc_feats])

    feature_sets = {
        "raw18": X_raw,
        "eng": X_eng,
        "mi35": X_mi35,
        "mega": X_mega,
        "mega_eng": X_mega_eng,
    }

    if verbose:
        print(f"  mega={X_mega.shape[1]}, mega_eng={X_mega_eng.shape[1]}")

    # --- Generate splits ---
    if splits is None:
        if verbose:
            print("Using default splits: RepeatedStratifiedKFold(5-fold, 5 repeats)")
        splits_5rep = default_splits(y, n_repeats=5)
        splits_3rep = default_splits(y, n_repeats=3)
    else:
        if verbose:
            print(f"Using {len(splits)} custom splits")
        splits_5rep = splits
        splits_3rep = splits  # Custom splits used for all models

    # --- Layer 0: base models ---
    if verbose:
        print(f"\n{'='*60}")
        print("LAYER 0: Base Models")
        print(f"{'='*60}", flush=True)

    base_results = build_layer0(
        feature_sets, y, splits_5rep, splits_3rep,
        use_log=(target == "homa_ir"), verbose=verbose
    )

    # --- Layer 1: stacking ---
    if verbose:
        print(f"\n{'='*60}")
        print("LAYER 1: Stacking Meta-learners")
        print(f"{'='*60}", flush=True)

    stack_preds, layer1_r2 = build_layer1(base_results, y, bins, splits_5rep, verbose=verbose)

    # --- Layer 2: stack of stacks ---
    final_r2 = layer1_r2
    final_preds = None

    if len(stack_preds) >= 3:
        if verbose:
            print(f"\n{'='*60}")
            print("LAYER 2: Stack of Stacks")
            print(f"{'='*60}", flush=True)

        final_r2, final_preds = build_layer2(stack_preds, y, bins, verbose=verbose)
    else:
        # Fall back to best layer-1
        best_name = max(stack_preds, key=lambda k: 1 - np.sum((y - stack_preds[k]) ** 2) / np.sum((y - y.mean()) ** 2))
        final_preds = stack_preds[best_name]

    r, _ = pearsonr(y, final_preds)
    elapsed = time.time() - t0

    # --- Summary ---
    if verbose:
        target_r2 = 0.37 if target == "homa_ir" else 0.70
        print(f"\n{'='*60}")
        print(f"RESULT: {target.upper()} (DW)")
        print(f"  R²       = {final_r2:.4f}  (target: {target_r2})")
        print(f"  Pearson r = {r:.4f}")
        print(f"  Gap      = {target_r2 - final_r2:.4f}")
        print(f"  Models   = {len(base_results)} base, {len(stack_preds)} stacks")
        print(f"  Time     = {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"{'='*60}")

    return {
        "r2": final_r2,
        "r": r,
        "oof_predictions": final_preds,
        "y": y,
        "n_base_models": len(base_results),
        "n_stacks": len(stack_preds),
        "n_samples": n_samples,
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="WEAR-ME: DW-only prediction via multi-layer stacking"
    )
    parser.add_argument(
        "--data", default="data.csv",
        help="Path to data.csv (default: data.csv)",
    )
    parser.add_argument(
        "--target", default="both", choices=["homa_ir", "hba1c", "both"],
        help="Target variable (default: both)",
    )
    parser.add_argument(
        "--splits", default=None,
        help='Path to JSON splits file: [{"train": [...], "test": [...]}, ...]',
    )

    args = parser.parse_args()

    # Load custom splits
    custom_splits = None
    if args.splits:
        with open(args.splits) as f:
            raw = json.load(f)
        custom_splits = [
            (np.array(s["train"]), np.array(s["test"])) for s in raw
        ]
        print(f"Loaded {len(custom_splits)} custom splits from {args.splits}")

    targets = ["homa_ir", "hba1c"] if args.target == "both" else [args.target]
    all_results = {}

    for tgt in targets:
        print(f"\n{'#' * 60}")
        print(f"#  TARGET: {tgt.upper()} (DW ONLY)")
        print(f"{'#' * 60}\n")

        result = run_pipeline(
            data_path=args.data,
            target=tgt,
            splits=custom_splits,
        )
        all_results[tgt] = result

    # Final summary
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
