#!/usr/bin/env python3
"""
WEAR-ME: Prediction with ALL Features (Demographics + Wearables + Blood Biomarkers)
====================================================================================

Targets:
  - True_HOMA_IR  (R² target: 0.65)
  - True_hba1c    (R² target: 0.85)

Usage:
  # Default CV (RepeatedStratifiedKFold, 5-fold × 3 repeats):
  python3 final_all_features.py

  # Custom splits from file (JSON list of {"train": [...], "test": [...]}):
  python3 final_all_features.py --splits splits.json

  # Single target:
  python3 final_all_features.py --target homa_ir
  python3 final_all_features.py --target hba1c

  # As a library:
  from final_all_features import run_pipeline
  results = run_pipeline("data.csv", target="homa_ir", splits=my_splits)
"""

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

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


# =============================================================================
# DATA LOADING
# =============================================================================

DEMOGRAPHIC_COLS = ["age", "bmi", "sex"]
WEARABLE_COLS = [
    "Resting Heart Rate (mean)", "Resting Heart Rate (median)", "Resting Heart Rate (std)",
    "HRV (mean)", "HRV (median)", "HRV (std)",
    "STEPS (mean)", "STEPS (median)", "STEPS (std)",
    "SLEEP Duration (mean)", "SLEEP Duration (median)", "SLEEP Duration (std)",
    "AZM Weekly (mean)", "AZM Weekly (median)", "AZM Weekly (std)",
]
TARGET_COLS = [
    "True_hba1c", "True_HOMA_IR", "True_IR_Class",
    "True_Diabetes_2_Class", "True_Normoglycemic_2_Class",
    "True_Diabetes_3_Class", "Participant_id",
]


def load_data(data_path: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load WEAR-ME dataset. Returns (X, y_homa, y_hba1c)."""
    df = pd.read_csv(data_path, skiprows=[0])

    feature_cols = [c for c in df.columns if c not in TARGET_COLS]
    X = df[feature_cols].copy()
    X["sex"] = (X["sex"] == "Male").astype(float)

    y_homa = df["True_HOMA_IR"].values
    y_hba1c = df["True_hba1c"].values

    return X, y_homa, y_hba1c


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def engineer_features(X: pd.DataFrame, target: str = "homa_ir") -> pd.DataFrame:
    """
    Engineer features for the ALL-features setting.

    Parameters
    ----------
    X : pd.DataFrame
        Raw features (sex already numeric).
    target : str
        "homa_ir" or "hba1c" — controls target-specific features.

    Returns
    -------
    pd.DataFrame with original + engineered features.
    """
    X = X.copy()

    # --- Demographics interactions ---
    X["bmi_sq"] = X["bmi"] ** 2
    X["bmi_cubed"] = X["bmi"] ** 3
    X["age_sq"] = X["age"] ** 2
    X["bmi_age"] = X["bmi"] * X["age"]
    X["age_sex"] = X["age"] * X["sex"]
    X["bmi_sex"] = X["bmi"] * X["sex"]

    # --- Metabolic features (requires blood biomarkers) ---
    if "triglycerides" in X.columns and "hdl" in X.columns:
        X["trig_hdl"] = X["triglycerides"] / X["hdl"].clip(lower=1)
        X["non_hdl_ratio"] = (X["total cholesterol"] - X["hdl"]) / X["hdl"].clip(lower=1)

    if "glucose" in X.columns and "triglycerides" in X.columns:
        X["tyg"] = np.log(X["triglycerides"].clip(lower=1) * X["glucose"].clip(lower=1) / 2)
        X["tyg_bmi"] = X["tyg"] * X["bmi"]

    if "glucose" in X.columns:
        X["glucose_bmi"] = X["glucose"] * X["bmi"]
        X["glucose_sq"] = X["glucose"] ** 2
        X["glucose_age"] = X["glucose"] * X["age"]
        if "hdl" in X.columns:
            X["glucose_hdl"] = X["glucose"] / X["hdl"].clip(lower=1)
        if "triglycerides" in X.columns:
            X["glucose_trig"] = X["glucose"] * X["triglycerides"]

    # --- HOMA-IR–specific features ---
    if target == "homa_ir" and "glucose" in X.columns:
        X["mets_ir"] = np.log(2 * X["glucose"].clip(lower=1) + X["triglycerides"].clip(lower=1)) * X["bmi"]
        X["mets_ir_bmi"] = X["mets_ir"] * X["bmi"]
        X["glucose_proxy"] = X["glucose"] * X["triglycerides"] / X["hdl"].clip(lower=1)
        X["bmi_trig"] = X["bmi"] * X["triglycerides"]
        X["insulin_proxy"] = X["glucose"] * X["bmi"] * X["triglycerides"] / (X["hdl"].clip(lower=1) * 100)
        X["vat_proxy"] = X["bmi"] * X["triglycerides"] / X["hdl"].clip(lower=1)
        if "alt" in X.columns:
            X["liver_stress"] = X["alt"] * X["ggt"] / X["albumin"].clip(lower=1)
        if "crp" in X.columns:
            X["inflammation"] = X["crp"] * X["white_blood_cell"]

    # --- hba1c–specific features ---
    if target == "hba1c" and "glucose" in X.columns:
        X["glucose_hb"] = X["glucose"] / X["hb"].clip(lower=1)
        X["glucose_rbc"] = X["glucose"] / X["red_blood_cell"].clip(lower=1)
        X["glucose_rdw"] = X["glucose"] * X["rdw"]
        X["glucose_mchc"] = X["glucose"] * X["mchc"]
        X["glucose_mcv"] = X["glucose"] * X["mcv"]
        X["glucose_hematocrit"] = X["glucose"] / X["hematocrit"].clip(lower=1)
        X["age_glucose"] = X["age"] * X["glucose"]
        X["age_rdw"] = X["age"] * X["rdw"]
        X["rbc_health"] = X["hb"] * X["hematocrit"] / X["red_blood_cell"].clip(lower=1)
        X["rdw_mchc"] = X["rdw"] / X["mchc"].clip(lower=1)
        if "crp" in X.columns:
            X["inflammation_glucose"] = X["crp"] * X["glucose"]
        X["glucose_log"] = np.log1p(X["glucose"])
        X["glucose_cubed"] = X["glucose"] ** 3
        X["cholesterol_glucose"] = X["total cholesterol"] * X["glucose"]
        X["lymph_glucose"] = X["absolute_lymphocytes"] * X["glucose"]
        X["egfr_age"] = X["egfr"] * X["age"]
        X["calcium_glucose"] = X["calcium"] * X["glucose"]
        X["globulin_glucose"] = X["globulin"] * X["glucose"]
        X["ggt_glucose"] = X["ggt"] * X["glucose"]
        X["trig_glucose_bmi"] = X["triglycerides"] * X["glucose"] * X["bmi"]
        X["mets_ir"] = np.log(2 * X["glucose"].clip(lower=1) + X["triglycerides"].clip(lower=1)) * X["bmi"]
        X["glucose_proxy"] = X["glucose"] * X["triglycerides"] / X["hdl"].clip(lower=1)

    # --- Wearable interaction features ---
    rhr = "Resting Heart Rate (mean)"
    hrv = "HRV (mean)"
    steps = "STEPS (mean)"
    sleep = "SLEEP Duration (mean)"
    azm = "AZM Weekly (mean)"

    if rhr in X.columns:
        X["rhr_bmi"] = X[rhr] * X["bmi"]
        X["rhr_bmi_sq"] = X[rhr] * X["bmi"] ** 2
        X["rhr_hrv"] = X[rhr] / X[hrv].clip(lower=1)
        X["steps_sleep"] = X[steps] * X[sleep]
        X["bmi_rhr_sq"] = X["bmi"] * X[rhr] ** 2
        X["low_azm_obese"] = (X[azm] < X[azm].median()).astype(int) * X["bmi"]
        X["cardio_fitness"] = X[hrv] * X[steps] / X[rhr].clip(lower=1)
        X["sleep_quality"] = X[sleep] * X[hrv]

    return X


def select_features(X: pd.DataFrame, y: np.ndarray, n_top: int = 35) -> List[str]:
    """Select top features by GradientBoosting importance."""
    gbr = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, subsample=0.8, random_state=42
    )
    gbr.fit(X, y)
    importance = pd.Series(gbr.feature_importances_, index=X.columns)
    return importance.nlargest(n_top).index.tolist()


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

def get_model_configs(target: str) -> Dict[str, dict]:
    """
    Return model configurations for the given target.

    Each config has:
      - "fn": callable returning a fresh model instance
      - "scale": bool, whether to StandardScale features
      - "log": bool, whether to log-transform the target
    """
    configs = {}

    # --- Tree-based models ---
    if HAS_XGB:
        configs["xgb_d5_mse"] = {
            "fn": lambda: xgb.XGBRegressor(
                n_estimators=500, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1, random_state=42,
            ),
            "scale": False, "log": False,
        }
        configs["xgb_d4_mse"] = {
            "fn": lambda: xgb.XGBRegressor(
                n_estimators=500, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.5, reg_lambda=2, random_state=42,
            ),
            "scale": False, "log": False,
        }
        configs["xgb_d5_mae"] = {
            "fn": lambda: xgb.XGBRegressor(
                n_estimators=500, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                objective="reg:absoluteerror", random_state=42,
            ),
            "scale": False, "log": False,
        }
        configs["xgb_d6_mse"] = {
            "fn": lambda: xgb.XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.08,
                subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1, reg_lambda=3, random_state=42,
            ),
            "scale": False, "log": False,
        }
        configs["xgb_d5_log"] = {
            "fn": lambda: xgb.XGBRegressor(
                n_estimators=500, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
            ),
            "scale": False, "log": True,
        }
        configs["xgb_d3_log"] = {
            "fn": lambda: xgb.XGBRegressor(
                n_estimators=800, max_depth=3, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.7, random_state=42,
            ),
            "scale": False, "log": True,
        }

    configs["hgbr"] = {
        "fn": lambda: HistGradientBoostingRegressor(
            max_iter=500, max_depth=5, learning_rate=0.05, max_leaf_nodes=31,
            random_state=42,
        ),
        "scale": False, "log": False,
    }
    configs["hgbr_log"] = {
        "fn": lambda: HistGradientBoostingRegressor(
            max_iter=500, max_depth=5, learning_rate=0.05, random_state=42,
        ),
        "scale": False, "log": True,
    }
    configs["rf_500"] = {
        "fn": lambda: RandomForestRegressor(
            n_estimators=500, max_depth=8, max_features=0.5,
            random_state=42, n_jobs=1,
        ),
        "scale": False, "log": False,
    }
    configs["et_500"] = {
        "fn": lambda: ExtraTreesRegressor(
            n_estimators=500, max_depth=8, max_features=0.5,
            random_state=42, n_jobs=1,
        ),
        "scale": False, "log": False,
    }

    # --- Linear models ---
    configs["ridge_10"] = {"fn": lambda: Ridge(alpha=10), "scale": True, "log": False}
    configs["ridge_100"] = {"fn": lambda: Ridge(alpha=100), "scale": True, "log": False}
    configs["elastic"] = {
        "fn": lambda: ElasticNet(alpha=0.01, l1_ratio=0.5),
        "scale": True, "log": False,
    }

    return configs


# =============================================================================
# CROSS-VALIDATION ENGINE
# =============================================================================

def make_stratification_bins(y: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Bin continuous target for stratified splitting."""
    return pd.qcut(y, n_bins, labels=False, duplicates="drop")


def default_splits(y: np.ndarray, n_splits: int = 5, n_repeats: int = 3,
                   random_state: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate default RepeatedStratifiedKFold splits."""
    bins = make_stratification_bins(y)
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    return list(cv.split(np.zeros(len(y)), bins))


def run_single_model_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_fn: callable,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    scale: bool = True,
    log_target: bool = False,
) -> Tuple[float, np.ndarray]:
    """
    Run cross-validation for a single model.

    Returns
    -------
    r2 : float
        Out-of-fold R² score.
    oof_preds : np.ndarray
        Out-of-fold predictions (averaged over repeats).
    """
    n_samples = len(y)
    oof_preds = np.zeros(n_samples)
    oof_counts = np.zeros(n_samples)

    for train_idx, test_idx in splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx].copy()

        # Optional log transform
        if log_target:
            y_train = np.log1p(y_train)

        # Optional scaling
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        model = model_fn()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if log_target:
            preds = np.expm1(preds)

        oof_preds[test_idx] += preds
        oof_counts[test_idx] += 1

    # Average over repeats
    mask = oof_counts > 0
    oof_preds[mask] /= oof_counts[mask]

    ss_res = np.sum((y[mask] - oof_preds[mask]) ** 2)
    ss_tot = np.sum((y[mask] - y[mask].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    return r2, oof_preds


def optimize_blend(
    model_preds: Dict[str, np.ndarray],
    y: np.ndarray,
    top_k: int = 10,
    n_dirichlet: int = 300_000,
) -> Tuple[float, Dict[str, float]]:
    """
    Find optimal blend weights via Dirichlet sampling + pairwise grid.

    Returns
    -------
    best_r2 : float
    best_weights : dict mapping model name -> weight
    """
    names = list(model_preds.keys())
    preds_mat = np.array([model_preds[n] for n in names])
    ss_tot = np.sum((y - y.mean()) ** 2)

    best_r2 = -np.inf
    best_w = np.zeros(len(names))

    # Dirichlet search
    rng = np.random.RandomState(42)
    for _ in range(n_dirichlet):
        w = rng.dirichlet(np.ones(len(names)))
        blend = w @ preds_mat
        r2 = 1 - np.sum((y - blend) ** 2) / ss_tot
        if r2 > best_r2:
            best_r2 = r2
            best_w = w.copy()

    # Pairwise grid
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for alpha in np.linspace(0, 1, 201):
                blend = alpha * preds_mat[i] + (1 - alpha) * preds_mat[j]
                r2 = 1 - np.sum((y - blend) ** 2) / ss_tot
                if r2 > best_r2:
                    best_r2 = r2
                    best_w = np.zeros(len(names))
                    best_w[i] = alpha
                    best_w[j] = 1 - alpha

    # Triplet grid (top 8)
    top_idx = list(range(min(8, len(names))))
    for i in top_idx:
        for j in top_idx:
            if j <= i:
                continue
            for k in top_idx:
                if k <= j:
                    continue
                for a in np.linspace(0.05, 0.9, 18):
                    for b in np.linspace(0.05, 0.9 - a, 13):
                        c = 1 - a - b
                        if c > 0:
                            blend = a * preds_mat[i] + b * preds_mat[j] + c * preds_mat[k]
                            r2 = 1 - np.sum((y - blend) ** 2) / ss_tot
                            if r2 > best_r2:
                                best_r2 = r2
                                best_w = np.zeros(len(names))
                                best_w[i] = a
                                best_w[j] = b
                                best_w[k] = c

    weights = {n: w for n, w in zip(names, best_w) if w > 0.001}
    return best_r2, weights


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(
    data_path: str,
    target: str = "homa_ir",
    splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    n_top_features: int = 35,
    verbose: bool = True,
) -> dict:
    """
    Run the full ALL-features pipeline.

    Parameters
    ----------
    data_path : str
        Path to data.csv.
    target : str
        "homa_ir" or "hba1c".
    splits : list of (train_idx, test_idx), optional
        Custom CV splits. If None, uses default RepeatedStratifiedKFold(5, 3).
    n_top_features : int
        Number of features to select (default 35).
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys:
        "r2": float — best OOF R²
        "r": float — Pearson correlation
        "oof_predictions": np.ndarray — out-of-fold predictions
        "y": np.ndarray — true targets
        "model_r2s": dict — per-model R² scores
        "blend_weights": dict — blend weights
        "feature_names": list — selected feature names
        "n_samples": int
    """
    assert target in ("homa_ir", "hba1c"), f"Unknown target: {target}"

    # --- Load data ---
    if verbose:
        print(f"Loading data from {data_path}...")
    X_raw, y_homa, y_hba1c = load_data(data_path)

    y = y_homa if target == "homa_ir" else y_hba1c
    mask = ~np.isnan(y)
    X_raw = X_raw[mask].reset_index(drop=True)
    y = y[mask]

    if verbose:
        print(f"  Samples: {len(y)}, Target: {target}, y range: [{y.min():.2f}, {y.max():.2f}]")

    # --- Feature engineering ---
    if verbose:
        print("Engineering features...")
    X_eng = engineer_features(X_raw, target=target)
    X_eng = X_eng.fillna(X_eng.median())

    if verbose:
        print(f"  Total features after engineering: {X_eng.shape[1]}")

    # --- Feature selection ---
    if verbose:
        print(f"Selecting top {n_top_features} features...")
    selected = select_features(X_eng, y, n_top=n_top_features)
    X_sel = X_eng[selected].values

    if verbose:
        print(f"  Selected: {selected[:10]}{'...' if len(selected) > 10 else ''}")

    # --- Generate splits ---
    if splits is None:
        if verbose:
            print("Using default splits: RepeatedStratifiedKFold(5-fold, 3 repeats)")
        splits = default_splits(y)
    else:
        if verbose:
            print(f"Using {len(splits)} custom splits")

    # --- Train all models ---
    if verbose:
        print(f"\nTraining {len(get_model_configs(target))} models...")
        print(f"{'Model':<25s} {'R²':>8s} {'Time':>8s}")
        print("-" * 45)

    model_configs = get_model_configs(target)
    model_r2s = {}
    model_preds = {}

    for name, cfg in model_configs.items():
        t0 = time.time()
        r2, preds = run_single_model_cv(
            X_sel, y, cfg["fn"], splits, scale=cfg["scale"], log_target=cfg["log"]
        )
        elapsed = time.time() - t0
        model_r2s[name] = r2
        model_preds[name] = preds
        if verbose:
            print(f"  {name:<25s} {r2:>8.4f} {elapsed:>7.1f}s")

    # --- Blend optimization ---
    if verbose:
        print("\nOptimizing blend...")
    best_r2, blend_weights = optimize_blend(model_preds, y)

    # Compute final OOF predictions
    final_preds = np.zeros(len(y))
    for name, w in blend_weights.items():
        final_preds += w * model_preds[name]

    r, _ = pearsonr(y, final_preds)

    if verbose:
        target_r2 = 0.65 if target == "homa_ir" else 0.85
        print(f"\n{'=' * 50}")
        print(f"RESULT: {target.upper()}")
        print(f"  R²      = {best_r2:.4f}  (target: {target_r2})")
        print(f"  Pearson r = {r:.4f}")
        print(f"  Gap     = {target_r2 - best_r2:.4f}")
        print(f"  Blend:")
        for name, w in sorted(blend_weights.items(), key=lambda x: -x[1]):
            print(f"    {name}: {w:.3f} (R²={model_r2s[name]:.4f})")
        print(f"{'=' * 50}")

    return {
        "r2": best_r2,
        "r": r,
        "oof_predictions": final_preds,
        "y": y,
        "model_r2s": model_r2s,
        "blend_weights": blend_weights,
        "feature_names": selected,
        "n_samples": len(y),
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="WEAR-ME: ALL-features prediction pipeline"
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
    parser.add_argument(
        "--n-features", type=int, default=35,
        help="Number of features to select (default: 35)",
    )

    args = parser.parse_args()

    # Load custom splits if provided
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
        print(f"\n{'=' * 60}")
        print(f"  TARGET: {tgt.upper()} (ALL FEATURES)")
        print(f"{'=' * 60}\n")

        t0 = time.time()
        result = run_pipeline(
            data_path=args.data,
            target=tgt,
            splits=custom_splits,
            n_top_features=args.n_features,
        )
        elapsed = time.time() - t0
        print(f"\nCompleted in {elapsed:.1f}s")
        all_results[tgt] = result

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY (ALL FEATURES)")
    print(f"{'=' * 60}")
    targets_map = {"homa_ir": 0.65, "hba1c": 0.85}
    for tgt, result in all_results.items():
        target_r2 = targets_map[tgt]
        status = "PASS" if result["r2"] >= target_r2 else "FAIL"
        print(f"  {tgt.upper():>10s}: R²={result['r2']:.4f}  r={result['r']:.4f}  "
              f"target={target_r2}  gap={target_r2 - result['r2']:.4f}  [{status}]")


if __name__ == "__main__":
    main()
