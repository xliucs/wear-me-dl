#!/usr/bin/env python3
"""
WEAR-ME Standardized Evaluation Library
========================================
Forces identical preprocessing, CV splits, and evaluation across ALL versions.

Usage:
    from wearme_eval import load_data, get_cv_splits, evaluate_oof, r2_score

    data = load_data()
    splits = get_cv_splits(data['y_homa'], n_repeats=5, seed=42)
    
    # Generate OOF predictions
    oof = np.zeros(data['n_samples'])
    for fold_idx, (tr, te) in enumerate(splits):
        model.fit(data['X_raw'][tr], data['y_homa'][tr])
        oof[te] += model.predict(data['X_raw'][te])
    oof /= data['n_repeats']  # average across repeats
    
    r2 = evaluate_oof(oof, data['y_homa'])
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr
import os

# ============================================================
# CONSTANTS
# ============================================================
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data.csv')

DW_COLS = ['age', 'bmi', 'sex',
    'Resting Heart Rate (mean)', 'Resting Heart Rate (median)', 'Resting Heart Rate (std)',
    'HRV (mean)', 'HRV (median)', 'HRV (std)',
    'STEPS (mean)', 'STEPS (median)', 'STEPS (std)',
    'SLEEP Duration (mean)', 'SLEEP Duration (median)', 'SLEEP Duration (std)',
    'AZM Weekly (mean)', 'AZM Weekly (median)', 'AZM Weekly (std)']

# Shorthand names for feature engineering
RHR_M = 'Resting Heart Rate (mean)'; RHR_MD = 'Resting Heart Rate (median)'; RHR_S = 'Resting Heart Rate (std)'
HRV_M = 'HRV (mean)'; HRV_MD = 'HRV (median)'; HRV_S = 'HRV (std)'
STP_M = 'STEPS (mean)'; STP_MD = 'STEPS (median)'; STP_S = 'STEPS (std)'
SLP_M = 'SLEEP Duration (mean)'; SLP_MD = 'SLEEP Duration (median)'; SLP_S = 'SLEEP Duration (std)'
AZM_M = 'AZM Weekly (mean)'; AZM_MD = 'AZM Weekly (median)'; AZM_S = 'AZM Weekly (std)'

# Standard CV config
DEFAULT_N_SPLITS = 5
DEFAULT_N_REPEATS = 5
DEFAULT_SEED = 42
DEFAULT_N_BINS = 5

# ============================================================
# DATA LOADING (Canonical)
# ============================================================
def load_data(data_path=None):
    """Load and preprocess the WEAR-ME dataset.
    
    Returns dict with:
        df: raw DataFrame
        X_dw: DataFrame with 18 DW features (sex encoded)
        X_raw: numpy array of raw 18 DW features
        X_eng: numpy array of 94 engineered features
        X_mi35: numpy array of top 35 MI-selected engineered features
        y_homa: HOMA-IR target (NaN-filtered)
        y_hba1c: HbA1c target (NaN-filtered, same mask as HOMA)
        homa_mask: boolean mask for valid HOMA-IR samples
        hba1c_mask: boolean mask for valid HbA1c samples
        bins_homa: stratification bins for HOMA-IR
        bins_hba1c: stratification bins for HbA1c
        n_samples_homa: number of valid HOMA samples
        n_samples_hba1c: number of valid HbA1c samples
        ss_tot_homa: total sum of squares for HOMA
        ss_tot_hba1c: total sum of squares for HbA1c
    """
    if data_path is None:
        data_path = DATA_PATH
    
    df = pd.read_csv(data_path, skiprows=[0])
    
    # DW features
    X_dw = df[DW_COLS].copy()
    X_dw['sex'] = (X_dw['sex'] == 'Male').astype(float)
    
    # HOMA-IR target
    y_homa_full = df['True_HOMA_IR'].values
    homa_mask = ~np.isnan(y_homa_full)
    y_homa = y_homa_full[homa_mask]
    X_dw_homa = X_dw[homa_mask].reset_index(drop=True)
    X_raw_homa = X_dw_homa.values
    
    # HbA1c target (use same mask structure)
    y_hba1c_full = df['True_hba1c'].values
    hba1c_mask = ~np.isnan(y_hba1c_full)
    y_hba1c = y_hba1c_full[hba1c_mask]
    X_dw_hba1c = X_dw[hba1c_mask].reset_index(drop=True)
    X_raw_hba1c = X_dw_hba1c.values
    
    # Engineered features (HOMA)
    X_eng_homa = engineer_features(X_dw_homa).fillna(0).values
    mi_homa = mutual_info_regression(X_eng_homa, y_homa, random_state=DEFAULT_SEED)
    X_mi35_homa = X_eng_homa[:, np.argsort(mi_homa)[-35:]]
    
    # Engineered features (HbA1c)
    X_eng_hba1c = engineer_features(X_dw_hba1c).fillna(0).values
    mi_hba1c = mutual_info_regression(X_eng_hba1c, y_hba1c, random_state=DEFAULT_SEED)
    X_mi35_hba1c = X_eng_hba1c[:, np.argsort(mi_hba1c)[-35:]]
    
    # Bins
    bins_homa = pd.qcut(y_homa, DEFAULT_N_BINS, labels=False, duplicates='drop')
    bins_hba1c = pd.qcut(y_hba1c, DEFAULT_N_BINS, labels=False, duplicates='drop')
    
    return {
        'df': df,
        # HOMA
        'X_dw_homa': X_dw_homa,
        'X_raw_homa': X_raw_homa,
        'X_eng_homa': X_eng_homa,
        'X_mi35_homa': X_mi35_homa,
        'y_homa': y_homa,
        'homa_mask': homa_mask,
        'bins_homa': bins_homa,
        'n_samples_homa': len(y_homa),
        'ss_tot_homa': np.sum((y_homa - y_homa.mean()) ** 2),
        # HbA1c
        'X_dw_hba1c': X_dw_hba1c,
        'X_raw_hba1c': X_raw_hba1c,
        'X_eng_hba1c': X_eng_hba1c,
        'X_mi35_hba1c': X_mi35_hba1c,
        'y_hba1c': y_hba1c,
        'hba1c_mask': hba1c_mask,
        'bins_hba1c': bins_hba1c,
        'n_samples_hba1c': len(y_hba1c),
        'ss_tot_hba1c': np.sum((y_hba1c - y_hba1c.mean()) ** 2),
    }


# ============================================================
# FEATURE ENGINEERING (Canonical)
# ============================================================
def engineer_features(X_df):
    """Create 94 engineered features from 18 raw DW features.
    
    Input: DataFrame with 18 DW columns (sex already encoded as 0/1).
    Output: DataFrame with ~94 features.
    """
    X = X_df.copy()
    
    # Distribution shape features
    for pfx, m, md, s in [('rhr', RHR_M, RHR_MD, RHR_S), ('hrv', HRV_M, HRV_MD, HRV_S),
                           ('stp', STP_M, STP_MD, STP_S), ('slp', SLP_M, SLP_MD, SLP_S),
                           ('azm', AZM_M, AZM_MD, AZM_S)]:
        X[f'{pfx}_skew'] = (X[m] - X[md]) / X[s].clip(lower=0.01)
        X[f'{pfx}_cv'] = X[s] / X[m].clip(lower=0.01)
    
    # Polynomial + log + inverse transforms
    for col, nm in [(X['bmi'], 'bmi'), (X['age'], 'age'), (X[RHR_M], 'rhr'),
                     (X[HRV_M], 'hrv'), (X[STP_M], 'stp')]:
        X[f'{nm}_sq'] = col ** 2
        X[f'{nm}_log'] = np.log1p(col.clip(lower=0))
        X[f'{nm}_inv'] = 1 / (col.clip(lower=0.01))
    
    # BMI interactions
    X['bmi_rhr'] = X['bmi'] * X[RHR_M]
    X['bmi_sq_rhr'] = X['bmi'] ** 2 * X[RHR_M]
    X['bmi_hrv'] = X['bmi'] * X[HRV_M]
    X['bmi_hrv_inv'] = X['bmi'] / X[HRV_M].clip(lower=1)
    X['bmi_stp'] = X['bmi'] * X[STP_M]
    X['bmi_stp_inv'] = X['bmi'] / X[STP_M].clip(lower=1) * 1000
    X['bmi_slp'] = X['bmi'] * X[SLP_M]
    X['bmi_azm'] = X['bmi'] * X[AZM_M]
    X['bmi_age'] = X['bmi'] * X['age']
    X['bmi_sq_age'] = X['bmi'] ** 2 * X['age']
    X['bmi_sex'] = X['bmi'] * X['sex']
    X['bmi_rhr_hrv'] = X['bmi'] * X[RHR_M] / X[HRV_M].clip(lower=1)
    X['bmi_rhr_stp'] = X['bmi'] * X[RHR_M] / X[STP_M].clip(lower=1) * 1000
    
    # Age interactions
    X['age_rhr'] = X['age'] * X[RHR_M]
    X['age_hrv_inv'] = X['age'] / X[HRV_M].clip(lower=1)
    X['age_stp'] = X['age'] * X[STP_M]
    X['age_slp'] = X['age'] * X[SLP_M]
    X['age_sex'] = X['age'] * X['sex']
    X['age_bmi_sex'] = X['age'] * X['bmi'] * X['sex']
    
    # Wearable interactions
    X['rhr_hrv'] = X[RHR_M] / X[HRV_M].clip(lower=1)
    X['stp_hrv'] = X[STP_M] * X[HRV_M]
    X['stp_rhr'] = X[STP_M] / X[RHR_M].clip(lower=1)
    X['azm_stp'] = X[AZM_M] / X[STP_M].clip(lower=1)
    X['slp_hrv'] = X[SLP_M] * X[HRV_M]
    X['slp_rhr'] = X[SLP_M] / X[RHR_M].clip(lower=1)
    
    # Composite health indices
    X['cardio'] = X[HRV_M] * X[STP_M] / X[RHR_M].clip(lower=1)
    X['cardio_log'] = np.log1p(X['cardio'].clip(lower=0))
    X['met_load'] = X['bmi'] * X[RHR_M] / X[STP_M].clip(lower=1) * 1000
    X['met_load_log'] = np.log1p(X['met_load'].clip(lower=0))
    X['recovery'] = X[HRV_M] / X[RHR_M].clip(lower=1) * X[SLP_M]
    X['activity_bmi'] = (X[STP_M] + X[AZM_M]) / X['bmi']
    X['sed_risk'] = X['bmi'] ** 2 * X[RHR_M] / (X[STP_M].clip(lower=1) * X[HRV_M].clip(lower=1))
    X['sed_risk_log'] = np.log1p(X['sed_risk'].clip(lower=0))
    X['auto_health'] = X[HRV_M] / X[RHR_M].clip(lower=1)
    X['hr_reserve'] = (220 - X['age'] - X[RHR_M]) / X['bmi']
    X['fitness_age'] = X['age'] * X[RHR_M] / X[HRV_M].clip(lower=1)
    X['bmi_fitness'] = X['bmi'] * X[RHR_M] / (X[HRV_M].clip(lower=1) * X[STP_M].clip(lower=1)) * 10000
    
    # Conditional features
    X['obese'] = (X['bmi'] >= 30).astype(float)
    X['older'] = (X['age'] >= 50).astype(float)
    X['obese_rhr'] = X['obese'] * X[RHR_M]
    X['obese_low_hrv'] = X['obese'] * (X[HRV_M] < X[HRV_M].median()).astype(float)
    X['older_bmi'] = X['older'] * X['bmi']
    X['older_rhr'] = X['older'] * X[RHR_M]
    
    # CV interactions
    X['rhr_cv_bmi'] = X['rhr_cv'] * X['bmi']
    X['hrv_cv_bmi'] = X['hrv_cv'] * X['bmi']
    X['rhr_cv_age'] = X['rhr_cv'] * X['age']
    
    # Rank features
    for col in ['bmi', 'age', RHR_M, HRV_M, STP_M]:
        X[f'rank_{col[:3]}'] = X[col].rank(pct=True)
    
    return X


# ============================================================
# CV SPLITS (Canonical)
# ============================================================
def get_cv_splits(y, bins=None, n_splits=DEFAULT_N_SPLITS, n_repeats=DEFAULT_N_REPEATS, seed=DEFAULT_SEED):
    """Get canonical CV splits.
    
    Returns list of (train_indices, test_indices) tuples.
    Always uses the same seed/n_splits/n_repeats for reproducibility.
    """
    if bins is None:
        bins = pd.qcut(y, DEFAULT_N_BINS, labels=False, duplicates='drop')
    
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return list(rkf.split(np.zeros(len(y)), bins))


# ============================================================
# OOF PREDICTION HELPER
# ============================================================
def generate_oof(X, y, model_fn, splits, scale=True, log_target=False):
    """Generate out-of-fold predictions using canonical splits.
    
    Args:
        X: feature array (n_samples, n_features)
        y: target array (n_samples,)
        model_fn: callable that returns a fresh model instance
        splits: list of (train_idx, test_idx) from get_cv_splits()
        scale: whether to StandardScale features
        log_target: whether to log1p-transform target
    
    Returns:
        preds: averaged OOF predictions (n_samples,)
        counts: number of times each sample was predicted (for verification)
    """
    n_samples = len(y)
    preds = np.zeros(n_samples)
    counts = np.zeros(n_samples)
    
    for tr, te in splits:
        Xtr, Xte = X[tr].copy(), X[te].copy()
        ytr = y[tr].copy()
        
        if log_target:
            ytr = np.log1p(ytr)
        
        if scale:
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(Xtr)
            Xte = scaler.transform(Xte)
        
        model = model_fn()
        model.fit(Xtr, ytr)
        p = model.predict(Xte)
        
        if log_target:
            p = np.expm1(p)
        
        preds[te] += p
        counts[te] += 1
    
    # Average predictions
    mask = counts > 0
    preds[mask] /= counts[mask]
    
    return preds, counts


# ============================================================
# EVALUATION (Canonical)
# ============================================================
def evaluate_oof(preds, y):
    """Evaluate OOF predictions.
    
    Returns dict with R², Pearson r, MAE, RMSE.
    """
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    r, p_value = pearsonr(preds, y)
    mae = np.mean(np.abs(y - preds))
    rmse = np.sqrt(np.mean((y - preds) ** 2))
    
    return {
        'r2': r2,
        'pearson_r': r,
        'pearson_p': p_value,
        'mae': mae,
        'rmse': rmse,
        'n_samples': len(y),
    }


def print_eval(name, metrics):
    """Pretty-print evaluation metrics."""
    print(f"  {name:50s}: R²={metrics['r2']:.4f}  r={metrics['pearson_r']:.4f}  "
          f"MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}")


# ============================================================
# STACKING HELPERS
# ============================================================
def stack_predictions(pred_dict, y, bins=None, stacker_fn=None, 
                      n_splits=DEFAULT_N_SPLITS, n_repeats=DEFAULT_N_REPEATS, seed=DEFAULT_SEED):
    """Stack multiple OOF predictions using canonical splits.
    
    Args:
        pred_dict: dict of {name: predictions_array}
        y: target array
        bins: stratification bins (auto-computed if None)
        stacker_fn: callable returning stacker model (default: Ridge(alpha=1))
        seed: CV seed for stacking (should differ from base model seed for cross-seed stacking)
    
    Returns:
        stacked_preds, stacker_r2
    """
    from sklearn.linear_model import Ridge
    
    if stacker_fn is None:
        stacker_fn = lambda: Ridge(alpha=1)
    
    names = sorted(pred_dict.keys())
    pred_mat = np.array([pred_dict[nm] for nm in names])  # (n_models, n_samples)
    
    splits = get_cv_splits(y, bins, n_splits, n_repeats, seed)
    n_samples = len(y)
    stacked = np.zeros(n_samples)
    counts = np.zeros(n_samples)
    
    for tr, te in splits:
        m = stacker_fn()
        m.fit(pred_mat[:, tr].T, y[tr])
        stacked[te] += m.predict(pred_mat[:, te].T)
        counts[te] += 1
    
    mask = counts > 0
    stacked[mask] /= counts[mask]
    
    metrics = evaluate_oof(stacked, y)
    return stacked, metrics


# ============================================================
# SELF-TEST
# ============================================================
if __name__ == '__main__':
    print("WEAR-ME Evaluation Library Self-Test")
    print("=" * 60)
    
    data = load_data()
    
    print(f"\nData loaded:")
    print(f"  HOMA-IR: n={data['n_samples_homa']}, mean={data['y_homa'].mean():.2f}, "
          f"std={data['y_homa'].std():.2f}, skew={pd.Series(data['y_homa']).skew():.2f}")
    print(f"  HbA1c:   n={data['n_samples_hba1c']}, mean={data['y_hba1c'].mean():.2f}, "
          f"std={data['y_hba1c'].std():.2f}, skew={pd.Series(data['y_hba1c']).skew():.2f}")
    print(f"  Raw features: {data['X_raw_homa'].shape[1]}")
    print(f"  Engineered features: {data['X_eng_homa'].shape[1]}")
    print(f"  MI-35 features: {data['X_mi35_homa'].shape[1]}")
    
    # Verify CV splits are deterministic
    splits1 = get_cv_splits(data['y_homa'])
    splits2 = get_cv_splits(data['y_homa'])
    assert all(np.array_equal(s1[0], s2[0]) and np.array_equal(s1[1], s2[1]) 
               for s1, s2 in zip(splits1, splits2)), "CV splits not deterministic!"
    print(f"\n  CV splits: {len(splits1)} folds ({DEFAULT_N_SPLITS}-fold × {DEFAULT_N_REPEATS} repeats)")
    print(f"  Deterministic: PASS")
    
    # Quick baseline: Ridge on raw features
    from sklearn.linear_model import Ridge
    
    splits = get_cv_splits(data['y_homa'])
    preds, counts = generate_oof(data['X_raw_homa'], data['y_homa'], 
                                  lambda: Ridge(alpha=100), splits, scale=True)
    metrics = evaluate_oof(preds, data['y_homa'])
    print(f"\n  Baseline (Ridge α=100, raw18):")
    print_eval("HOMA_IR DW", metrics)
    
    # Verify all samples predicted equally
    assert np.all(counts == DEFAULT_N_REPEATS), f"Unequal prediction counts: {np.unique(counts)}"
    print(f"  All samples predicted {DEFAULT_N_REPEATS} times: PASS")
    
    print(f"\n{'=' * 60}")
    print("Self-test PASSED. Library ready for use.")
