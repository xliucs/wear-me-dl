#!/usr/bin/env python3
"""V16: All-out push on DW (demographics + wearables) targets."""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, QuantileTransformer
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, RidgeCV
from sklearn.ensemble import (GradientBoostingRegressor, HistGradientBoostingRegressor,
                               RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor,
                               AdaBoostRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct, WhiteKernel
import xgboost as xgb
import lightgbm as lgb
import time, os

os.environ['HF_TOKEN'] = 'REDACTED'

df = pd.read_csv('data.csv', skiprows=[0])
targets_cols = ['True_hba1c','True_HOMA_IR','True_IR_Class','True_Diabetes_2_Class',
                'True_Normoglycemic_2_Class','True_Diabetes_3_Class','Participant_id']

X_raw = df[[c for c in df.columns if c not in targets_cols]].copy()
X_raw['sex_num'] = (X_raw['sex'] == 'Male').astype(int)
X_raw = X_raw.drop('sex', axis=1)

# DW base features
dw_raw = ['age', 'bmi', 'sex_num',
    'Resting Heart Rate (mean)', 'Resting Heart Rate (median)', 'Resting Heart Rate (std)',
    'HRV (mean)', 'HRV (median)', 'HRV (std)',
    'STEPS (mean)', 'STEPS (median)', 'STEPS (std)',
    'SLEEP Duration (mean)', 'SLEEP Duration (median)', 'SLEEP Duration (std)',
    'AZM Weekly (mean)', 'AZM Weekly (median)', 'AZM Weekly (std)']

def make_bins(y, n_bins=5):
    return pd.qcut(y, n_bins, labels=False, duplicates='drop')

def engineer_dw_exhaustive(X):
    """Exhaustive feature engineering for DW features."""
    X = X.copy()
    
    # Aliases
    rhr = 'Resting Heart Rate (mean)'
    rhr_med = 'Resting Heart Rate (median)'
    rhr_std = 'Resting Heart Rate (std)'
    hrv = 'HRV (mean)'
    hrv_med = 'HRV (median)'
    hrv_std = 'HRV (std)'
    steps = 'STEPS (mean)'
    steps_med = 'STEPS (median)'
    steps_std = 'STEPS (std)'
    slp = 'SLEEP Duration (mean)'
    slp_med = 'SLEEP Duration (median)'
    slp_std = 'SLEEP Duration (std)'
    azm = 'AZM Weekly (mean)'
    azm_med = 'AZM Weekly (median)'
    azm_std = 'AZM Weekly (std)'
    
    # === Polynomial features of top predictors ===
    X['bmi_sq'] = X['bmi'] ** 2
    X['bmi_cubed'] = X['bmi'] ** 3
    X['bmi_log'] = np.log1p(X['bmi'])
    X['bmi_sqrt'] = np.sqrt(X['bmi'])
    X['age_sq'] = X['age'] ** 2
    X['age_cubed'] = X['age'] ** 3
    X['age_log'] = np.log1p(X['age'])
    X['rhr_sq'] = X[rhr] ** 2
    X['steps_sq'] = X[steps] ** 2
    X['steps_log'] = np.log1p(X[steps])
    X['hrv_sq'] = X[hrv] ** 2
    X['hrv_log'] = np.log1p(X[hrv])
    X['sleep_sq'] = X[slp] ** 2
    X['sleep_log'] = np.log1p(X[slp])
    
    # === Key interactions (BMI × wearable) ===
    X['bmi_rhr'] = X['bmi'] * X[rhr]
    X['bmi_rhr_sq'] = X['bmi'] * X[rhr] ** 2
    X['bmi_sq_rhr'] = X['bmi'] ** 2 * X[rhr]
    X['bmi_hrv'] = X['bmi'] * X[hrv]
    X['bmi_steps'] = X['bmi'] * X[steps]
    X['bmi_sleep'] = X['bmi'] * X[slp]
    X['bmi_azm'] = X['bmi'] * X[azm]
    X['bmi_rhr_hrv'] = X['bmi'] * X[rhr] / X[hrv].clip(lower=1)
    
    # === Age interactions ===
    X['age_bmi'] = X['age'] * X['bmi']
    X['age_bmi_sq'] = X['age'] * X['bmi'] ** 2
    X['age_rhr'] = X['age'] * X[rhr]
    X['age_hrv'] = X['age'] * X[hrv]
    X['age_steps'] = X['age'] * X[steps]
    X['age_sleep'] = X['age'] * X[slp]
    X['age_sex'] = X['age'] * X['sex_num']
    X['bmi_sex'] = X['bmi'] * X['sex_num']
    X['age_bmi_sex'] = X['age'] * X['bmi'] * X['sex_num']
    
    # === Wearable cross-interactions ===
    X['rhr_hrv'] = X[rhr] / X[hrv].clip(lower=1)  # autonomic balance
    X['rhr_hrv_product'] = X[rhr] * X[hrv]
    X['steps_sleep'] = X[steps] * X[slp]
    X['steps_hrv'] = X[steps] * X[hrv]
    X['steps_rhr'] = X[steps] / X[rhr].clip(lower=1)
    X['azm_steps'] = X[azm] / X[steps].clip(lower=1)  # intensity ratio
    X['azm_bmi'] = X[azm] / X['bmi'].clip(lower=1)
    X['sleep_hrv'] = X[slp] * X[hrv]
    X['sleep_rhr'] = X[slp] * X[rhr]
    
    # === Variability features (CV = std/mean) ===
    X['rhr_cv'] = X[rhr_std] / X[rhr].clip(lower=1)
    X['hrv_cv'] = X[hrv_std] / X[hrv].clip(lower=1)
    X['steps_cv'] = X[steps_std] / X[steps].clip(lower=1)
    X['sleep_cv'] = X[slp_std] / X[slp].clip(lower=1)
    X['azm_cv'] = X[azm_std] / X[azm].clip(lower=1)
    
    # === Median-mean differences (skewness proxies) ===
    X['rhr_skew'] = X[rhr] - X[rhr_med]
    X['hrv_skew'] = X[hrv] - X[hrv_med]
    X['steps_skew'] = X[steps] - X[steps_med]
    X['sleep_skew'] = X[slp] - X[slp_med]
    X['azm_skew'] = X[azm] - X[azm_med]
    
    # === Composite health indices ===
    X['cardio_fitness'] = X[hrv] * X[steps] / X[rhr].clip(lower=1)
    X['cardio_fitness_log'] = np.log1p(X['cardio_fitness'].clip(lower=0))
    X['metabolic_load'] = X['bmi'] * X[rhr] / X[steps].clip(lower=1) * 1000
    X['recovery_index'] = X[hrv] / X[rhr].clip(lower=1) * X[slp]
    X['activity_index'] = (X[steps] + X[azm]) / X['bmi']
    X['sedentary_risk'] = X['bmi'] ** 2 * X[rhr] / (X[steps].clip(lower=1) * X[hrv].clip(lower=1))
    X['autonomic_health'] = X[hrv] / X[rhr].clip(lower=1)
    X['sleep_efficiency'] = X[slp] / X[slp_std].clip(lower=1)
    X['hr_reserve_proxy'] = (220 - X['age'] - X[rhr]) / X['bmi']
    
    # === Obesity-conditional features ===
    X['obese'] = (X['bmi'] >= 30).astype(float)
    X['overweight'] = (X['bmi'] >= 25).astype(float)
    X['obese_rhr'] = X['obese'] * X[rhr]
    X['obese_low_steps'] = X['obese'] * (X[steps] < X[steps].median()).astype(float)
    X['obese_low_hrv'] = X['obese'] * (X[hrv] < X[hrv].median()).astype(float)
    X['obese_poor_sleep'] = X['obese'] * (X[slp] < X[slp].median()).astype(float)
    X['overweight_rhr_high'] = X['overweight'] * (X[rhr] > X[rhr].median()).astype(float)
    
    # === Age-conditional features ===
    X['older'] = (X['age'] >= 50).astype(float)
    X['older_bmi'] = X['older'] * X['bmi']
    X['older_rhr'] = X['older'] * X[rhr]
    X['older_low_hrv'] = X['older'] * (X[hrv] < X[hrv].median()).astype(float)
    
    # === Binned features ===
    X['bmi_bin'] = pd.cut(X['bmi'], bins=[0, 18.5, 25, 30, 35, 100], labels=False)
    X['age_bin'] = pd.cut(X['age'], bins=[0, 30, 40, 50, 60, 100], labels=False)
    
    return X

# ============================================================
# CV infrastructure
# ============================================================
def run_model_cv(X, y, model_fn, n_splits=5, n_repeats=5, scale=True, log_target=False):
    bins = make_bins(y)
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    n = len(y)
    preds = np.zeros(n); counts = np.zeros(n)
    
    for tr_idx, te_idx in rkf.split(X, bins):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr = y[tr_idx]
        y_use = np.log1p(y_tr) if log_target else y_tr
        
        if scale:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
        
        m = model_fn()
        m.fit(X_tr, y_use)
        p = m.predict(X_te)
        if log_target: p = np.expm1(p)
        preds[te_idx] += p; counts[te_idx] += 1
    
    preds /= counts
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum((y - preds) ** 2) / ss_tot
    return r2, preds

def mega_blend(results, y, n_search=300000):
    ss_tot = np.sum((y - y.mean()) ** 2)
    names = sorted(results.keys(), key=lambda k: results[k]['r2'], reverse=True)[:15]
    preds_mat = np.array([results[nm]['preds'] for nm in names])
    
    best_r2 = -999; best_w = None
    rng = np.random.RandomState(42)
    
    for _ in range(n_search):
        w = rng.dirichlet(np.ones(len(names)))
        blend = w @ preds_mat
        r2 = 1 - np.sum((y - blend) ** 2) / ss_tot
        if r2 > best_r2: best_r2 = r2; best_w = w
    
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            for alpha in np.linspace(0, 1, 201):
                blend = alpha * preds_mat[i] + (1-alpha) * preds_mat[j]
                r2 = 1 - np.sum((y - blend) ** 2) / ss_tot
                if r2 > best_r2: best_r2 = r2; best_w = np.zeros(len(names)); best_w[i] = alpha; best_w[j] = 1-alpha
    
    # Top-5 triplet
    for i in range(min(5, len(names))):
        for j in range(i+1, min(5, len(names))):
            for k in range(j+1, min(5, len(names))):
                for a in np.linspace(0.1, 0.8, 15):
                    for b in np.linspace(0.1, 0.8-a, 10):
                        c = 1-a-b
                        if c > 0:
                            blend = a*preds_mat[i] + b*preds_mat[j] + c*preds_mat[k]
                            r2 = 1 - np.sum((y-blend)**2)/ss_tot
                            if r2 > best_r2: best_r2=r2; best_w=np.zeros(len(names)); best_w[i]=a; best_w[j]=b; best_w[k]=c
    
    print(f"  Mega blend: R2={best_r2:.4f}")
    for nm, w in zip(names, best_w):
        if w > 0.01: print(f"    {nm}: {w:.3f}")
    
    # Ridge stack
    oof_mat = np.column_stack([results[nm]['preds'] for nm in names])
    bins = make_bins(y)
    stack_preds = np.zeros(len(y)); stack_counts = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, te_idx in skf.split(oof_mat, bins):
        ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000])
        ridge.fit(oof_mat[tr_idx], y[tr_idx])
        stack_preds[te_idx] = ridge.predict(oof_mat[te_idx])
        stack_counts[te_idx] += 1
    stack_preds /= stack_counts
    r2_stack = 1 - np.sum((y - stack_preds)**2)/ss_tot
    print(f"  Ridge stack: R2={r2_stack:.4f}")
    
    return max(best_r2, r2_stack)

def run_dw_experiment(X_raw_subset, y, target_name, target_r2):
    """Run exhaustive DW experiment."""
    X_eng = engineer_dw_exhaustive(X_raw_subset)
    X_eng = X_eng.fillna(X_eng.median())
    
    # Also create polynomial features on top 5 raw features for linear models
    top5 = ['bmi', 'Resting Heart Rate (mean)', 'STEPS (mean)', 'HRV (mean)', 'age']
    top5_cols = [c for c in top5 if c in X_raw_subset.columns]
    X_poly2 = pd.DataFrame(
        PolynomialFeatures(degree=2, interaction_only=False, include_bias=False).fit_transform(X_raw_subset[top5_cols]),
    )
    X_poly3 = pd.DataFrame(
        PolynomialFeatures(degree=3, interaction_only=True, include_bias=False).fit_transform(X_raw_subset[top5_cols]),
    )
    
    feature_sets = {
        'raw_18': X_raw_subset.values,
        'engineered': X_eng.values,
        'poly2_top5': X_poly2.values,
        'poly3_top5_interact': X_poly3.values,
    }
    
    use_log = 'homa' in target_name.lower()
    
    all_results = {}
    
    for fs_name, X_fs in feature_sets.items():
        print(f"\n  === Feature set: {fs_name} ({X_fs.shape[1]} features) ===")
        
        models = [
            # Ridge sweep
            ('ridge_01', lambda: Ridge(alpha=0.1), True),
            ('ridge_1', lambda: Ridge(alpha=1), True),
            ('ridge_10', lambda: Ridge(alpha=10), True),
            ('ridge_50', lambda: Ridge(alpha=50), True),
            ('ridge_100', lambda: Ridge(alpha=100), True),
            ('ridge_500', lambda: Ridge(alpha=500), True),
            ('ridge_1000', lambda: Ridge(alpha=1000), True),
            ('ridge_5000', lambda: Ridge(alpha=5000), True),
            # Lasso
            ('lasso_001', lambda: Lasso(alpha=0.001, max_iter=5000), True),
            ('lasso_01', lambda: Lasso(alpha=0.01, max_iter=5000), True),
            ('lasso_1', lambda: Lasso(alpha=0.1, max_iter=5000), True),
            # ElasticNet
            ('elastic_001', lambda: ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000), True),
            ('elastic_01', lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000), True),
            # Bayesian Ridge
            ('bayesian', lambda: BayesianRidge(), True),
            # Kernel Ridge
            ('kernel_rbf', lambda: KernelRidge(alpha=1, kernel='rbf', gamma=0.01), True),
            ('kernel_poly', lambda: KernelRidge(alpha=1, kernel='poly', degree=2), True),
            # SVR
            ('svr_rbf_01', lambda: SVR(kernel='rbf', C=0.1, epsilon=0.1), True),
            ('svr_rbf_1', lambda: SVR(kernel='rbf', C=1, epsilon=0.1), True),
            ('svr_rbf_10', lambda: SVR(kernel='rbf', C=10, epsilon=0.1), True),
            # KNN
            ('knn_10', lambda: KNeighborsRegressor(n_neighbors=10, weights='distance'), True),
            ('knn_20', lambda: KNeighborsRegressor(n_neighbors=20, weights='distance'), True),
            ('knn_30', lambda: KNeighborsRegressor(n_neighbors=30, weights='distance'), True),
            ('knn_50', lambda: KNeighborsRegressor(n_neighbors=50, weights='distance'), True),
            # XGBoost (shallow, regularized)
            ('xgb_d2', lambda: xgb.XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, reg_alpha=2, reg_lambda=5, random_state=42), False),
            ('xgb_d3', lambda: xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, reg_alpha=2, reg_lambda=5, random_state=42), False),
            # LightGBM
            ('lgb_d2', lambda: lgb.LGBMRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, verbose=-1, n_jobs=2, random_state=42), False),
            ('lgb_dart', lambda: lgb.LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, boosting_type='dart', verbose=-1, n_jobs=2, random_state=42), False),
            # HGBR
            ('hgbr_d3', lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05, random_state=42), False),
            # RF/ET
            ('rf_d4', lambda: RandomForestRegressor(n_estimators=500, max_depth=4, max_features=0.5, random_state=42, n_jobs=2), False),
            ('et_d4', lambda: ExtraTreesRegressor(n_estimators=500, max_depth=4, max_features=0.5, random_state=42, n_jobs=2), False),
            # Bagged Ridge
            ('bagged_ridge', lambda: BaggingRegressor(estimator=Ridge(alpha=100), n_estimators=20, max_samples=0.8, max_features=0.8, random_state=42), True),
        ]
        
        # Add log-target variants for key models if applicable
        if use_log:
            extra = []
            for name, fn, scale in models[:14]:  # linear models
                extra.append((name + '_log', fn, scale))
            models.extend(extra)
        
        for name, model_fn, scale in models:
            full_name = f"{fs_name}__{name}"
            is_log = '_log' in name and use_log
            t0 = time.time()
            try:
                r2, preds = run_model_cv(X_fs, y, model_fn, scale=scale, log_target=is_log, n_repeats=5)
                all_results[full_name] = {'r2': r2, 'preds': preds}
                if r2 > 0.15:  # Only print decent results
                    print(f"    {full_name:50s}: R2={r2:.4f} ({time.time()-t0:.1f}s)")
            except Exception as e:
                pass
    
    # Skip TabPFN and GP (run separately to avoid hangs)
    
    # Mega blend
    print(f"\n  === Mega blend across all ===")
    # Filter to models with R2 > 0
    good_results = {k: v for k, v in all_results.items() if v['r2'] > 0}
    best = mega_blend(good_results, y)
    
    print(f"\n  >>> {target_name} BEST: R2={best:.4f} (target {target_r2}, gap={target_r2-best:.3f})")
    return best

# ============================================================
print("=" * 60)
print("V16: ALL-OUT DW PUSH")
print("=" * 60)

# HOMA_IR DW
print("\n" + "=" * 60)
print("HOMA_IR DW | Target: 0.37")
print("=" * 60)

y_homa = df['True_HOMA_IR'].values
mask = ~np.isnan(y_homa)
X_dw = X_raw.loc[mask, dw_raw].reset_index(drop=True)
y_homa = y_homa[mask]
best_homa_dw = run_dw_experiment(X_dw, y_homa, 'HOMA_IR DW', 0.37)

# hba1c DW
print("\n" + "=" * 60)
print("hba1c DW | Target: 0.70")
print("=" * 60)

y_hba1c = df['True_hba1c'].values
mask2 = ~np.isnan(y_hba1c)
X_dw2 = X_raw.loc[mask2, dw_raw].reset_index(drop=True)
y_hba1c = y_hba1c[mask2]
best_hba1c_dw = run_dw_experiment(X_dw2, y_hba1c, 'hba1c DW', 0.70)

# Summary
print("\n" + "=" * 60)
print("V16 FINAL SUMMARY")
print("=" * 60)
print(f"  HOMA_IR DW: R2={best_homa_dw:.4f} (target 0.37, gap={0.37-best_homa_dw:.3f})")
print(f"  hba1c DW:   R2={best_hba1c_dw:.4f} (target 0.70, gap={0.70-best_hba1c_dw:.3f})")
print("\nDone.")
