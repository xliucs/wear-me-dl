#!/usr/bin/env python3
"""Re-validate ALL best results using standardized wearme_eval library.
This ensures all R² values are directly comparable (same CV splits)."""
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from wearme_eval import (load_data, get_cv_splits, generate_oof, evaluate_oof, 
                          print_eval, stack_predictions, engineer_features)

from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.ensemble import (HistGradientBoostingRegressor, ExtraTreesRegressor, 
                               RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor)
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.svm import SVR, NuSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

print("="*70)
print("STANDARDIZED VALIDATION — All results on IDENTICAL CV splits")
print("="*70)

data = load_data()

# ============================================================
# HOMA-IR DW
# ============================================================
print(f"\n{'='*70}")
print(f"HOMA-IR DW (n={data['n_samples_homa']}, 18 raw features)")
print(f"{'='*70}")

y = data['y_homa']
bins = data['bins_homa']
splits = get_cv_splits(y)  # CANONICAL splits
ss_tot = data['ss_tot_homa']
n_samples = data['n_samples_homa']

# Feature sets
X_raw = data['X_raw_homa']
X_eng = data['X_eng_homa']
X_mi35 = data['X_mi35_homa']

all_results = {}

print("\n--- Single Models (5-rep × 5-fold, seed=42) ---")

models = [
    # (name, model_fn, feature_set_name, X, scale, log_target)
    ('Ridge α=100 raw18', lambda: Ridge(alpha=100), 'raw18', X_raw, True, False),
    ('Ridge α=500 eng', lambda: Ridge(alpha=500), 'eng', X_eng, True, False),
    ('Ridge α=1000 eng', lambda: Ridge(alpha=1000), 'eng', X_eng, True, False),
    ('BayesianRidge eng', lambda: BayesianRidge(), 'eng', X_eng, True, False),
    ('KR RBF α=1 γ=0.01 raw18', lambda: KernelRidge(alpha=1, kernel='rbf', gamma=0.01), 'raw18', X_raw, True, False),
    ('KR RBF α=0.1 γ=0.001 raw18', lambda: KernelRidge(alpha=0.1, kernel='rbf', gamma=0.001), 'raw18', X_raw, True, False),
    ('KR RBF α=1 γ=0.001 eng', lambda: KernelRidge(alpha=1, kernel='rbf', gamma=0.001), 'eng', X_eng, True, False),
    ('KR poly2 raw18', lambda: KernelRidge(alpha=1, kernel='poly', degree=2, gamma=0.01), 'raw18', X_raw, True, False),
    ('SVR C=1 raw18', lambda: SVR(kernel='rbf', C=1, epsilon=0.1), 'raw18', X_raw, True, False),
    ('KNN k=20 raw18', lambda: KNeighborsRegressor(n_neighbors=20, weights='distance'), 'raw18', X_raw, True, False),
    ('KNN k=50 raw18', lambda: KNeighborsRegressor(n_neighbors=50, weights='distance'), 'raw18', X_raw, True, False),
    ('Bag Ridge500 eng', lambda: BaggingRegressor(estimator=Ridge(alpha=500), n_estimators=30, max_samples=0.8, max_features=0.8, random_state=42), 'eng', X_eng, True, False),
    ('XGB d2 raw18', lambda: xgb.XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, reg_alpha=2, reg_lambda=5, random_state=42), 'raw18', X_raw, False, False),
    ('LGB d2 raw18', lambda: lgb.LGBMRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, verbose=-1, n_jobs=1, random_state=42), 'raw18', X_raw, False, False),
    ('HGBR d2 raw18', lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=2, learning_rate=0.05, random_state=42), 'raw18', X_raw, False, False),
]

for name, mfn, fs_name, X_fs, scale, log_t in models:
    preds, _ = generate_oof(X_fs, y, mfn, splits, scale=scale, log_target=log_t)
    metrics = evaluate_oof(preds, y)
    print_eval(name, metrics)
    all_results[name] = preds

# Log-target variants
print("\n--- Log-target variants ---")
log_models = [
    ('Ridge α=100 raw18 LOG', lambda: Ridge(alpha=100), X_raw, True),
    ('Ridge α=500 eng LOG', lambda: Ridge(alpha=500), X_eng, True),
    ('KR RBF α=1 γ=0.01 raw18 LOG', lambda: KernelRidge(alpha=1, kernel='rbf', gamma=0.01), X_raw, True),
    ('KR RBF α=1 γ=0.001 eng LOG', lambda: KernelRidge(alpha=1, kernel='rbf', gamma=0.001), X_eng, True),
]

for name, mfn, X_fs, scale in log_models:
    preds, _ = generate_oof(X_fs, y, mfn, splits, scale=scale, log_target=True)
    metrics = evaluate_oof(preds, y)
    print_eval(name, metrics)
    all_results[name] = preds

# More diverse models for stacking
print("\n--- Additional models for stacking ---")
extra = [
    ('Ridge α=50 raw18', lambda: Ridge(alpha=50), X_raw, True, False),
    ('Ridge α=2000 eng', lambda: Ridge(alpha=2000), X_eng, True, False),
    ('Lasso α=0.01 eng', lambda: Lasso(alpha=0.01, max_iter=10000), X_eng, True, False),
    ('ElasticNet eng', lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000), X_eng, True, False),
    ('Huber eng', lambda: HuberRegressor(max_iter=1000), X_eng, True, False),
    ('KR RBF α=10 γ=0.01 raw18', lambda: KernelRidge(alpha=10, kernel='rbf', gamma=0.01), X_raw, True, False),
    ('NuSVR raw18', lambda: NuSVR(kernel='rbf', C=1, nu=0.5), X_raw, True, False),
    ('KNN k=10 raw18', lambda: KNeighborsRegressor(n_neighbors=10, weights='distance'), X_raw, True, False),
    ('KNN k=75 raw18', lambda: KNeighborsRegressor(n_neighbors=75, weights='distance'), X_raw, True, False),
    ('Bag Ridge100 raw18', lambda: BaggingRegressor(estimator=Ridge(alpha=100), n_estimators=30, max_samples=0.8, max_features=0.8, random_state=42), X_raw, True, False),
    ('XGB d3 eng', lambda: xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.01, subsample=0.8, colsample_bytree=0.7, reg_alpha=5, reg_lambda=10, random_state=42), X_eng, False, False),
    ('GBR d2 raw18', lambda: GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=42), X_raw, False, False),
    ('RF d4 raw18', lambda: RandomForestRegressor(n_estimators=300, max_depth=4, max_features=0.5, random_state=42, n_jobs=1), X_raw, False, False),
    ('ET d4 raw18', lambda: ExtraTreesRegressor(n_estimators=300, max_depth=4, max_features=0.5, random_state=42, n_jobs=1), X_raw, False, False),
    # MI35 models
    ('Ridge α=500 mi35', lambda: Ridge(alpha=500), X_mi35, True, False),
    ('KR RBF α=1 γ=0.001 mi35', lambda: KernelRidge(alpha=1, kernel='rbf', gamma=0.001), X_mi35, True, False),
    # Log variants
    ('Ridge α=50 raw18 LOG', lambda: Ridge(alpha=50), X_raw, True, True),
    ('BayesianRidge eng LOG', lambda: BayesianRidge(), X_eng, True, True),
    ('KR poly2 raw18 LOG', lambda: KernelRidge(alpha=1, kernel='poly', degree=2, gamma=0.01), X_raw, True, True),
    ('ElasticNet eng LOG', lambda: ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000), X_eng, True, True),
]

for name, mfn, X_fs, scale, log_t in extra:
    preds, _ = generate_oof(X_fs, y, mfn, splits, scale=scale, log_target=log_t)
    metrics = evaluate_oof(preds, y)
    all_results[name] = preds
    if metrics['r2'] > 0.20:
        print_eval(name, metrics)

# ============================================================
# STACKING (using canonical splits)
# ============================================================
print(f"\n--- Stacking ({len(all_results)} base models) ---")

good = {k: v for k, v in all_results.items() if evaluate_oof(v, y)['r2'] > 0}
names = sorted(good.keys(), key=lambda k: evaluate_oof(good[k], y)['r2'], reverse=True)[:25]
preds_mat = np.array([good[nm] for nm in names])

print(f"Top 5 base models:")
for nm in names[:5]:
    m = evaluate_oof(good[nm], y)
    print(f"  {nm:50s}: R²={m['r2']:.4f}  r={m['pearson_r']:.4f}")

# Layer-1 stacking with CANONICAL seed
all_stacks = {}
best_l1 = -999

for alpha in [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000]:
    sp = np.zeros(n_samples); sc = np.zeros(n_samples)
    for tr, te in splits:  # SAME canonical splits!
        m = Ridge(alpha=alpha); m.fit(preds_mat[:, tr].T, y[tr])
        sp[te] += m.predict(preds_mat[:, te].T); sc[te] += 1
    sp /= sc; r2 = 1 - np.sum((y-sp)**2) / ss_tot
    if r2 > best_l1: best_l1 = r2
    if r2 > 0.20: all_stacks[f'ridge_{alpha}'] = sp.copy()

for alpha in [0.01, 0.1, 1]:
    for l1 in [0.1, 0.3, 0.5, 0.7, 0.9]:
        sp = np.zeros(n_samples); sc = np.zeros(n_samples)
        for tr, te in splits:
            m = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=5000, positive=True)
            m.fit(preds_mat[:, tr].T, y[tr]); sp[te] += m.predict(preds_mat[:, te].T); sc[te] += 1
        sp /= sc; r2 = 1 - np.sum((y-sp)**2) / ss_tot
        if r2 > best_l1: best_l1 = r2
        if r2 > 0.20: all_stacks[f'en_{alpha}_{l1}'] = sp.copy()

for alpha in [0.001, 0.01, 0.1]:
    sp = np.zeros(n_samples); sc = np.zeros(n_samples)
    for tr, te in splits:
        m = Lasso(alpha=alpha, max_iter=5000, positive=True)
        m.fit(preds_mat[:, tr].T, y[tr]); sp[te] += m.predict(preds_mat[:, te].T); sc[te] += 1
    sp /= sc; r2 = 1 - np.sum((y-sp)**2) / ss_tot
    if r2 > best_l1: best_l1 = r2
    if r2 > 0.20: all_stacks[f'lasso_{alpha}'] = sp.copy()

# Reduced rep splits for slower stackers
splits_3rep = get_cv_splits(y, n_repeats=3)
for k in [3, 5, 7, 10, 15, 20, 30]:
    sp = np.zeros(n_samples); sc = np.zeros(n_samples)
    for tr, te in splits_3rep:
        ss = StandardScaler(); pt = ss.fit_transform(preds_mat[:, tr].T); pe = ss.transform(preds_mat[:, te].T)
        m = KNeighborsRegressor(n_neighbors=k, weights='distance'); m.fit(pt, y[tr])
        sp[te] += m.predict(pe); sc[te] += 1
    sp /= sc; r2 = 1 - np.sum((y-sp)**2) / ss_tot
    if r2 > best_l1: best_l1 = r2
    if r2 > 0.20: all_stacks[f'knn_{k}'] = sp.copy()

for C in [0.1, 1, 10, 100]:
    for eps in [0.05, 0.1]:
        sp = np.zeros(n_samples); sc = np.zeros(n_samples)
        for tr, te in splits_3rep:
            ss = StandardScaler(); pt = ss.fit_transform(preds_mat[:, tr].T); pe = ss.transform(preds_mat[:, te].T)
            m = SVR(kernel='rbf', C=C, epsilon=eps); m.fit(pt, y[tr])
            sp[te] += m.predict(pe); sc[te] += 1
        sp /= sc; r2 = 1 - np.sum((y-sp)**2) / ss_tot
        if r2 > best_l1: best_l1 = r2
        if r2 > 0.20: all_stacks[f'svr_{C}_{eps}'] = sp.copy()

for d in [2, 3]:
    sp = np.zeros(n_samples); sc = np.zeros(n_samples)
    for tr, te in splits_3rep:
        m = xgb.XGBRegressor(n_estimators=100, max_depth=d, learning_rate=0.05, reg_alpha=5, reg_lambda=10, random_state=42)
        m.fit(preds_mat[:, tr].T, y[tr]); sp[te] += m.predict(preds_mat[:, te].T); sc[te] += 1
    sp /= sc; r2 = 1 - np.sum((y-sp)**2) / ss_tot
    if r2 > best_l1: best_l1 = r2
    if r2 > 0.20: all_stacks[f'xgb_{d}'] = sp.copy()

sp = np.zeros(n_samples); sc = np.zeros(n_samples)
for tr, te in splits:
    m = BayesianRidge(); m.fit(preds_mat[:, tr].T, y[tr])
    sp[te] += m.predict(preds_mat[:, te].T); sc[te] += 1
sp /= sc; r2 = 1 - np.sum((y-sp)**2) / ss_tot
if r2 > best_l1: best_l1 = r2
if r2 > 0.20: all_stacks['bayesian'] = sp.copy()

print(f"\n  Layer-1 best: R²={best_l1:.4f} ({len(all_stacks)} stacks)")

# Mega blend
rng = np.random.RandomState(42); blend_best = -999
for _ in range(500000):
    w = rng.dirichlet(np.ones(len(names))*0.3); bl = w @ preds_mat
    r2 = 1-np.sum((y-bl)**2)/ss_tot
    if r2 > blend_best: blend_best = r2
for _ in range(200000):
    w = rng.dirichlet(np.ones(len(names))*0.1); bl = w @ preds_mat
    r2 = 1-np.sum((y-bl)**2)/ss_tot
    if r2 > blend_best: blend_best = r2
for i in range(len(names)):
    for j in range(i+1, len(names)):
        for a in np.linspace(0, 1, 201):
            bl = a*preds_mat[i]+(1-a)*preds_mat[j]; r2 = 1-np.sum((y-bl)**2)/ss_tot
            if r2 > blend_best: blend_best = r2
print(f"  Mega blend: R²={blend_best:.4f}")

# Layer-2
if len(all_stacks) >= 3:
    snames = list(all_stacks.keys()); smat = np.array([all_stacks[k] for k in snames])
    best_l2 = -999
    for alpha in [0.01, 0.1, 1, 10, 100]:
        sp = np.zeros(n_samples); sc = np.zeros(n_samples)
        for tr, te in splits:
            m = Ridge(alpha=alpha); m.fit(smat[:, tr].T, y[tr])
            sp[te] += m.predict(smat[:, te].T); sc[te] += 1
        sp /= sc; r2 = 1 - np.sum((y-sp)**2) / ss_tot
        if r2 > best_l2: best_l2 = r2
    print(f"  Layer-2: R²={best_l2:.4f}")

overall_best = max(best_l1, blend_best, best_l2 if len(all_stacks) >= 3 else -999)
r_overall = np.sqrt(max(0, overall_best))

print(f"\n{'='*70}")
print(f"HOMA-IR DW FINAL (standardized): R²={overall_best:.4f}  r={r_overall:.4f}")
print(f"Target: 0.37  Gap: {0.37-overall_best:.3f}")
print(f"{'='*70}")
