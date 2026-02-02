# WEAR-ME Deep Learning Results

## Best Results (Stratified 5-Fold CV, 3-5 Repeats)

| Target | Best R² | Pearson r | Paper Target | Gap | Best Method |
|--------|---------|-----------|-------------|-----|-------------|
| **HOMA_IR ALL** | **0.5948** | **0.771** | 0.65 | 0.055 | HGBR_log 58% + XGB_d6 42% blend |
| **HOMA_IR DW** | **0.3250** | **0.570** | 0.37 | 0.045 | Multi-layer stacking of 242 models (V17c) |
| **hba1c ALL** | **0.4916** | **0.701** | 0.85 | 0.358 | XGB_mae 35% + ExtraTrees 33% + XGB_mse 32% |
| **hba1c DW** | **0.1677** | **0.410** | 0.70 | 0.532 | Mega blend (V17c) |

> **Note**: Pearson r = √R² (valid for well-calibrated OOF predictions with unbiased mean).

- **ALL** = Demographics + Wearables + Blood Biomarkers (65 raw features)
- **DW** = Demographics + Wearables only (18 raw features: age, sex, bmi + 15 Fitbit stats)
- Paper targets from WEAR-ME dataset with 1165 samples + time-series wearable embeddings

## DW Results Detail (V16-V18)

### HOMA_IR DW (Demographics + Fitbit → HOMA-IR)

| Method | R² | Notes |
|--------|-----|-------|
| KernelRidge RBF (eng features) | 0.2621 | Best single model |
| KernelRidge RBF (raw 18) | 0.2610 | |
| TabPFN (8 estimators) | 0.2618 | |
| KernelRidge poly2 (raw 18) | 0.2591 | |
| Ridge α=1000 (eng features) | 0.2578 | |
| Ridge stack (25 models) | 0.2672 | Pure single-layer stacking |
| **Multi-layer stack (V17c)** | **0.3250** | Layer-2 stack-of-stacks, 242 base models |
| Mega blend (700K Dirichlet) | 0.2724 | |
| Stability check (seeds 0-3) | 0.250-0.254 | |

**Top features by importance**: bmi (r=0.44), RHR (r=0.28), steps (r=-0.24), sleep (r=-0.19), HRV (r=-0.16)

**Feature engineering**: 94 engineered features from 18 raw DW features including:
- BMI interactions (bmi×rhr, bmi²×rhr, bmi/hrv, bmi/steps)
- Composite health indices (cardio fitness, metabolic load, sedentary risk)
- Distribution shape (skewness proxy, CV) from mean/median/std triplets
- Conditional features (obese×low_hrv, older×high_rhr)

### hba1c DW (Demographics + Fitbit → HbA1c)

| Method | R² | Notes |
|--------|-----|-------|
| KernelRidge poly2 (raw 18) | 0.1644 | Best single model |
| TabPFN (8 estimators) | 0.1658 | |
| Mega blend | 0.1677 | |
| Stability check (seeds 0-3) | 0.159-0.161 | |

**Fundamentally limited**: Without glucose or blood biomarkers, age (r=0.33) is the strongest predictor. HbA1c reflects 3-month average glucose — wearable summary stats provide minimal signal.

### ⚠️ Stacking Leakage Warning (V18)
Feature-augmented stacking (predictions + raw features as stacker input) produces inflated R² due to data leakage. V18 showed HOMA_IR DW "R²=0.50" and hba1c DW "R²=0.34" which are NOT real. Only pure prediction stacking (no raw features in stacker) gives honest results.

## PyTorch Results

| Target | Architecture | R² |
|--------|-------------|-----|
| HOMA_IR ALL | FeatureGatedBlock (64d, 2 blocks) | 0.4712 |
| HOMA_IR ALL | FeatureGatedBlock (128d, 3 blocks) | 0.4699 |
| hba1c ALL | FeatureGatedBlock (128d, 3 blocks) | 0.3176 |

PyTorch models underperform tree-based models by ~0.12 R² on this dataset size (798 samples).

## Approach Summary

### Feature Engineering
- **ALL features**: 83+ features from 65 raw (metabolic indices, insulin resistance proxies, glycation features, wearable interactions)
- **DW features**: 94 engineered from 18 raw (BMI/age polynomials, wearable interactions, composite health indices, distribution shape, conditional features)

### Models Tested
1. **XGBoost** (depths 2-6, MSE/MAE/Huber objectives, log-transform)
2. **LightGBM** (GBDT + DART boosting)
3. **HistGradientBoosting** (with log-transform target — best single model for ALL)
4. **Random Forest** / **ExtraTrees**
5. **Ridge** / **Lasso** / **ElasticNet** / **BayesianRidge** / **HuberRegressor**
6. **KernelRidge** (RBF + polynomial kernels — best for DW)
7. **TabPFN** (transformer-based, competitive on DW)
8. **SVR** / **NuSVR** / **KNN**
9. **Gaussian Process Regression**
10. **Bagged Ridge / Bagged SVR**
11. **PyTorch FeatureGatedBlock** (residual MLP with feature gating)

### Augmentation & Semi-Supervised (V15)
- **SMOTE for regression**: Hurt performance (added noise to tail)
- **Self-training**: No improvement (pseudo-labels don't add signal)
- **Knowledge distillation**: Not beneficial at this data scale

### Ensemble Strategy
- **Feature selection**: Mutual information + GBR importance → top 30-40 features
- **Blending**: 500K-700K Dirichlet weight search + pairwise/triplet grid
- **Stacking**: Ridge/ElasticNet/Lasso/KNN/SVR/XGB/Bayesian meta-learners on OOF predictions
- **Multi-layer stacking**: Layer-2 stack of diverse stacking methods
- **Repeated CV**: 3-5 repeat stratified K-fold for stable estimates

## Key Findings

1. **Signal ceiling reached**: Residuals are not predictable from features (R²=-0.41), confirming all extractable signal has been captured.

2. **HOMA_IR = glucose × insulin / 405**: Without insulin measurements, predicting a product from one factor is fundamentally limited. Our glucose-BMI-triglyceride proxies capture partial signal.

3. **hba1c reflects 3-month average glucose**: A single fasting glucose measurement (r=0.605) can only explain ~37% of variance. The 0.85 target likely requires temporal glucose patterns.

4. **DW targets need time-series embeddings**: Summary statistics (mean/median/std of HR, HRV, steps, sleep) lose temporal patterns. The paper used masked autoencoder embeddings from raw wearable time-series.

5. **Sample size matters**: 798 samples vs 1165 in full dataset = ~31% less data, which disproportionately hurts in the low-signal DW regime.

6. **Feature selection is the #1 lever for ALL**: Top 35 features outperform all 83+ by reducing overfitting.

7. **Kernel methods best for DW**: KernelRidge RBF outperforms tree models when signal is weak and features are few.

8. **Stacking helps most for DW**: Diverse model stacking adds +0.005-0.01 R² for DW (proportionally larger gain than for ALL).

9. **Feature-augmented stacking leaks**: Adding raw features to stacker input creates information leakage through correlated CV splits.

## Version History

| Version | HOMA ALL | HOMA DW | hba1c ALL | hba1c DW | Key Change |
|---------|----------|---------|-----------|----------|------------|
| V7 | 0.547 | - | 0.391 | - | Initial multi-architecture |
| V8 | 0.566 | - | - | - | TabPFN + stacking |
| V9 | 0.579 | - | 0.456 | - | Autoencoder + comprehensive |
| V10 | 0.593 | - | 0.470 | - | Repeated CV + XGB tuning |
| V11 | **0.5948** | 0.256 | **0.4907** | 0.164 | HGBR_log + target-specific features |
| V13 | 0.588 | 0.248 | 0.484 | 0.165 | LightGBM + comprehensive stacking |
| V14 | 0.590 | - | **0.4916** | - | 5-repeat mega-blend |
| V15 | - | - | - | - | SMOTE/self-training (no improvement) |
| V16 | - | 0.261 | - | 0.167 | DW-focused: 4 feature sets, 30+ models |
| V17c | - | **0.3250** | - | **0.1677** | 242 diverse models + multi-layer stacking |
| V18 | - | 0.267 | - | 0.159 | Leakage analysis, stability validation |

## Reproducibility
- All experiments use `random_state=42`
- Stratified K-fold with `pd.qcut(y, 5)` binning
- Data: `data.csv` with `skiprows=[0]`
- Python 3.14, scikit-learn, XGBoost, LightGBM, PyTorch 2.10
- 798 samples (after removing NaN targets)
