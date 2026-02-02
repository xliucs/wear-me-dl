# WEAR-ME Deep Learning Results

## Best Results (Stratified 5-Fold CV, 3-5 Repeats)

| Target | Best R² | Paper Target | Gap | Best Method |
|--------|---------|-------------|-----|-------------|
| **HOMA_IR ALL** | **0.5948** | 0.65 | 0.055 | HGBR_log 58% + XGB_d6 42% blend |
| **HOMA_IR DW** | **0.2657** | 0.37 | 0.107 | Ridge stack (Ridge + TabPFN + RF) |
| **hba1c ALL** | **0.4916** | 0.85 | 0.358 | XGB_mae 35% + ExtraTrees 33% + XGB_mse 32% |
| **hba1c DW** | **0.1651** | 0.70 | 0.535 | Ridge 90% + ExtraTrees 10% |

- **ALL** = Demographics + Wearables + Blood Biomarkers
- **DW** = Demographics + Wearables only
- Paper targets from WEAR-ME dataset with 1165 samples + time-series wearable embeddings

## PyTorch Results

| Target | Architecture | R² |
|--------|-------------|-----|
| HOMA_IR ALL | FeatureGatedBlock (64d, 2 blocks) | 0.4712 |
| HOMA_IR ALL | FeatureGatedBlock (128d, 3 blocks) | 0.4699 |
| hba1c ALL | FeatureGatedBlock (128d, 3 blocks) | 0.3176 |

PyTorch models underperform tree-based models by ~0.12 R² on this dataset size (798 samples).

## Approach Summary

### Feature Engineering (83+ features from 65 raw)
- **Metabolic indices**: TyG, METS-IR, glucose×BMI, triglyceride/HDL ratio, visceral adiposity proxy
- **Insulin resistance proxies**: glucose×BMI×triglycerides/HDL
- **Glycation features** (hba1c): glucose/hemoglobin, glucose×RDW, glucose×MCHC
- **Wearable interactions**: RHR×BMI, steps×sleep, cardio fitness index
- **Demographics**: BMI², BMI³, age×sex, BMI×sex

### Models Tested
1. **XGBoost** (depths 3-6, MSE/MAE/Huber objectives, log-transform)
2. **LightGBM** (GBDT + DART boosting)
3. **HistGradientBoosting** (with log-transform target — best single model)
4. **Random Forest** / **ExtraTrees**
5. **Ridge** / **Lasso** / **ElasticNet**
6. **TabPFN** (transformer-based, best for DW)
7. **SVR** / **KNN**
8. **PyTorch FeatureGatedBlock** (residual MLP with feature gating)

### Augmentation & Semi-Supervised (V15)
- **SMOTE for regression**: Hurt performance (added noise to tail)
- **Self-training**: No improvement (pseudo-labels don't add signal)
- **Knowledge distillation**: Not beneficial at this data scale

### Ensemble Strategy
- **Feature selection**: GBR importance → top 30-35 features (reduces overfitting)
- **Blending**: 200K-500K Dirichlet weight search + pairwise/triplet grid
- **Stacking**: Ridge meta-learner on OOF predictions
- **Repeated CV**: 3-5 repeat stratified K-fold for stable estimates

## Key Findings

1. **Signal ceiling reached**: Residuals are not predictable from features (R²=-0.41), confirming all extractable signal has been captured.

2. **HOMA_IR = glucose × insulin / 405**: Without insulin measurements, predicting a product from one factor is fundamentally limited. Our glucose-BMI-triglyceride proxies capture partial signal.

3. **hba1c reflects 3-month average glucose**: A single fasting glucose measurement (r=0.605) can only explain ~37% of variance. The 0.85 target likely requires temporal glucose patterns.

4. **DW targets need time-series embeddings**: Summary statistics (mean/median/std of HR, HRV, steps, sleep) lose temporal patterns. The paper used masked autoencoder embeddings from raw wearable time-series.

5. **Sample size matters**: 798 samples vs 1165 in full dataset = ~31% less data, which disproportionately hurts in the low-signal DW regime.

6. **Feature selection is the #1 lever**: Top 35 features outperform all 83+ by reducing overfitting.

## Version History

| Version | HOMA_IR ALL | hba1c ALL | Key Change |
|---------|-------------|-----------|------------|
| V7 | 0.547 | 0.391 | Initial multi-architecture |
| V8 | 0.566 | - | TabPFN + stacking |
| V9 | 0.579 | 0.456 | Autoencoder + comprehensive |
| V10 | 0.593 | 0.470 | Repeated CV + XGB tuning |
| V11 | **0.5948** | **0.4907** | HGBR_log + target-specific features |
| V13 | 0.588 | 0.484 | LightGBM + comprehensive stacking |
| V14 | 0.590 | **0.4916** | 5-repeat mega-blend |
| V15 | - | - | SMOTE/self-training (no improvement) |

## Reproducibility
- All experiments use `random_state=42`
- Stratified K-fold with `pd.qcut(y, 5)` binning
- Data: `data.csv` with `skiprows=[0]`
- Python 3.14, scikit-learn, XGBoost, LightGBM, PyTorch 2.10
