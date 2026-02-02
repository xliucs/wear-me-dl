# Results Summary

All results use 5-fold cross-validation. V7 uses stratified k-fold for more realistic estimates.

## HOMA_IR — All Features (demographics + wearables + blood biomarkers)

| Version | Method | R² (mean ± std) | Notes |
|---------|--------|-----------------|-------|
| V7 | Two-Stage Ensemble (5 seeds, weighted) | **0.547 ± 0.018** | Stratified CV, inverse-freq weights |
| V7 | sqrt_inverse + asymmetric loss | 0.524 ± 0.014 | Stratified CV |
| V7 | Baseline ResNet 256×6 | 0.514 ± 0.022 | Stratified CV |
| Final | ResNet 256×6 standard transform | 0.551 ± 0.097 | Regular CV (inflated) |
| Final | Ensemble 10 models log | 0.544 ± 0.091 | Regular CV |
| V3 | ResNet 256×6 log | 0.555 ± 0.097 | Regular CV (inflated) |
| V5 | MT_ResNet 128×4 log | 0.533 ± 0.081 | Regular CV |
| V6 | AE_Regressor log | 0.508 ± 0.091 | Regular CV |
| Baseline | sklearn HistGBR log | ~0.61 | Regular CV |

**Goal: R² ≥ 0.85** — Gap: 0.30

## hba1c — All Features

| Version | Method | R² (mean ± std) | Notes |
|---------|--------|-----------------|-------|
| V7 | sqrt_inverse + focal loss | **0.364 ± 0.093** | Stratified CV |
| V7 | inverse_freq + focal loss | 0.360 ± 0.085 | Stratified CV |
| V7 | Baseline ResNet 256×6 | 0.350 ± 0.120 | Stratified CV |
| V6 | AE_Regressor log | 0.456 ± 0.111 | Regular CV |
| Baseline | sklearn HistGBR | ~0.45 | Regular CV |

**Goal: R² ≥ 0.85** — Gap: ~0.49

*Note: V7 two-stage ensemble for hba1c still running*

## HOMA_IR — Demographics + Wearables Only

| Version | Method | R² (mean ± std) | Notes |
|---------|--------|-----------------|-------|
| V6 | WideShallow 1024 log | **0.177 ± 0.057** | Regular CV |
| V6 | AE_Regressor log | 0.228 ± 0.029 | Regular CV |
| V6 | WideShallow 1024 quantile | 0.180 ± 0.045 | Regular CV |

**Goal: R² ≥ 0.70** — Gap: ~0.47

*V7 DW experiments queued, will run after hba1c ALL completes*

## hba1c — Demographics + Wearables Only

| Version | Method | R² (mean ± std) | Notes |
|---------|--------|-----------------|-------|
| V6 | (not yet completed) | TBD | |

**Goal: R² ≥ 0.70**

*V7 DW experiments queued*

---

## Key Findings

### Why R² = 0.85 is extremely difficult for this dataset:

1. **Missing key variable**: HOMA_IR = fasting_insulin × glucose / 405. Fasting insulin is NOT in the dataset. We're predicting a formula where one of two inputs is missing.

2. **Severe class imbalance in regression**: 56% of HOMA_IR values are in 0-2 range, only 9.3% above 5. The model compresses predictions toward the mean.

3. **Prediction compression**: Model pred std is only 70% of true std. Top 10% true values (mean=7.93) are predicted as 5.23 (34% under-prediction).

4. **Irreducible error from normal-glucose/high-insulin cases**: The 32 samples with HOMA_IR > 8 have normal glucose (80-103) but extremely high insulin. These are fundamentally unpredictable without insulin data.

5. **Small dataset**: Only 798 samples with 96+ features leads to overfitting.

### Demographics + Wearables Only:
- Without blood biomarkers (especially glucose, triglycerides), there is very little signal for insulin resistance
- BMI alone is the strongest wearable-adjacent predictor (r≈0.35 with HOMA_IR)
- Wearable features (HR, HRV, steps, sleep) have weak individual correlations (r < 0.15)

### Error Analysis Highlights:
- **Obese** patients easier to predict (R²=0.47) than **normal BMI** (R²=0.33)
- **Female** patients easier (R²=0.61) than **male** (R²=0.48)
- **Diabetic glucose range** hardest (R²=0.23) — most variance in insulin levels
- See `analysis_True_HOMA_IR.png` and `analysis_True_hba1c.png` for t-SNE plots
