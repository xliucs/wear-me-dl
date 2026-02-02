# WEAR-ME Deep Learning Results

## Best Results (5-fold Stratified CV, OOF R²)

| Target | Features | Best R² | Method | Target R² | Gap |
|--------|----------|---------|--------|-----------|-----|
| HOMA_IR | ALL (top35) | 0.587 | XGB ensemble blend (MAE+log+tree) | 0.65 | 0.063 |
| HOMA_IR | DW only | 0.262 | TabPFN + Ridge blend | 0.37 | 0.108 |
| hba1c | ALL | 0.484 | TabPFN (n_est=16) | 0.85 | 0.366 |
| hba1c | DW only | 0.165 | TabPFN + Ridge blend | 0.70 | 0.535 |

## Key Findings

### 1. Feature Selection is the #1 Lever
- Top 35 features (by XGB importance) outperform all 87 features
- Reducing from 87 → 35 features improves R² by ~0.02-0.03
- Top features for HOMA_IR: glucose_bmi, mets_ir_bmi, glucose_proxy, glucose, tyg

### 2. Overfitting is Extreme
- Train R² = 1.0 for all tree models, Test R² ≈ 0.55
- Gap = 0.45 even with regularized XGBoost
- Early stopping helps marginally (50 rounds patience)
- 798 samples × 87 features = severe curse of dimensionality

### 3. Prediction Compression is the Core Issue
- Model pred std = 1.53 vs True std = 2.21 (compression = 0.69)
- HOMA_IR 8+ (n=32): predicted mean 6.15, true mean 10.69
- Residuals correlate r=0.43 with |glucose_bmi| — heteroscedastic errors
- Residuals are NOT predictable from features (R² = -0.41)

### 4. DW Ceiling is Low Without Time-Series
- BMI alone: r=0.35 with HOMA_IR → r² = 0.12
- Best DW model: R² = 0.26 (TabPFN + Ridge blend)
- Paper's 0.37 used masked autoencoder time-series embeddings, not summary stats
- Summary stats lose temporal patterns that encode metabolic health

### 5. Model Rankings
- XGBoost with MAE/log target: consistently best single models
- TabPFN v2.5: excellent on DW (less prone to overfitting)
- Ridge/Lasso: competitive on DW (strong regularization helps)
- Random Forest: good baseline, implicit regularization
- Neural networks: generally underperform tree models on this dataset size

## Approaches Tried

1. **V1-V6**: Various NN architectures (ResNet MLP, SNN, DCN, FT-Transformer, Multi-task)
2. **V7**: Informed training with loss weighting, two-stage ensemble, quantile transforms
3. **V8**: Stacking framework (GBR base → NN meta-learner)
4. **V9**: XGBoost with early stopping + Ridge stacking + autoencoder for DW
5. **V10**: Repeated stratified CV (3×5-fold) for variance reduction

## Dataset Notes
- N = 798 (subset of full WEAR-ME N=1165)
- Published paper best: R² = 0.50 (on N=1165 with time-series embeddings)
- Xin's internal baseline: R² = 0.65 (ALL), R² = 0.37 (DW) on full N=1165
