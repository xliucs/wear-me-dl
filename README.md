# WEAR-ME: Deep Learning for Insulin Resistance & HbA1c Prediction

Predicting **True_HOMA_IR** and **True_hba1c** from anonymized wearable (Fitbit) and clinical data.

## Best Results (Stratified 5-Fold CV, 3-5 Repeats)

| Target | Features | R² Target | **Best R²** | Method |
|--------|----------|-----------|-------------|--------|
| HOMA_IR | ALL | 0.65 | **0.5948** | HGBR_log + XGB blend |
| HOMA_IR | DW only | 0.37 | **0.2657** | Ridge stack |
| hba1c | ALL | 0.85 | **0.4916** | XGB + ExtraTrees blend |
| hba1c | DW only | 0.70 | **0.1651** | Ridge + ExtraTrees |

- **ALL** = Demographics + Wearables + Blood Biomarkers (35 selected from 83+ engineered)
- **DW** = Demographics + Wearables only (32 engineered features)

See [RESULTS.md](RESULTS.md) for detailed results, version history, and analysis.

## Dataset

- **Source:** Anonymized WEAR-ME study data
- **Samples:** 798 participants (full dataset: 1165)
- **Features:**
  - Demographics (3): age, sex, BMI
  - Fitbit Wearable (15): Resting HR, HRV, steps, sleep duration, active zone minutes (mean/median/std)
  - Blood Biomarkers (46): Lipid panel, CBC, metabolic panel, inflammation markers
  - Engineered (40+): TyG, METS-IR, glucose×BMI, insulin proxies, glycation features

## Approach

### Feature Engineering
- **Metabolic indices**: TyG, METS-IR, glucose×BMI, triglyceride/HDL ratio
- **Insulin resistance proxies**: glucose×BMI×triglycerides/HDL
- **Glycation features**: glucose/hemoglobin, glucose×RDW, glucose×MCHC
- **Wearable interactions**: RHR×BMI, steps×sleep, cardio fitness index

### Models (15+ tested)
- **Tree-based**: XGBoost, LightGBM, HistGradientBoosting, RandomForest, ExtraTrees
- **Linear**: Ridge, Lasso, ElasticNet
- **Neural**: PyTorch FeatureGatedBlock (residual MLP with feature gating)
- **Transformer**: TabPFN (pretrained tabular transformer)
- **Other**: SVR, KNN

### Ensemble Strategy
- Feature selection via GBR importance (top 30-35 features)
- Dirichlet weight search (200K-500K random blends)
- Ridge stacking on out-of-fold predictions
- Repeated stratified CV (3-5 repeats) for stable estimates

## Key Findings

1. **HGBR with log-transform** is the best single model for HOMA_IR (R²=0.589)
2. **Feature selection is the #1 lever**: top 35 beats all 83+ features
3. **Signal ceiling reached**: residuals not predictable from features (R²=-0.41)
4. **HOMA_IR = glucose × insulin / 405**: without insulin, there's a hard information ceiling
5. **hba1c = 3-month avg glucose**: single fasting glucose (r=0.605) can only explain ~37%
6. **DW targets need time-series embeddings**: summary statistics lose temporal patterns
7. **PyTorch NNs underperform trees by ~0.12 R²** at this data scale (798 samples)

## Project Structure

```
├── README.md                    # This file
├── RESULTS.md                   # Detailed results & analysis
├── data.csv                     # Dataset (not tracked)
├── explore_data.py              # Data exploration
├── quick_baseline.py            # Sklearn baselines
├── analyze_errors.py            # Error analysis
├── deep_analysis.py             # Deep diagnostic analysis
├── pytorch_model.py             # PyTorch FeatureGatedBlock architecture
├── train_v11_comprehensive.py   # V11: Best results (HGBR + target-specific features)
├── train_v13_lgbm.py            # V13: LightGBM + comprehensive stacking
├── train_v14_megablend.py       # V14: 5-repeat mega-blend optimization
├── train_v15_augment.py         # V15: SMOTE + self-training + PyTorch
├── train_v[1-8]*.py             # Earlier versions (architecture exploration)
└── .gitignore
```

## Running

```bash
pip install torch pandas scikit-learn xgboost lightgbm tabpfn numpy

# Best results (V11)
python train_v11_comprehensive.py

# LightGBM experiments
python train_v13_lgbm.py

# PyTorch model
python pytorch_model.py

# Quick baselines
python quick_baseline.py
```

## Requirements
- Python 3.10+
- PyTorch 2.0+
- scikit-learn, XGBoost, LightGBM
- pandas, numpy
- TabPFN (optional, requires HuggingFace token)
