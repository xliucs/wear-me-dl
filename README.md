# WEAR-ME: Deep Learning for Insulin Resistance & HbA1c Prediction

Predicting **True_HOMA_IR** and **True_hba1c** from anonymized wearable (Fitbit) and clinical data using PyTorch deep learning models.

## 🎯 Goals

| Target | Features | R² Goal | Best R² So Far | Status |
|--------|----------|---------|----------------|--------|
| True_HOMA_IR | All (demographics + wearables + blood) | 0.85 | 0.555 | 🔄 In Progress |
| True_HOMA_IR | Demographics + Wearables only | 0.70 | TBD | 🔄 In Progress |
| True_hba1c | All (demographics + wearables + blood) | 0.85 | ~0.45 | 🔄 In Progress |
| True_hba1c | Demographics + Wearables only | 0.70 | TBD | 🔄 In Progress |

## 📊 Dataset

- **Source:** Anonymized WEAR-ME study data
- **Samples:** 798 participants
- **Feature Groups:**
  - **Demographics (3):** age, sex, BMI
  - **Fitbit Wearable (15):** Resting HR, HRV, steps, sleep duration, active zone minutes (mean/median/std)
  - **Blood Biomarkers (46):** Lipid panel, CBC, metabolic panel, inflammation markers
  - **Engineered Features (30+):** TyG index, TyG-BMI, trig/HDL ratio, metabolic syndrome score, etc.
- **Labels:**
  - `True_HOMA_IR` — Homeostatic Model Assessment for Insulin Resistance (continuous)
  - `True_hba1c` — Glycated hemoglobin (continuous)
  - Classification labels used as auxiliary tasks: IR class, diabetes class

## 🏗️ Architecture & Approach

### Models Implemented (PyTorch)

1. **ResNet MLP** — Deep residual MLP with LayerNorm + GELU activation
   - Skip connections prevent gradient degradation
   - Best single-model performance so far
   
2. **FT-Transformer** — Feature Tokenizer + Transformer encoder
   - Each feature tokenized into learned embedding
   - CLS token for prediction
   - State-of-the-art for tabular data (Gorishniy et al., 2021)

3. **Deep Cross Network v2 (DCNv2)** — Explicit feature crossing + deep network
   - Cross layers learn bounded-degree feature interactions
   - Combined with deep MLP branch

4. **Self-Normalizing Neural Network (SNN)** — SELU activation + AlphaDropout
   - Automatically maintains normalized activations
   - Good for small datasets

5. **Multi-Task ResNet** — Shared backbone with auxiliary classification heads
   - Auxiliary IR classification + diabetes classification
   - Extra supervision signal improves representation learning
   - Classification weight decays during training

6. **Wide Shallow Networks** — 1024-2048 hidden units, 2 layers
   - Heavy dropout regularization

7. **Gradient-Boosted Neural Networks** — Sequential residual learning
   - Train small NNs on residuals (neural XGBoost)

8. **KNN-Augmented Networks** — KNN predictions as extra features
   - Combines local (KNN) and global (NN) patterns

9. **Autoencoder + Regressor** — Denoising AE pretraining + regression finetuning
   - Learn robust representations first

### Training Strategy

- **5-Fold Cross Validation** with fixed seed (42) for reproducibility
- **Target Transforms:** log1p (for right-skewed HOMA_IR), quantile normalization
- **Feature Scaling:** StandardScaler per fold
- **Feature Selection:** Mutual Information Regression (top 25-60 features)
- **Data Augmentation:** Mixup (α=0.2) applied 50% of batches
- **Loss Function:** Huber loss (δ=1.0) for robustness to outliers
- **Optimizer:** AdamW with cosine annealing + warmup (15 epochs)
- **Early Stopping:** patience=40-60 epochs
- **Deep Ensembling:** Multiple seeds × multiple architectures
- **Hardware:** Apple Mac Mini M-series (MPS + CPU)

### Feature Engineering (Domain Knowledge)

Key engineered features based on clinical literature:
- **TyG Index** = log(triglycerides × glucose / 2) — validated HOMA-IR proxy
- **TyG-BMI** = TyG × BMI — enhanced insulin resistance proxy
- **Triglyceride/HDL ratio** — metabolic syndrome indicator
- **Neutrophil-Lymphocyte Ratio (NLR)** — inflammation marker
- **Metabolic Syndrome Score** — composite of BMI>30, TG>150, glucose>100, HDL<40
- **Wearable variability features** — CV of HR, HRV, steps, sleep

## 📁 Project Structure

```
insulin_resistance/
├── README.md                  # This file
├── data.csv                   # Dataset (not tracked in git)
├── explore_data.py            # Data exploration & analysis
├── train.py                   # V1: Broad architecture search
├── train_v2.py                # V2: Feature engineering + multi-arch
├── train_v3.py                # V3: CPU-optimized lean training
├── train_lean.py              # V3.5: Memory-efficient pipeline
├── train_fast.py              # V4: Focused fast experiments
├── train_v4.py                # V4: MI selection + multi-task learning
├── train_v5.py                # V5: TabPFN + stacking (MPS)
├── train_v6_boosted.py        # V6: Boosted NN + wide nets + KNN
├── quick_baseline.py          # Sklearn baselines (Ridge, GBR, RF)
├── results_*.json             # CV results from each version
└── .gitignore
```

## 🔬 Key Findings So Far

### Data Ceiling Analysis
- **HOMA_IR** = fasting_insulin × fasting_glucose / 405
  - Fasting insulin is NOT in the dataset — this is the fundamental bottleneck
  - Glucose alone correlates r=0.56 with HOMA_IR
  - Best proxy: TyG-BMI index (TyG × BMI)
  - Sklearn HistGBR baseline: R² ≈ 0.61 (log target)
  
- **hba1c** (glycated hemoglobin)
  - Glucose correlates r=0.60 with hba1c
  - Age correlates r=0.33 (age-related glycation)
  - RDW has unexpected correlation r=0.19

### Model Comparison (HOMA_IR, all features, 5-fold CV)
| Model | Transform | R² |
|-------|-----------|-----|
| ResNet 256×6 | log | 0.555 ± 0.097 |
| Multi-Task ResNet 128×4 | log | 0.533 ± 0.081 |
| Ensemble (40 feat, MT+SNN) | log | 0.540 ± 0.104 |
| Wide Shallow 1024 | log | 0.397 ± 0.146 |
| SNN 256×6 | log | 0.430 ± 0.153 |
| HistGBR (sklearn baseline) | log | 0.613 ± 0.068 |

### Observations
1. ResNet MLP with log target transform is the best DL architecture
2. Multi-task auxiliary classification provides modest improvement
3. Feature selection (MI) helps reduce overfitting (40 > 96 features)
4. Deep ensembling improves stability but not R² ceiling
5. Wide shallow networks underperform deep residual ones
6. MPS (Apple GPU) shows training instability — CPU more reliable

## 🚀 Running

```bash
# Install dependencies
pip install torch pandas scikit-learn numpy matplotlib

# Explore data
python explore_data.py

# Run latest training (V6: boosted + wide + KNN)
python train_v6_boosted.py

# Run focused training (V4: multi-task + MI selection)
python train_v4.py

# Quick sklearn baselines
python quick_baseline.py
```

## 📚 References

- Gorishniy et al. (2021) "Revisiting Deep Learning Models for Tabular Data" (FT-Transformer)
- Wang et al. (2021) "DCN V2: Improved Deep & Cross Network" 
- Klambauer et al. (2017) "Self-Normalizing Neural Networks"
- Simental-Mendía et al. (2008) "TyG index as insulin resistance marker"
- Arik & Pfister (2021) "TabNet: Attentive Interpretable Tabular Learning"

## ⏱️ Timeline

- **V1-V3:** Architecture exploration (ResNet, FT-Transformer, DCN, SNN)
- **V4:** Multi-task learning with auxiliary classification
- **V5:** TabPFN + stacking (blocked by HF auth)
- **V6:** Boosted neural nets + KNN augmentation + autoencoder pretraining
- **Next:** Hyperparameter Bayesian optimization, neural architecture search
