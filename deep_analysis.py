#!/usr/bin/env python3
"""
Deep sub-analysis of model performance.
Informed by WEAR-ME paper findings and error analysis.

Goals:
1. Compare our performance vs paper (R²=0.50) on same feature subsets
2. Analyze METS-IR as a standalone predictor
3. Check impact of outlier capping at HOMA-IR=15
4. Feature importance via permutation
5. Subgroup analysis matching paper's stratifications
6. Information-theoretic analysis (mutual information ceiling)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

def load_and_engineer():
    df = pd.read_csv('data.csv', skiprows=[0])
    df['sex_num'] = (df['sex'] == 'Male').astype(float)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    
    # All engineered features
    df['trig_hdl'] = df['triglycerides'] / (df['hdl'] + 0.1)
    df['tyg'] = np.log(df['triglycerides'] * df['glucose'] / 2 + 1)
    df['tyg_bmi'] = df['tyg'] * df['bmi']
    df['glucose_bmi'] = df['glucose'] * df['bmi']
    df['glucose_sq'] = df['glucose'] ** 2 / 1000
    df['bmi_sq'] = df['bmi'] ** 2
    df['log_trig'] = np.log1p(df['triglycerides'])
    df['log_glucose'] = np.log1p(df['glucose'])
    df['glucose_proxy'] = df['glucose'] * df['trig_hdl']
    df['ldl_hdl'] = df['ldl'] / (df['hdl'] + 0.1)
    df['ast_alt'] = df['ast'] / (df['alt'] + 0.1)
    df['rhr_hrv'] = df['Resting Heart Rate (mean)'] / (df['HRV (mean)'] + 0.1)
    df['bmi_age'] = df['bmi'] * df['age']
    df['glucose_trig'] = df['glucose'] * df['triglycerides'] / 1000
    df['glucose_hdl'] = df['glucose'] / (df['hdl'] + 0.1)
    df['non_hdl_hdl'] = df['non hdl'] / (df['hdl'] + 0.1)
    df['log_crp'] = np.log1p(df['crp'])
    df['nlr'] = df['absolute_neutrophils'] / (df['absolute_lymphocytes'] + 0.1)
    df['bun_creat'] = df['bun'] / (df['creatinine'] + 0.01)
    df['ggt_alt'] = df['ggt'] / (df['alt'] + 0.1)
    df['met_score'] = ((df['bmi']>30).astype(float) + (df['triglycerides']>150).astype(float) + 
                       (df['glucose']>100).astype(float) + (df['hdl']<40).astype(float))
    df['insulin_proxy'] = df['bmi'] * df['trig_hdl']
    df['insulin_proxy2'] = df['bmi_sq'] * df['log_trig'] / 100
    df['liver_stress'] = df['alt'] * df['ggt'] / 100
    df['bmi_cubed'] = df['bmi'] ** 3 / 10000
    
    # H2: METS-IR (non-insulin IR surrogate)
    df['mets_ir'] = np.log((2 * df['glucose'] + df['triglycerides']) / (df['hdl'] + 0.1))
    df['mets_ir_bmi'] = df['mets_ir'] * df['bmi']
    df['vat_proxy'] = df['bmi'] * df['triglycerides'] / (df['hdl'] + 0.1)
    df['atherogenic_idx'] = np.log10(df['triglycerides'] / (df['hdl'] + 0.1))
    
    mask = df[['True_HOMA_IR','True_hba1c']].notna().all(axis=1)
    df = df[mask].reset_index(drop=True)
    
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(df[col].median())
    
    return df


def run_gbr_cv(X, y, label=""):
    """Quick HistGBR cross-validation."""
    y_bins = pd.qcut(y, q=5, labels=False, duplicates='drop')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    preds = np.zeros(len(y))
    for fold, (ti, vi) in enumerate(skf.split(X, y_bins)):
        m = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=5,
                                          min_samples_leaf=10, l2_regularization=0.1, random_state=42)
        m.fit(X[ti], y[ti])
        preds[vi] = m.predict(X[vi])
    
    r2 = r2_score(y, preds)
    mae = mean_absolute_error(y, preds)
    print(f"  {label:45s}: R²={r2:.4f}, MAE={mae:.3f}")
    return r2, mae, preds


if __name__ == '__main__':
    df = load_and_engineer()
    print(f"Loaded {len(df)} samples")
    
    # ================================================================
    # 1. FEATURE SET ABLATION (matching paper's experiments)
    # ================================================================
    print("\n" + "="*70)
    print("1. FEATURE SET ABLATION (GBR baseline, matching paper)")
    print("="*70)
    
    demo = ['age', 'bmi', 'sex_num']
    wearables = ['Resting Heart Rate (mean)', 'Resting Heart Rate (median)', 'Resting Heart Rate (std)',
                 'HRV (mean)', 'HRV (median)', 'HRV (std)', 'STEPS (mean)', 'STEPS (median)', 'STEPS (std)',
                 'SLEEP Duration (mean)', 'SLEEP Duration (median)', 'SLEEP Duration (std)',
                 'AZM Weekly (mean)', 'AZM Weekly (median)', 'AZM Weekly (std)']
    glucose_only = ['glucose']
    lipid = ['triglycerides', 'hdl', 'ldl', 'non hdl', 'total_cholesterol']
    metabolic = ['glucose', 'bun', 'creatinine', 'calcium', 'sodium', 'potassium', 'chloride', 
                 'carbon_dioxide', 'total_protein', 'albumin', 'globulin', 'ag_ratio', 
                 'total_bilirubin', 'alkaline_phosphatase', 'ast', 'alt']
    hba1c_feat = ['hba1c']
    cbc = ['white_blood_cell', 'rbc', 'hemoglobin', 'hematocrit', 'mcv', 'mch', 'mchc', 'rdw',
           'platelet_count', 'absolute_neutrophils', 'absolute_lymphocytes', 'absolute_monocytes',
           'absolute_eosinophils', 'absolute_basophils']
    other_blood = ['ggt', 'crp', 'uric_acid', 'testosterone']
    
    y = df['True_HOMA_IR'].values
    
    # Paper's Experiment configs
    feature_sets = {
        # Minimal model (paper: demographics + wearables)
        'Demographics only': demo,
        'Wearables only': wearables,
        'Demographics + Wearables (DW)': demo + wearables,
        # Moderate model (paper: DW + glucose)
        'DW + Glucose': demo + wearables + glucose_only,
        # Optimal model (paper: DW + lipid + metabolic)
        'DW + Lipid panel': demo + wearables + lipid,
        'DW + Metabolic panel': demo + wearables + metabolic,
        'DW + Lipid + Metabolic (Optimal)': demo + wearables + lipid + metabolic,
        # Extended
        'DW + All blood': demo + wearables + lipid + metabolic + hba1c_feat + cbc + other_blood,
        # With engineering
        'ALL features (engineered)': None,  # use all
    }
    
    print("\n  HOMA_IR:")
    for name, cols in feature_sets.items():
        if cols is None:
            label_cols = ['True_hba1c','True_HOMA_IR','True_IR_Class','True_Diabetes_2_Class',
                         'True_Normoglycemic_2_Class','True_Diabetes_3_Class']
            cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                    if c not in label_cols + ['Participant_id']]
        valid_cols = [c for c in cols if c in df.columns]
        X = df[valid_cols].values
        run_gbr_cv(X, y, f"{name} ({len(valid_cols)} feat)")
    
    print("\n  hba1c:")
    y_hba = df['True_hba1c'].values
    for name, cols in feature_sets.items():
        if cols is None:
            label_cols = ['True_hba1c','True_HOMA_IR','True_IR_Class','True_Diabetes_2_Class',
                         'True_Normoglycemic_2_Class','True_Diabetes_3_Class']
            cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                    if c not in label_cols + ['Participant_id']]
        valid_cols = [c for c in cols if c in df.columns]
        X = df[valid_cols].values
        run_gbr_cv(X, y_hba, f"{name} ({len(valid_cols)} feat)")
    
    # ================================================================
    # 2. H3: IMPACT OF OUTLIER CAPPING
    # ================================================================
    print("\n" + "="*70)
    print("2. IMPACT OF OUTLIER CAPPING (HOMA-IR)")
    print("="*70)
    
    label_cols = ['True_hba1c','True_HOMA_IR','True_IR_Class','True_Diabetes_2_Class',
                 'True_Normoglycemic_2_Class','True_Diabetes_3_Class']
    all_feat = [c for c in df.select_dtypes(include=[np.number]).columns 
                if c not in label_cols + ['Participant_id']]
    X = df[all_feat].values
    
    for cap in [None, 20, 15, 12, 10]:
        y_capped = df['True_HOMA_IR'].values.copy()
        if cap:
            n_above = (y_capped >= cap).sum()
            y_capped = np.clip(y_capped, None, cap - 0.01)
            name = f"Cap at {cap} (removed {n_above} outliers)"
        else:
            name = "No capping (original)"
        run_gbr_cv(X, y_capped, name)
    
    # ================================================================
    # 3. METS-IR AS STANDALONE PREDICTOR
    # ================================================================
    print("\n" + "="*70)
    print("3. METS-IR AND OTHER SURROGATES VS HOMA-IR")
    print("="*70)
    
    y = df['True_HOMA_IR'].values
    
    surrogates = {
        'METS-IR only': ['mets_ir'],
        'METS-IR + BMI': ['mets_ir', 'bmi'],
        'TyG index': ['tyg'],
        'TyG + BMI': ['tyg', 'bmi'],
        'Trig/HDL ratio': ['trig_hdl'],
        'Insulin proxy (BMI × Trig/HDL)': ['insulin_proxy'],
        'VAT proxy (BMI × Trig/HDL)': ['vat_proxy'],
        'Atherogenic index': ['atherogenic_idx'],
        'Top 5 surrogate combo': ['mets_ir', 'tyg_bmi', 'insulin_proxy', 'vat_proxy', 'atherogenic_idx'],
        'Surrogates + demographics': ['mets_ir', 'tyg_bmi', 'insulin_proxy', 'vat_proxy', 
                                       'atherogenic_idx', 'age', 'bmi', 'sex_num'],
    }
    
    for name, cols in surrogates.items():
        valid_cols = [c for c in cols if c in df.columns]
        X = df[valid_cols].values
        run_gbr_cv(X, y, f"{name} ({len(valid_cols)} feat)")
    
    # ================================================================
    # 4. MUTUAL INFORMATION ANALYSIS
    # ================================================================
    print("\n" + "="*70)
    print("4. MUTUAL INFORMATION WITH HOMA-IR (top 20 features)")
    print("="*70)
    
    all_feat_valid = [c for c in all_feat if c in df.columns]
    X = df[all_feat_valid].values
    y = df['True_HOMA_IR'].values
    
    mi = mutual_info_regression(X, y, random_state=42, n_neighbors=5)
    mi_df = pd.DataFrame({'feature': all_feat_valid, 'MI': mi}).sort_values('MI', ascending=False)
    
    print(f"  {'Feature':40s} {'MI':>8s}")
    print("  " + "-"*50)
    for _, row in mi_df.head(20).iterrows():
        print(f"  {row['feature']:40s} {row['MI']:8.4f}")
    
    # ================================================================
    # 5. SUBGROUP ANALYSIS (matching paper's Table 3/Figure 5)
    # ================================================================
    print("\n" + "="*70)
    print("5. SUBGROUP ANALYSIS")
    print("="*70)
    
    # Get GBR predictions for subgroup analysis
    X = df[all_feat_valid].values
    y = df['True_HOMA_IR'].values
    _, _, preds = run_gbr_cv(X, y, "Full model (for subgroup analysis)")
    
    # IR categories (matching paper)
    df['pred_homa'] = preds
    df['ir_class'] = pd.cut(df['True_HOMA_IR'], bins=[0, 1.5, 2.9, 100], 
                            labels=['IS', 'Impaired', 'IR'])
    
    print("\n  By IR class:")
    for cls in ['IS', 'Impaired', 'IR']:
        mask = df['ir_class'] == cls
        if mask.sum() > 5:
            r2 = r2_score(df.loc[mask, 'True_HOMA_IR'], df.loc[mask, 'pred_homa'])
            mae = mean_absolute_error(df.loc[mask, 'True_HOMA_IR'], df.loc[mask, 'pred_homa'])
            print(f"    {cls:12s}: n={mask.sum():4d}, R²={r2:.4f}, MAE={mae:.3f}")
    
    print("\n  By BMI category:")
    bmi_cats = {'Underweight (<18.5)': (0, 18.5), 'Normal (18.5-25)': (18.5, 25), 
                'Overweight (25-30)': (25, 30), 'Obese (30+)': (30, 100)}
    for name, (lo, hi) in bmi_cats.items():
        mask = (df['bmi'] >= lo) & (df['bmi'] < hi)
        if mask.sum() > 5:
            r2 = r2_score(df.loc[mask, 'True_HOMA_IR'], df.loc[mask, 'pred_homa'])
            mae = mean_absolute_error(df.loc[mask, 'True_HOMA_IR'], df.loc[mask, 'pred_homa'])
            print(f"    {name:25s}: n={mask.sum():4d}, R²={r2:.4f}, MAE={mae:.3f}")
    
    print("\n  By sex:")
    for sex in ['Male', 'Female']:
        mask = df['sex'] == sex
        if mask.sum() > 5:
            r2 = r2_score(df.loc[mask, 'True_HOMA_IR'], df.loc[mask, 'pred_homa'])
            mae = mean_absolute_error(df.loc[mask, 'True_HOMA_IR'], df.loc[mask, 'pred_homa'])
            print(f"    {sex:12s}: n={mask.sum():4d}, R²={r2:.4f}, MAE={mae:.3f}")
    
    print("\n  By age group:")
    age_cats = {'Young (21-35)': (21, 35), 'Middle (35-50)': (35, 50), 'Older (50+)': (50, 100)}
    for name, (lo, hi) in age_cats.items():
        mask = (df['age'] >= lo) & (df['age'] < hi)
        if mask.sum() > 5:
            r2 = r2_score(df.loc[mask, 'True_HOMA_IR'], df.loc[mask, 'pred_homa'])
            mae = mean_absolute_error(df.loc[mask, 'True_HOMA_IR'], df.loc[mask, 'pred_homa'])
            print(f"    {name:25s}: n={mask.sum():4d}, R²={r2:.4f}, MAE={mae:.3f}")
    
    # Classification performance at threshold 2.9 (matching paper)
    print("\n  Classification at HOMA-IR > 2.9 threshold:")
    true_ir = (df['True_HOMA_IR'] > 2.9).astype(int)
    pred_ir = (preds > 2.9).astype(int)
    
    tp = ((true_ir == 1) & (pred_ir == 1)).sum()
    tn = ((true_ir == 0) & (pred_ir == 0)).sum()
    fp = ((true_ir == 0) & (pred_ir == 1)).sum()
    fn = ((true_ir == 1) & (pred_ir == 0)).sum()
    
    sensitivity = tp / (tp + fn + 1e-10)
    specificity = tn / (tn + fp + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    
    from sklearn.metrics import roc_auc_score
    auroc = roc_auc_score(true_ir, preds)
    
    print(f"    Sensitivity: {sensitivity:.3f}")
    print(f"    Specificity: {specificity:.3f}")
    print(f"    Precision:   {precision:.3f}")
    print(f"    AUROC:       {auroc:.3f}")
    print(f"    (Paper:      Sens=0.76, Spec=0.84, AUROC=0.80)")
    
    print("\n  For obese + sedentary subgroup:")
    obese_sed = (df['bmi'] >= 30) & (df['STEPS (mean)'] < 5000)
    if obese_sed.sum() > 5:
        true_ir_sub = (df.loc[obese_sed, 'True_HOMA_IR'] > 2.9).astype(int)
        pred_ir_sub = (preds[obese_sed] > 2.9).astype(int)
        tp = ((true_ir_sub == 1) & (pred_ir_sub == 1)).sum()
        tn = ((true_ir_sub == 0) & (pred_ir_sub == 0)).sum()
        fp = ((true_ir_sub == 0) & (pred_ir_sub == 1)).sum()
        fn = ((true_ir_sub == 1) & (pred_ir_sub == 0)).sum()
        sens = tp / (tp + fn + 1e-10)
        spec = tn / (tn + fp + 1e-10)
        print(f"    n={obese_sed.sum()}, Sensitivity={sens:.3f}, Specificity={spec:.3f}")
        print(f"    (Paper: Sens=0.93, Spec=0.95 for obese+sedentary)")
    
    print("\nDone!")
