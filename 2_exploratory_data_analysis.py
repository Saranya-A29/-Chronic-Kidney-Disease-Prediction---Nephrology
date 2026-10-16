# Exploratory Data Analysis (EDA) & Publication-Ready Visualizations
# Student Name: SARANYA A
# Project: Chronic Kidney Disease Prediction (Nephrology)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set overall style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

def run_eda(data_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    print("=" * 65)
    print("  EXPLORATORY DATA ANALYSIS - CHRONIC KIDNEY DISEASE")
    print("  Student Name: SARANYA A | Nephrology Domain")
    print("=" * 65)
    print(f"\n[+] Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\n[+] Column Data Types & Missing Counts:")
    print(df.isna().sum())
    
    # -------------------------------------------------------------
    # 1. Target Class Distribution (CKD vs Non-CKD)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    palette = {'ckd': '#d9534f', 'notckd': '#2ecc71'}
    
    # Bar Chart
    sns.countplot(x='classification', hue='classification', data=df, ax=axes[0], palette=palette, edgecolor='black', linewidth=1, legend=False)
    axes[0].set_title("Target Class Distribution (Counts)", fontweight='bold')
    axes[0].set_xlabel("Diagnosis Category")
    axes[0].set_ylabel("Patient Count")
    for p in axes[0].patches:
        height = p.get_height()
        axes[0].annotate(f"{int(height)}", (p.get_x() + p.get_width() / 2., height / 2),
                         ha='center', va='center', fontsize=12, color='white', fontweight='bold')
    
    # Pie Chart
    counts = df['classification'].value_counts()
    axes[1].pie(counts, labels=[f"CKD Positive\n({counts['ckd']})", f"Non-CKD / Healthy\n({counts['notckd']})"],
                autopct='%1.1f%%', colors=['#d9534f', '#2ecc71'], startangle=140,
                explode=(0.05, 0), textprops={'fontsize': 11, 'fontweight': 'bold'})
    axes[1].set_title("Target Class Proportion (%)", fontweight='bold')
    
    plt.suptitle("Saranya A - Nephrology Project: CKD vs Healthy Cohort Distribution", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    target_dist_path = os.path.join(output_dir, 'target_distribution.png')
    plt.savefig(target_dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved: {target_dist_path}")

    # -------------------------------------------------------------
    # 2. Correlation Heatmap of Numerical Features
    # -------------------------------------------------------------
    num_cols = df.select_dtypes(include=[np.number]).columns
    # Add target encoded as binary for correlation analysis
    df_corr = df[num_cols].copy()
    df_corr['ckd_target'] = (df['classification'] == 'ckd').astype(int)

    plt.figure(figsize=(14, 11))
    corr_matrix = df_corr.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                annot=True, fmt='.2f', square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Pearson Correlation Matrix: Clinical Biomarkers vs CKD Diagnosis\n(Saranya A - Nephrology Dataset)",
              fontsize=14, fontweight='bold', pad=15)
    heatmap_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved: {heatmap_path}")

    # -------------------------------------------------------------
    # 3. Serum Creatinine vs Blood Urea (Nephrology Keystone Markers)
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 6))
    scatter = sns.scatterplot(
        data=df, x='blood_urea', y='serum_creatinine',
        hue='classification', style='classification',
        palette=palette, alpha=0.85, s=70, edgecolor='black', linewidth=0.5
    )
    plt.axvline(x=45, color='gray', linestyle='--', label='Normal Blood Urea Threshold (45 mg/dL)')
    plt.axhline(y=1.2, color='crimson', linestyle='--', label='Normal Creatinine Threshold (1.2 mg/dL)')
    plt.title("Nephrology Biomarker Scatter: Serum Creatinine vs. Blood Urea", fontsize=13, fontweight='bold')
    plt.xlabel("Blood Urea (mg/dL)")
    plt.ylabel("Serum Creatinine (mg/dL)")
    plt.legend(frameon=True)
    urea_creat_path = os.path.join(output_dir, 'creatinine_vs_urea.png')
    plt.tight_layout()
    plt.savefig(urea_creat_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved: {urea_creat_path}")

    # -------------------------------------------------------------
    # 4. Hemoglobin vs Albuminuria (Glomerular Breakdown & Anemia)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Hemoglobin distribution by CKD
    sns.kdeplot(data=df[df['classification'] == 'ckd']['hemoglobin'], ax=axes[0],
                label='CKD Positive', fill=True, color='#d9534f', alpha=0.4, linewidth=2)
    sns.kdeplot(data=df[df['classification'] == 'notckd']['hemoglobin'], ax=axes[0],
                label='Healthy (Not CKD)', fill=True, color='#2ecc71', alpha=0.4, linewidth=2)
    axes[0].axvline(12.0, color='darkorange', linestyle=':', label='Anemia Threshold (12 g/dL)')
    axes[0].set_title("Hemoglobin Concentration Density (g/dL)", fontweight='bold')
    axes[0].set_xlabel("Hemoglobin (g/dL)")
    axes[0].legend()
    
    # Albumin level impact on Hemoglobin
    sns.boxplot(x='albumin', y='hemoglobin', hue='classification', data=df,
                palette=palette, ax=axes[1], boxprops=dict(alpha=0.8))
    axes[1].set_title("Hemoglobin Levels across Urine Albuminuria Grades (0 to 4)", fontweight='bold')
    axes[1].set_xlabel("Urine Albumin (Protein Leakage Grade)")
    axes[1].set_ylabel("Hemoglobin (g/dL)")
    
    plt.suptitle("Erythropoietin Loss & Glomerular Damage Correlates in Nephrology", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    hemo_alb_path = os.path.join(output_dir, 'hemoglobin_vs_albumin.png')
    plt.savefig(hemo_alb_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved: {hemo_alb_path}")

    # -------------------------------------------------------------
    # 5. Comorbidity Breakdown (Hypertension, Diabetes, Edema, Anemia)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    comorbidities = ['hypertension', 'diabetes_mellitus', 'pedal_edema', 'anemia']
    titles = [
        'Hypertension Prevalence',
        'Diabetes Mellitus Prevalence',
        'Pedal Edema (Fluid Retention)',
        'Clinical Anemia Prevalence'
    ]

    for idx, (col, title) in enumerate(zip(comorbidities, titles)):
        r, c = idx // 2, idx % 2
        sns.countplot(x=col, hue='classification', data=df, palette=palette, ax=axes[r, c], edgecolor='black')
        axes[r, c].set_title(title, fontweight='bold')
        axes[r, c].set_xlabel(f"{col.replace('_', ' ').title()} Present?")
        axes[r, c].set_ylabel("Patient Count")
        axes[r, c].legend(title="Diagnosis", loc='upper right')

    plt.suptitle("Clinical Comorbidities & Secondary Symptom Prevalence in CKD Cohort", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    comorbid_path = os.path.join(output_dir, 'comorbidities_ckd_risk.png')
    plt.savefig(comorbid_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved: {comorbid_path}")

    print("\n[+] All EDA Visualizations successfully generated and stored in:", output_dir)

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(base_dir, 'data', 'chronic_kidney_disease.csv')
    viz_dir = os.path.join(base_dir, 'visualizations')
    run_eda(data_file, viz_dir)
