# Chronic Kidney Disease (CKD) Dataset Generator & Verifier
# Student Name: SARANYA A
# Domain: Nephrology (Data Science & Machine Learning)

import os
import numpy as np
import pandas as pd

def generate_ckd_dataset(n_samples=150, random_state=42):
    """
    Generates a medically consistent Chronic Kidney Disease (CKD) dataset
    mirroring the Apollo Hospitals / UCI Machine Learning Repository benchmark.
    Total: 150 records
    with 24 clinical features representing nephrology indicators.
    Includes realistic clinical borderline cases (early CKD Stage 1-2 vs controlled diabetes/hypertension).
    """
    np.random.seed(random_state)
    
    n_ckd = int(n_samples * 0.625)
    n_notckd = n_samples - n_ckd
    
    # --- 1. HEALTHY / NON-CKD PATIENTS (n_notckd = 150) ---
    age_notckd = np.clip(np.random.normal(loc=43, scale=13, size=n_notckd).astype(int), 18, 80)
    bp_notckd = np.clip(np.random.choice([70, 80, 80, 80, 70, 90, 100], size=n_notckd, p=[0.2, 0.45, 0.1, 0.1, 0.05, 0.05, 0.05]), 60, 110)
    sg_notckd = np.random.choice([1.015, 1.020, 1.025], size=n_notckd, p=[0.10, 0.45, 0.45])
    al_notckd = np.random.choice([0, 1], size=n_notckd, p=[0.94, 0.06]) # rare benign transient microalbuminuria
    su_notckd = np.random.choice([0, 1], size=n_notckd, p=[0.93, 0.07])
    rbc_notckd = np.random.choice(['normal', 'abnormal'], size=n_notckd, p=[0.95, 0.05])
    pc_notckd = np.random.choice(['normal', 'abnormal'], size=n_notckd, p=[0.94, 0.06])
    pcc_notckd = np.random.choice(['notpresent', 'present'], size=n_notckd, p=[0.98, 0.02])
    ba_notckd = np.random.choice(['notpresent', 'present'], size=n_notckd, p=[0.97, 0.03])
    
    # Blood Chemistry
    bgr_notckd = np.clip(np.random.normal(loc=110, scale=22, size=n_notckd), 70, 175).round(1)
    bu_notckd = np.clip(np.random.normal(loc=32, scale=9, size=n_notckd), 15, 54).round(1)
    sc_notckd = np.clip(np.random.normal(loc=0.92, scale=0.22, size=n_notckd), 0.5, 1.35).round(2)
    sod_notckd = np.clip(np.random.normal(loc=139, scale=3.5, size=n_notckd), 132, 148).round(1)
    pot_notckd = np.clip(np.random.normal(loc=4.3, scale=0.45, size=n_notckd), 3.5, 5.2).round(2)
    
    # Hematology
    hemo_notckd = np.clip(np.random.normal(loc=14.8, scale=1.6, size=n_notckd), 11.5, 17.8).round(1)
    pcv_notckd = np.clip((hemo_notckd * 3.0 + np.random.normal(0, 2.0, size=n_notckd)).astype(int), 34, 54)
    wc_notckd = np.clip(np.random.normal(loc=7400, scale=1600, size=n_notckd).astype(int), 4200, 11500)
    rc_notckd = np.clip(np.random.normal(loc=5.1, scale=0.6, size=n_notckd), 3.8, 6.5).round(1)
    
    # Medical History & Symptoms
    htn_notckd = np.random.choice(['no', 'yes'], size=n_notckd, p=[0.85, 0.15])
    dm_notckd = np.random.choice(['no', 'yes'], size=n_notckd, p=[0.86, 0.14])
    cad_notckd = np.random.choice(['no', 'yes'], size=n_notckd, p=[0.94, 0.06])
    appet_notckd = np.random.choice(['good', 'poor'], size=n_notckd, p=[0.93, 0.07])
    pe_notckd = np.random.choice(['no', 'yes'], size=n_notckd, p=[0.95, 0.05])
    ane_notckd = np.random.choice(['no', 'yes'], size=n_notckd, p=[0.96, 0.04])
    class_notckd = np.full(n_notckd, 'notckd')

    df_notckd = pd.DataFrame({
        'age': age_notckd, 'blood_pressure': bp_notckd, 'specific_gravity': sg_notckd,
        'albumin': al_notckd, 'sugar': su_notckd, 'red_blood_cells': rbc_notckd,
        'pus_cell': pc_notckd, 'pus_cell_clumps': pcc_notckd, 'bacteria': ba_notckd,
        'blood_glucose_random': bgr_notckd, 'blood_urea': bu_notckd, 'serum_creatinine': sc_notckd,
        'sodium': sod_notckd, 'potassium': pot_notckd, 'hemoglobin': hemo_notckd,
        'packed_cell_volume': pcv_notckd, 'white_blood_cell_count': wc_notckd,
        'red_blood_cell_count': rc_notckd, 'hypertension': htn_notckd, 'diabetes_mellitus': dm_notckd,
        'coronary_artery_disease': cad_notckd, 'appetite': appet_notckd, 'pedal_edema': pe_notckd,
        'anemia': ane_notckd, 'classification': class_notckd
    })

    # --- 2. CKD POSITIVE PATIENTS (n_ckd = 250) ---
    age_ckd = np.clip(np.random.normal(loc=55, scale=14, size=n_ckd).astype(int), 20, 88)
    bp_ckd = np.clip(np.random.choice([70, 80, 90, 100, 110, 120], size=n_ckd, p=[0.12, 0.35, 0.25, 0.15, 0.08, 0.05]), 60, 130)
    sg_ckd = np.random.choice([1.005, 1.010, 1.015, 1.020], size=n_ckd, p=[0.24, 0.40, 0.26, 0.10])
    al_ckd = np.random.choice([0, 1, 2, 3, 4], size=n_ckd, p=[0.10, 0.30, 0.30, 0.18, 0.12])
    su_ckd = np.random.choice([0, 1, 2, 3, 4], size=n_ckd, p=[0.52, 0.22, 0.12, 0.08, 0.06])
    rbc_ckd = np.random.choice(['abnormal', 'normal'], size=n_ckd, p=[0.42, 0.58])
    pc_ckd = np.random.choice(['abnormal', 'normal'], size=n_ckd, p=[0.36, 0.64])
    pcc_ckd = np.random.choice(['present', 'notpresent'], size=n_ckd, p=[0.22, 0.78])
    ba_ckd = np.random.choice(['present', 'notpresent'], size=n_ckd, p=[0.16, 0.84])
    
    # Blood Chemistry in CKD (Including Stage 1-3 milder cases and Stage 4-5 severe cases)
    bgr_ckd = np.clip(np.random.normal(loc=158, scale=62, size=n_ckd), 75, 410).round(1)
    bu_ckd = np.clip(np.random.exponential(scale=48, size=n_ckd) + 32, 22, 310).round(1)
    sc_ckd = np.clip(np.random.exponential(scale=2.6, size=n_ckd) + 1.25, 1.05, 14.8).round(2)
    sod_ckd = np.clip(np.random.normal(loc=134, scale=6.2, size=n_ckd), 112, 146).round(1)
    pot_ckd = np.clip(np.random.normal(loc=4.7, scale=0.75, size=n_ckd), 3.1, 7.4).round(2)
    
    # Hematology in CKD
    hemo_ckd = np.clip(np.random.normal(loc=10.6, scale=2.3, size=n_ckd), 5.2, 14.8).round(1)
    pcv_ckd = np.clip((hemo_ckd * 3.0 + np.random.normal(0, 2.2, size=n_ckd)).astype(int), 17, 45)
    wc_ckd = np.clip(np.random.normal(loc=8900, scale=3100, size=n_ckd).astype(int), 3400, 21000)
    rc_ckd = np.clip(np.random.normal(loc=3.9, scale=0.75, size=n_ckd), 2.3, 5.4).round(1)
    
    # Medical History & Symptoms
    htn_ckd = np.random.choice(['yes', 'no'], size=n_ckd, p=[0.70, 0.30])
    dm_ckd = np.random.choice(['yes', 'no'], size=n_ckd, p=[0.56, 0.44])
    cad_ckd = np.random.choice(['yes', 'no'], size=n_ckd, p=[0.22, 0.78])
    appet_ckd = np.random.choice(['poor', 'good'], size=n_ckd, p=[0.38, 0.62])
    pe_ckd = np.random.choice(['yes', 'no'], size=n_ckd, p=[0.36, 0.64])
    ane_ckd = np.random.choice(['yes', 'no'], size=n_ckd, p=[0.32, 0.68])
    class_ckd = np.full(n_ckd, 'ckd')

    df_ckd = pd.DataFrame({
        'age': age_ckd, 'blood_pressure': bp_ckd, 'specific_gravity': sg_ckd,
        'albumin': al_ckd, 'sugar': su_ckd, 'red_blood_cells': rbc_ckd,
        'pus_cell': pc_ckd, 'pus_cell_clumps': pcc_ckd, 'bacteria': ba_ckd,
        'blood_glucose_random': bgr_ckd, 'blood_urea': bu_ckd, 'serum_creatinine': sc_ckd,
        'sodium': sod_ckd, 'potassium': pot_ckd, 'hemoglobin': hemo_ckd,
        'packed_cell_volume': pcv_ckd, 'white_blood_cell_count': wc_ckd,
        'red_blood_cell_count': rc_ckd, 'hypertension': htn_ckd, 'diabetes_mellitus': dm_ckd,
        'coronary_artery_disease': cad_ckd, 'appetite': appet_ckd, 'pedal_edema': pe_ckd,
        'anemia': ane_ckd, 'classification': class_ckd
    })

    # Concatenate and shuffle
    full_df = pd.concat([df_notckd, df_ckd], ignore_index=True)
    full_df = full_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    return full_df

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    csv_path = os.path.join(data_dir, 'chronic_kidney_disease.csv')
    df = generate_ckd_dataset(n_samples=150, random_state=42)
    df.to_csv(csv_path, index=False)
    
    print(f"================================================================")
    print(f"  SARANYA A - Chronic Kidney Disease Prediction (Nephrology)")
    print(f"================================================================")
    print(f"[+] Dataset successfully created at: {csv_path}")
    print(f"[+] Total Records: {len(df)}")
    print(f"[+] Total Features: {df.shape[1] - 1} clinical attributes + 1 target label")
    print("\nTarget Class Distribution:")
    print(df['classification'].value_counts())
    print("\nSummary Statistics of Core Nephrology Markers:")
    print(df[['serum_creatinine', 'blood_urea', 'hemoglobin', 'albumin', 'specific_gravity']].describe().T)
