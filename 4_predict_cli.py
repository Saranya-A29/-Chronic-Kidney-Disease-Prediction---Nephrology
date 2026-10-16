# Interactive Command-Line Predictor for Chronic Kidney Disease
# Student Name: SARANYA A
# Domain: Nephrology

import os
import json
import joblib
import numpy as np
import pandas as pd

def estimate_egfr_and_stage(creatinine, age, is_female=False):
    """
    Estimates eGFR (Glomerular Filtration Rate) using the simplified CKD-EPI formula
    and maps to clinical KDIGO CKD Stages (Stage 1 to 5 / Kidney Failure).
    """
    if creatinine <= 0:
        return 90, "Stage 1 (Normal kidney function)"
    
    # Simplified CKD-EPI approximation (mL/min/1.73m2)
    k = 0.7 if is_female else 0.9
    alpha = -0.329 if is_female else -0.411
    gender_mult = 1.018 if is_female else 1.0
    
    min_sc = min(creatinine / k, 1) ** alpha
    max_sc = max(creatinine / k, 1) ** -1.209
    age_mult = 0.993 ** age
    
    egfr = 141 * min_sc * max_sc * age_mult * gender_mult
    egfr = round(max(5.0, min(130.0, egfr)), 1)

    if egfr >= 90:
        stage = "Stage 1: Normal or high kidney function (eGFR >= 90)"
    elif egfr >= 60:
        stage = "Stage 2: Mildly decreased kidney function (eGFR 60 - 89)"
    elif egfr >= 45:
        stage = "Stage 3A: Mild-to-moderately decreased function (eGFR 45 - 59)"
    elif egfr >= 30:
        stage = "Stage 3B: Moderately-to-severely decreased function (eGFR 30 - 44)"
    elif egfr >= 15:
        stage = "Stage 4: Severely decreased kidney function (eGFR 15 - 29)"
    else:
        stage = "Stage 5: Kidney failure / End-Stage Renal Disease (eGFR < 15)"

    return egfr, stage

def predict_patient(patient_dict):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'saved_models')

    model = joblib.load(os.path.join(models_dir, 'best_ckd_model.pkl'))
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    with open(os.path.join(models_dir, 'feature_columns.json'), 'r') as f:
        feature_cols = json.load(f)

    # Convert binary fields
    binary_maps = {
        'red_blood_cells': {'normal': 0, 'abnormal': 1, 0: 0, 1: 1},
        'pus_cell': {'normal': 0, 'abnormal': 1, 0: 0, 1: 1},
        'pus_cell_clumps': {'notpresent': 0, 'present': 1, 0: 0, 1: 1},
        'bacteria': {'notpresent': 0, 'present': 1, 0: 0, 1: 1},
        'hypertension': {'no': 0, 'yes': 1, 0: 0, 1: 1},
        'diabetes_mellitus': {'no': 0, 'yes': 1, 0: 0, 1: 1},
        'coronary_artery_disease': {'no': 0, 'yes': 1, 0: 0, 1: 1},
        'appetite': {'good': 0, 'poor': 1, 0: 0, 1: 1},
        'pedal_edema': {'no': 0, 'yes': 1, 0: 0, 1: 1},
        'anemia': {'no': 0, 'yes': 1, 0: 0, 1: 1}
    }

    patient_clean = {}
    for col in feature_cols:
        val = patient_dict.get(col, 0)
        if col in binary_maps and isinstance(val, str):
            patient_clean[col] = binary_maps[col].get(val.lower().strip(), 0)
        else:
            patient_clean[col] = float(val)

    df_sample = pd.DataFrame([patient_clean])[feature_cols]
    scaled_features = scaler.transform(df_sample)

    pred_class = model.predict(scaled_features)[0]
    pred_proba = model.predict_proba(scaled_features)[0]

    ckd_risk_prob = pred_proba[1] * 100
    egfr, stage = estimate_egfr_and_stage(patient_clean['serum_creatinine'], patient_clean['age'])

    return {
        'prediction': 'CHRONIC KIDNEY DISEASE (CKD)' if pred_class == 1 else 'HEALTHY / NORMAL RENAL FUNCTION',
        'is_ckd': bool(pred_class == 1),
        'confidence': round(float(pred_proba[pred_class] * 100), 2),
        'ckd_risk_percentage': round(float(ckd_risk_prob), 2),
        'estimated_egfr': egfr,
        'kdigo_stage': stage
    }

def main():
    print("=" * 70)
    print("  SARANYA A - Chronic Kidney Disease Clinical Diagnosis Engine")
    print("  Nephrology Machine Learning Predictor")
    print("=" * 70)

    # Benchmark Test Cases
    cases = [
        {
            "name": "Case 1: Healthy Patient Screen",
            "data": {
                'age': 34, 'blood_pressure': 80, 'specific_gravity': 1.025,
                'albumin': 0, 'sugar': 0, 'red_blood_cells': 'normal',
                'pus_cell': 'normal', 'pus_cell_clumps': 'notpresent', 'bacteria': 'notpresent',
                'blood_glucose_random': 95, 'blood_urea': 28, 'serum_creatinine': 0.8,
                'sodium': 140, 'potassium': 4.1, 'hemoglobin': 15.5,
                'packed_cell_volume': 48, 'white_blood_cell_count': 6800,
                'red_blood_cell_count': 5.2, 'hypertension': 'no', 'diabetes_mellitus': 'no',
                'coronary_artery_disease': 'no', 'appetite': 'good', 'pedal_edema': 'no', 'anemia': 'no'
            }
        },
        {
            "name": "Case 2: Advanced Chronic Kidney Disease (Uremic Patient)",
            "data": {
                'age': 62, 'blood_pressure': 90, 'specific_gravity': 1.010,
                'albumin': 3, 'sugar': 1, 'red_blood_cells': 'abnormal',
                'pus_cell': 'abnormal', 'pus_cell_clumps': 'present', 'bacteria': 'notpresent',
                'blood_glucose_random': 210, 'blood_urea': 118, 'serum_creatinine': 4.6,
                'sodium': 130, 'potassium': 5.4, 'hemoglobin': 8.2,
                'packed_cell_volume': 26, 'white_blood_cell_count': 11200,
                'red_blood_cell_count': 3.1, 'hypertension': 'yes', 'diabetes_mellitus': 'yes',
                'coronary_artery_disease': 'yes', 'appetite': 'poor', 'pedal_edema': 'yes', 'anemia': 'yes'
            }
        }
    ]

    for case in cases:
        print(f"\n--- Testing: {case['name']} ---")
        result = predict_patient(case['data'])
        print(f"[*] Diagnostic Output   : {result['prediction']}")
        print(f"[*] Confidence Score    : {result['confidence']}%")
        print(f"[*] CKD Risk Score      : {result['ckd_risk_percentage']}%")
        print(f"[*] Estimated eGFR      : {result['estimated_egfr']} mL/min/1.73m2")
        print(f"[*] KDIGO Classification: {result['kdigo_stage']}")

if __name__ == '__main__':
    main()
