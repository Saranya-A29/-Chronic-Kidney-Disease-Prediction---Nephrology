# Flask Web Application for Chronic Kidney Disease Prediction
# Student Name: SARANYA A
# Domain: Nephrology

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
VIZ_DIR = os.path.join(BASE_DIR, 'visualizations')

# Load Model, Scaler, and Metadata
model = joblib.load(os.path.join(MODELS_DIR, 'best_ckd_model.pkl'))
scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))

with open(os.path.join(MODELS_DIR, 'feature_columns.json'), 'r') as f:
    feature_cols = json.load(f)

with open(os.path.join(MODELS_DIR, 'model_metrics.json'), 'r') as f:
    model_metrics = json.load(f)

def estimate_egfr_and_stage(creatinine, age, is_female=False):
    """
    Simplified CKD-EPI formula approximation for eGFR (mL/min/1.73m2)
    and KDIGO clinical staging.
    """
    try:
        creatinine = float(creatinine)
        age = float(age)
    except (ValueError, TypeError):
        return 90.0, "Stage 1", "Normal renal function"

    if creatinine <= 0:
        return 90.0, "Stage 1", "Normal or high filtration"

    k = 0.7 if is_female else 0.9
    alpha = -0.329 if is_female else -0.411
    min_sc = min(creatinine / k, 1) ** alpha
    max_sc = max(creatinine / k, 1) ** -1.209
    age_mult = 0.993 ** age
    
    egfr = round(max(5.0, min(130.0, 141 * min_sc * max_sc * age_mult)), 1)

    if egfr >= 90:
        stage = "Stage 1"
        desc = "Normal or high filtration (eGFR >= 90)"
        severity = "low"
    elif egfr >= 60:
        stage = "Stage 2"
        desc = "Mildly decreased filtration (eGFR 60 - 89)"
        severity = "moderate"
    elif egfr >= 45:
        stage = "Stage 3A"
        desc = "Mild-to-moderately decreased filtration (eGFR 45 - 59)"
        severity = "warning"
    elif egfr >= 30:
        stage = "Stage 3B"
        desc = "Moderately-to-severely decreased filtration (eGFR 30 - 44)"
        severity = "high"
    elif egfr >= 15:
        stage = "Stage 4"
        desc = "Severely decreased kidney function (eGFR 15 - 29)"
        severity = "critical"
    else:
        stage = "Stage 5"
        desc = "End-Stage Renal Disease / Kidney Failure (eGFR < 15)"
        severity = "terminal"

    return egfr, stage, desc, severity

@app.route('/')
def home():
    return render_template('index.html', metrics=model_metrics)

@app.route('/visualizations/<path:filename>')
def serve_visualization(filename):
    return send_from_directory(VIZ_DIR, filename)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json or request.form
        
        binary_maps = {
            'red_blood_cells': {'normal': 0, 'abnormal': 1, '0': 0, '1': 1, 0: 0, 1: 1},
            'pus_cell': {'normal': 0, 'abnormal': 1, '0': 0, '1': 1, 0: 0, 1: 1},
            'pus_cell_clumps': {'notpresent': 0, 'present': 1, '0': 0, '1': 1, 0: 0, 1: 1},
            'bacteria': {'notpresent': 0, 'present': 1, '0': 0, '1': 1, 0: 0, 1: 1},
            'hypertension': {'no': 0, 'yes': 1, '0': 0, '1': 1, 0: 0, 1: 1},
            'diabetes_mellitus': {'no': 0, 'yes': 1, '0': 0, '1': 1, 0: 0, 1: 1},
            'coronary_artery_disease': {'no': 0, 'yes': 1, '0': 0, '1': 1, 0: 0, 1: 1},
            'appetite': {'good': 0, 'poor': 1, '0': 0, '1': 1, 0: 0, 1: 1},
            'pedal_edema': {'no': 0, 'yes': 1, '0': 0, '1': 1, 0: 0, 1: 1},
            'anemia': {'no': 0, 'yes': 1, '0': 0, '1': 1, 0: 0, 1: 1}
        }

        patient_vals = {}
        for col in feature_cols:
            raw_val = data.get(col, 0)
            if col in binary_maps:
                if str(raw_val).lower().strip() in binary_maps[col]:
                    patient_vals[col] = binary_maps[col][str(raw_val).lower().strip()]
                else:
                    patient_vals[col] = float(raw_val)
            else:
                patient_vals[col] = float(raw_val)

        df_input = pd.DataFrame([patient_vals])[feature_cols]
        scaled = scaler.transform(df_input)

        pred_class = int(model.predict(scaled)[0])
        pred_proba = model.predict_proba(scaled)[0]

        ckd_prob = round(float(pred_proba[1]) * 100, 2)
        healthy_prob = round(float(pred_proba[0]) * 100, 2)

        egfr, kdigo_stage, kdigo_desc, severity = estimate_egfr_and_stage(
            patient_vals['serum_creatinine'], patient_vals['age']
        )

        # Clinical Recommendation Generation
        recommendations = []
        if pred_class == 1 or ckd_prob >= 50.0:
            recommendations.append("Immediate Nephrologist Referral: Consult a board-certified nephrologist for complete renal workup.")
            recommendations.append("Renal Ultrasound: Schedule renal ultrasonography to assess kidney dimensions, cortical thickness, and echogenicity.")
            if patient_vals['blood_pressure'] >= 90 or patient_vals['hypertension'] == 1:
                recommendations.append("BP Control: Target blood pressure < 130/80 mmHg; discuss ACE inhibitors or ARBs with treating physician for renal protection.")
            if patient_vals['blood_glucose_random'] > 140 or patient_vals['diabetes_mellitus'] == 1:
                recommendations.append("Glycemic Optimization: Target HbA1c < 7.0%; review nephro-safe antidiabetic regimens.")
            if patient_vals['albumin'] > 0:
                recommendations.append(f"Proteinuria Management: Urine Albumin grade {int(patient_vals['albumin'])}+ detected. Recommend 24-hr urine protein quantification.")
            if patient_vals['hemoglobin'] < 12.0 or patient_vals['anemia'] == 1:
                recommendations.append(f"Renal Anemia Protocol: Low Hemoglobin ({patient_vals['hemoglobin']} g/dL). Screen serum ferritin and consider Erythropoietin-Stimulating Agents (ESA).")
            recommendations.append("Dietary Adjustment: Initiate low-sodium (< 2g/day), controlled potassium, and individualized dietary protein restriction (0.6 - 0.8 g/kg/day).")
        else:
            recommendations.append("Kidney Function Optimal: Current biomarkers reflect healthy glomerular filtration and electrolyte balance.")
            recommendations.append("Routine Health Monitoring: Annual serum creatinine, urinalysis, and blood pressure check recommended.")
            recommendations.append("Hydration & Lifestyle: Maintain adequate daily fluid intake (2.0 - 2.5 L) and avoid excessive non-steroidal anti-inflammatory drugs (NSAIDs).")

        return jsonify({
            'success': True,
            'prediction': 'CHRONIC KIDNEY DISEASE DETECTED' if pred_class == 1 else 'HEALTHY / NORMAL RENAL FUNCTION',
            'is_ckd': bool(pred_class == 1),
            'ckd_probability': ckd_prob,
            'healthy_probability': healthy_prob,
            'egfr': egfr,
            'kdigo_stage': kdigo_stage,
            'kdigo_description': kdigo_desc,
            'severity': severity,
            'recommendations': recommendations,
            'patient_data': patient_vals
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    return jsonify(model_metrics)

if __name__ == '__main__':
    print("==================================================================")
    print("  SARANYA A - Chronic Kidney Disease Decision Support System")
    print("  Server running at http://127.0.0.1:5000")
    print("==================================================================")
    app.run(host='127.0.0.1', port=5000, debug=False)
