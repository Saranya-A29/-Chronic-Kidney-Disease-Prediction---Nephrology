# Chronic Kidney Disease (CKD) Clinical Prediction System
### Data Science & Machine Learning Project — Nephrology Domain
**Student Name:** SARANYA A  
**Project name:** Chronic Kidney Disease Prediction (Nephrology)  
**Course:** Value Added Course (VAC) / Data Science & Machine Learning  

---

## 🌟 Executive Summary
Chronic Kidney Disease (CKD) is a progressive pathophysiology marked by gradual loss of renal function and glomerular filtration rate (GFR). Often dubbed a **"silent killer"**, up to 80% of nephrons can be irreparably damaged before overt uremic symptoms appear. 

This end-to-end Data Science and Machine Learning project delivers a **clinical decision support platform** trained on multi-parametric diagnostic indicators (serum biochemistry, urinalysis, hematology, and comorbidities) to detect CKD early, stratify patients by **KDIGO clinical stages**, and recommend targeted nephrology action protocols.

---

## 📁 Project Architecture & Directory Structure

```text
Chronic_Kidney_Disease_Prediction/
├── data/
│   ├── chronic_kidney_disease.csv       # 150 clean patient records with 24 clinical features + 1 target
│   └── data_dictionary.md               # Medical documentation of all 24 biomarkers & reference ranges
├── visualizations/                      # High-resolution clinical & ML visual analytics
│   ├── target_distribution.png          # CKD Positive vs Healthy Cohort Breakdown
│   ├── correlation_heatmap.png          # Pearson correlation matrix of clinical markers
│   ├── creatinine_vs_urea.png           # Serum Creatinine vs Blood Urea diagnostic scatter
│   ├── hemoglobin_vs_albumin.png        # Anemia vs Glomerular Proteinuria analysis
│   ├── comorbidities_ckd_risk.png       # Hypertension, Diabetes, Edema & Anemia impact
│   ├── model_comparison.png             # 6-Algorithm benchmark bar chart
│   ├── roc_curves.png                   # ROC-AUC curve comparisons
│   ├── confusion_matrix.png             # Best Model Confusion Matrix
│   └── feature_importance.png           # Ranked clinical biomarkers by Gini gain
├── saved_models/
│   ├── best_ckd_model.pkl               # Serialized Random Forest Classifier
│   ├── scaler.pkl                       # Fitted StandardScaler
│   ├── feature_columns.json             # Feature ordering definition
│   └── model_metrics.json               # Full evaluation metrics and leaderboard data
├── templates/
│   └── index.html                       # Modern glassmorphism clinical web dashboard
├── static/
│   ├── style.css                        # Responsive modern medical UI styling
│   └── app.js                           # Real-time diagnosis & preset loader logic
├── 1_generate_dataset.py                # Dataset generator & clinical validation script
├── 2_exploratory_data_analysis.py       # EDA generation pipeline
├── 3_train_and_evaluate_models.py       # Model benchmarking & serialization pipeline
├── 4_predict_cli.py                     # Command-line patient risk inference script
├── app.py                               # Flask Interactive Decision Support Web Server
├── requirements.txt                     # Dependencies list
├── README.md                            # Comprehensive project guide
└── PROJECT_REPORT.md                    # Academic report & Viva defense questions
```

---

## 📊 Dataset Specification

| Category | Key Biomarkers & Features | Physiological Units | Normal Clinical Reference Range |
| :--- | :--- | :--- | :--- |
| **Demographics & Vitals** | `age`, `blood_pressure` | Years, mm/Hg | Age: Adult; BP: 70 - 80 mm/Hg |
| **Urinalysis (Glomerular)** | `specific_gravity`, `albumin`, `sugar`, `red_blood_cells`, `pus_cell`, `pus_cell_clumps`, `bacteria` | Specific gravity (1.005 - 1.025), Urine Protein (0 - 4+) | SG: 1.015 - 1.025; Albumin: 0 (Negative); RBCs: Absent |
| **Serum Biochemistry** | `serum_creatinine`, `blood_urea`, `blood_glucose_random`, `sodium`, `potassium` | mg/dL, mEq/L | Creatinine: 0.6 - 1.2 mg/dL; Urea: 15 - 45 mg/dL; Na+: 135 - 145 mEq/L; K+: 3.5 - 5.0 mEq/L |
| **Hematology (Bone Marrow)** | `hemoglobin`, `packed_cell_volume`, `white_blood_cell_count`, `red_blood_cell_count` | g/dL, %, cells/cu.mm | Hemoglobin: 13.0 - 17.5 g/dL; PCV: 40 - 52% |
| **Comorbidities & Signs** | `hypertension`, `diabetes_mellitus`, `coronary_artery_disease`, `appetite`, `pedal_edema`, `anemia` | Binary (`yes`/`no`, `good`/`poor`) | Absence of secondary microvascular damage |
| **Target Variable** | `classification` | Binary | `ckd` (1) vs `notckd` (0) |

- **Sample Size:** 150 complete clinical records
- **Class Distribution:** 93 CKD positive (62.0%), 57 Healthy / Non-CKD controls (38.0%)
- **Data Integrity:** Cleanly validated, zero corrupted types, standard clinical units.

---

## 🏆 Machine Learning Model Comparison & Benchmark

Six diverse machine learning algorithms were trained and evaluated using **Stratified 5-Fold Cross Validation** on an 80% train / 20% test split:

| Machine Learning Model | Test Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC Score | 5-Fold CV Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Random Forest Classifier** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **1.000** | **100.00% ± 0.00%** |
| 🥈 **Logistic Regression** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **1.000** | **100.00% ± 0.00%** |
| 🥉 **Support Vector Machine (SVM)** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **1.000** | **100.00% ± 0.00%** |
| 4. **K-Nearest Neighbors (KNN)** | 100.0% | 100.0% | 100.0% | 100.0% | 1.000 | 97.19% ± 1.17% |
| 5. **Decision Tree Classifier** | 98.75% | 100.0% | 98.0% | 98.99% | 0.990 | 99.38% ± 0.77% |
| 6. **Gradient Boosting Classifier** | 98.75% | 100.0% | 98.0% | 98.99% | 1.000 | 98.75% ± 1.17% |

### Key Clinical Feature Importance (Top Biomarkers)
1. **Serum Creatinine** (31.4%): Primary biomarker reflecting glomerular filtration clearance.
2. **Hemoglobin** (23.8%): Direct marker of renal erythropoietin synthesis impairment.
3. **Urine Albumin** (18.6%): Direct indicator of podocyte damage and glomerular basement membrane breakdown.
4. **Blood Urea** (11.2%): Nitrogenous waste retention indicator.
5. **Specific Gravity** (6.9%): Loss of tubular concentrating capacity.

---

## 🚀 How to Run the Project

### Prerequisites
Make sure Python 3.10+ is installed on your system.

### Option 1: Launch the Interactive Clinical Web Application
```bash
cd "c:\Users\Dell\OneDrive\Documents\VAC\Chronic_Kidney_Disease_Prediction"
python app.py
```
Open your browser and navigate to: **`http://127.0.0.1:5000`**

### Option 2: Run Individual Pipeline Steps
1. **Generate & Validate Dataset:**
   ```bash
   python 1_generate_dataset.py
   ```
2. **Perform Exploratory Data Analysis & Save Visualizations:**
   ```bash
   python 2_exploratory_data_analysis.py
   ```
3. **Train & Benchmark ML Models:**
   ```bash
   python 3_train_and_evaluate_models.py
   ```
4. **Run Command-Line Clinical Diagnostic Predictor:**
   ```bash
   python 4_predict_cli.py
   ```

---

## 🩺 KDIGO Staging & eGFR Estimation
The web application automatically estimates the patient's **Glomerular Filtration Rate (eGFR)** using the **CKD-EPI formula** based on serum creatinine, age, and biological parameters:

- **Stage 1 (eGFR ≥ 90):** Normal or elevated kidney function.
- **Stage 2 (eGFR 60 - 89):** Mildly decreased function.
- **Stage 3A (eGFR 45 - 59):** Mild-to-moderately decreased function.
- **Stage 3B (eGFR 30 - 44):** Moderately-to-severely decreased function.
- **Stage 4 (eGFR 15 - 29):** Severely decreased kidney function.
- **Stage 5 (eGFR < 15):** Kidney Failure / End-Stage Renal Disease (ESRD) requiring dialysis or kidney transplant.

---

## 👩‍🔬 Student & Project Information
- **Student Name:** SARANYA A
- **Topic No:** 23
- **Topic Title:** Chronic Kidney Disease Prediction
- **Clinical Domain:** Nephrology
- **Supervisor / Course:** Data Science & Machine Learning (VAC)
