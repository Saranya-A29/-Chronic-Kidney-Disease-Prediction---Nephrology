# Machine Learning Model Training, Comparison & Clinical Evaluation
# Student Name: SARANYA A
# Project: Chronic Kidney Disease Prediction (Nephrology)

import os
import json
import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# Models to benchmark
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def preprocess_data(df):
    """
    Encodes categorical features and prepares feature matrix X and target y.
    """
    df_clean = df.copy()

    # Binary mappings
    binary_maps = {
        'red_blood_cells': {'normal': 0, 'abnormal': 1},
        'pus_cell': {'normal': 0, 'abnormal': 1},
        'pus_cell_clumps': {'notpresent': 0, 'present': 1},
        'bacteria': {'notpresent': 0, 'present': 1},
        'hypertension': {'no': 0, 'yes': 1},
        'diabetes_mellitus': {'no': 0, 'yes': 1},
        'coronary_artery_disease': {'no': 0, 'yes': 1},
        'appetite': {'good': 0, 'poor': 1},
        'pedal_edema': {'no': 0, 'yes': 1},
        'anemia': {'no': 0, 'yes': 1},
        'classification': {'notckd': 0, 'ckd': 1}
    }

    for col, mapping in binary_maps.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].map(mapping)

    X = df_clean.drop(columns=['classification'])
    y = df_clean['classification']

    feature_cols = list(X.columns)
    return X, y, feature_cols, binary_maps

def train_and_evaluate():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'chronic_kidney_disease.csv')
    models_dir = os.path.join(base_dir, 'saved_models')
    viz_dir = os.path.join(base_dir, 'visualizations')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    print("=" * 70)
    print("  MACHINE LEARNING MODEL BENCHMARKING & CLINICAL VALIDATION")
    print("  Student Name: SARANYA A | Nephrology Domain")
    print("=" * 70)

    df = pd.read_csv(data_path)
    X, y, feature_cols, binary_maps = preprocess_data(df)

    # Train / Test split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Standard scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Benchmark Classifiers
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Support Vector Machine': SVC(probability=True, kernel='rbf', C=1.0, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }

    results = []
    trained_models = {}
    roc_data = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, clf in models.items():
        # Train model
        clf.fit(X_train_scaled, y_train)
        trained_models[name] = clf

        # Predictions
        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)[:, 1] if hasattr(clf, 'predict_proba') else None

        # Cross validation
        cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv, scoring='accuracy')

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0

        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_data[name] = (fpr, tpr, auc)

        results.append({
            'Model': name,
            'Accuracy': round(acc * 100, 2),
            'Precision': round(prec * 100, 2),
            'Recall (Sensitivity)': round(rec * 100, 2),
            'F1-Score': round(f1 * 100, 2),
            'ROC-AUC': round(auc, 4),
            'CV Mean Acc': round(cv_scores.mean() * 100, 2),
            'CV Std': round(cv_scores.std() * 100, 2)
        })

    results_df = pd.DataFrame(results).sort_values(by=['F1-Score', 'Accuracy'], ascending=False)
    print("\n[+] Model Evaluation Leaderboard:")
    print(results_df.to_string(index=False))

    # Identify Best Model (Highest F1 and Accuracy)
    best_model_name = results_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]
    print(f"\n[*] Selected Best Performing Model: {best_model_name}")

    # Best Model Detailed Test Report
    y_best_pred = best_model.predict(X_test_scaled)
    print("\nClassification Report for Best Model:")
    print(classification_report(y_test, y_best_pred, target_names=['Healthy (notckd)', 'CKD Patient']))

    # -------------------------------------------------------------
    # 1. Plot Model Comparison Bar Chart
    # -------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    melted = results_df.melt(id_vars=['Model'], value_vars=['Accuracy', 'Recall (Sensitivity)', 'F1-Score'],
                             var_name='Metric', value_name='Score (%)')
    ax = sns.barplot(x='Model', y='Score (%)', hue='Metric', data=melted, palette='viridis', edgecolor='black')
    plt.title("Comparative Evaluation of Machine Learning Algorithms for CKD Prediction", fontsize=13, fontweight='bold')
    plt.ylim(80, 105)
    plt.xticks(rotation=20, ha='right')
    plt.legend(loc='lower right')
    for p in ax.patches:
        val = p.get_height()
        if not np.isnan(val) and val > 0:
            ax.annotate(f"{val:.1f}%", (p.get_x() + p.get_width() / 2., val + 0.6),
                        ha='center', va='bottom', fontsize=8, rotation=0)
    plt.tight_layout()
    comparison_path = os.path.join(viz_dir, 'model_comparison.png')
    plt.savefig(comparison_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved: {comparison_path}")

    # -------------------------------------------------------------
    # 2. Plot ROC Curves
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Chance Line (AUC = 0.50)')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity / Recall)')
    plt.title('Receiver Operating Characteristic (ROC) Curves - Nephrology Model Suite', fontweight='bold')
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    roc_path = os.path.join(viz_dir, 'roc_curves.png')
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved: {roc_path}")

    # -------------------------------------------------------------
    # 3. Best Model Confusion Matrix
    # -------------------------------------------------------------
    cm = confusion_matrix(y_test, y_best_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Healthy (notckd)', 'CKD'],
                yticklabels=['Healthy (notckd)', 'CKD'],
                annot_kws={'size': 14, 'fontweight': 'bold'})
    plt.title(f"Confusion Matrix: {best_model_name}\n(Saranya A - Nephrology Evaluation)", fontweight='bold')
    plt.xlabel('Predicted Diagnosis')
    plt.ylabel('Actual Clinical Diagnosis')
    plt.tight_layout()
    cm_path = os.path.join(viz_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved: {cm_path}")

    # -------------------------------------------------------------
    # 4. Feature Importance (Top Nephrology Biomarkers)
    # -------------------------------------------------------------
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    else:
        # Fallback to Random Forest feature importance
        rf_model = trained_models['Random Forest']
        importances = rf_model.feature_importances_

    feat_imp = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(feat_imp['Feature'], feat_imp['Importance'], color='#3498db', edgecolor='black')
    plt.title(f"Clinical Biomarker Feature Importance (Nephrology Relevance)\nModel: {best_model_name}",
              fontsize=13, fontweight='bold')
    plt.xlabel('Relative Feature Importance (MDI / Gini Gain)')
    plt.tight_layout()
    feat_imp_path = os.path.join(viz_dir, 'feature_importance.png')
    plt.savefig(feat_imp_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Saved: {feat_imp_path}")

    # -------------------------------------------------------------
    # 5. Save Artifacts for Web App and Inference
    # -------------------------------------------------------------
    model_save_path = os.path.join(models_dir, 'best_ckd_model.pkl')
    scaler_save_path = os.path.join(models_dir, 'scaler.pkl')
    metrics_save_path = os.path.join(models_dir, 'model_metrics.json')
    features_save_path = os.path.join(models_dir, 'feature_columns.json')

    joblib.dump(best_model, model_save_path)
    joblib.dump(scaler, scaler_save_path)

    with open(features_save_path, 'w') as f:
        json.dump(feature_cols, f, indent=4)

    metrics_payload = {
        'best_model_name': best_model_name,
        'student_name': 'SARANYA A',
        'project_title': 'Chronic Kidney Disease Prediction (Nephrology)',
        'leaderboard': results_df.to_dict(orient='records'),
        'feature_importance': feat_imp.sort_values(by='Importance', ascending=False).to_dict(orient='records'),
        'confusion_matrix': cm.tolist()
    }

    with open(metrics_save_path, 'w') as f:
        json.dump(metrics_payload, f, indent=4)

    print(f"\n[+] Trained Model saved to: {model_save_path}")
    print(f"[+] Scaler saved to: {scaler_save_path}")
    print(f"[+] Evaluation Metrics saved to: {metrics_save_path}")
    print(f"[+] Feature Columns saved to: {features_save_path}")

if __name__ == '__main__':
    train_and_evaluate()
