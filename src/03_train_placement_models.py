"""
=============================================================================
CampusPlacement-IQ - Script 03: Dual ML Engine (Classification & Regression)
=============================================================================
Trains two production Machine Learning pipelines:
1. Classifier: Predicts Placement Likelihood (Placed vs Unplaced)
2. Regressor: Estimates Expected Salary Package (CTC in INR Lakhs / Annum)
3. Evaluates Precision, Recall, ROC-AUC, R2-Score, and MAE
4. Scores entire student body with Placement Probabilities & Career Readiness Tiers
=============================================================================
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, mean_absolute_error, r2_score, classification_report
)

# Windows console encoding safeguard
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'Segoe UI'

def train_placement_engines():
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "..", "data")
    models_dir = os.path.join(base_dir, "..", "models")
    output_dir = os.path.join(base_dir, "..", "outputs")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    input_file = os.path.join(data_dir, "student_placement_processed.csv")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Processed file not found at: {input_file}")

    print("=" * 75)
    print("🤖 [CampusPlacement-IQ Step 3] Training Dual ML Placement & Salary Engines")
    print("=" * 75)

    df = pd.read_csv(input_file)
    print(f"[+] Loaded {len(df):,} student records.")

    num_features = [
        'CGPA',
        'Active_Backlogs',
        'Internships_Count',
        'Coding_DSA_Score',
        'Projects_Completed',
        'Certifications_Count',
        'Aptitude_Test_Score',
        'Soft_Skills_Rating',
        'Hackathons_Attended'
    ]
    cat_features = ['Degree_Stream']

    X = df[num_features + cat_features]
    y_class = df['Placement_Status']

    # -------------------------------------------------------------------------
    # 1. TRAIN PLACEMENT CLASSIFIER (Placed vs Unplaced)
    # -------------------------------------------------------------------------
    print("\n--- Training Model A: Placement Status Classifier ---")
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_class, test_size=0.20, random_state=42, stratify=y_class
    )

    preprocessor_c = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_features)
        ]
    )

    clf_pipeline = Pipeline([
        ('preprocessor', preprocessor_c),
        ('classifier', RandomForestClassifier(n_estimators=150, max_depth=9, random_state=42))
    ])

    clf_pipeline.fit(X_train_c, y_train_c)
    y_pred_c = clf_pipeline.predict(X_test_c)
    y_prob_c = clf_pipeline.predict_proba(X_test_c)[:, 1]

    acc_c = accuracy_score(y_test_c, y_pred_c)
    prec_c = precision_score(y_test_c, y_pred_c)
    rec_c = recall_score(y_test_c, y_pred_c)
    roc_c = roc_auc_score(y_test_c, y_prob_c)

    print(f"[+] Classifier Accuracy: {acc_c*100:.2f}% | Precision: {prec_c*100:.2f}% | Recall: {rec_c*100:.2f}% | ROC-AUC: {roc_c:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test_c, y_pred_c, target_names=["Unplaced (0)", "Placed (1)"]))

    # -------------------------------------------------------------------------
    # 2. TRAIN SALARY REGRESSOR (Package in LPA)
    # -------------------------------------------------------------------------
    print("\n--- Training Model B: Salary Package (CTC in LPA) Regressor ---")
    placed_mask = (df['Placement_Status'] == 1)
    X_reg = df[placed_mask][num_features + cat_features]
    y_reg = df[placed_mask]['Salary_Package_LPA']

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.20, random_state=42
    )

    preprocessor_r = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_features)
        ]
    )

    reg_pipeline = Pipeline([
        ('preprocessor', preprocessor_r),
        ('regressor', RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42))
    ])

    reg_pipeline.fit(X_train_r, y_train_r)
    y_pred_r = reg_pipeline.predict(X_test_r)

    mae_r = mean_absolute_error(y_test_r, y_pred_r)
    r2_r = r2_score(y_test_r, y_pred_r)
    print(f"[+] Salary Regressor R2-Score: {r2_r:.4f} | MAE: INR {mae_r:.2f} Lakhs")

    # -------------------------------------------------------------------------
    # 3. SAVE VISUALIZATIONS
    # -------------------------------------------------------------------------
    print("\n[+] Generating & Saving Model Performance Visualizations...")

    # Plot 1: Confusion Matrix
    cm = confusion_matrix(y_test_c, y_pred_c)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=["Pred: Unplaced", "Pred: Placed"],
                yticklabels=["Actual: Unplaced", "Actual: Placed"],
                annot_kws={"size": 13, "weight": "bold"})
    ax.set_title("Confusion Matrix (Placement Classifier)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_placement_confusion_matrix.png"), dpi=200)
    plt.close()

    # Plot 2: Feature Importance (What drives placements?)
    ohe_cols = list(clf_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_features))
    feature_names = num_features + ohe_cols
    importances = clf_pipeline.named_steps['classifier'].feature_importances_
    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(fi_df['Feature'][::-1], fi_df['Importance'][::-1], color='#2563eb')
    ax.set_title("Key Drivers of Campus Placement Success", fontsize=13, fontweight='bold')
    ax.set_xlabel("Relative Importance Weight")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_feature_importance_placement.png"), dpi=200)
    plt.close()

    # Plot 3: Regressor Actual vs Predicted Salary
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test_r, y_pred_r, alpha=0.35, color='#10b981', s=25)
    ax.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], color='#ef4444', lw=2, linestyle='--')
    ax.set_title("Actual vs Predicted CTC Package (LPA)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Actual Offered Package (INR LPA)")
    ax.set_ylabel("Model Predicted Package (INR LPA)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "06_salary_prediction_scatter.png"), dpi=200)
    plt.close()

    # -------------------------------------------------------------------------
    # 4. SAVE TRAINED PIPELINES
    # -------------------------------------------------------------------------
    clf_path = os.path.join(models_dir, "placement_classifier.pkl")
    reg_path = os.path.join(models_dir, "salary_regressor.pkl")
    joblib.dump(clf_pipeline, clf_path)
    joblib.dump(reg_pipeline, reg_path)
    print(f"[+] Saved Classifier Pipeline: {os.path.abspath(clf_path)}")
    print(f"[+] Saved Salary Regressor: {os.path.abspath(reg_path)}")

    # -------------------------------------------------------------------------
    # 5. SCORE FULL DATASET & ASSIGN READINESS TIERS
    # -------------------------------------------------------------------------
    all_probs = clf_pipeline.predict_proba(X)[:, 1]
    all_est_ctc = reg_pipeline.predict(X)

    df['Placement_Probability'] = np.round(all_probs, 4)
    df['Predicted_CTC_LPA'] = np.round(all_est_ctc, 2)

    def assign_readiness(row):
        prob = row['Placement_Probability']
        ctc = row['Predicted_CTC_LPA']
        if prob >= 0.75 and ctc >= 9.0:
            return "Tier-1 Dream Company (10+ LPA)"
        elif prob >= 0.50:
            return "Tier-2 Service & Growth IT (4-9 LPA)"
        else:
            return "At-Risk (Needs Skill Intervention)"

    df['Readiness_Tier'] = df.apply(assign_readiness, axis=1)

    final_csv = os.path.join(data_dir, "student_placement_final_scored.csv")
    df.to_csv(final_csv, index=False)

    print(f"\n[SUCCESS] Scored Student Dataset Saved: {os.path.abspath(final_csv)}")
    print("Readiness Tier Distribution:")
    print(df['Readiness_Tier'].value_counts().to_string())
    print("=" * 75)

    return clf_pipeline, reg_pipeline, df

if __name__ == "__main__":
    train_placement_engines()
