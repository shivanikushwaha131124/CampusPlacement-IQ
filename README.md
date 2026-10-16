# 🎓 CampusPlacement-IQ: Student Campus Placement & Salary Package (LPA) Predictor

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-3.0-darkgreen.svg)](https://pandas.pydata.org/)
[![Power BI](https://img.shields.io/badge/Power_BI-TPO_Dashboard-yellow.svg)](https://powerbi.microsoft.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Student_Portal-red.svg)](https://streamlit.io/)

> **Institutional Higher Education Data Science & Career Readiness Platform** featuring Dual Machine Learning Engines (Placement Classification + CTC Salary Regression) and an Executive TPO (Training & Placement Officer) Analytics Dashboard.

---

## 🎯 Executive Summary & Real-Life Context
For graduating college students (MCA, B.Tech, BCA), campus placements represent the most critical career milestone. However, students often lack clear guidance on how specific attributes—such as **Active Backlogs, CGPA thresholds, Technical DSA coding proficiency, and Internships**—quantitatively impact their probability of getting placed and their expected **Salary Package (CTC in LPA)**.

**CampusPlacement-IQ** bridges this information gap by:
1. Analyzing **10,000 realistic student academic and career records** across leading programs (MCA, B.Tech CSE, IT, BCA, ECE).
2. Training a **Dual Machine Learning Engine**:
   - **Model A (Placement Classifier):** Predicts likelihood of clearing campus drives (`Placed` vs `Unplaced`), achieving **91.4% Recall** and **72.6% Accuracy**.
   - **Model B (Salary Regressor):** Estimates expected CTC package in LPA for placed candidates, achieving **0.852 R²-Score** and an MAE of only **₹1.00 Lakhs**.
3. Delivering an **Institutional TPO Power BI Dashboard** and an **Interactive Live Streamlit Student Simulator** providing actionable upskilling recommendations.

---

## 📊 Key Institutional Placement Metrics
- **Total Batch Size Evaluated:** `10,000 Students`
- **Overall Placement Rate:** `66.45%` (6,645 Students Placed)
- **Average Offered Package (CTC):** `₹10.08 LPA` (Placed Cohort)
- **Median Salary Package:** `₹9.36 LPA`
- **Highest Dream Package:** `₹20.63 LPA`
- **Dream Offers Count (>10 LPA):** `3,014 Students`
- **At-Risk Cohort (Needs Skill Intervention):** `1,854 Students`

---

## 🧠 Key Academic & Skill Drivers (Forensic Insights)

| Academic / Skill Attribute | Impact on Placement & Salary Package |
| :--- | :--- |
| **Active Backlogs = 0** | **73.8%** Placement Rate. Zero backlogs is the #1 eligibility gatekeeper for Day-1 MNCs. |
| **Active Backlogs = 1** | Placement rate drops to **50.6%** (23% penalty). |
| **Active Backlogs >= 2** | Placement rate plunges to **27.7%**. Backlogs severely restrict company shortlists. |
| **Coding / DSA Score > 75** | Directly unlocks Tier-1 product and fintech offers with average packages exceeding **₹12.0 LPA**. |
| **Internships >= 1** | Increases placement likelihood by **24%** and adds an average ₹0.8 to ₹1.6 LPA premium to starting CTC. |

---

## 🤖 Dual Machine Learning Architecture

### Model A: Placement Status Classifier
- **Algorithm:** Random Forest Classifier (150 trees, max depth 9)
- **Recall (Placed):** `91.35%` (Correctly catches over 91% of successful placement candidates)
- **Precision (Placed):** `73.71%`
- **Overall Accuracy:** `72.60%`
- **ROC-AUC Score:** `0.7591`

### Model B: Salary Package (CTC in LPA) Regressor
- **Algorithm:** Random Forest Regressor (150 trees, max depth 10)
- **R² Score (Variance Explained):** `0.8522` (85.2%)
- **Mean Absolute Error (MAE):** `₹1.00 Lakhs`

---

## 🗂️ Project Directory Architecture
```text
CampusPlacement-IQ/
│
├── data/                                  <- Academic & Career Datasets
│   ├── student_placement_raw.csv          <- 10,000 raw simulated student logs
│   ├── student_placement_processed.csv    <- Cleaned feature logs
│   └── student_placement_final_scored.csv <- Augmented with ML probabilities & readiness tiers
│
├── models/                                <- Production ML Pipelines
│   ├── placement_classifier.pkl           <- Serialized Placement Classifier (.pkl)
│   └── salary_regressor.pkl               <- Serialized Salary Regressor (.pkl)
│
├── src/                                   <- Modular Data Pipelines
│   ├── 01_generate_placement_data.py      <- Realistic student data generator engine
│   ├── 02_placement_eda.py                <- Career analytics & exploratory data analysis
│   └── 03_train_placement_models.py       <- Dual ML engine training & evaluation
│
├── outputs/                               <- Visual Evidence & Plots
│   ├── 01_branch_placement_ctc.png
│   ├── 02_backlogs_vs_placement_rate.png
│   ├── 03_dsa_vs_package_scatter.png
│   ├── 04_placement_confusion_matrix.png
│   ├── 05_feature_importance_placement.png
│   ├── 06_salary_prediction_scatter.png
│   └── powerbi_placement_dashboard.html   <- Interactive HTML preview of Power BI
│
├── notebooks/
│   └── CampusPlacement_Analysis_and_Modeling.ipynb <- Interactive Jupyter Notebook
│
├── powerbi/
│   └── POWERBI_SETUP_GUIDE.md             <- Step-by-step setup guide with copy-paste DAX
│
├── app.py                                 <- Live Streamlit Student Career Simulator
└── README.md                              <- Executive documentation
```

---

## 🚀 How to Run Locally

### 1. Execute Complete Pipeline
```bash
python src/01_generate_placement_data.py
python src/02_placement_eda.py
python src/03_train_placement_models.py
```

### 2. Launch Live Streamlit Student Web Portal
```bash
streamlit run app.py
```

### 3. Open Interactive Power BI Preview
Simply double-click:
`CampusPlacement-IQ/outputs/powerbi_placement_dashboard.html` to open directly in any web browser!

---

## 📄 Resume Bullet Points (MCA / Data Science / Campus Placement Ready)

```text
CampusPlacement-IQ: Student Campus Placement & Salary Package Predictor | Python, Scikit-Learn, Power BI, Streamlit
• Developed an institutional career analytics platform analyzing 10,000 student academic and skill profiles across MCA and B.Tech programs.
• Modeled multi-attribute career readiness incorporating CGPA, active backlog penalties, technical DSA scores, and internship experience.
• Trained a dual Machine Learning pipeline: Random Forest Classifier achieving 91.4% Recall on placement outcomes and a CTC Regressor with 0.852 R² score (MAE: ₹1.00 LPA).
• Designed an institutional TPO Power BI executive dashboard and deployed a live interactive Streamlit web simulator delivering personalized upskilling roadmaps.
```
