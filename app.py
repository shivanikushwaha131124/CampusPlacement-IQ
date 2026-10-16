"""
=============================================================================
CampusPlacement-IQ - Live Student Placement & Salary Package Web Portal
=============================================================================
Institutional Career Analytics & Real-Time Placement Readiness Simulator
=============================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure Page
st.set_page_config(
    page_title="CampusPlacement-IQ | Student Placement & Salary Estimator",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #0b0f19; }
    .kpi-card {
        background: #111827;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #1f2937;
        margin-bottom: 12px;
    }
    .kpi-title { font-size: 13px; color: #9ca3af; text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; }
    .kpi-val { font-size: 26px; font-weight: 800; color: #ffffff; margin-top: 4px; }
    .kpi-sub { font-size: 12px; color: #38bdf8; font-weight: 500; margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# Helper function to load datasets & models
@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "data", "student_placement_final_scored.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join(base_dir, "data", "student_placement_processed.csv")
    df = pd.read_csv(data_path)
    return df

@st.cache_resource
def load_models():
    base_dir = os.path.dirname(__file__)
    clf_path = os.path.join(base_dir, "models", "placement_classifier.pkl")
    reg_path = os.path.join(base_dir, "models", "salary_regressor.pkl")
    clf = joblib.load(clf_path) if os.path.exists(clf_path) else None
    reg = joblib.load(reg_path) if os.path.exists(reg_path) else None
    return clf, reg

df = load_data()
clf_model, reg_model = load_models()

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=64)
st.sidebar.title("Placement-IQ")
st.sidebar.markdown("**College TPO Cell & Student Career Engine**")
st.sidebar.divider()

placed_count = (df['Placement_Status'] == 1).sum()
placement_rate = df['Placement_Status'].mean() * 100
avg_ctc = df[df['Placement_Status'] == 1]['Salary_Package_LPA'].mean()

st.sidebar.metric("Evaluated Batch", f"{len(df):,} Students")
st.sidebar.metric("Overall Placement Rate", f"{placement_rate:.1f}%")
st.sidebar.metric("Average Salary (CTC)", f"₹{avg_ctc:.2f} LPA")
st.sidebar.divider()

st.sidebar.info("💡 **Student Tip:** Enter your CGPA & skills in Tab 3 to check your personal Placement & Package forecast!")

# Main Tabs
tabs = st.tabs([
    "📊 TPO Institutional Analytics", 
    "🔍 Student Cohort Explorer", 
    "🎯 Live Placement & CTC Estimator", 
    "📈 Dual ML Engine Proof"
])

# -----------------------------------------------------------------------------
# TAB 1: Institutional Analytics
# -----------------------------------------------------------------------------
with tabs[0]:
    st.subheader("🎓 College Campus Placement & Salary Benchmark Overview")
    
    # KPI Row
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid #3b82f6;">
            <div class="kpi-title">Total Batch Size</div>
            <div class="kpi-val">{len(df):,}</div>
            <div class="kpi-sub">MCA, B.Tech, BCA</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid #10b981;">
            <div class="kpi-title">Placement Rate</div>
            <div class="kpi-val" style="color:#10b981;">{placement_rate:.1f}%</div>
            <div class="kpi-sub">{placed_count:,} Placed Offers</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid #8b5cf6;">
            <div class="kpi-title">Average Salary (CTC)</div>
            <div class="kpi-val" style="color:#8b5cf6;">₹{avg_ctc:.2f} LPA</div>
            <div class="kpi-sub">Placed Cohort</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid #f59e0b;">
            <div class="kpi-title">Highest Package</div>
            <div class="kpi-val" style="color:#f59e0b;">₹{df['Salary_Package_LPA'].max():.2f} LPA</div>
            <div class="kpi-sub">Tier-1 Product Offer</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        dream_offers = (df['Salary_Package_LPA'] >= 10.0).sum()
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid #ec4899;">
            <div class="kpi-title">Dream Offers (>10 LPA)</div>
            <div class="kpi-val" style="color:#ec4899;">{dream_offers:,}</div>
            <div class="kpi-sub">Top Performers</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col_chart1, col_chart2 = st.columns([1, 1.3])
    
    with col_chart1:
        st.markdown("##### 👥 Career Readiness Tiers Distribution")
        tier_df = df['Readiness_Tier'].value_counts().reset_index()
        tier_df.columns = ['Readiness_Tier', 'Count']
        fig_tier = px.pie(
            tier_df, names='Readiness_Tier', values='Count',
            hole=0.45,
            color_discrete_sequence=['#6366f1', '#10b981', '#ef4444']
        )
        fig_tier.update_layout(margin=dict(t=20, b=20, l=10, r=10), template="plotly_dark")
        st.plotly_chart(fig_tier, use_container_width=True)

    with col_chart2:
        st.markdown("##### 💼 Average Salary Package (INR LPA) Across Degrees")
        deg_ctc = df[df['Placement_Status'] == 1].groupby('Degree_Stream')['Salary_Package_LPA'].mean().reset_index()
        deg_ctc = deg_ctc.sort_values(by='Salary_Package_LPA', ascending=True)

        fig_deg = px.bar(
            deg_ctc, y='Degree_Stream', x='Salary_Package_LPA', orientation='h',
            labels={'Salary_Package_LPA': 'Average CTC (INR LPA)', 'Degree_Stream': 'Degree'},
            color='Salary_Package_LPA',
            color_continuous_scale='Teal'
        )
        fig_deg.update_layout(showlegend=False, margin=dict(t=20, b=20, l=10, r=10), template="plotly_dark")
        st.plotly_chart(fig_deg, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: Student Directory & Filter Explorer
# -----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("🔍 Institutional Student Career Roster & Filter Matrix")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        selected_degree = st.multiselect("Filter by Degree:", options=df['Degree_Stream'].unique(), default=df['Degree_Stream'].unique())
    with col_f2:
        selected_tier = st.multiselect("Filter by Readiness Tier:", options=df['Readiness_Tier'].unique(), default=df['Readiness_Tier'].unique())
    with col_f3:
        filter_status = st.selectbox("Placement Filter:", ["All Students", "Placed Only (1)", "Unplaced Only (0)"])

    filtered_df = df[
        (df['Degree_Stream'].isin(selected_degree)) & 
        (df['Readiness_Tier'].isin(selected_tier))
    ]
    if filter_status == "Placed Only (1)":
        filtered_df = filtered_df[filtered_df['Placement_Status'] == 1]
    elif filter_status == "Unplaced Only (0)":
        filtered_df = filtered_df[filtered_df['Placement_Status'] == 0]

    st.markdown(f"**Showing {len(filtered_df):,} matching student profiles:**")
    st.dataframe(
        filtered_df[[
            'Student_ID', 'Degree_Stream', 'CGPA', 'Active_Backlogs', 'Internships_Count', 
            'Coding_DSA_Score', 'Projects_Completed', 'Placement_Status', 
            'Salary_Package_LPA', 'Placement_Probability', 'Readiness_Tier'
        ]].head(100),
        use_container_width=True
    )

    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Export TPO Student Placement CSV", data=csv_data, file_name="campus_placement_report.csv", mime="text/csv")

# -----------------------------------------------------------------------------
# TAB 3: Live Placement & CTC Estimator
# -----------------------------------------------------------------------------
with tabs[2]:
    st.subheader("🎯 Personal Placement Probability & Salary Package Estimator")
    st.markdown("Enter your academic & technical background below to calculate your placement forecast:")

    sim_c1, sim_c2, sim_c3 = st.columns(3)
    
    with sim_c1:
        st.markdown("**🎓 Academic Track Record**")
        sim_degree = st.selectbox("Degree Stream", ["MCA", "B.Tech CSE", "B.Tech IT", "BCA", "B.Tech ECE"])
        sim_cgpa = st.slider("Current Cumulative CGPA", 5.0, 10.0, 7.8, step=0.1)
        sim_backlogs = st.slider("Active Pending Backlogs", 0, 3, 0)
        sim_internships = st.slider("Technical Internships Completed", 0, 3, 1)

    with sim_c2:
        st.markdown("**💻 Technical & Coding Skills**")
        sim_dsa = st.slider("Coding / DSA Assessment Score (0-100)", 20, 100, 72)
        sim_projects = st.slider("Practical Projects Built", 1, 5, 3)
        sim_certs = st.slider("Industry Certifications (Cloud, AI, Python)", 0, 4, 1)

    with sim_c3:
        st.markdown("**🗣️ Soft Skills & Competitions**")
        sim_aptitude = st.slider("Aptitude & Problem Solving Score", 30, 100, 68)
        sim_soft_skills = st.slider("HR & Soft Skills Mock Rating (1 to 5)", 1.0, 5.0, 3.8, step=0.1)
        sim_hackathons = st.slider("Hackathons Attended", 0, 3, 1)

    st.markdown("---")
    
    if st.button("🚀 Predict My Placement & Salary Package", type="primary", use_container_width=True):
        if clf_model is None or reg_model is None:
            st.error("Model pipelines not loaded. Please ensure 03_train_placement_models.py has been executed.")
        else:
            input_df = pd.DataFrame([{
                'CGPA': sim_cgpa,
                'Active_Backlogs': sim_backlogs,
                'Internships_Count': sim_internships,
                'Coding_DSA_Score': sim_dsa,
                'Projects_Completed': sim_projects,
                'Certifications_Count': sim_certs,
                'Aptitude_Test_Score': sim_aptitude,
                'Soft_Skills_Rating': sim_soft_skills,
                'Hackathons_Attended': sim_hackathons,
                'Degree_Stream': sim_degree
            }])

            prob = clf_model.predict_proba(input_df)[0, 1]
            prob_percent = prob * 100
            est_ctc = reg_model.predict(input_df)[0]

            res_c1, res_c2 = st.columns([1, 1.4])
            
            with res_c1:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_percent,
                    title={'text': "Placement Probability (%)", 'font': {'size': 18, 'color': "white"}},
                    number={'suffix': "%", 'font': {'color': "white"}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': "white"},
                        'bar': {'color': "#10b981" if prob_percent >= 65 else "#f59e0b" if prob_percent >= 45 else "#ef4444"},
                        'steps': [
                            {'range': [0, 45], 'color': "rgba(239, 68, 68, 0.25)"},
                            {'range': [45, 75], 'color': "rgba(245, 158, 11, 0.25)"},
                            {'range': [75, 100], 'color': "rgba(16, 185, 129, 0.25)"}
                        ]
                    }
                ))
                fig_gauge.update_layout(height=260, margin=dict(t=30, b=10, l=20, r=20), template="plotly_dark")
                st.plotly_chart(fig_gauge, use_container_width=True)

            with res_c2:
                if prob_percent >= 75:
                    st.success(f"🎉 **HIGH PLACEMENT READINESS ({prob_percent:.1f}%)**")
                    st.markdown(f"### 💼 **Estimated Salary Package: ₹{est_ctc:.2f} LPA**")
                    st.markdown("""
                    **🎯 Next-Step Career Advice:**
                    - 🌟 You qualify for **Tier-1 Product & Unicorn hiring drives**!
                    - Focus on System Design (LLD/HLD) and advanced LeetCode medium/hard topics to push towards ₹15+ LPA!
                    """)
                elif prob_percent >= 50:
                    st.warning(f"⚠️ **MODERATE PLACEMENT READINESS ({prob_percent:.1f}%)**")
                    st.markdown(f"### 💼 **Estimated Salary Package: ₹{est_ctc:.2f} LPA**")
                    st.markdown("""
                    **🎯 How to Boost into Tier-1 (>10 LPA):**
                    - 💻 **DSA Push:** Increase coding score from your current level above 80 to unlock higher package brackets.
                    - 🛠️ **Internship Impact:** Complete 1 more production internship to increase placement probability by ~15%!
                    """)
                else:
                    st.error(f"🚨 **SKILL INTERVENTION RECOMMENDED ({prob_percent:.1f}%)**")
                    st.markdown("""
                    **🚨 Priority Action Items:**
                    - ⚠️ Clear pending backlogs immediately; companies enforce a strict 0-backlog eligibility rule.
                    - 📈 Focus on core DSA foundations (Arrays, Trees, Graphs) and build 1 full-stack/AI capstone project.
                    """)

# -----------------------------------------------------------------------------
# TAB 4: Dual ML Engine Proof
# -----------------------------------------------------------------------------
with tabs[3]:
    st.subheader("📈 Machine Learning Performance & Dual-Engine Verification")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("##### 🏆 Placement Classifier Metrics")
        metrics_clf = pd.DataFrame({
            "Metric": ["Accuracy", "Recall (Placed)", "Precision", "ROC-AUC Score"],
            "Score": ["72.60%", "91.35%", "73.71%", "0.7591"]
        })
        st.dataframe(metrics_clf, use_container_width=True)

    with col_m2:
        st.markdown("##### 💼 Salary Package Regressor Metrics")
        metrics_reg = pd.DataFrame({
            "Metric": ["R2-Score (Variance Explained)", "Mean Absolute Error (MAE)", "Trained Samples"],
            "Score": ["0.8522 (85.2%)", "INR 1.00 Lakhs", "6,645 Placed Students"]
        })
        st.dataframe(metrics_reg, use_container_width=True)

    st.markdown("---")
    img_col1, img_col2, img_col3 = st.columns(3)
    base_dir = os.path.dirname(__file__)
    
    p1 = os.path.join(base_dir, "outputs", "04_placement_confusion_matrix.png")
    p2 = os.path.join(base_dir, "outputs", "05_feature_importance_placement.png")
    p3 = os.path.join(base_dir, "outputs", "06_salary_prediction_scatter.png")
    
    if os.path.exists(p1):
        with img_col1:
            st.image(p1, caption="Placement Confusion Matrix")
    if os.path.exists(p2):
        with img_col2:
            st.image(p2, caption="Key Placement Drivers")
    if os.path.exists(p3):
        with img_col3:
            st.image(p3, caption="Actual vs Predicted Salary (CTC)")
