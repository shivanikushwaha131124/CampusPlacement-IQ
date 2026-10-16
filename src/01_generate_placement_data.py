"""
=============================================================================
CampusPlacement-IQ - Script 01: High-Fidelity Student Placement Simulator
=============================================================================
Generates 10,000 realistic Indian college student records (MCA, B.Tech, BCA)
capturing academic rigor (CGPA, backlogs), technical aptitude (DSA coding score,
projects, certifications), soft skills, placement outcome, and CTC package (LPA).
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

# Safe UTF-8 console output for Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def generate_placement_dataset(num_students=10000, random_seed=42):
    np.random.seed(random_seed)
    print("=" * 70)
    print("🎓 [CampusPlacement-IQ Step 1] Generating Student Academic & Career Logs")
    print(f"[+] Total Simulated Students: {num_students:,}")
    print("=" * 70)

    # 1. Degree & Academic Programs Configuration
    degree_streams = ["MCA", "B.Tech CSE", "B.Tech IT", "BCA", "B.Tech ECE"]
    degree_weights = [0.30, 0.35, 0.15, 0.12, 0.08]

    records = []

    for i in range(1, num_students + 1):
        stu_id = f"STU-{20260000 + i}"
        stream = np.random.choice(degree_streams, p=degree_weights)

        # CGPA (Normal distribution centered at 7.6)
        cgpa = float(np.clip(np.random.normal(loc=7.6, scale=0.9), 5.2, 9.9))
        cgpa = round(cgpa, 2)

        # Backlogs: Most have 0, some have 1 or 2
        backlogs = int(np.random.choice([0, 1, 2, 3], p=[0.76, 0.16, 0.06, 0.02]))

        # Internships: 0 to 3
        internships = int(np.random.choice([0, 1, 2, 3], p=[0.42, 0.38, 0.15, 0.05]))

        # Technical Assessment: Coding / DSA Test Score (0 to 100)
        # Students with more internships or high CGPA tend to have higher coding scores
        base_coding = 45.0 + (cgpa * 3.5) + (internships * 5.0)
        coding_score = int(np.clip(np.random.normal(loc=base_coding, scale=12.0), 20, 100))

        # Academic Projects Completed (1 to 5)
        projects = int(np.random.choice([1, 2, 3, 4, 5], p=[0.15, 0.35, 0.30, 0.15, 0.05]))

        # Industry Certifications (AWS, Python, Data Science, etc.)
        certs = int(np.random.choice([0, 1, 2, 3], p=[0.40, 0.38, 0.16, 0.06]))

        # General Aptitude Test Score (30 to 100)
        aptitude = int(np.clip(np.random.normal(loc=65, scale=14), 30, 100))

        # Communication / Soft Skills Mock Interview Rating (1.0 to 5.0)
        soft_skills = float(np.clip(np.random.normal(loc=3.5, scale=0.7), 1.0, 5.0))
        soft_skills = round(soft_skills, 1)

        # Hackathons attended
        hackathons = int(np.random.choice([0, 1, 2, 3], p=[0.60, 0.25, 0.11, 0.04]))

        # --- REALISTIC PLACEMENT PROBABILITY LOGIC ---
        readiness_score = 0.0
        # CGPA Contribution
        if cgpa >= 8.5:
            readiness_score += 0.35
        elif cgpa >= 7.0:
            readiness_score += 0.25
        else:
            readiness_score += 0.10

        # Coding / DSA Contribution
        if coding_score >= 80:
            readiness_score += 0.35
        elif coding_score >= 60:
            readiness_score += 0.25
        else:
            readiness_score += 0.08

        # Internships & Projects
        readiness_score += (internships * 0.10)
        readiness_score += (projects * 0.03)
        readiness_score += (certs * 0.03)

        # Soft Skills & Aptitude
        if soft_skills >= 4.0:
            readiness_score += 0.12
        if aptitude >= 70:
            readiness_score += 0.10

        # Backlogs Penalty (Severe in campus drives)
        if backlogs == 1:
            readiness_score -= 0.25
        elif backlogs >= 2:
            readiness_score -= 0.50

        placement_prob = float(np.clip(readiness_score, 0.05, 0.96))
        placed_status = 1 if np.random.rand() < placement_prob else 0

        # --- SALARY PACKAGE (CTC in LPA) PREDICTION ---
        if placed_status == 1:
            # Baseline entry salary (Service MNCs: 3.6 to 4.5 LPA)
            base_ctc = 3.6
            # Premium for DSA coding expertise
            if coding_score >= 85:
                base_ctc += float(np.random.uniform(5.0, 13.0)) # Tier-1 Product / Unicorn
            elif coding_score >= 70:
                base_ctc += float(np.random.uniform(2.5, 5.5))  # Differential / Growth Tech
            elif coding_score >= 55:
                base_ctc += float(np.random.uniform(0.8, 2.2))

            # Premium for Internships & Projects
            base_ctc += (internships * 0.8)
            base_ctc += (projects * 0.25)

            # Premium for CGPA
            if cgpa >= 9.0:
                base_ctc += 1.5
            elif cgpa >= 8.0:
                base_ctc += 0.8

            package_lpa = round(float(np.clip(base_ctc, 3.2, 24.0)), 2)
        else:
            package_lpa = 0.0

        records.append({
            "Student_ID": stu_id,
            "Degree_Stream": stream,
            "CGPA": cgpa,
            "Active_Backlogs": backlogs,
            "Internships_Count": internships,
            "Coding_DSA_Score": coding_score,
            "Projects_Completed": projects,
            "Certifications_Count": certs,
            "Aptitude_Test_Score": aptitude,
            "Soft_Skills_Rating": soft_skills,
            "Hackathons_Attended": hackathons,
            "Placement_Status": placed_status,
            "Salary_Package_LPA": package_lpa
        })

    df = pd.DataFrame(records)

    # Save to data directory
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(output_dir, exist_ok=True)
    raw_csv = os.path.join(output_dir, "student_placement_raw.csv")
    df.to_csv(raw_csv, index=False)

    print("\n[SUCCESS] Generated Student Placement Dataset Summary:")
    print(f" • File Saved: {os.path.abspath(raw_csv)}")
    print(f" • Total Students: {len(df):,}")
    print(f" • Overall Placement Rate: {(df['Placement_Status'] == 1).mean()*100:.2f}% ({(df['Placement_Status'] == 1).sum():,} Placed)")
    print(f" • Average CTC for Placed Students: INR {df[df['Placement_Status'] == 1]['Salary_Package_LPA'].mean():.2f} LPA")
    print(f" • Highest Package Offered: INR {df['Salary_Package_LPA'].max():.2f} LPA")
    print("=" * 70)

    return df

if __name__ == "__main__":
    generate_placement_dataset()
