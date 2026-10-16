"""
=============================================================================
CampusPlacement-IQ - Script 02: Exploratory Data Analysis & Career Metrics
=============================================================================
Analyzes 10,000 student profiles to uncover:
1. Degree & Branch placement success rates (MCA vs B.Tech CSE vs BCA)
2. Backlog penalties and eligibility cutoffs
3. Correlation between Coding/DSA proficiency and high-tier CTC offers
4. Generates visual charts in outputs/ for Power BI and portfolio
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Windows console encoding safeguard
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'Segoe UI'

def run_placement_eda():
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "..", "data")
    output_dir = os.path.join(base_dir, "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    input_file = os.path.join(data_dir, "student_placement_raw.csv")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found at: {input_file}")

    print("=" * 70)
    print("📊 [CampusPlacement-IQ Step 2] Exploratory Career Analytics & EDA")
    print("=" * 70)

    df = pd.read_csv(input_file)
    print(f"[+] Loaded {len(df):,} student records.")

    placed_df = df[df['Placement_Status'] == 1]
    overall_placed_pct = df['Placement_Status'].mean() * 100
    avg_package = placed_df['Salary_Package_LPA'].mean()
    median_package = placed_df['Salary_Package_LPA'].median()
    max_package = placed_df['Salary_Package_LPA'].max()

    print(f"[+] Overall Placement Rate: {overall_placed_pct:.2f}% ({len(placed_df):,} Placed)")
    print(f"[+] Average Salary Package (Placed): INR {avg_package:.2f} LPA")
    print(f"[+] Median Package: INR {median_package:.2f} LPA | Highest: INR {max_package:.2f} LPA")

    # 1. Degree Branch Performance
    branch_summary = df.groupby('Degree_Stream').agg(
        Total_Students=('Student_ID', 'count'),
        Placed_Count=('Placement_Status', 'sum'),
        Avg_CGPA=('CGPA', 'mean'),
        Avg_DSA_Score=('Coding_DSA_Score', 'mean')
    )
    branch_summary['Placement_Rate_%'] = (branch_summary['Placed_Count'] / branch_summary['Total_Students']) * 100
    
    # Calculate Avg Package only for placed students in each branch
    avg_pkgs = placed_df.groupby('Degree_Stream')['Salary_Package_LPA'].mean()
    branch_summary['Avg_CTC_LPA'] = avg_pkgs

    print("\n--- Branch & Degree Performance Summary ---")
    print(branch_summary[['Total_Students', 'Placement_Rate_%', 'Avg_CTC_LPA', 'Avg_DSA_Score']].to_string())

    # 2. Backlogs Impact
    backlog_impact = df.groupby('Active_Backlogs')['Placement_Status'].agg(['count', 'mean']).rename(
        columns={'count': 'Total_Students', 'mean': 'Placement_Rate'}
    )
    backlog_impact['Placement_Rate_%'] = backlog_impact['Placement_Rate'] * 100
    print("\n--- Impact of Active Backlogs on Placement ---")
    print(backlog_impact[['Total_Students', 'Placement_Rate_%']].to_string())

    # 3. Visualizations
    print("\n[+] Generating Career Analytics Visualizations...")

    # Plot 1: Placement Rate & Avg CTC across Degrees
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    branch_summary['Placement_Rate_%'].plot(kind='bar', ax=axes[0], color='#2563eb', rot=0)
    axes[0].set_title("Campus Placement Rate % by Degree Program", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Placement Rate (%)")
    axes[0].set_ylim(0, 100)

    branch_summary['Avg_CTC_LPA'].plot(kind='bar', ax=axes[1], color='#10b981', rot=0)
    axes[1].set_title("Average CTC Package Offered (INR LPA)", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Package (LPA)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_branch_placement_ctc.png"), dpi=200)
    plt.close()

    # Plot 2: Backlogs Impact Bar Chart
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(backlog_impact.index.astype(str), backlog_impact['Placement_Rate_%'], color=['#10b981', '#f59e0b', '#ef4444', '#b91c1c'])
    ax.set_title("Placement Rate Penalty by Active Backlogs Count", fontsize=13, fontweight='bold')
    ax.set_xlabel("Number of Active Backlogs")
    ax.set_ylabel("Placement Probability (%)")
    ax.set_ylim(0, 100)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 2, f"{h:.1f}%", ha='center', fontweight='bold', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_backlogs_vs_placement_rate.png"), dpi=200)
    plt.close()

    # Plot 3: Coding DSA Score vs Salary Package (LPA)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(
        data=placed_df,
        x='Coding_DSA_Score',
        y='Salary_Package_LPA',
        hue='Degree_Stream',
        alpha=0.6,
        s=35,
        ax=ax
    )
    ax.set_title("Technical Coding (DSA) Proficiency vs CTC Package (LPA)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Coding / DSA Assessment Score (0 - 100)")
    ax.set_ylabel("Offered Package (INR Lakhs / Annum)")
    ax.legend(title="Degree Program", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_dsa_vs_package_scatter.png"), dpi=200)
    plt.close()

    # 4. Save processed dataset
    processed_file = os.path.join(data_dir, "student_placement_processed.csv")
    df.to_csv(processed_file, index=False)

    print("\n[SUCCESS] Placement EDA Complete!")
    print(f" • Processed Dataset: {os.path.abspath(processed_file)}")
    print(f" • Plots Saved in: {os.path.abspath(output_dir)}")
    print("=" * 70)

    return df

if __name__ == "__main__":
    run_placement_eda()
