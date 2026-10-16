# 🎓 CampusPlacement-IQ - College TPO Power BI Setup & DAX Guide

This guide details how to build the **College Training & Placement Officer (TPO) Executive Dashboard** in Power BI Desktop using `data/student_placement_final_scored.csv`.

---

## 🚀 Step 1: Connect Dataset
1. Open **Power BI Desktop**.
2. Click **Get Data** -> **Text/CSV**.
3. Select: `CampusPlacement-IQ/data/student_placement_final_scored.csv`.
4. Click **Close & Apply**.

---

## 🧮 Step 2: Essential DAX Measures (Copy & Paste)

### 1. Total Student Pool
```dax
Total Students = DISTINCTCOUNT(student_placement_final_scored[Student_ID])
```

### 2. Overall Placement Rate %
```dax
Placement Rate % = 
DIVIDE(
    CALCULATE(COUNTROWS(student_placement_final_scored), student_placement_final_scored[Placement_Status] = 1),
    COUNTROWS(student_placement_final_scored),
    0
)
```

### 3. Average CTC Package Offered (Placed Students)
```dax
Average Package (LPA) = 
CALCULATE(
    AVERAGE(student_placement_final_scored[Salary_Package_LPA]),
    student_placement_final_scored[Placement_Status] = 1
)
```

### 4. Highest CTC Offer
```dax
Highest Package (LPA) = MAX(student_placement_final_scored[Salary_Package_LPA])
```

### 5. Dream Offers Count (>10 LPA)
```dax
Dream Offers Count = 
CALCULATE(
    COUNTROWS(student_placement_final_scored),
    student_placement_final_scored[Salary_Package_LPA] >= 10.0
)
```

### 6. Students with Active Backlogs Count
```dax
Students with Backlogs = 
CALCULATE(
    COUNTROWS(student_placement_final_scored),
    student_placement_final_scored[Active_Backlogs] > 0
)
```

---

## 🖥️ Step 3: 3-Page Dashboard Layout

### 📑 Page 1: College TPO Executive Overview
- **Top KPIs:**
  - `[Total Students]` (10,000)
  - `[Placement Rate %]` (66.5%)
  - `[Average Package (LPA)]` (₹10.08 LPA)
  - `[Highest Package (LPA)]` (₹20.63 LPA)
  - `[Dream Offers Count]` (3,014 students)
- **Visual 1 (Clustered Column Chart):** Placement Rate % by `Degree_Stream` (MCA, B.Tech CSE, IT, BCA)
- **Visual 2 (Bar Chart):** Average Salary Package (LPA) by `Degree_Stream`
- **Visual 3 (Donut Chart):** Career Readiness Tiers (Dream Tier-1 vs Service MNC vs At-Risk)

### 📑 Page 2: Skill Gaps & Backlogs Impact
- **Visual 1 (Bar Chart):** Placement Probability by `Active_Backlogs` (0 vs 1 vs 2+ backlogs)
- **Visual 2 (Scatter Plot):** `Coding_DSA_Score` (X) vs `Salary_Package_LPA` (Y) colored by `Degree_Stream`.
- **Slicers:** `Degree_Stream`, `Active_Backlogs`, `Internships_Count`.

### 📑 Page 3: Student Career Triage & Upskilling Action List
- **Table of At-Risk Students:**
  - Columns: `Student_ID`, `Degree_Stream`, `CGPA`, `Active_Backlogs`, `Coding_DSA_Score`, `Placement_Probability`, `Readiness_Tier`.
  - Filtered where `Readiness_Tier = "At-Risk (Needs Skill Intervention)"`.
  - Allows TPO cell to assign remedial coding workshops and mock interviews.
