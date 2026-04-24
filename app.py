import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Employee Attrition Risk by Business Travel", layout="wide")

st.title("Employee Attrition Risk by Business Travel Frequency")
st.markdown("Adjust the employee profile and view predicted attrition risk across business travel levels.")

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

ROLE_DEPARTMENT = {
    "Sales Executive": "Sales",
    "Sales Representative": "Sales",
    "Human Resources": "Human Resources",
    "Healthcare Representative": "Research & Development",
    "Laboratory Technician": "Research & Development",
    "Manager": "Research & Development",
    "Manufacturing Director": "Research & Development",
    "Research Director": "Research & Development",
    "Research Scientist": "Research & Development",
}

# CV F1 std from 5-fold cross-validation (see notebook)
CV_ERROR_PCT = 5.47

TRAVEL_LEVELS = [
    ("No", "Non-Travel"),
    ("Rare", "Travel_Rarely"),
    ("Frequent", "Travel_Frequently"),
]

st.sidebar.header("Employee Profile")

age = st.sidebar.slider("Age", 18, 60, 30)
years_at_company = st.sidebar.slider("Years at Company", 0, 40, 5)
job_satisfaction = st.sidebar.slider("Job Satisfaction", 1, 4, 3)
environment_satisfaction = st.sidebar.slider("Environment Satisfaction", 1, 4, 3)
job_involvement = st.sidebar.slider("Job Involvement", 1, 4, 3)
num_companies_worked = st.sidebar.slider("Number of Companies Worked", 0, 10, 2)
years_since_last_promotion = st.sidebar.slider("Years Since Last Promotion", 0, 15, 2)
years_with_curr_manager = st.sidebar.slider("Years With Current Manager", 0, 17, 3)
total_working_years = st.sidebar.slider("Total Working Years", 0, 40, 10)
distance_from_home = st.sidebar.slider("Distance From Home", 1, 30, 10)
job_level = st.sidebar.slider("Job Level", 1, 5, 2)

overtime = st.sidebar.selectbox("Overtime", ["No", "Yes"])
job_role = st.sidebar.selectbox("Job Role", list(ROLE_DEPARTMENT.keys()))
marital_status = st.sidebar.selectbox("Marital Status", ["Divorced", "Married", "Single"])
education_field = st.sidebar.selectbox(
    "Education Field",
    ["Life Sciences", "Medical", "Marketing", "Other", "Technical Degree", "Human Resources"]
)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])

department = ROLE_DEPARTMENT[job_role]


def build_row(business_travel: str) -> pd.DataFrame:
    row = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)

    numeric_values = {
        "Age": age,
        "YearsAtCompany": years_at_company,
        "JobSatisfaction": job_satisfaction,
        "EnvironmentSatisfaction": environment_satisfaction,
        "JobInvolvement": job_involvement,
        "NumCompaniesWorked": num_companies_worked,
        "YearsSinceLastPromotion": years_since_last_promotion,
        "YearsWithCurrManager": years_with_curr_manager,
        "TotalWorkingYears": total_working_years,
        "DistanceFromHome": distance_from_home,
        "JobLevel": job_level,
    }

    for col, val in numeric_values.items():
        if col in row.columns:
            row.at[0, col] = val

    # Baselines: BusinessTravel=Non-Travel, Department=R&D, EducationField=Life Sciences,
    # Gender=Female, JobRole=Sales Executive, MaritalStatus=Divorced, OverTime=No
    dummy_values = {
        "OverTime_Yes": int(overtime == "Yes"),
        "BusinessTravel_Travel_Rarely": int(business_travel == "Travel_Rarely"),
        "BusinessTravel_Travel_Frequently": int(business_travel == "Travel_Frequently"),
        "MaritalStatus_Married": int(marital_status == "Married"),
        "MaritalStatus_Single": int(marital_status == "Single"),
        "Department_Sales": int(department == "Sales"),
        "Department_Human Resources": int(department == "Human Resources"),
        "EducationField_Medical": int(education_field == "Medical"),
        "EducationField_Marketing": int(education_field == "Marketing"),
        "EducationField_Other": int(education_field == "Other"),
        "EducationField_Technical Degree": int(education_field == "Technical Degree"),
        "EducationField_Human Resources": int(education_field == "Human Resources"),
        "Gender_Male": int(gender == "Male"),
        "JobRole_Healthcare Representative": int(job_role == "Healthcare Representative"),
        "JobRole_Laboratory Technician": int(job_role == "Laboratory Technician"),
        "JobRole_Manager": int(job_role == "Manager"),
        "JobRole_Manufacturing Director": int(job_role == "Manufacturing Director"),
        "JobRole_Research Director": int(job_role == "Research Director"),
        "JobRole_Research Scientist": int(job_role == "Research Scientist"),
        "JobRole_Sales Representative": int(job_role == "Sales Representative"),
        "JobRole_Human Resources": int(job_role == "Human Resources"),
    }

    for col, val in dummy_values.items():
        if col in row.columns:
            row.at[0, col] = val

    return row


labels, probs = [], []
for label, travel_code in TRAVEL_LEVELS:
    row = build_row(travel_code)
    prob = model.predict_proba(scaler.transform(row))[0][1] * 100
    labels.append(label)
    probs.append(prob)

top_idx = int(np.argmax(probs))
st.subheader(f"Highest attrition risk score: {labels[top_idx]} travel ({probs[top_idx]:.1f})")

fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(len(labels))
bar_width = 0.5

ax.bar(x, probs, width=bar_width, color="steelblue")

for i, prob in enumerate(probs):
    upper = min(prob + CV_ERROR_PCT, 100)
    lower = max(prob - CV_ERROR_PCT, 0)
    half = bar_width * 0.35
    ax.plot([x[i] - half, x[i] + half], [prob, prob], color="green", linewidth=2)
    ax.plot([x[i] - half, x[i] + half], [upper, upper], color="red", linewidth=1.5)
    ax.plot([x[i] - half, x[i] + half], [lower, lower], color="red", linewidth=1.5)
    ax.text(x[i], 5, f"{prob:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold", color="white")
    ax.text(x[i], 1.5, f"±{CV_ERROR_PCT:.1f}%", ha="center", va="bottom", fontsize=8, color="white")

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 100)
ax.set_ylabel("Attrition Risk Score (0–100)")
ax.set_xlabel("Business Travel Frequency")
ax.set_title("Attrition Risk Score by Business Travel Frequency")
plt.tight_layout()
st.pyplot(fig)

st.caption(
    "Scores reflect model-adjusted attrition risk, not raw population probabilities. "
    "The model was trained with class reweighting to improve detection of at-risk employees, "
    "which inflates absolute scores relative to the base attrition rate (~16%). "
    "Scores are best interpreted comparatively across groups. "
    "Error bounds represent ±1 SD from 5-fold cross-validation. "
    "Not causal — use as a screening signal alongside managerial judgment."
)
