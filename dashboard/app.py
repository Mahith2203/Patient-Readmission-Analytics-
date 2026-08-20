"""
app.py
------
STEP 5: INTERACTIVE DASHBOARD

Explore the real Diabetes 130-US Hospitals dataset, including a live
hypothesis-testing panel and the propensity-matching result.

Run locally:
    streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats

st.set_page_config(page_title="Patient Readmission Analytics", layout="wide", page_icon="🏥")


@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/cohort_one_per_patient.csv")
    return df


df = load_data()

st.title("🏥 Patient Readmission & Hospital Utilization Analytics")
st.caption("Real data: 101,766 encounters, 130 US hospitals, 1999-2008 "
           "(UCI Diabetes 130-US Hospitals dataset, Strack et al. 2014) — "
           f"analyzed here on {len(df):,} deduped independent patients")

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")
age_range = st.sidebar.select_slider(
    "Age group range", options=sorted(df["age"].unique(), key=lambda x: int(x.strip("[)").split("-")[0])),
    value=(df["age"].min(), df["age"].max())
)
races = st.sidebar.multiselect("Race", sorted(df["race"].unique()), default=sorted(df["race"].unique()))

ages_sorted = sorted(df["age"].unique(), key=lambda x: int(x.strip("[)").split("-")[0]))
lo_idx, hi_idx = ages_sorted.index(age_range[0]), ages_sorted.index(age_range[1])
allowed_ages = ages_sorted[lo_idx:hi_idx + 1]

filtered = df[df["age"].isin(allowed_ages) & df["race"].isin(races)]

# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Patients", f"{len(filtered):,}")
col2.metric("30-Day Readmission Rate", f"{filtered['readmit_30'].mean()*100:.1f}%")
col3.metric("Avg Length of Stay", f"{filtered['time_in_hospital'].mean():.1f} days")
col4.metric("Avg # Medications", f"{filtered['num_medications'].mean():.1f}")

st.divider()

# ---------------------------------------------------------------------------
# Readmission by prior utilization
# ---------------------------------------------------------------------------
st.subheader("Readmission Rate by Prior Inpatient Visits (strongest predictor)")
filtered = filtered.copy()
filtered["prior_inpatient_bucket"] = filtered["number_inpatient"].clip(upper=4).astype(str)
filtered.loc[filtered["number_inpatient"] >= 4, "prior_inpatient_bucket"] = "4+"
rate_by_bucket = (filtered.groupby("prior_inpatient_bucket")["readmit_30"].mean() * 100).reindex(
    ["0", "1", "2", "3", "4+"]).reset_index()
fig1 = px.bar(rate_by_bucket, x="prior_inpatient_bucket", y="readmit_30",
              labels={"readmit_30": "Readmission rate (%)", "prior_inpatient_bucket": "Prior inpatient visits"})
st.plotly_chart(fig1, use_container_width=True)

# ---------------------------------------------------------------------------
# Length of stay by admission type / diabetes med
# ---------------------------------------------------------------------------
c1, c2 = st.columns(2)
with c1:
    st.subheader("Length of Stay by Admission Type")
    los_by_type = filtered.groupby("admission_type")["time_in_hospital"].mean().sort_values(ascending=False).reset_index()
    fig2 = px.bar(los_by_type, x="time_in_hospital", y="admission_type", orientation="h",
                  labels={"time_in_hospital": "Avg days", "admission_type": ""})
    st.plotly_chart(fig2, use_container_width=True)

with c2:
    st.subheader("Readmission Rate by Race")
    rate_by_race = (filtered.groupby("race")["readmit_30"].mean() * 100).sort_values(ascending=False).reset_index()
    fig3 = px.bar(rate_by_race, x="readmit_30", y="race", orientation="h",
                  labels={"readmit_30": "Readmission rate (%)", "race": ""})
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Live hypothesis test tool
# ---------------------------------------------------------------------------
st.subheader("🔬 Live Hypothesis Test Tool")
st.caption("Pick any two groups and run a real Welch's t-test on the filtered data, "
           "live, on a numeric outcome of your choice.")

test_col1, test_col2, test_col3 = st.columns(3)
group_var = test_col1.selectbox("Group by", ["admission_type", "race", "gender", "diabetesMed", "change"])
outcome_var = test_col2.selectbox("Outcome variable", [
    "time_in_hospital", "num_medications", "num_lab_procedures",
    "number_inpatient", "number_diagnoses"
])
available_groups = sorted(filtered[group_var].dropna().unique())
group_a = test_col3.selectbox("Group A", available_groups, index=0)
group_b = test_col3.selectbox("Group B", available_groups, index=min(1, len(available_groups) - 1))

if group_a != group_b:
    sample_a = filtered.loc[filtered[group_var] == group_a, outcome_var].dropna()
    sample_b = filtered.loc[filtered[group_var] == group_b, outcome_var].dropna()
    t_stat, p_val = stats.ttest_ind(sample_a, sample_b, equal_var=False)
    pooled_std = np.sqrt((sample_a.var() + sample_b.var()) / 2)
    cohens_d = (sample_a.mean() - sample_b.mean()) / pooled_std if pooled_std > 0 else 0

    r1, r2, r3, r4 = st.columns(4)
    r1.metric(f"{group_a} mean", f"{sample_a.mean():.2f}")
    r2.metric(f"{group_b} mean", f"{sample_b.mean():.2f}")
    r3.metric("p-value", f"{p_val:.4g}", delta="significant" if p_val < 0.05 else "not significant")
    r4.metric("Cohen's d", f"{cohens_d:.3f}")
else:
    st.info("Pick two different groups to compare.")

st.divider()

# ---------------------------------------------------------------------------
# Propensity matching result (pre-computed, displayed for context)
# ---------------------------------------------------------------------------
st.subheader("Causal-Style Finding: Does a Medication Change Affect Readmission?")
try:
    pm = pd.read_csv("outputs/propensity_matching_results.csv")
    pc1, pc2 = st.columns(2)
    naive = pm[pm["comparison"] == "naive"].iloc[0]
    matched = pm[pm["comparison"] == "propensity_matched"].iloc[0]
    pc1.metric("Naive difference", f"{naive['difference_pp']:+.2f} pp",
               help="Unadjusted comparison -- confounded by patient severity")
    pc2.metric("Propensity-matched difference", f"{matched['difference_pp']:+.2f} pp",
               help=f"p={matched.get('p_value', float('nan')):.3f} -- after balancing on 8 severity/utilization covariates")
    st.caption("The naive comparison overstates the effect of medication changes on readmission. "
               "After matching patients on how sick they were to begin with, most of the apparent "
               "effect disappears -- a classic confounding-by-indication pattern. "
               "See outputs/insights_report.md for full methodology.")
except FileNotFoundError:
    st.info("Run analysis/07_propensity_matching.py first to populate this section.")

st.divider()
st.caption("Data: UCI Diabetes 130-US Hospitals dataset (real, CC BY 4.0) | "
           "Built with Streamlit + Plotly + statsmodels + scikit-learn")
