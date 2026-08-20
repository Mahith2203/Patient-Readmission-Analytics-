"""
03_statistical_tests.py
------------------------
STEP 3: CORE HYPOTHESIS TESTS

Runs the standard hypothesis-testing toolkit against the REAL, deduped
patient cohort (data/processed/cohort_one_per_patient.csv):
    - Chi-square tests of independence (categorical vs. readmission)
    - Welch's t-test (continuous variable vs. readmission group)
    - One-way ANOVA (continuous variable across >2 groups)

All tests use the one-encounter-per-patient cohort so observations are
independent (see 02_clean_data.py for why). Raw p-values from every test
run here are collected and passed through multiple-testing correction in
06_effect_sizes_multiple_testing.py -- do not treat any p-value in THIS
script's output as final without checking that correction.

Run:
    python analysis/03_statistical_tests.py
"""

import pandas as pd
from scipy import stats

df = pd.read_csv("data/processed/cohort_one_per_patient.csv")

results = []  # collect (test_name, statistic, p_value, n) for later correction


def record(name, stat, p, n):
    results.append({"test": name, "statistic": round(stat, 4), "p_value": p, "n": n})
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"{name:55s} stat={stat:10.4f}  p={p:.6g}  n={n:,}  {sig}")


print("=" * 100)
print("CHI-SQUARE TESTS OF INDEPENDENCE (categorical variable vs. 30-day readmission)")
print("=" * 100)

for col in ["race", "gender", "admission_type", "change", "diabetesMed", "A1Cresult", "max_glu_serum"]:
    ct = pd.crosstab(df[col], df["readmit_30"])
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    record(f"chi2: {col} vs readmit_30", chi2, p, ct.values.sum())

print()
print("=" * 100)
print("WELCH'S T-TEST (continuous variable, readmitted-within-30-days vs. not)")
print("=" * 100)

readmit_yes = df[df["readmit_30"] == 1]
readmit_no = df[df["readmit_30"] == 0]

for col in ["time_in_hospital", "num_medications", "num_lab_procedures",
            "number_inpatient", "number_emergency", "number_diagnoses"]:
    t_stat, p = stats.ttest_ind(readmit_yes[col], readmit_no[col], equal_var=False)
    record(f"welch t-test: {col}", t_stat, p, len(df))
    print(f"    mean readmitted<30={readmit_yes[col].mean():.2f}  "
          f"mean not={readmit_no[col].mean():.2f}")

print()
print("=" * 100)
print("ONE-WAY ANOVA (time_in_hospital across admission_type groups)")
print("=" * 100)

groups = [g["time_in_hospital"].values for _, g in df.groupby("admission_type") if len(g) > 30]
f_stat, p = stats.f_oneway(*groups)
record("ANOVA: time_in_hospital by admission_type", f_stat, p, len(df))

print()
print("=" * 100)
print("ONE-WAY ANOVA (num_medications across age groups)")
print("=" * 100)

groups_age = [g["num_medications"].values for _, g in df.groupby("age")]
f_stat2, p2 = stats.f_oneway(*groups_age)
record("ANOVA: num_medications by age group", f_stat2, p2, len(df))

# Save raw results for downstream correction script
pd.DataFrame(results).to_csv("outputs/raw_test_results.csv", index=False)
print(f"\nSaved {len(results)} raw test results to outputs/raw_test_results.csv "
      f"(uncorrected -- see 06_effect_sizes_multiple_testing.py for corrected p-values)")
