"""
07_propensity_matching.py
---------------------------
EXTENSION 4: PROPENSITY SCORE MATCHING -- CAUSAL-STYLE COMPARISON

Question: does changing a patient's diabetes medication during their stay
("change" == "Ch") affect their 30-day readmission risk?

The naive comparison (just compare readmission rates between the "changed"
and "not changed" groups) is confounded: doctors don't change medications
at random -- they're more likely to change medication for sicker patients
(more diagnoses, longer stays, more prior admissions), who are ALSO more
likely to be readmitted regardless of the medication change. A naive
comparison would make medication changes look worse than they are.

Propensity score matching addresses this: estimate each patient's
probability of receiving a medication change based on their observed
covariates (the "propensity score"), then match "changed" patients to
"not changed" patients with similar propensity scores. Comparing
readmission rates WITHIN matched pairs approximates a randomized
comparison on the observed confounders -- this is the same logic used in
observational clinical research when a randomized trial isn't available.

IMPORTANT CAVEAT (worth saying explicitly in an interview): this only
controls for confounders we measured. Unmeasured confounding (e.g. disease
severity not captured in the diagnosis codes) can still bias the estimate.
Propensity matching is a step up from a naive comparison, not a randomized
trial -- and that distinction is exactly what a good analytics interviewer
will ask about.

Run:
    python analysis/07_propensity_matching.py
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy import stats

df = pd.read_csv("data/processed/cohort_one_per_patient.csv")

# Treatment: diabetes medication was changed during this encounter
df["treatment"] = (df["change"] == "Ch").astype(int)

# Confounders: variables that plausibly affect BOTH whether a doctor changes
# medication AND readmission risk
covariates = ["age_midpoint", "time_in_hospital", "num_medications", "number_diagnoses",
              "number_inpatient", "number_emergency", "number_outpatient", "num_lab_procedures"]

model_df = df[["treatment", "readmit_30"] + covariates].dropna().copy()
print(f"Propensity matching sample: {len(model_df):,} patients "
      f"({model_df['treatment'].sum():,} treated / medication changed, "
      f"{(1 - model_df['treatment']).sum():,} control / not changed)\n")

# --- Step 1: naive (unadjusted) comparison, for contrast ---
naive_treated_rate = model_df.loc[model_df["treatment"] == 1, "readmit_30"].mean()
naive_control_rate = model_df.loc[model_df["treatment"] == 0, "readmit_30"].mean()
print(f"NAIVE comparison (no adjustment for confounders):")
print(f"  Readmission rate, medication changed:     {naive_treated_rate*100:.2f}%")
print(f"  Readmission rate, medication not changed: {naive_control_rate*100:.2f}%")
print(f"  Naive difference: {(naive_treated_rate - naive_control_rate)*100:+.2f} percentage points\n")

# --- Step 2: estimate propensity scores ---
X = model_df[covariates].values
y = model_df["treatment"].values

ps_model = LogisticRegression(max_iter=1000)
ps_model.fit(X, y)
model_df["propensity_score"] = ps_model.predict_proba(X)[:, 1]

print(f"Propensity model AUC (how well covariates predict treatment assignment): "
      f"{ps_model.score(X, y):.4f} accuracy")

# --- Step 3: nearest-neighbor matching (1:1, on propensity score) ---
treated = model_df[model_df["treatment"] == 1].reset_index(drop=True)
control = model_df[model_df["treatment"] == 0].reset_index(drop=True)

nn = NearestNeighbors(n_neighbors=1)
nn.fit(control[["propensity_score"]])
distances, indices = nn.kneighbors(treated[["propensity_score"]])

# Caliper: only keep matches within 0.01 propensity-score distance (standard practice)
CALIPER = 0.01
good_matches = distances.flatten() < CALIPER
matched_treated = treated[good_matches].reset_index(drop=True)
matched_control = control.iloc[indices.flatten()[good_matches]].reset_index(drop=True)

print(f"\nMatched {len(matched_treated):,} pairs within caliper={CALIPER} "
      f"({good_matches.sum()}/{len(treated)} treated patients found a close match)")

# --- Step 4: covariate balance check (did matching actually work?) ---
print("\nCovariate balance BEFORE vs AFTER matching (standardized mean difference):")
balance_rows = []
for cov in covariates:
    smd_before = (treated[cov].mean() - control[cov].mean()) / np.sqrt(
        (treated[cov].var() + control[cov].var()) / 2)
    smd_after = (matched_treated[cov].mean() - matched_control[cov].mean()) / np.sqrt(
        (matched_treated[cov].var() + matched_control[cov].var()) / 2)
    balance_rows.append({"covariate": cov, "smd_before": round(smd_before, 4),
                          "smd_after": round(smd_after, 4)})
    print(f"  {cov:20s} before={smd_before:+.4f}  after={smd_after:+.4f}  "
          f"{'OK (<0.1)' if abs(smd_after) < 0.1 else 'still imbalanced'}")

pd.DataFrame(balance_rows).to_csv("outputs/propensity_covariate_balance.csv", index=False)

# --- Step 5: matched comparison ---
matched_treated_rate = matched_treated["readmit_30"].mean()
matched_control_rate = matched_control["readmit_30"].mean()

# McNemar-style paired comparison since these are matched pairs
t_stat, p_val = stats.ttest_rel(matched_treated["readmit_30"], matched_control["readmit_30"])

print(f"\nMATCHED comparison (confounders balanced via propensity score):")
print(f"  Readmission rate, medication changed (matched):     {matched_treated_rate*100:.2f}%")
print(f"  Readmission rate, medication not changed (matched):  {matched_control_rate*100:.2f}%")
print(f"  Matched difference: {(matched_treated_rate - matched_control_rate)*100:+.2f} percentage points")
print(f"  Paired t-test: t={t_stat:.4f}, p={p_val:.6g}")

print(f"\nBOTTOM LINE:")
print(f"  Naive difference:   {(naive_treated_rate - naive_control_rate)*100:+.2f} pp")
print(f"  Matched difference: {(matched_treated_rate - matched_control_rate)*100:+.2f} pp")
if abs(naive_treated_rate - naive_control_rate) > abs(matched_treated_rate - matched_control_rate):
    print("  => The naive comparison OVERSTATED the effect. Once we match patients on "
          "how sick they were to begin with, most of the apparent effect of medication "
          "changes on readmission shrinks -- consistent with confounding-by-indication "
          "(sicker patients get their meds changed AND are more likely to be readmitted).")
else:
    print("  => The effect persists even after matching on observed severity indicators, "
          "which is somewhat more (though not conclusively) suggestive of a real effect.")

results_summary = pd.DataFrame([{
    "comparison": "naive", "treated_rate": naive_treated_rate, "control_rate": naive_control_rate,
    "difference_pp": (naive_treated_rate - naive_control_rate) * 100
}, {
    "comparison": "propensity_matched", "treated_rate": matched_treated_rate,
    "control_rate": matched_control_rate,
    "difference_pp": (matched_treated_rate - matched_control_rate) * 100,
    "p_value": p_val
}])
results_summary.to_csv("outputs/propensity_matching_results.csv", index=False)
print("\nSaved: outputs/propensity_matching_results.csv, outputs/propensity_covariate_balance.csv")
