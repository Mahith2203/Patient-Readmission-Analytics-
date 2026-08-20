"""
05_logistic_regression.py
--------------------------
EXTENSION 2: MULTIPLE LOGISTIC REGRESSION -- PREDICTING 30-DAY READMISSION

Moves beyond "is X associated with readmission" (a single chi-square/t-test
at a time) to "how much does X matter AFTER controlling for everything else."
This is the difference between a bivariate finding and a real predictive
model, and it's the natural next step once several individual tests come
back significant.

Reports odds ratios (not just coefficients) with 95% CIs, since odds ratios
are what a non-technical stakeholder can actually interpret ("each extra
prior inpatient visit multiplies readmission odds by X").

Run:
    python analysis/05_logistic_regression.py
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score

df = pd.read_csv("data/processed/cohort_one_per_patient.csv")

# Feature set: mix of demographics, utilization history, and encounter severity.
# number_inpatient/emergency/outpatient = visits in the year BEFORE this encounter
# -- these are the strongest, most clinically sensible predictors of readmission.
features = [
    "age_midpoint", "time_in_hospital", "num_medications", "num_lab_procedures",
    "num_procedures", "number_diagnoses", "number_inpatient", "number_emergency",
    "number_outpatient",
]
categorical = ["race", "gender", "diabetesMed", "change"]

model_df = df[["readmit_30"] + features + categorical].dropna().copy()
model_df = model_df[model_df["gender"] != "Unknown/Invalid"]  # 3 rows, drop for clean modeling

print(f"Logistic regression sample: {len(model_df):,} patients\n")

formula = ("readmit_30 ~ age_midpoint + time_in_hospital + num_medications + "
           "num_lab_procedures + num_procedures + number_diagnoses + "
           "number_inpatient + number_emergency + number_outpatient + "
           "C(race) + C(gender) + C(diabetesMed) + C(change)")

model = smf.logit(formula, data=model_df).fit(disp=0)

print("=" * 100)
print(model.summary())
print("=" * 100)

# Build a clean odds-ratio table (more interpretable than raw log-odds coefficients)
conf = model.conf_int()
conf.columns = ["ci_lower", "ci_upper"]
or_table = pd.DataFrame({
    "coef": model.params,
    "odds_ratio": np.exp(model.params),
    "or_ci_lower": np.exp(conf["ci_lower"]),
    "or_ci_upper": np.exp(conf["ci_upper"]),
    "p_value": model.pvalues,
}).round(4)
or_table = or_table.sort_values("p_value")

print("\nOdds Ratio Table (sorted by significance):")
print(or_table.to_string())
or_table.to_csv("outputs/logistic_regression_odds_ratios.csv")

# Model quality metrics
pred_probs = model.predict(model_df)
auc = roc_auc_score(model_df["readmit_30"], pred_probs)
print(f"\nPseudo R-squared (McFadden): {model.prsquared:.4f}")
print(f"AUC-ROC: {auc:.4f}")
print("\nNote: readmission is a genuinely hard prediction problem in this literature "
      "(published models on this exact dataset report AUC in the 0.60-0.68 range) -- "
      "a modest AUC here is expected and consistent with prior published work, not a bug.")

with open("outputs/logistic_regression_summary.txt", "w") as f:
    f.write(str(model.summary()))
    f.write(f"\n\nPseudo R-squared (McFadden): {model.prsquared:.4f}\n")
    f.write(f"AUC-ROC: {auc:.4f}\n")
print("\nFull model summary saved to outputs/logistic_regression_summary.txt")
