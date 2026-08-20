"""
04_two_way_anova.py
--------------------
EXTENSION 1: TWO-WAY ANOVA WITH INTERACTION EFFECTS

A one-way ANOVA can tell you admission_type affects length of stay. It
CANNOT tell you whether that effect is the same for every age group, or
whether diabetes medication status changes the picture. A two-way ANOVA
tests both main effects AND the interaction term -- this is the step most
student projects skip, and it's exactly where the interesting findings
often live (e.g. "X matters, but only for group Y").

Model: time_in_hospital ~ admission_type * diabetesMed
    - Main effect of admission_type
    - Main effect of diabetesMed
    - Interaction effect (admission_type x diabetesMed)

Run:
    python analysis/04_two_way_anova.py
"""

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

df = pd.read_csv("data/processed/cohort_one_per_patient.csv")

# Keep only admission types with reasonable sample size for stable estimates
valid_types = df["admission_type"].value_counts()
valid_types = valid_types[valid_types > 500].index
model_df = df[df["admission_type"].isin(valid_types)].copy()

print(f"Two-way ANOVA sample: {len(model_df):,} patients across "
      f"{model_df['admission_type'].nunique()} admission types x 2 diabetesMed groups\n")

# statsmodels needs clean variable names for the formula API
model_df["admission_type_clean"] = model_df["admission_type"].str.replace(" ", "_").str.replace("/", "_")

formula = "time_in_hospital ~ C(admission_type_clean) * C(diabetesMed)"
model = ols(formula, data=model_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print("=" * 90)
print("TWO-WAY ANOVA: time_in_hospital ~ admission_type * diabetesMed")
print("=" * 90)
print(anova_table.round(6).to_string())

anova_table.to_csv("outputs/two_way_anova_results.csv")

# Interpretation helper -- pull out the interaction row specifically
interaction_row = [i for i in anova_table.index if ":" in i]
if interaction_row:
    p_interaction = anova_table.loc[interaction_row[0], "PR(>F)"]
    print(f"\nInteraction term p-value: {p_interaction:.6g}")
    if p_interaction < 0.05:
        print("=> SIGNIFICANT interaction: the effect of admission_type on length of "
              "stay DIFFERS depending on whether the patient is on diabetes medication. "
              "A one-way ANOVA would have hidden this.")
    else:
        print("=> No significant interaction: admission_type and diabetesMed affect "
              "length of stay independently/additively.")

# Show the actual cell means so the interaction (or lack of it) is visible
print("\nCell means (time_in_hospital) by admission_type x diabetesMed:")
pivot = model_df.pivot_table(index="admission_type", columns="diabetesMed",
                               values="time_in_hospital", aggfunc="mean").round(2)
print(pivot.to_string())
pivot.to_csv("outputs/two_way_anova_cell_means.csv")
