"""
08_generate_charts.py
-----------------------
STEP 4: CHART GENERATION

Generates charts supporting each major finding: readmission by utilization
history, the two-way ANOVA interaction, logistic regression odds ratios,
and the propensity matching before/after comparison.

Run:
    python analysis/08_generate_charts.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120
CHART_DIR = "outputs/charts"

df = pd.read_csv("data/processed/cohort_one_per_patient.csv")

# ---------------------------------------------------------------------------
# 1. Readmission rate by number of prior inpatient visits (strongest predictor)
# ---------------------------------------------------------------------------
df["prior_inpatient_bucket"] = df["number_inpatient"].clip(upper=4).astype(str)
df.loc[df["number_inpatient"] >= 4, "prior_inpatient_bucket"] = "4+"
rate_by_bucket = df.groupby("prior_inpatient_bucket")["readmit_30"].mean().reindex(
    ["0", "1", "2", "3", "4+"]) * 100

plt.figure(figsize=(8, 5))
sns.barplot(x=rate_by_bucket.index, y=rate_by_bucket.values, color="#C44E52")
plt.title("30-Day Readmission Rate by Prior Inpatient Visits (past year)")
plt.xlabel("Number of prior inpatient visits")
plt.ylabel("Readmission rate (%)")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/01_readmit_by_prior_inpatient.png")
plt.close()

# ---------------------------------------------------------------------------
# 2. Two-way ANOVA interaction plot
# ---------------------------------------------------------------------------
valid_types = df["admission_type"].value_counts()
valid_types = valid_types[valid_types > 500].index
interaction_df = df[df["admission_type"].isin(valid_types)]
pivot = interaction_df.pivot_table(index="admission_type", columns="diabetesMed",
                                     values="time_in_hospital", aggfunc="mean")

plt.figure(figsize=(8, 5))
for col in pivot.columns:
    plt.plot(pivot.index, pivot[col], marker="o", label=f"diabetesMed={col}")
plt.title("Interaction Effect: Admission Type x Diabetes Medication\non Length of Stay")
plt.ylabel("Mean time in hospital (days)")
plt.xlabel("Admission type")
plt.legend(title="On diabetes medication")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/02_two_way_anova_interaction.png")
plt.close()

# ---------------------------------------------------------------------------
# 3. Logistic regression odds ratios (forest plot style)
# ---------------------------------------------------------------------------
or_table = pd.read_csv("outputs/logistic_regression_odds_ratios.csv", index_col=0)
or_table = or_table[or_table.index != "Intercept"].sort_values("odds_ratio")

plt.figure(figsize=(9, 7))
colors = ["#C44E52" if p < 0.05 else "#AAAAAA" for p in or_table["p_value"]]
plt.errorbar(or_table["odds_ratio"], range(len(or_table)),
             xerr=[or_table["odds_ratio"] - or_table["or_ci_lower"],
                   or_table["or_ci_upper"] - or_table["odds_ratio"]],
             fmt="o", color="black", ecolor="gray", capsize=3)
for i, (idx, row) in enumerate(or_table.iterrows()):
    plt.scatter(row["odds_ratio"], i, color=colors[i], zorder=5, s=60)
plt.axvline(1, linestyle="--", color="black", alpha=0.5)
plt.yticks(range(len(or_table)), or_table.index)
plt.xlabel("Odds Ratio (95% CI) for 30-day readmission")
plt.title("Logistic Regression: Odds Ratios\n(red = significant at p<0.05)")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/03_odds_ratios_forest_plot.png")
plt.close()

# ---------------------------------------------------------------------------
# 4. Propensity matching: naive vs matched comparison
# ---------------------------------------------------------------------------
pm_results = pd.read_csv("outputs/propensity_matching_results.csv")
plt.figure(figsize=(7, 5))
x = pm_results["comparison"]
plt.bar(x, pm_results["difference_pp"], color=["#C44E52", "#4C72B0"])
plt.axhline(0, color="black", linewidth=0.8)
plt.ylabel("Readmission rate difference\n(medication changed - not changed, pp)")
plt.title("Naive vs. Propensity-Matched Comparison\n(Medication Change -> Readmission)")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/04_propensity_naive_vs_matched.png")
plt.close()

# ---------------------------------------------------------------------------
# 5. Effect sizes summary (Cohen's d for all t-tests)
# ---------------------------------------------------------------------------
corrected = pd.read_csv("outputs/corrected_test_results.csv")
d_rows = corrected[corrected["effect_type"] == "Cohens_d"].copy()
d_rows["test"] = d_rows["test"].str.replace("welch t-test: ", "")
d_rows = d_rows.sort_values("effect_size")

plt.figure(figsize=(8, 5))
sns.barplot(x=d_rows["effect_size"], y=d_rows["test"], color="#55A868")
plt.axvline(0.2, linestyle="--", color="gray", alpha=0.6, label="small (0.2)")
plt.axvline(0.5, linestyle="--", color="orange", alpha=0.6, label="medium (0.5)")
plt.title("Effect Sizes (Cohen's d) — Statistically Significant\nDoesn't Always Mean Practically Large")
plt.xlabel("Cohen's d")
plt.legend()
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/05_effect_sizes_cohens_d.png")
plt.close()

print("All charts saved to outputs/charts/")
print("Files:", sorted([f for f in __import__("os").listdir(CHART_DIR)]))
