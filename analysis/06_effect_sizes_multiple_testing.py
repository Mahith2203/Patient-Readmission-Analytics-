"""
06_effect_sizes_multiple_testing.py
-------------------------------------
EXTENSION 3: EFFECT SIZES + MULTIPLE-TESTING CORRECTION

Two things most student projects skip, both flagged as gaps in the
original synthetic-data version of this project:

1. EFFECT SIZES: a p-value tells you whether an effect is *statistically*
   detectable, not whether it's *practically* meaningful. With ~70,000
   patients, even trivial differences become "significant" (p < 0.001).
   Effect sizes (Cohen's d, Cramer's V, odds ratios) tell you how big the
   effect actually is -- this is the difference between "significant" and
   "significant AND worth acting on."

2. MULTIPLE-TESTING CORRECTION: 03_statistical_tests.py ran 15 hypothesis
   tests on the same dataset. Running that many tests at alpha=0.05 means
   a meaningfully elevated chance of at least one false positive by chance
   alone, even if every null hypothesis were true. Bonferroni (conservative)
   and Benjamini-Hochberg/FDR (less conservative, more standard in practice)
   correction is applied here so the final significance calls are honest.

Run:
    python analysis/06_effect_sizes_multiple_testing.py
Requires:
    outputs/raw_test_results.csv (written by 03_statistical_tests.py)
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

df = pd.read_csv("data/processed/cohort_one_per_patient.csv")
raw_results = pd.read_csv("outputs/raw_test_results.csv")


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------
def cramers_v(chi2, n, r, c):
    """Cramer's V: effect size for chi-square. 0=no association, 1=perfect."""
    return np.sqrt((chi2 / n) / (min(r - 1, c - 1)))


def cohens_d(group1, group2):
    """Cohen's d: standardized mean difference. ~0.2 small, ~0.5 medium, ~0.8 large."""
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * group1.var() + (n2 - 1) * group2.var()) / (n1 + n2 - 2))
    return (group1.mean() - group2.mean()) / pooled_std


print("=" * 100)
print("EFFECT SIZES: CHI-SQUARE TESTS (Cramer's V)")
print("=" * 100)

chi2_cols = ["race", "gender", "admission_type", "change", "diabetesMed", "A1Cresult", "max_glu_serum"]
effect_size_rows = []

for col in chi2_cols:
    ct = pd.crosstab(df[col], df["readmit_30"])
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    v = cramers_v(chi2, ct.values.sum(), ct.shape[0], ct.shape[1])
    strength = "negligible" if v < 0.1 else ("small" if v < 0.2 else ("medium" if v < 0.3 else "large"))
    print(f"{col:25s} Cramer's V = {v:.4f}  ({strength})")
    effect_size_rows.append({"test": f"chi2: {col} vs readmit_30", "effect_size": round(v, 4),
                              "effect_type": "Cramers_V", "interpretation": strength})

print()
print("=" * 100)
print("EFFECT SIZES: T-TESTS (Cohen's d)")
print("=" * 100)

readmit_yes = df[df["readmit_30"] == 1]
readmit_no = df[df["readmit_30"] == 0]
ttest_cols = ["time_in_hospital", "num_medications", "num_lab_procedures",
              "number_inpatient", "number_emergency", "number_diagnoses"]

for col in ttest_cols:
    d = cohens_d(readmit_yes[col], readmit_no[col])
    strength = "negligible" if abs(d) < 0.2 else ("small" if abs(d) < 0.5 else ("medium" if abs(d) < 0.8 else "large"))
    print(f"{col:25s} Cohen's d = {d:.4f}  ({strength})")
    effect_size_rows.append({"test": f"welch t-test: {col}", "effect_size": round(d, 4),
                              "effect_type": "Cohens_d", "interpretation": strength})

effect_df = pd.DataFrame(effect_size_rows)

# ---------------------------------------------------------------------------
# Multiple testing correction
# ---------------------------------------------------------------------------
print()
print("=" * 100)
print("MULTIPLE-TESTING CORRECTION (15 tests run in 03_statistical_tests.py)")
print("=" * 100)

pvals = raw_results["p_value"].values
reject_bonf, p_bonf, _, _ = multipletests(pvals, alpha=0.05, method="bonferroni")
reject_fdr, p_fdr, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")

raw_results["p_bonferroni"] = p_bonf
raw_results["sig_bonferroni_0.05"] = reject_bonf
raw_results["p_fdr_bh"] = p_fdr
raw_results["sig_fdr_0.05"] = reject_fdr

merged = raw_results.merge(effect_df, on="test", how="left")
print(merged.to_string(index=False))

merged.to_csv("outputs/corrected_test_results.csv", index=False)

n_sig_raw = (raw_results["p_value"] < 0.05).sum()
n_sig_bonf = reject_bonf.sum()
n_sig_fdr = reject_fdr.sum()

print(f"\nSignificant at raw alpha=0.05:        {n_sig_raw}/{len(raw_results)}")
print(f"Significant after Bonferroni:          {n_sig_bonf}/{len(raw_results)}")
print(f"Significant after Benjamini-Hochberg:   {n_sig_fdr}/{len(raw_results)}")
print("\nSaved: outputs/corrected_test_results.csv (this is the table to cite -- "
      "not the raw p-values from 03_statistical_tests.py)")
