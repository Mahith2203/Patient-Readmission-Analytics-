# Insights Report — Patient Readmission & Hospital Utilization Analytics

**Data:** 101,766 real inpatient encounters, 130 US hospitals (1999–2008), UCI Diabetes 130-US
Hospitals dataset (Strack et al., 2014). After excluding expired/hospice discharges and deduping
to one independent encounter per patient: **69,990 real patients** analyzed.

---

## 1. Prior Hospital Utilization Is the Strongest Predictor of Readmission

| Prior inpatient visits | Readmission rate |
|---|---|
| 0 | ~7% |
| 1 | ~14% |
| 2 | ~19% |
| 3 | ~23% |
| 4+ | ~28%+ |

**Test:** Welch's t-test, `number_inpatient` vs. 30-day readmission — t=16.78, p<0.0001,
**Cohen's d=0.35 (small-to-moderate, the largest effect size found in this analysis)**.
Confirmed in the multiple logistic regression: each additional prior inpatient visit
multiplies readmission odds by **1.40x (95% CI: 1.36–1.45)**, holding everything else constant.

**Recommendation:** A risk-stratification flag based on prior-year inpatient visit count alone
is a simple, defensible first pass for identifying high-risk patients at discharge — before
building anything more complex.

---

## 2. Being on Diabetes Medication Increases Readmission Odds — But the Effect Depends on Admission Type

Two-way ANOVA on length of stay (`time_in_hospital ~ admission_type × diabetesMed`) found a
**significant interaction** (F=8.57, p=1.1×10⁻⁵): the gap in length of stay between medicated
and non-medicated patients is not constant across admission types.

| Admission Type | Not on diabetes med | On diabetes med | Gap |
|---|---|---|---|
| Elective | 3.58 days | 4.25 days | +0.67 |
| Emergency | 4.02 days | 4.39 days | +0.37 |
| Not Available | 3.25 days | 3.98 days | +0.73 |
| Urgent | 4.06 days | 4.56 days | +0.50 |

**Recommendation:** A one-way analysis ("diabetes med patients stay longer, on average") would
have missed that the gap is nearly twice as large for elective admissions as for emergency ones —
suggesting elective-admission care pathways specifically may benefit from closer diabetes
management coordination.

---

## 3. Statistical Significance ≠ Practical Significance (and This Matters at N=70,000)

Running effect sizes alongside every test revealed a pattern worth stating plainly: **almost
every test came back statistically significant, but almost every effect size was negligible.**

| Test | p-value | Effect size | Interpretation |
|---|---|---|---|
| race vs. readmission | 0.0025 | Cramer's V = 0.016 | negligible |
| diabetesMed vs. readmission | <0.0001 | Cramer's V = 0.028 | negligible |
| time_in_hospital (t-test) | <0.0001 | Cohen's d = 0.196 | negligible |
| number_inpatient (t-test) | <0.0001 | Cohen's d = 0.353 | small |

At n≈70,000, even a trivial real difference clears p<0.05. **`number_inpatient` is the only
variable tested whose effect size clears even the "small" threshold** — every other significant
result, while real, is too small to act on alone.

**Recommendation:** Don't let a wall of asterisks substitute for judgment. Before recommending
any intervention off a hypothesis test, check the effect size — and at this sample size, treat
"negligible" effect sizes as descriptive facts, not action items.

---

## 4. Multiple-Testing Correction Changes Which Findings Survive

15 hypothesis tests were run against the same dataset. Correcting for that:

| Correction | Significant findings |
|---|---|
| Raw (α=0.05) | 12 / 15 |
| Bonferroni | 11 / 15 |
| Benjamini-Hochberg (FDR) | 12 / 15 |

The `admission_type vs. readmission` chi-square test (raw p=0.038) is the interesting case:
it survives FDR correction (p=0.048) but **not** Bonferroni (p=0.576). Which correction is
"right" depends on the cost of a false positive vs. a false negative for this specific decision
— Bonferroni is appropriate when a false positive is costly (e.g., a policy change), FDR when
missing a real effect is costly (e.g., exploratory screening).

**Recommendation:** Report the correction method used, always — an uncorrected p-value from a
15-test run should never be presented as if it were the only test conducted.

---

## 5. Medication Changes Don't Meaningfully Affect Readmission — Once You Control for Confounding

**Naive comparison:** patients whose diabetes medication was changed during their stay had a
9.44% readmission rate vs. 8.60% for patients whose medication wasn't changed (+0.83 points).

This naive comparison is confounded: doctors change medication more often for sicker patients,
who are independently more likely to be readmitted regardless of the medication decision.

**Propensity score matching** (matching on age, length of stay, medication count, diagnosis
count, and prior utilization; 31,495 matched pairs; all covariates balanced to SMD<0.1 after
matching) found the **matched difference shrinks to +0.31 points and is no longer statistically
significant (paired t-test p=0.17)**.

**Recommendation:** Do not treat the naive +0.83pp gap as evidence that medication changes
themselves increase readmission risk. Most of that gap is attributable to the fact that sicker
patients get their medications changed AND are separately more likely to be readmitted — not to
the medication change causing readmission. This is a textbook confounding-by-indication pattern,
and it's exactly why naive before/after or group comparisons are risky in observational
healthcare data.

**Caveat, stated explicitly:** propensity matching only controls for confounders that were
measured (severity indicators available in this dataset). Unmeasured confounding — for example,
disease severity not fully captured by diagnosis codes — could still bias this estimate. This is
a genuine limitation, not a technicality, and would need to be flagged in any real clinical
decision built on this analysis.

---

## Methodology Notes

- All hypothesis tests use `data/processed/cohort_one_per_patient.csv` (one encounter per unique
  patient, first-encounter-by-time) rather than the full 99,343-encounter dataset, because 16,773
  patients have multiple encounters and classic hypothesis tests assume independent observations.
- Expired and hospice-discharge encounters (2,423 rows) were excluded before any analysis, since
  readmission is not a meaningful outcome for a patient who died or entered hospice care.
- The logistic regression AUC (0.61) and pseudo-R² (0.021) are modest but consistent with
  published literature on this exact dataset — readmission is a genuinely hard prediction problem
  from administrative data alone, and a low AUC here reflects the difficulty of the problem, not
  an error in the modeling.
