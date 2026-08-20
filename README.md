# 🏥 Patient Readmission & Hospital Utilization Analytics

An end-to-end statistical analytics project built on **real, verified clinical data** —
101,766 real de-identified inpatient encounters across 130 real US hospitals — going from
data verification through cleaning, hypothesis testing, advanced statistical extensions, and
an interactive dashboard.

> **This is real data, not synthetic.** Dataset: "Diabetes 130-US Hospitals for Years
> 1999-2008" (Strack, B., DeShazo, J.P., Gennings, C., et al. "Impact of HbA1c Measurement on
> Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records." *BioMed
> Research International*, 2014). Source: UCI ML Repository, CC BY 4.0. Row count and
> readmission distribution are verified against the published paper in `data/01_load_real_data.py`.

---

## 🧱 Project Structure

```
patient-readmission-analytics/
├── data/
│   ├── 01_load_real_data.py       # Verifies real data matches published dataset stats
│   ├── 02_clean_data.py           # Cleaning: missing sentinels, exclusions, dedup for independence
│   ├── raw/                       # diabetic_data.csv (101,766 rows), IDs_mapping.csv
│   └── processed/                 # encounters_clean.csv, cohort_one_per_patient.csv
├── analysis/
│   ├── 03_statistical_tests.py            # Chi-square, Welch's t-test, one-way ANOVA
│   ├── 04_two_way_anova.py                # EXTENSION: interaction effects
│   ├── 05_logistic_regression.py          # EXTENSION: multiple logistic regression, odds ratios
│   ├── 06_effect_sizes_multiple_testing.py # EXTENSION: Cohen's d, Cramer's V, Bonferroni/FDR
│   ├── 07_propensity_matching.py          # EXTENSION: causal-style propensity score matching
│   └── 08_generate_charts.py
├── outputs/
│   ├── charts/                    # 5 PNG charts
│   ├── raw_test_results.csv
│   ├── corrected_test_results.csv        # <- the table to cite (post-correction)
│   ├── logistic_regression_odds_ratios.csv
│   ├── propensity_matching_results.csv
│   ├── propensity_covariate_balance.csv
│   └── insights_report.md         # Full findings & recommendations
├── dashboard/
│   └── app.py                     # Interactive Streamlit dashboard + live hypothesis test tool
├── requirements.txt
└── README.md
```

## ▶️ How to Run

```bash
pip install -r requirements.txt

python data/01_load_real_data.py          # Verify real data integrity
python data/02_clean_data.py              # Clean, exclude, dedupe

python analysis/03_statistical_tests.py           # Core hypothesis tests
python analysis/04_two_way_anova.py                # Interaction effects
python analysis/05_logistic_regression.py          # Predictive model
python analysis/06_effect_sizes_multiple_testing.py # Effect sizes + correction
python analysis/07_propensity_matching.py           # Causal-style comparison
python analysis/08_generate_charts.py               # Charts

streamlit run dashboard/app.py            # Interactive dashboard
```

Deploy `dashboard/app.py` free on **Streamlit Community Cloud** for a live demo link.

---

## 🔧 Tech Stack

| Layer | Tool |
|---|---|
| Statistical testing | SciPy (t-test, ANOVA, chi-square), statsmodels (two-way ANOVA, logistic regression) |
| Effect sizes / correction | Custom Cohen's d / Cramer's V, `statsmodels.stats.multitest` |
| Causal-style analysis | scikit-learn (propensity model, nearest-neighbor matching) |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit (includes a live hypothesis-test tool) |

---

## 📊 Key Findings (Real Data, Real Results)

| Analysis | Finding |
|---|---|
| Welch's t-test | Prior inpatient visits: **strongest predictor**, Cohen's d=0.35, t=16.78, p<0.0001 |
| Two-way ANOVA | Admission type × diabetes medication **interaction is significant** (p<0.0001) — a one-way test would have hidden this |
| Logistic regression | Each prior inpatient visit multiplies readmission odds by **1.40x** (95% CI 1.36–1.45); model AUC=0.61, consistent with published literature on this dataset |
| Effect sizes + correction | 12/15 tests significant at raw α=0.05 → 11/15 after Bonferroni; **most significant effects are practically negligible** at this sample size |
| Propensity matching | Naive medication-change effect (+0.83pp) **shrinks to +0.31pp (not significant, p=0.17)** after matching on 8 confounders — classic confounding-by-indication |

Full write-up with methodology, caveats, and recommendations:
[`outputs/insights_report.md`](outputs/insights_report.md)

---

## 💡 What This Project Demonstrates

- **Working with real, imperfect clinical data** — missing-value sentinels, administrative
  exclusion criteria (expired/hospice discharges), and non-independent repeat observations,
  all handled explicitly rather than glossed over
- **Beyond single hypothesis tests** — two-way ANOVA (interaction effects), multiple logistic
  regression (holding confounders constant), not just isolated bivariate tests
- **Statistical maturity** — effect sizes reported alongside every p-value, multiple-testing
  correction applied to the full test family, explicit acknowledgment of what each test can
  and cannot claim
- **Causal reasoning under observational data constraints** — propensity score matching to
  distinguish correlation from a defensible causal estimate, with covariate balance checks and
  an explicit statement of what confounding this can and cannot rule out
- **A genuinely interactive tool** — the dashboard runs a live t-test with live effect size on
  whatever grouping/outcome combination a stakeholder picks

## 🚀 Possible Further Extensions

- Multilevel/mixed-effects model to account for hospital-level clustering (patients within the
  same hospital aren't independent either — a natural next rigor step beyond patient-level dedup)
- Survival analysis (time-to-readmission) if a continuous time variable were available
- Full covariate set (all 23 medication columns) in the logistic regression with regularization
  (LASSO) for feature selection instead of a hand-picked subset
