"""
02_clean_data.py
-----------------
STEP 2: CLEANING REAL EHR DATA

Real clinical data brings real problems that synthetic data doesn't:
    - '?' used as the missing-value sentinel instead of NaN
    - Administrative discharge codes that mean "died" or "hospice" and
      must be EXCLUDED from readmission analysis (a dead patient cannot
      be "readmitted" -- including them silently biases every result)
    - `weight` is 97% missing (a known, documented limitation of this
      dataset -- dropped rather than imputed, since imputing a 97%-missing
      column would mostly be fabricating data)
    - Repeat patients: 16,773 of 71,518 patients have more than one
      encounter. Classic hypothesis tests (t-test, chi-square, ANOVA)
      assume independent observations. We keep every encounter for the
      dashboard's descriptive views, but build a DEDUPED cohort (first
      encounter per patient) specifically for the hypothesis-testing
      scripts, and say so explicitly in the outputs.

Run:
    python data/02_clean_data.py
Output:
    data/processed/encounters_clean.csv   (all valid encounters, for dashboard/descriptives)
    data/processed/cohort_one_per_patient.csv (deduped, for hypothesis tests / regression)
"""

import pandas as pd
import numpy as np

RAW = "data/raw"
OUT = "data/processed"

# Discharge disposition codes meaning the patient died or entered hospice --
# these encounters cannot meaningfully be labeled "readmitted" or "not readmitted"
# and are excluded from readmission analysis (standard practice for this dataset,
# per Strack et al. 2014 and subsequent literature).
EXPIRED_HOSPICE_CODES = {11, 13, 14, 19, 20, 21}


def load_id_mappings():
    """Parse the 3-section IDs_mapping.csv into three lookup dicts."""
    raw = pd.read_csv(f"{RAW}/IDs_mapping.csv", header=None, dtype=str)
    sections = {}
    current_key = None
    current_rows = []
    for _, row in raw.iterrows():
        c0, c1 = row[0], row[1]
        if c0 in ("admission_type_id", "discharge_disposition_id", "admission_source_id"):
            if current_key:
                sections[current_key] = dict(current_rows)
            current_key = c0
            current_rows = []
        elif pd.notna(c0) and c0 != "":
            try:
                current_rows.append((int(c0), c1))
            except ValueError:
                continue
    if current_key:
        sections[current_key] = dict(current_rows)
    return sections


def clean(df, id_maps):
    before = len(df)

    # Standardize missing-value sentinel
    df = df.replace("?", np.nan)

    # Drop weight (97% missing -- documented dataset limitation, not imputable)
    df = df.drop(columns=["weight"])

    # Exclude expired/hospice discharges from readmission analysis
    df = df[~df["discharge_disposition_id"].isin(EXPIRED_HOSPICE_CODES)].copy()
    print(f"[clean] Excluded {before - len(df)} expired/hospice-discharge encounters "
          f"(readmission is not a meaningful outcome for these).")

    # Human-readable labels for the three ID-coded columns
    df["admission_type"] = df["admission_type_id"].map(id_maps["admission_type_id"])
    df["discharge_disposition"] = df["discharge_disposition_id"].map(id_maps["discharge_disposition_id"])
    df["admission_source"] = df["admission_source_id"].map(id_maps["admission_source_id"])

    # Binary readmission target used throughout the literature on this dataset:
    # readmitted within 30 days vs. not (collapses ">30" and "NO" into 0)
    df["readmit_30"] = (df["readmitted"] == "<30").astype(int)

    # Fill remaining categorical missingness with an explicit "Unknown" category
    # rather than dropping rows (payer_code and medical_specialty are ~40-49%
    # missing -- dropping those rows would throw away roughly half the dataset)
    for col in ["race", "payer_code", "medical_specialty"]:
        df[col] = df[col].fillna("Unknown")

    # Convert bucketed age string like "[70-80)" to a numeric midpoint for
    # correlation/regression use, while keeping the original bucket for grouping
    def age_midpoint(bucket):
        lo, hi = bucket.strip("[)").split("-")
        return (int(lo) + int(hi)) / 2

    df["age_midpoint"] = df["age"].apply(age_midpoint)

    print(f"[clean] Final dataset: {len(df):,} rows, {df.shape[1]} columns")
    return df


if __name__ == "__main__":
    df_raw = pd.read_csv(f"{RAW}/diabetic_data.csv")
    id_maps = load_id_mappings()
    df_clean = clean(df_raw, id_maps)

    df_clean.to_csv(f"{OUT}/encounters_clean.csv", index=False)
    print(f"Saved: {OUT}/encounters_clean.csv ({len(df_clean):,} rows)")

    # Deduped cohort: keep first encounter per patient (by encounter_id order,
    # which is monotonically increasing with time in this dataset) for
    # hypothesis tests that require independent observations.
    cohort = df_clean.sort_values("encounter_id").drop_duplicates(subset="patient_nbr", keep="first")
    cohort.to_csv(f"{OUT}/cohort_one_per_patient.csv", index=False)
    print(f"Saved: {OUT}/cohort_one_per_patient.csv ({len(cohort):,} rows, "
          f"one encounter per unique patient -- used for all hypothesis tests)")
