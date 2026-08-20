"""
01_load_real_data.py
---------------------
STEP 1: DATA SOURCE (REAL DATA — NOT SYNTHETIC)

This project uses the "Diabetes 130-US Hospitals for Years 1999-2008" dataset:
    - 101,766 REAL de-identified inpatient encounters
    - 130 REAL US hospitals and integrated delivery networks
    - Collected 1999-2008, donated to UCI ML Repository in 2014
    - Original source: Health Facts database (Cerner EHR extraction)
    - Published in: Strack, B., et al. (2014). "Impact of HbA1c Measurement
      on Hospital Readmission Rates: Analysis of 70,000 Clinical Database
      Patient Records." BioMed Research International, 2014.
    - License: CC BY 4.0
    - Canonical source: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

This is genuinely real clinical data, not simulated. Every number in this
project's insights report comes from an actual hospital encounter.

Files used (already placed in data/raw/ by the extraction step):
    - diabetic_data.csv   : 101,766 rows x 50 columns, one row per encounter
    - IDs_mapping.csv     : decodes admission_type_id, discharge_disposition_id,
                            admission_source_id into human-readable labels

Run:
    python data/01_load_real_data.py
Output:
    Prints a data profile to confirm the file matches the published dataset
    (row count, readmission distribution) before any cleaning happens --
    this is a basic but important sanity check when working with real data
    pulled from a third-party source.
"""

import pandas as pd

RAW = "data/raw"

EXPECTED_ROWS = 101766
EXPECTED_READMIT_DIST = {"NO": 54864, ">30": 35545, "<30": 11357}


def load_and_verify():
    df = pd.read_csv(f"{RAW}/diabetic_data.csv")

    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns")
    assert len(df) == EXPECTED_ROWS, "Row count doesn't match published dataset!"

    actual_dist = df["readmitted"].value_counts().to_dict()
    for k, v in EXPECTED_READMIT_DIST.items():
        assert actual_dist.get(k) == v, f"Readmission count mismatch for {k}"

    print("Verified: row count and readmission distribution match the published dataset.")
    print(f"Unique patients: {df['patient_nbr'].nunique():,} "
          f"({(df['patient_nbr'].value_counts() > 1).sum():,} patients have multiple encounters)")

    return df


if __name__ == "__main__":
    df = load_and_verify()
    print("\nColumn list:")
    print(list(df.columns))
