import pandas as pd
import numpy as np

IN_PATH = "data/raw/women_healthcare_activity_2019.csv"
OUT_PATH = "data/processed/analysis_ready_2019.csv"

df = pd.read_csv(IN_PATH)

# Derived outcome
if "female_activity_rate_pct" not in df.columns:
    df["female_activity_rate_pct"] = 100 - df["insuff_pa_female_pct"]

# Log GDP per capita (drop non-positive safely)
df["log_gdp_pc"] = np.where(df["NY.GDP.PCAP.CD"] > 0, np.log(df["NY.GDP.PCAP.CD"]), np.nan)

# Consistent short names
df = df.rename(columns={
    "insuff_pa_female_pct": "insuff_pa_f",
    "skilled_birth_attendance_pct": "sba_pct",
    "NY.GDP.PCAP.CD": "gdp_pc",
    "SE.SEC.ENRR.FE": "sec_enroll_f"
})

# Keep only columns used
keep = ["iso3", "insuff_pa_f", "female_activity_rate_pct", "sba_pct", "gdp_pc", "log_gdp_pc", "sec_enroll_f"]
df = df[keep].copy()

df.to_csv(OUT_PATH, index=False)
print(f"Saved: {OUT_PATH} (rows={len(df)})")
