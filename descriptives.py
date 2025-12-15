import pandas as pd

IN_PATH = "data/processed/analysis_ready_2019.csv"
OUT_TABLE = "outputs/tables/descriptives_2019.csv"
OUT_TEXT = "outputs/tables/descriptives_2019.txt"

df = pd.read_csv(IN_PATH)

vars_to_describe = [
    "female_activity_rate_pct",
    "insuff_pa_f",
    "sba_pct",
    "gdp_pc",
    "log_gdp_pc",
    "sec_enroll_f",
]

desc = df[vars_to_describe].describe().T  # count, mean, std, min, 25%, 50%, 75%, max
missing = df[vars_to_describe].isna().sum().rename("missing")
desc = desc.join(missing)

# Add N used for regressions (common reporting)
n_all = df["female_activity_rate_pct"].notna().sum()
n_with_gdp = df[["female_activity_rate_pct", "sba_pct", "log_gdp_pc"]].dropna().shape[0]
n_with_edu = df[["female_activity_rate_pct", "sba_pct", "log_gdp_pc", "sec_enroll_f"]].dropna().shape[0]

desc.to_csv(OUT_TABLE)

with open(OUT_TEXT, "w") as f:
    f.write("Descriptive statistics (2019)\n")
    f.write(desc.to_string())
    f.write("\n\nKey sample sizes:\n")
    f.write(f"- N (activity outcome non-missing): {n_all}\n")
    f.write(f"- N (Model with GDP): {n_with_gdp}\n")
    f.write(f"- N (Model with GDP + education): {n_with_edu}\n")

print(f"Saved: {OUT_TABLE}")
print(f"Saved: {OUT_TEXT}")
