import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

IN_PATH = "data/processed/analysis_ready_2019.csv"
OUT_TEXT = "outputs/tables/key_numbers_2019.txt"

df = pd.read_csv(IN_PATH)

# Correlation (bivariate)
corr = df[["female_activity_rate_pct", "sba_pct"]].corr().iloc[0, 1]

# SDs for context
sd_activity = df["female_activity_rate_pct"].std()
sd_sba = df["sba_pct"].std()

# Main regression (Model 2 is usually the headline: controls for GDP)
m2 = smf.ols("female_activity_rate_pct ~ sba_pct + log_gdp_pc", data=df).fit(cov_type="HC3")
beta_sba = m2.params["sba_pct"]
ci = m2.conf_int().loc["sba_pct"].tolist()

effect_10pp = beta_sba * 10
ci_10pp = [ci[0] * 10, ci[1] * 10]

# Sample sizes
n_all = df.shape[0]
n_m2 = int(m2.nobs)
n_m3 = df[["female_activity_rate_pct", "sba_pct", "log_gdp_pc", "sec_enroll_f"]].dropna().shape[0]

with open(OUT_TEXT, "w") as f:
    f.write("Key numbers (2019)\n\n")
    f.write(f"N total (with outcome & SBA): {n_all}\n")
    f.write(f"N Model 2 (adds log GDP): {n_m2}\n")
    f.write(f"N Model 3 (adds education): {n_m3}\n\n")

    f.write(f"Correlation(activity, SBA): {corr:.3f}\n\n")

    f.write("Standard deviations:\n")
    f.write(f"- SD female activity rate: {sd_activity:.3f}\n")
    f.write(f"- SD skilled birth attendance: {sd_sba:.3f}\n\n")

    f.write("Interpretable effect size (Model 2):\n")
    f.write(f"- Beta on SBA (%-pt activity per 1%-pt SBA): {beta_sba:.4f}\n")
    f.write(f"- 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]\n")
    f.write(f"- Effect of +10pp SBA: {effect_10pp:.3f} percentage points\n")
    f.write(f"- 95% CI for +10pp: [{ci_10pp[0]:.3f}, {ci_10pp[1]:.3f}]\n")

print(f"Saved: {OUT_TEXT}")
