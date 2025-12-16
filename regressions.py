import pandas as pd
import statsmodels.formula.api as smf

IN_PATH = "data/processed/analysis_ready_2019.csv"
OUT_TEXT = "outputs/tables/regression_results_2019.txt"
OUT_COEFS = "outputs/tables/regression_coefs_2019.csv"

df = pd.read_csv(IN_PATH)

# Models (robust SEs)
m1 = smf.ols("female_activity_rate_pct ~ sba_pct", data=df).fit(cov_type="HC3")
m2 = smf.ols("female_activity_rate_pct ~ sba_pct + log_gdp_pc", data=df).fit(cov_type="HC3")
m3 = smf.ols("female_activity_rate_pct ~ sba_pct + log_gdp_pc + sec_enroll_f", data=df).fit(cov_type="HC3")

models = {"Model1_bivariate": m1, "Model2_plus_gdp": m2, "Model3_plus_gdp_edu": m3}

# Save readable summaries
with open(OUT_TEXT, "w") as f:
    for name, model in models.items():
        f.write("=" * 80 + "\n")
        f.write(name + "\n")
        f.write(model.summary().as_text())
        f.write("\n\n")

# Coefficient table
rows = []
for name, model in models.items():
    params = model.params
    ses = model.bse
    pvals = model.pvalues
    conf = model.conf_int()
    conf.columns = ["ci_low", "ci_high"]

    for term in params.index:
        rows.append({
            "model": name,
            "term": term,
            "coef": params[term],
            "se": ses[term],
            "pvalue": pvals[term],
            "ci_low": conf.loc[term, "ci_low"],
            "ci_high": conf.loc[term, "ci_high"],
            "n": int(model.nobs),
            "r2": float(model.rsquared),
        })

coef_df = pd.DataFrame(rows)
coef_df.to_csv(OUT_COEFS, index=False)

print(f"Saved: {OUT_TEXT}")
print(f"Saved: {OUT_COEFS}")
