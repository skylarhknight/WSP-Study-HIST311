import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

IN_PATH = "data/processed/analysis_ready_2019.csv"

df = pd.read_csv(IN_PATH)

# --- Figure 1: Scatter (SBA vs Female activity rate)
plt.figure()
plt.scatter(df["sba_pct"], df["female_activity_rate_pct"])
plt.xlabel("Births attended by skilled health personnel (%)")
plt.ylabel("Female physical activity rate (%)")
plt.title("Healthcare access and female physical activity (2019)")
plt.tight_layout()
plt.savefig("outputs/figures/fig1_scatter_sba_activity.png", dpi=200)
plt.close()

# --- Figure 2: Scatter + regression line
x = df["sba_pct"].values
y = df["female_activity_rate_pct"].values

# Drop NaNs for line fit
mask = ~np.isnan(x) & ~np.isnan(y)
m, b = np.polyfit(x[mask], y[mask], 1)

plt.figure()
plt.scatter(x, y)
plt.plot(x[mask], m * x[mask] + b)
plt.xlabel("Births attended by skilled health personnel (%)")
plt.ylabel("Female physical activity rate (%)")
plt.title("Linear association (2019)")
plt.tight_layout()
plt.savefig("outputs/figures/fig2_scatter_line.png", dpi=200)
plt.close()

# --- Figure 3 (optional but useful): GDP vs activity (shows confounding)
plt.figure()
plt.scatter(df["log_gdp_pc"], df["female_activity_rate_pct"])
plt.xlabel("Log GDP per capita")
plt.ylabel("Female physical activity rate (%)")
plt.title("Economic development and female physical activity (2019)")
plt.tight_layout()
plt.savefig("outputs/figures/fig3_scatter_gdp_activity.png", dpi=200)
plt.close()

print("Saved figures to outputs/figures/")


# Figure 4
df = pd.read_csv("data/processed/analysis_ready_2019.csv")

df["sba_bin"] = pd.cut(df["sba_pct"], bins=[0,70,80,90,95,100])

bin_means = df.groupby("sba_bin")["female_activity_rate_pct"].mean()

plt.figure()
bin_means.plot(marker="o")
plt.xlabel("Skilled birth attendance (%) – bins")
plt.ylabel("Mean female physical activity rate (%)")
plt.title("Average female activity by healthcare access level (2019)")
plt.tight_layout()
plt.savefig("outputs/figures/fig4_binned_means.png", dpi=200)
plt.close()



# Figure 5
df = pd.read_csv("data/processed/analysis_ready_2019.csv").dropna(subset=["female_activity_rate_pct","sba_pct","log_gdp_pc"])

# Residualize outcome
y_resid = sm.OLS(df["female_activity_rate_pct"], sm.add_constant(df["log_gdp_pc"])).fit().resid

# Residualize predictor
x_resid = sm.OLS(df["sba_pct"], sm.add_constant(df["log_gdp_pc"])).fit().resid

m, b = np.polyfit(x_resid, y_resid, 1)

plt.figure()
plt.scatter(x_resid, y_resid)
plt.plot(x_resid, m*x_resid + b)
plt.xlabel("Healthcare access residual (net of GDP)")
plt.ylabel("Female activity residual (net of GDP)")
plt.title("Partial association controlling for GDP (2019)")
plt.tight_layout()
plt.savefig("outputs/figures/fig5_partial_sba.png", dpi=200)
plt.close()


# Figure 6 
plt.figure()
plt.hist(df["female_activity_rate_pct"], bins=15)
plt.xlabel("Female physical activity rate (%)")
plt.ylabel("Number of countries")
plt.title("Distribution of female physical activity across countries (2019)")
plt.tight_layout()
plt.savefig("outputs/figures/fig6_activity_distribution.png", dpi=200)
plt.close()
