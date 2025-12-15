import pandas as pd
import numpy as np

df = pd.read_csv("women_healthcare_activity_2019.csv")

print(df.describe())

# Range checks
print("Insufficient PA out of bounds:",
      ((df["insuff_pa_female_pct"] < 0) | (df["insuff_pa_female_pct"] > 100)).sum())
print("Skilled birth attendance out of bounds:",
      ((df["skilled_birth_attendance_pct"] < 0) | (df["skilled_birth_attendance_pct"] > 100)).sum())

df["log_gdp_pc"] = np.log(df["NY.GDP.PCAP.CD"])