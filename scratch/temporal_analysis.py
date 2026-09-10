import pandas as pd
import json

df_clean = pd.read_csv("data/processed/apple_support/support_pairs_clean.csv", low_memory=False)
df_clean['dt'] = pd.to_datetime(df_clean['customer_timestamp'], format='%a %b %d %H:%M:%S +0000 %Y')

print("Total clean pairs:", len(df_clean))
df_sorted = df_clean.sort_values('dt').reset_index(drop=True)

# 70% cutoff
idx_70 = int(len(df_sorted) * 0.70)
idx_85 = int(len(df_sorted) * 0.85)

print(f"Earliest customer timestamp: {df_sorted['dt'].min()}")
print(f"Latest customer timestamp: {df_sorted['dt'].max()}")
print(f"70% Chronological Cutoff date: {df_sorted['dt'].iloc[idx_70]}")
print(f"85% Chronological Cutoff date: {df_sorted['dt'].iloc[idx_85]}")

print("\nYearly Distribution of Customer Queries:")
print(df_sorted['dt'].dt.year.value_counts().sort_index())

print("\nMonthly Distribution for 2017:")
print(df_sorted[df_sorted['dt'].dt.year == 2017]['dt'].dt.month.value_counts().sort_index())
