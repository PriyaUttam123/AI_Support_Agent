import pandas as pd
import numpy as np

sample_path = "data/evaluation/apple_support_manual_sample.csv"
pairs_path = "data/processed/apple_support/support_pairs.csv"

df_sample = pd.read_csv(sample_path)
df_pairs = pd.read_csv(pairs_path)

sample_convs = set(df_sample['conversation_id'].unique())
all_convs = set(df_pairs['conversation_id'].unique())

print(f"Total conversations in support_pairs: {len(all_convs):,}")
print(f"Total conversations in manual sample: {len(sample_convs):,}")
print(f"Are all sample convs in support_pairs: {sample_convs.issubset(all_convs)}")
