import pandas as pd

sample = pd.read_csv("data/evaluation/apple_support_manual_sample.csv")
clean = pd.read_csv("data/processed/apple_support/support_pairs_clean.csv")

sample_convs = set(sample['conversation_id'])
clean_convs = set(clean['conversation_id'])

diff = sample_convs - clean_convs
print(f"Sample convs count: {len(sample_convs)}")
print(f"Missing from clean: {len(diff)}")
if len(diff) > 0:
    for c in diff:
        row = sample[sample['conversation_id'] == c].iloc[0]
        print(f"Conv {c}: customer_text='{row['customer_text']}'")
