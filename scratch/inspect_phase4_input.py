import os
import json
import pandas as pd

tweets_path = "data/processed/apple_support/tweets.csv"
convs_path = "data/processed/apple_support/conversations.jsonl"
pairs_path = "data/processed/apple_support/support_pairs.csv"
sample_path = "data/evaluation/apple_support_manual_sample.csv"

def inspect_df(name, path):
    print(f"\n{'='*20} {name} {'='*20}")
    df = pd.read_csv(path, low_memory=False)
    print(f"Path: {path}")
    print(f"Rows: {len(df):,}, Cols: {len(df.columns)}")
    print(f"Duplicates (entire row): {df.duplicated().sum()}")
    print("\nColumns & Types:")
    for c in df.columns:
        null_cnt = df[c].isnull().sum()
        null_pct = null_cnt / len(df) * 100
        print(f"  - {c} ({df[c].dtype}): {null_cnt:,} nulls ({null_pct:.2f}%)")
    return df

df_t = inspect_df("TWEETS", tweets_path)
df_p = inspect_df("SUPPORT PAIRS", pairs_path)
df_s = inspect_df("MANUAL SAMPLE", sample_path)

print(f"\n{'='*20} CONVERSATIONS.JSONL {'='*20}")
conv_count = 0
total_turns_jsonl = 0
null_conv_keys = 0
sample_conv = None

with open(convs_path, 'r', encoding='utf-8') as f:
    for line in f:
        conv_count += 1
        data = json.loads(line)
        if sample_conv is None:
            sample_conv = data
        if 'conversation_id' not in data or 'tweets' not in data:
            null_conv_keys += 1
        total_turns_jsonl += len(data.get('tweets', []))

print(f"Path: {convs_path}")
print(f"Conversations: {conv_count:,}")
print(f"Total Turns in JSONL: {total_turns_jsonl:,}")
print(f"Null/Invalid Keys: {null_conv_keys}")
print(f"Sample JSON keys: {list(sample_conv.keys()) if sample_conv else None}")
