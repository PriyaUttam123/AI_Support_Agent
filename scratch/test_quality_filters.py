import pandas as pd
import re

pairs = pd.read_csv("data/processed/apple_support/support_pairs.csv", low_memory=False)
print(f"Total pairs in raw support_pairs.csv: {len(pairs):,}")

# Let's test various quality issues:
# 1. Broken tweet references
broken_refs = (pairs['customer_tweet_id'].isnull()) | (pairs['response_tweet_id'].isnull())
print(f"Broken tweet references: {broken_refs.sum()}")

# 2. Chronology: response_timestamp < customer_timestamp
chrono_issue = pairs['turn_delay_seconds'] < 0
print(f"Impossible chronology (delay < 0): {chrono_issue.sum()}")

# 3. Duplicate pairs:
dup_pairs = pairs.duplicated(subset=['customer_tweet_id', 'response_tweet_id'])
print(f"Duplicate (customer_tweet_id, response_tweet_id): {dup_pairs.sum()}")

dup_cust_tweet = pairs.duplicated(subset=['customer_tweet_id'], keep=False)
print(f"Customer tweets with multiple AppleSupport responses: {pairs.duplicated(subset=['customer_tweet_id']).sum()}")

# 4. Text emptiness / artifacts
# Customer text: strip mentions, URLs, whitespace
mention_url_pat = re.compile(r'(@[A-Za-z0-9_]+|https?://\S+|t\.co/\S+)')
cust_substantive = pairs['customer_text'].fillna('').apply(lambda t: mention_url_pat.sub('', t).strip())

cust_empty = cust_substantive.str.len() == 0
cust_too_short = (cust_substantive.str.len() > 0) & (cust_substantive.str.len() < 3)
print(f"Customer text completely empty of words (only URL or mention): {cust_empty.sum()}")
print(f"Customer text ultra short (<3 chars of substantive text): {cust_too_short.sum()}")

# Response text: strip mentions, URLs, whitespace
resp_substantive = pairs['response_text'].fillna('').apply(lambda t: mention_url_pat.sub('', t).strip())
resp_empty = resp_substantive.str.len() == 0
resp_too_short = (resp_substantive.str.len() > 0) & (resp_substantive.str.len() < 5)
print(f"Response text empty of words (only URL/mentions): {resp_empty.sum()}")
print(f"Response text ultra short (<5 chars): {resp_too_short.sum()}")

# Sample some of the cust_empty rows to see what they look like:
print("\nSample cust_empty texts:")
for idx, row in pairs[cust_empty].head(5).iterrows():
    print(f"  [{row['customer_tweet_id']}]: '{row['customer_text']}' -> Response: '{row['response_text']}'")
