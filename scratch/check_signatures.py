import pandas as pd
import re

pairs = pd.read_csv("data/processed/apple_support/support_pairs.csv", low_memory=False)

# Check signatures in response_text
sig_hat = re.compile(r'\^[A-Z]{1,3}\b')
sig_slash = re.compile(r'/[A-Z]{1,3}\b')

has_hat = pairs['response_text'].str.contains(sig_hat, regex=True).sum()
has_slash = pairs['response_text'].str.contains(sig_slash, regex=True).sum()

print(f"Responses with ^XX: {has_hat:,} ({has_hat/len(pairs)*100:.2f}%)")
print(f"Responses with /XX: {has_slash:,} ({has_slash/len(pairs)*100:.2f}%)")

# Sample some responses with signatures
print("\nSample responses with signatures:")
for t in pairs[pairs['response_text'].str.contains(sig_hat, regex=True)]['response_text'].head(5):
    print(" ", t)
