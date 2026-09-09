import pandas as pd
import json

df_pairs = pd.read_csv("data/processed/apple_support/support_pairs.csv", low_memory=False)

def print_example(query_filter, category_name):
    matches = df_pairs[query_filter]
    if len(matches) > 0:
        row = matches.iloc[0]
        print(f"### Category: {category_name}")
        print(f"- Customer Tweet ID: {row['customer_tweet_id']}")
        print(f"- Customer Text: {row['customer_text']}")
        print(f"- Response Tweet ID: {row['response_tweet_id']}")
        print(f"- Response Text: {row['response_text']}\n")

# 1. Pure troubleshooting
print_example(
    (~df_pairs['response_requests_dm']) & (df_pairs['response_has_troubleshooting']) & (df_pairs['customer_text'].str.contains('battery', case=False)),
    "Pure Troubleshooting (Self-Serve Instructions - Battery/Performance)"
)

# 2. Immediate Escalation (DM)
print_example(
    (df_pairs['response_requests_dm']) & (~df_pairs['response_has_troubleshooting']) & (df_pairs['customer_text'].str.contains('apple id|password|account', case=False)),
    "Immediate Escalation (DM Request - Account/Authentication)"
)

# 3. Troubleshooting + Escalation
print_example(
    (df_pairs['response_requests_dm']) & (df_pairs['response_has_troubleshooting']),
    "Troubleshooting + Escalation (Conditional Deflection)"
)

# 4. Documentation / URL Only
print_example(
    (df_pairs['response_has_url']) & (~df_pairs['response_requests_dm']) & (~df_pairs['response_has_troubleshooting']),
    "Documentation / Help Article URL"
)

# 5. Clarification Question / Diagnostic
print_example(
    (~df_pairs['response_has_url']) & (~df_pairs['response_requests_dm']) & (df_pairs['response_text'].str.contains(r'\?', regex=True)),
    "General Inquiry / Clarification Question"
)
