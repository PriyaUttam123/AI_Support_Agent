import os
import json
import pandas as pd
import numpy as np
from collections import Counter
import re

tweets_csv = r"data/processed/apple_support/tweets.csv"
pairs_csv = r"data/processed/apple_support/support_pairs.csv"
convs_jsonl = r"data/processed/apple_support/conversations.jsonl"

print("Loading processed AppleSupport datasets...")
df_tweets = pd.read_csv(tweets_csv, low_memory=False)
df_pairs = pd.read_csv(pairs_csv, low_memory=False)

print(f"Tweets shape: {df_tweets.shape}")
print(f"Pairs shape: {df_pairs.shape}")

# A. Dataset Size & Breakdown
total_tweets = len(df_tweets)
outbound_apple = len(df_tweets[(df_tweets['author_id'] == 'AppleSupport') & (~df_tweets['inbound'])])
inbound_cust = len(df_tweets[df_tweets['inbound']])
other_tweets = total_tweets - outbound_apple - inbound_cust
unique_customers = df_tweets[df_tweets['inbound']]['author_id'].nunique()

# B. Conversation Statistics
total_conversations = df_tweets['conversation_id'].nunique()
conv_lengths = df_tweets.groupby('conversation_id').size()
mean_len = conv_lengths.mean()
median_len = conv_lengths.median()
std_len = conv_lengths.std()
min_len = conv_lengths.min()
max_len = conv_lengths.max()

length_dist = Counter(conv_lengths)
length_breakdown = {
    "1 turn (unanswered/single)": sum(v for k, v in length_dist.items() if k == 1),
    "2 turns (single Q&A pair)": length_dist.get(2, 0),
    "3 turns": length_dist.get(3, 0),
    "4 turns": length_dist.get(4, 0),
    "5+ turns (extended dialogue)": sum(v for k, v in length_dist.items() if k >= 5),
}

# C. Support-Pair Statistics
clean_pairs = len(df_pairs)
unique_customers_in_pairs = df_pairs['customer_author_id'].nunique()
avg_turn_delay_sec = df_pairs['turn_delay_seconds'].dropna().mean()
median_turn_delay_sec = df_pairs['turn_delay_seconds'].dropna().median()

# D. Response-Type & Escalation Statistics
has_url_cnt = df_pairs['response_has_url'].sum()
req_dm_cnt = df_pairs['response_requests_dm'].sum()
genius_cnt = df_pairs['response_mentions_genius_bar'].sum()
troubleshoot_cnt = df_pairs['response_has_troubleshooting'].sum()

# Mutually exclusive or composite response categorization
def categorize_response(row):
    is_dm = row['response_requests_dm']
    is_genius = row['response_mentions_genius_bar']
    is_trouble = row['response_has_troubleshooting']
    is_url = row['response_has_url']
    
    if is_trouble and (is_dm or is_genius):
        return "Troubleshooting + Escalation (Conditional Deflection)"
    elif is_trouble and not (is_dm or is_genius):
        return "Pure Troubleshooting (Self-Serve Instructions)"
    elif (is_dm or is_genius) and not is_trouble:
        return "Immediate Escalation (DM / Store Referral)"
    elif is_url and not (is_dm or is_genius or is_trouble):
        return "Documentation / URL Only"
    else:
        return "General Inquiry / Clarification Question"

response_categories = df_pairs.apply(categorize_response, axis=1).value_counts()

# E. Common Issue Patterns (Keywords, N-Grams)
stopwords = set("""
a about above after again against all am an and any are aren't as at be because been before being below
between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down during each
few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself
him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most
mustn't my myself no nor not of off on once only or other ought our ours ourselves out over own same shan't
she she'd she'll she's should shouldn't so some such than that that's the their theirs them themselves then
there there's these they they'd they'll they're they've this those through to too under until up very was
wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who who's
whom why why's with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves
applesupport apple support http https t co com please help thanks hi hello new getting got get
""".split())

word_pat = re.compile(r'\b[a-z]{3,}\b')

def get_words(text):
    return [w for w in word_pat.findall(str(text).lower()) if w not in stopwords]

unigram_counts = Counter()
bigram_counts = Counter()
trigram_counts = Counter()

for text in df_pairs['customer_text_normalized']:
    words = get_words(text)
    unigram_counts.update(words)
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    bigram_counts.update(bigrams)
    trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words)-2)]
    trigram_counts.update(trigrams)

results = {
    'total_tweets': total_tweets,
    'outbound_apple': outbound_apple,
    'inbound_cust': inbound_cust,
    'other_tweets': other_tweets,
    'unique_customers': unique_customers,
    'total_conversations': total_conversations,
    'mean_len': round(float(mean_len), 2),
    'median_len': float(median_len),
    'std_len': round(float(std_len), 2),
    'min_len': int(min_len),
    'max_len': int(max_len),
    'length_breakdown': length_breakdown,
    'clean_pairs': clean_pairs,
    'unique_customers_in_pairs': unique_customers_in_pairs,
    'avg_turn_delay_min': round(float(avg_turn_delay_sec / 60), 1) if avg_turn_delay_sec else None,
    'median_turn_delay_min': round(float(median_turn_delay_sec / 60), 1) if median_turn_delay_sec else None,
    'has_url_cnt': int(has_url_cnt),
    'has_url_pct': round(float(has_url_cnt / clean_pairs * 100), 2),
    'req_dm_cnt': int(req_dm_cnt),
    'req_dm_pct': round(float(req_dm_cnt / clean_pairs * 100), 2),
    'genius_cnt': int(genius_cnt),
    'genius_pct': round(float(genius_cnt / clean_pairs * 100), 2),
    'troubleshoot_cnt': int(troubleshoot_cnt),
    'troubleshoot_pct': round(float(troubleshoot_cnt / clean_pairs * 100), 2),
    'response_categories': {k: int(v) for k, v in response_categories.items()},
    'top_unigrams': unigram_counts.most_common(20),
    'top_bigrams': bigram_counts.most_common(20),
    'top_trigrams': trigram_counts.most_common(15),
}

out_path = r"reports/apple_stats.json"
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

print(f"Stats computed and saved to {out_path}!")
