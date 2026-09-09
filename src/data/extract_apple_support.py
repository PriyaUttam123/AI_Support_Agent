"""Extract, filter, and structure AppleSupport conversations and support pairs from TWCS."""

import os
import re
import json
import time
from collections import defaultdict
import pandas as pd
import numpy as np

RAW_TWCS_PATH = os.path.join("data", "raw", "twcs", "twcs.csv")
PROCESSED_DIR = os.path.join("data", "processed", "apple_support")
EVAL_DIR = os.path.join("data", "evaluation")
REPORTS_DIR = os.path.join("reports")

CHUNK_SIZE = 350000

# Regex patterns for signal detection
DM_PATTERN = re.compile(r'\b(?:dm|direct message|private message)\b', re.IGNORECASE)
URL_PATTERN = re.compile(r'https?://|t\.co/', re.IGNORECASE)
GENIUS_BAR_PATTERN = re.compile(r'\b(?:genius bar|apple store|appointment|reservation|service location|authorized service)\b', re.IGNORECASE)
TROUBLESHOOTING_PATTERN = re.compile(
    r'\b(?:restart|reboot|settings|update|ios\s*\d+|backup|restore|unpair|re-pair|toggle|clear cache|force close|reset|safari|wifi|bluetooth|turn off|turn on|volume|charge|charging|cable)\b',
    re.IGNORECASE
)

# Text normalization patterns
USER_HANDLE_PATTERN = re.compile(r'@\d+\b')
MULTIPLE_SPACES_PATTERN = re.compile(r'\s+')


def normalize_text(text: str) -> str:
    """Safely normalize text by standardizing whitespace and anonymized user handles."""
    if not isinstance(text, str):
        return ""
    # Standardize spaces
    cleaned = MULTIPLE_SPACES_PATTERN.sub(" ", text).strip()
    # Normalize anonymized numeric handles (e.g., @115854 -> @user)
    cleaned = USER_HANDLE_PATTERN.sub("@user", cleaned)
    return cleaned


def extract_apple_support_data():
    """Main function to extract and process AppleSupport dataset."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("=" * 70)
    print("PHASE 3: Extracting AppleSupport Dataset from TWCS")
    print("=" * 70)
    t_start = time.time()

    # Step 1: Find all AppleSupport outbound tweets and immediate parent/child IDs
    print("\n[Step 1/5] Identifying AppleSupport outbound tweets and conversation linkages...")
    apple_outbound_ids = set()
    candidate_tweet_ids = set()
    parent_map = {} # tid -> in_response_to_tweet_id
    children_map = defaultdict(list) # tid -> [child_ids]

    for chunk in pd.read_csv(RAW_TWCS_PATH, chunksize=CHUNK_SIZE, 
                             usecols=['tweet_id', 'author_id', 'inbound', 'in_response_to_tweet_id', 'response_tweet_id'],
                             low_memory=False):
        # AppleSupport outbound
        apple_out = chunk[(chunk['author_id'] == 'AppleSupport') & (~chunk['inbound'])]
        for tid, in_resp, resp_str in zip(apple_out['tweet_id'], apple_out['in_response_to_tweet_id'], apple_out['response_tweet_id']):
            tid = int(tid)
            apple_outbound_ids.add(tid)
            candidate_tweet_ids.add(tid)
            if pd.notnull(in_resp):
                in_resp_id = int(in_resp)
                parent_map[tid] = in_resp_id
                candidate_tweet_ids.add(in_resp_id)
            if pd.notnull(resp_str):
                for r in str(resp_str).split(','):
                    r = r.strip()
                    if r.isdigit():
                        r_id = int(r)
                        children_map[tid].append(r_id)
                        candidate_tweet_ids.add(r_id)

    print(f"  -> Outbound AppleSupport tweets: {len(apple_outbound_ids):,}")
    print(f"  -> Initial candidate tweet IDs in conversation graph: {len(candidate_tweet_ids):,}")

    # Step 2: Multi-pass expansion to capture full conversation threads (ancestors and follow-ups)
    print("\n[Step 2/5] Expanding ancestors and follow-ups across full conversation trees...")
    for pass_num in [1, 2]:
        ids_to_fetch = set(candidate_tweet_ids)
        new_added = 0
        for chunk in pd.read_csv(RAW_TWCS_PATH, chunksize=CHUNK_SIZE, 
                                 usecols=['tweet_id', 'in_response_to_tweet_id', 'response_tweet_id'],
                                 low_memory=False):
            matching = chunk[chunk['tweet_id'].isin(ids_to_fetch)]
            for tid, in_resp, resp_str in zip(matching['tweet_id'], matching['in_response_to_tweet_id'], matching['response_tweet_id']):
                tid = int(tid)
                if pd.notnull(in_resp):
                    in_resp_id = int(in_resp)
                    if tid not in parent_map:
                        parent_map[tid] = in_resp_id
                    if in_resp_id not in candidate_tweet_ids:
                        candidate_tweet_ids.add(in_resp_id)
                        new_added += 1
                if pd.notnull(resp_str):
                    for r in str(resp_str).split(','):
                        r = r.strip()
                        if r.isdigit():
                            r_id = int(r)
                            children_map[tid].append(r_id)
                            if r_id not in candidate_tweet_ids:
                                candidate_tweet_ids.add(r_id)
                                new_added += 1
        print(f"  -> Expansion Pass {pass_num}: {new_added:,} new tweets discovered (Total: {len(candidate_tweet_ids):,})")

    # Step 3: Stream raw data and pull full records for all candidate tweets
    print("\n[Step 3/5] Extracting full tweet metadata for AppleSupport conversations...")
    extracted_records = []
    found_tweet_ids = set()

    for chunk in pd.read_csv(RAW_TWCS_PATH, chunksize=CHUNK_SIZE, low_memory=False):
        matching = chunk[chunk['tweet_id'].isin(candidate_tweet_ids)]
        if len(matching) > 0:
            extracted_records.append(matching)
            found_tweet_ids.update(matching['tweet_id'].values)

    df_tweets = pd.concat(extracted_records, ignore_index=True)
    df_tweets['tweet_id'] = df_tweets['tweet_id'].astype(int)
    # Deduplicate just in case
    df_tweets = df_tweets.drop_duplicates(subset=['tweet_id']).reset_index(drop=True)
    print(f"  -> Total AppleSupport-related tweets successfully extracted: {len(df_tweets):,}")

    # Build root conversation mapping via backwards traversal
    print("  -> Resolving conversation trees to root thread IDs...")
    tweet_id_set = set(df_tweets['tweet_id'].values)
    parent_lookup = {}
    for tid, in_resp in zip(df_tweets['tweet_id'], df_tweets['in_response_to_tweet_id']):
        if pd.notnull(in_resp) and int(in_resp) in tweet_id_set:
            parent_lookup[int(tid)] = int(in_resp)

    def find_root(tid):
        curr = tid
        visited = set()
        while curr in parent_lookup and curr not in visited:
            visited.add(curr)
            curr = parent_lookup[curr]
        return curr

    df_tweets['conversation_id'] = df_tweets['tweet_id'].map(find_root)
    
    # Calculate conversation lengths
    conv_lengths = df_tweets['conversation_id'].value_counts().to_dict()
    df_tweets['conversation_length'] = df_tweets['conversation_id'].map(conv_lengths)

    # Save tweets.csv
    tweets_csv_path = os.path.join(PROCESSED_DIR, "tweets.csv")
    df_tweets.to_csv(tweets_csv_path, index=False, encoding='utf-8')
    print(f"  -> Saved tweets.csv ({len(df_tweets):,} rows) to {tweets_csv_path}")

    # Step 4: Export conversations.jsonl
    print("\n[Step 4/5] Building and exporting conversations.jsonl...")
    df_tweets_sorted = df_tweets.sort_values(by=['conversation_id', 'created_at'])
    conversations_jsonl_path = os.path.join(PROCESSED_DIR, "conversations.jsonl")

    with open(conversations_jsonl_path, 'w', encoding='utf-8') as f_out:
        for conv_id, grp in df_tweets_sorted.groupby('conversation_id'):
            turns = []
            for _, row in grp.iterrows():
                turns.append({
                    'tweet_id': int(row['tweet_id']),
                    'author_id': str(row['author_id']),
                    'inbound': bool(row['inbound']),
                    'created_at': str(row['created_at']),
                    'text': str(row['text']),
                    'in_response_to_tweet_id': int(row['in_response_to_tweet_id']) if pd.notnull(row['in_response_to_tweet_id']) else None,
                    'response_tweet_id': str(row['response_tweet_id']) if pd.notnull(row['response_tweet_id']) else None
                })
            f_out.write(json.dumps({
                'conversation_id': int(conv_id),
                'conversation_length': len(turns),
                'tweets': turns
            }) + "\n")
    print(f"  -> Exported {len(conv_lengths):,} conversation threads to {conversations_jsonl_path}")

    # Step 5: Extract clean customer -> AppleSupport support pairs
    print("\n[Step 5/5] Extracting clean customer -> AppleSupport support pairs...")
    # Index tweets for fast lookup
    tweet_dict = {}
    for _, row in df_tweets.iterrows():
        tweet_dict[int(row['tweet_id'])] = row

    support_pairs = []
    pair_id = 1

    for _, row in df_tweets.iterrows():
        # Look for AppleSupport outbound response
        if (not row['inbound']) and row['author_id'] == 'AppleSupport':
            in_resp = row['in_response_to_tweet_id']
            if pd.notnull(in_resp):
                cust_tid = int(in_resp)
                if cust_tid in tweet_dict:
                    cust_row = tweet_dict[cust_tid]
                    # Verify customer side is inbound
                    if cust_row['inbound']:
                        resp_text = str(row['text'])
                        cust_text = str(cust_row['text'])
                        
                        cust_ts = pd.to_datetime(cust_row['created_at'], format='%a %b %d %H:%M:%S +0000 %Y')
                        resp_ts = pd.to_datetime(row['created_at'], format='%a %b %d %H:%M:%S +0000 %Y')
                        delay_sec = int((resp_ts - cust_ts).total_seconds()) if (pd.notnull(cust_ts) and pd.notnull(resp_ts)) else None
                        
                        has_url = bool(URL_PATTERN.search(resp_text))
                        req_dm = bool(DM_PATTERN.search(resp_text))
                        has_genius = bool(GENIUS_BAR_PATTERN.search(resp_text))
                        has_troubleshoot = bool(TROUBLESHOOTING_PATTERN.search(resp_text))

                        support_pairs.append({
                            'pair_id': pair_id,
                            'conversation_id': int(row['conversation_id']),
                            'customer_tweet_id': cust_tid,
                            'response_tweet_id': int(row['tweet_id']),
                            'customer_author_id': str(cust_row['author_id']),
                            'customer_text': cust_text,
                            'response_text': resp_text,
                            'customer_text_normalized': normalize_text(cust_text),
                            'response_text_normalized': normalize_text(resp_text),
                            'customer_timestamp': str(cust_row['created_at']),
                            'response_timestamp': str(row['created_at']),
                            'turn_delay_seconds': delay_sec,
                            'response_has_url': has_url,
                            'response_requests_dm': req_dm,
                            'response_mentions_genius_bar': has_genius,
                            'response_has_troubleshooting': has_troubleshoot,
                            'conversation_length': int(row['conversation_length']),
                        })
                        pair_id += 1

    df_pairs = pd.DataFrame(support_pairs)
    pairs_csv_path = os.path.join(PROCESSED_DIR, "support_pairs.csv")
    df_pairs.to_csv(pairs_csv_path, index=False, encoding='utf-8')
    print(f"  -> Generated {len(df_pairs):,} clean support pairs saved to {pairs_csv_path}")

    # Generate reproducible manual sample (250 pairs) with fixed seed 42
    print("\nGenerating reproducible manual evaluation sample (250 pairs, seed=42)...")
    sample_size = min(250, len(df_pairs))
    df_sample = df_pairs.sample(n=sample_size, random_state=42).reset_index(drop=True)
    sample_csv_path = os.path.join(EVAL_DIR, "apple_support_manual_sample.csv")
    df_sample.to_csv(sample_csv_path, index=False, encoding='utf-8')
    print(f"  -> Saved {sample_size} manual inspection pairs to {sample_csv_path}")

    print(f"\nProcessing completed in {time.time() - t_start:.1f}s.")
    return df_tweets, df_pairs


if __name__ == "__main__":
    extract_apple_support_data()
