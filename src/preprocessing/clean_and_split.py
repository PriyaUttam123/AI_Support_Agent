"""Phase 4: Data Cleaning, Quality Filtering, Response-Type Labeling, and Leakage-Safe Splitting."""

import os
import re
import json
import time
import pandas as pd
import numpy as np

# File paths
RAW_PAIRS_PATH = os.path.join("data", "processed", "apple_support", "support_pairs.csv")
TWEETS_PATH = os.path.join("data", "processed", "apple_support", "tweets.csv")
SAMPLE_PATH = os.path.join("data", "evaluation", "apple_support_manual_sample.csv")

PROCESSED_DIR = os.path.join("data", "processed", "apple_support")
SPLITS_DIR = os.path.join(PROCESSED_DIR, "splits")
RETRIEVAL_DIR = os.path.join(PROCESSED_DIR, "retrieval")
REPORTS_DIR = os.path.join("reports")

# Regex patterns
MENTION_URL_PAT = re.compile(r'(@[A-Za-z0-9_]+|https?://\S+|t\.co/\S+)')
NUMERIC_HANDLE_PAT = re.compile(r'@\d+\b')
MULTIPLE_SPACES_PAT = re.compile(r'\s+')
URL_PAT = re.compile(r'https?://\S+|t\.co/\S+', re.IGNORECASE)
AGENT_SIG_PAT = re.compile(r'\s*[\^/][A-Za-z]{1,3}\s*$', re.IGNORECASE)

# Response categorization patterns
DM_PAT = re.compile(r'\b(?:dm|direct message|private message)\b', re.IGNORECASE)
GENIUS_BAR_PAT = re.compile(r'\b(?:genius bar|apple store|appointment|reservation|service location|authorized service)\b', re.IGNORECASE)
TROUBLESHOOT_PAT = re.compile(
    r'\b(?:restart|reboot|settings|update|ios\s*\d+|backup|restore|unpair|re-pair|toggle|clear cache|force close|reset|safari|wifi|bluetooth|turn off|turn on|volume|charge|charging|cable)\b',
    re.IGNORECASE
)


def normalize_customer_text(text: str) -> str:
    """Conservative normalization for customer text."""
    if not isinstance(text, str):
        return ""
    # Normalize whitespace
    t = MULTIPLE_SPACES_PAT.sub(" ", text).strip()
    # Normalize synthetic anonymized handles (@115854 -> @user)
    t = NUMERIC_HANDLE_PAT.sub("@user", t)
    return t


def normalize_response_text(text: str) -> str:
    """Conservative normalization for AppleSupport response text."""
    if not isinstance(text, str):
        return ""
    t = MULTIPLE_SPACES_PAT.sub(" ", text).strip()
    t = NUMERIC_HANDLE_PAT.sub("@user", t)
    # Strip obvious agent signatures at the very end of tweet (e.g. /LS, ^HP)
    t = AGENT_SIG_PAT.sub("", t).strip()
    return t


def url_normalized_text(text: str) -> str:
    """Replace URLs with a generic [URL] token."""
    if not isinstance(text, str):
        return ""
    return URL_PAT.sub("[URL]", text)


def assign_response_type(row) -> str:
    """Assign transparent rule-based response-type labels."""
    is_dm = bool(row['response_requests_dm'])
    is_genius = bool(row['response_mentions_genius_bar'])
    is_trouble = bool(row['response_has_troubleshooting'])
    is_url = bool(row['response_has_url'])

    if is_trouble and (is_dm or is_genius):
        return "Troubleshooting + Escalation"
    elif is_trouble and not (is_dm or is_genius):
        return "Pure Troubleshooting"
    elif (is_dm or is_genius) and not is_trouble:
        return "Immediate Escalation"
    elif is_url and not (is_dm or is_genius or is_trouble):
        return "Documentation / Help Article URL"
    else:
        return "General Inquiry / Clarification"


def run_cleaning_and_splitting():
    t0 = time.time()
    os.makedirs(SPLITS_DIR, exist_ok=True)
    os.makedirs(RETRIEVAL_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("=" * 70)
    print("PHASE 4: Data Cleaning, Quality Filtering, and Leakage-Safe Splitting")
    print("=" * 70)

    # Step 1: Load inputs
    print("\n[Step 1/7] Loading raw support pairs, tweets, and manual evaluation sample...")
    df_pairs = pd.read_csv(RAW_PAIRS_PATH, low_memory=False)
    df_tweets = pd.read_csv(TWEETS_PATH, low_memory=False)
    df_sample = pd.read_csv(SAMPLE_PATH, low_memory=False)

    total_raw_pairs = len(df_pairs)
    print(f"  -> Total raw support pairs loaded: {total_raw_pairs:,}")
    print(f"  -> Total tweets loaded: {len(df_tweets):,}")
    print(f"  -> Total manual sample rows: {len(df_sample):,}")

    # Step 2: Conservative Text Normalization
    print("\n[Step 2/7] Applying conservative text normalization (preserving originals)...")
    df_pairs['customer_text_normalized'] = df_pairs['customer_text'].apply(normalize_customer_text)
    df_pairs['response_text_normalized'] = df_pairs['response_text'].apply(normalize_response_text)
    df_pairs['customer_text_url_norm'] = df_pairs['customer_text_normalized'].apply(url_normalized_text)
    df_pairs['response_text_url_norm'] = df_pairs['response_text_normalized'].apply(url_normalized_text)

    # Step 3: Transparent Quality Filtering
    print("\n[Step 3/7] Performing quality assessment and rule-based filtering...")
    # Calculate substantive text length (ignoring handles & URLs)
    cust_substantive = df_pairs['customer_text'].fillna('').apply(lambda t: MENTION_URL_PAT.sub('', t).strip())
    resp_substantive = df_pairs['response_text'].fillna('').apply(lambda t: MENTION_URL_PAT.sub('', t).strip())

    quality_status = []
    quality_reason = []

    # Check for duplicate customer_tweet_ids (multi-response split)
    dup_cust_mask = df_pairs.duplicated(subset=['customer_tweet_id'], keep='first')

    for idx, row in df_pairs.iterrows():
        c_sub = cust_substantive.iloc[idx]
        r_sub = resp_substantive.iloc[idx]
        is_dup = dup_cust_mask.iloc[idx]
        delay = row['turn_delay_seconds']

        if pd.isnull(row['customer_text']) or len(str(row['customer_text']).strip()) == 0:
            quality_status.append("flagged")
            quality_reason.append("empty_customer_text")
        elif len(c_sub) == 0:
            # Customer message was only an image/link or mention
            quality_status.append("flagged")
            quality_reason.append("url_or_mention_only_customer")
        elif len(c_sub) < 3:
            # E.g. "?", "ok", "hi"
            quality_status.append("flagged")
            quality_reason.append("ultra_short_customer_message")
        elif pd.isnull(row['response_text']) or len(str(row['response_text']).strip()) == 0:
            quality_status.append("flagged")
            quality_reason.append("empty_response_text")
        elif delay < 0:
            quality_status.append("flagged")
            quality_reason.append("impossible_chronology")
        elif is_dup:
            quality_status.append("flagged")
            quality_reason.append("duplicate_customer_turn")
        else:
            quality_status.append("clean")
            quality_reason.append("valid")

    df_pairs['quality_status'] = quality_status
    df_pairs['quality_reason'] = quality_reason

    # Filter clean dataset
    df_clean = df_pairs[df_pairs['quality_status'] == 'clean'].copy()
    clean_pairs_count = len(df_clean)
    flagged_pairs_count = total_raw_pairs - clean_pairs_count

    print(f"  -> Retained clean support pairs: {clean_pairs_count:,} ({clean_pairs_count/total_raw_pairs*100:.2f}%)")
    print(f"  -> Flagged/removed pairs: {flagged_pairs_count:,} ({flagged_pairs_count/total_raw_pairs*100:.2f}%)")
    print("  -> Flag breakdown:")
    for reason, cnt in df_pairs['quality_reason'].value_counts().items():
        print(f"      * {reason}: {cnt:,}")

    # Save cleaned dataset
    clean_pairs_path = os.path.join(PROCESSED_DIR, "support_pairs_clean.csv")
    df_clean.to_csv(clean_pairs_path, index=False, encoding='utf-8')
    print(f"  -> Saved clean dataset to {clean_pairs_path}")

    # Step 4: Assign Response Type Labels
    print("\n[Step 4/7] Generating transparent rule-based response-type labels...")
    df_clean['response_type'] = df_clean.apply(assign_response_type, axis=1)
    # Also update clean_pairs_path with the response_type column
    df_clean.to_csv(clean_pairs_path, index=False, encoding='utf-8')
    print("  -> Response type distribution:")
    for r_type, cnt in df_clean['response_type'].value_counts().items():
        print(f"      * {r_type}: {cnt:,} ({cnt/clean_pairs_count*100:.2f}%)")

    # Step 5: Conversation-Level Leakage-Safe Splitting with Golden-Set Protection
    print("\n[Step 5/7] Executing conversation-level splitting with Golden Set protection...")
    # Get all conversation IDs present in the clean support pairs
    unique_clean_convs = df_clean['conversation_id'].unique()
    total_clean_convs = len(unique_clean_convs)

    # Protect manual sample conversations
    sample_conv_ids = set(df_sample['conversation_id'].unique())
    golden_in_clean = set(sample_conv_ids).intersection(set(unique_clean_convs))
    print(f"  -> Total unique clean conversations: {total_clean_convs:,}")
    print(f"  -> Golden manual evaluation conversations protected: {len(golden_in_clean):,}")

    # Remaining conversations to split
    remaining_convs = np.array([c for c in unique_clean_convs if c not in golden_in_clean])
    
    # Deterministic shuffle
    rng = np.random.RandomState(42)
    rng.shuffle(remaining_convs)

    n_remaining = len(remaining_convs)
    n_train = int(round(0.70 * total_clean_convs))
    n_val = int(round(0.15 * total_clean_convs))
    
    train_conv_ids = set(remaining_convs[:n_train])
    val_conv_ids = set(remaining_convs[n_train:n_train + n_val])
    # Test set gets remaining plus all protected golden sample conversations
    test_conv_ids = set(remaining_convs[n_train + n_val:]).union(golden_in_clean)

    print(f"  -> Conversation splits:")
    print(f"      * Train: {len(train_conv_ids):,} ({len(train_conv_ids)/total_clean_convs*100:.2f}%)")
    print(f"      * Validation: {len(val_conv_ids):,} ({len(val_conv_ids)/total_clean_convs*100:.2f}%)")
    print(f"      * Test (incl. golden): {len(test_conv_ids):,} ({len(test_conv_ids)/total_clean_convs*100:.2f}%)")

    # Partition support pairs
    df_train = df_clean[df_clean['conversation_id'].isin(train_conv_ids)].copy()
    df_val = df_clean[df_clean['conversation_id'].isin(val_conv_ids)].copy()
    df_test = df_clean[df_clean['conversation_id'].isin(test_conv_ids)].copy()

    train_path = os.path.join(SPLITS_DIR, "train.csv")
    val_path = os.path.join(SPLITS_DIR, "validation.csv")
    test_path = os.path.join(SPLITS_DIR, "test.csv")

    df_train.to_csv(train_path, index=False, encoding='utf-8')
    df_val.to_csv(val_path, index=False, encoding='utf-8')
    df_test.to_csv(test_path, index=False, encoding='utf-8')

    print(f"  -> Support pair splits:")
    print(f"      * Train pairs: {len(df_train):,} ({len(df_train)/clean_pairs_count*100:.2f}%)")
    print(f"      * Validation pairs: {len(df_val):,} ({len(df_val)/clean_pairs_count*100:.2f}%)")
    print(f"      * Test pairs: {len(df_test):,} ({len(df_test)/clean_pairs_count*100:.2f}%)")

    # Map tweet counts per split
    tweets_conv_map = df_tweets.groupby('conversation_id').size().to_dict()
    train_tweets_cnt = sum(tweets_conv_map.get(c, 0) for c in train_conv_ids)
    val_tweets_cnt = sum(tweets_conv_map.get(c, 0) for c in val_conv_ids)
    test_tweets_cnt = sum(tweets_conv_map.get(c, 0) for c in test_conv_ids)

    print(f"  -> Tweets per split:")
    print(f"      * Train tweets: {train_tweets_cnt:,}")
    print(f"      * Validation tweets: {val_tweets_cnt:,}")
    print(f"      * Test tweets: {test_tweets_cnt:,}")

    # Step 6: Create Retrieval-Safe Corpus
    print("\n[Step 6/7] Constructing isolated retrieval corpus from training split only...")
    retrieval_train_path = os.path.join(RETRIEVAL_DIR, "retrieval_train.csv")
    df_train.to_csv(retrieval_train_path, index=False, encoding='utf-8')
    print(f"  -> Exported retrieval_train.csv ({len(df_train):,} records) to {retrieval_train_path}")

    retrieval_readme_path = os.path.join(RETRIEVAL_DIR, "README.md")
    with open(retrieval_readme_path, "w", encoding="utf-8") as f_r:
        f_r.write(
            "# Retrieval Knowledge Base (Leakage Isolation)\n\n"
            "> [!IMPORTANT]\n"
            "> **Strict Leakage Protection**: Retrieval knowledge is constructed strictly from "
            "**training conversations** (`train.csv`). Support pairs and conversations from `validation.csv`, "
            "`test.csv`, and the golden evaluation benchmark are strictly barred from the retrieval index "
            "to prevent memorization and data leakage during downstream evaluation.\n\n"
            f"- **Corpus File**: `retrieval_train.csv`\n"
            f"- **Corpus Size**: {len(df_train):,} support pairs\n"
            f"- **Conversations Represented**: {len(train_conv_ids):,} distinct conversation trees\n"
        )
    print(f"  -> Written retrieval isolation documentation to {retrieval_readme_path}")

    # Step 7: Temporal Analysis
    print("\n[Step 7/7] Conducting temporal distribution analysis...")
    df_clean['resp_dt'] = pd.to_datetime(df_clean['response_timestamp'], format='%a %b %d %H:%M:%S +0000 %Y', errors='coerce')
    earliest_ts = df_clean['resp_dt'].min()
    latest_ts = df_clean['resp_dt'].max()
    yearly_dist = df_clean['resp_dt'].dt.year.value_counts().sort_index().to_dict()
    monthly_2017 = df_clean[df_clean['resp_dt'].dt.year == 2017]['resp_dt'].dt.month.value_counts().sort_index().to_dict()

    print(f"  -> Earliest Response Timestamp: {earliest_ts}")
    print(f"  -> Latest Response Timestamp: {latest_ts}")
    print(f"  -> Yearly Distribution: {yearly_dist}")

    # Compile comprehensive stats dictionary for report
    summary_stats = {
        'total_raw_pairs': int(total_raw_pairs),
        'clean_pairs_count': int(clean_pairs_count),
        'flagged_pairs_count': int(flagged_pairs_count),
        'flag_breakdown': {k: int(v) for k, v in df_pairs['quality_reason'].value_counts().items()},
        'response_type_distribution': {k: int(v) for k, v in df_clean['response_type'].value_counts().items()},
        'total_clean_convs': int(total_clean_convs),
        'golden_convs_protected': int(len(golden_in_clean)),
        'split_conversations': {
            'train': int(len(train_conv_ids)),
            'validation': int(len(val_conv_ids)),
            'test': int(len(test_conv_ids)),
        },
        'split_support_pairs': {
            'train': int(len(df_train)),
            'validation': int(len(df_val)),
            'test': int(len(df_test)),
        },
        'split_tweets': {
            'train': int(train_tweets_cnt),
            'validation': int(val_tweets_cnt),
            'test': int(test_tweets_cnt),
        },
        'retrieval_corpus_size': int(len(df_train)),
        'temporal_analysis': {
            'earliest_timestamp': str(earliest_ts),
            'latest_timestamp': str(latest_ts),
            'yearly_distribution': {int(k): int(v) for k, v in yearly_dist.items()},
            'monthly_2017': {int(k): int(v) for k, v in monthly_2017.items()},
        }
    }

    stats_out = os.path.join(REPORTS_DIR, "cleaning_stats.json")
    with open(stats_out, "w", encoding="utf-8") as f_s:
        json.dump(summary_stats, f_s, indent=2)
    print(f"  -> Saved cleaning statistics JSON to {stats_out}")

    print(f"\nPhase 4 processing completed successfully in {time.time() - t0:.1f}s.")
    return summary_stats


if __name__ == "__main__":
    run_cleaning_and_splitting()
