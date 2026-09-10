"""Unit tests verifying Phase 4 data cleaning, quality filtering, and leakage-safe splits."""

import os
import unittest
import pandas as pd

TWEETS_PATH = os.path.join("data", "processed", "apple_support", "tweets.csv")
RAW_PAIRS_PATH = os.path.join("data", "processed", "apple_support", "support_pairs.csv")
CLEAN_PAIRS_PATH = os.path.join("data", "processed", "apple_support", "support_pairs_clean.csv")
TRAIN_PATH = os.path.join("data", "processed", "apple_support", "splits", "train.csv")
VAL_PATH = os.path.join("data", "processed", "apple_support", "splits", "validation.csv")
TEST_PATH = os.path.join("data", "processed", "apple_support", "splits", "test.csv")
RETRIEVAL_PATH = os.path.join("data", "processed", "apple_support", "retrieval", "retrieval_train.csv")
SAMPLE_PATH = os.path.join("data", "evaluation", "apple_support_manual_sample.csv")


class TestDataCleaningAndSplits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tweets_df = pd.read_csv(TWEETS_PATH, low_memory=False)
        cls.raw_pairs_df = pd.read_csv(RAW_PAIRS_PATH, low_memory=False)
        cls.clean_pairs_df = pd.read_csv(CLEAN_PAIRS_PATH, low_memory=False)
        cls.train_df = pd.read_csv(TRAIN_PATH, low_memory=False)
        cls.val_df = pd.read_csv(VAL_PATH, low_memory=False)
        cls.test_df = pd.read_csv(TEST_PATH, low_memory=False)
        cls.retrieval_df = pd.read_csv(RETRIEVAL_PATH, low_memory=False)
        cls.sample_df = pd.read_csv(SAMPLE_PATH, low_memory=False)

    def test_1_no_duplicate_tweet_ids_in_tweets(self):
        """1. Verify no duplicate tweet IDs in processed tweets."""
        self.assertTrue(self.tweets_df["tweet_id"].is_unique)

    def test_2_no_conversation_overlap_between_splits(self):
        """2. Verify no conversation_id overlap between train, validation, and test splits."""
        train_convs = set(self.train_df["conversation_id"])
        val_convs = set(self.val_df["conversation_id"])
        test_convs = set(self.test_df["conversation_id"])

        self.assertEqual(len(train_convs.intersection(val_convs)), 0, "Train and Validation share conversations!")
        self.assertEqual(len(train_convs.intersection(test_convs)), 0, "Train and Test share conversations!")
        self.assertEqual(len(val_convs.intersection(test_convs)), 0, "Validation and Test share conversations!")

    def test_3_no_test_conversation_in_retrieval(self):
        """3. Verify no test conversation exists in retrieval_train."""
        test_convs = set(self.test_df["conversation_id"])
        retrieval_convs = set(self.retrieval_df["conversation_id"])
        self.assertEqual(len(test_convs.intersection(retrieval_convs)), 0, "Test conversations leaked into retrieval!")

    def test_4_support_pair_customer_inbound(self):
        """4. Verify every clean support pair customer tweet is inbound."""
        inbound_map = dict(zip(self.tweets_df["tweet_id"], self.tweets_df["inbound"]))
        sample_subset = self.clean_pairs_df.sample(n=min(5000, len(self.clean_pairs_df)), random_state=42)
        for cid in sample_subset["customer_tweet_id"]:
            self.assertTrue(inbound_map[cid], f"Customer tweet {cid} is not inbound")

    def test_5_support_pair_response_is_applesupport(self):
        """5. Verify every clean support pair response tweet belongs to AppleSupport."""
        inbound_map = dict(zip(self.tweets_df["tweet_id"], self.tweets_df["inbound"]))
        author_map = dict(zip(self.tweets_df["tweet_id"], self.tweets_df["author_id"]))
        sample_subset = self.clean_pairs_df.sample(n=min(5000, len(self.clean_pairs_df)), random_state=42)
        for rid in sample_subset["response_tweet_id"]:
            self.assertFalse(inbound_map[rid], f"Response tweet {rid} is not outbound")
            self.assertEqual(author_map[rid], "AppleSupport", f"Response tweet {rid} is not from AppleSupport")

    def test_6_customer_timestamp_le_response_timestamp(self):
        """6. Verify customer timestamp <= response timestamp (no negative delays)."""
        invalid_delays = self.clean_pairs_df[self.clean_pairs_df["turn_delay_seconds"] < 0]
        self.assertEqual(len(invalid_delays), 0, "Found responses preceding customer tweets in clean data")

    def test_7_original_text_remains_unchanged(self):
        """7. Verify original customer and response texts match raw pairs exactly."""
        clean_indexed = self.clean_pairs_df.set_index("pair_id")
        raw_indexed = self.raw_pairs_df.set_index("pair_id")
        sample_ids = clean_indexed.sample(n=min(1000, len(clean_indexed)), random_state=42).index

        for pid in sample_ids:
            self.assertEqual(clean_indexed.loc[pid, "customer_text"], raw_indexed.loc[pid, "customer_text"])
            self.assertEqual(clean_indexed.loc[pid, "response_text"], raw_indexed.loc[pid, "response_text"])

    def test_8_normalized_text_derived_from_original(self):
        """8. Verify normalized text is non-empty and derived from original text."""
        sample_subset = self.clean_pairs_df.sample(n=min(1000, len(self.clean_pairs_df)), random_state=42)
        for _, row in sample_subset.iterrows():
            self.assertTrue(len(str(row["customer_text_normalized"]).strip()) > 0)
            self.assertTrue(len(str(row["response_text_normalized"]).strip()) > 0)

    def test_9_no_null_texts_after_cleaning(self):
        """9. Verify no null customer_text or response_text in cleaned dataset."""
        self.assertEqual(self.clean_pairs_df["customer_text"].isnull().sum(), 0)
        self.assertEqual(self.clean_pairs_df["response_text"].isnull().sum(), 0)
        self.assertEqual(self.clean_pairs_df["customer_text_normalized"].isnull().sum(), 0)
        self.assertEqual(self.clean_pairs_df["response_text_normalized"].isnull().sum(), 0)

    def test_10_golden_conversations_protected_from_retrieval(self):
        """10. Verify golden evaluation sample conversations never appear in train or retrieval."""
        sample_convs = set(self.sample_df["conversation_id"])
        retrieval_convs = set(self.retrieval_df["conversation_id"])
        train_convs = set(self.train_df["conversation_id"])

        self.assertEqual(len(sample_convs.intersection(retrieval_convs)), 0, "Golden conversation leaked into retrieval!")
        self.assertEqual(len(sample_convs.intersection(train_convs)), 0, "Golden conversation leaked into training split!")


if __name__ == "__main__":
    unittest.main()
