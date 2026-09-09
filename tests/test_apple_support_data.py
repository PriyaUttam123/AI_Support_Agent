"""Unit tests verifying data integrity and consistency of the extracted AppleSupport dataset."""

import os
import unittest
import pandas as pd

TWEETS_PATH = os.path.join("data", "processed", "apple_support", "tweets.csv")
PAIRS_PATH = os.path.join("data", "processed", "apple_support", "support_pairs.csv")
SAMPLE_PATH = os.path.join("data", "evaluation", "apple_support_manual_sample.csv")


class TestAppleSupportData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert os.path.exists(TWEETS_PATH), f"Missing {TWEETS_PATH}"
        assert os.path.exists(PAIRS_PATH), f"Missing {PAIRS_PATH}"
        assert os.path.exists(SAMPLE_PATH), f"Missing {SAMPLE_PATH}"
        cls.tweets_df = pd.read_csv(TWEETS_PATH, low_memory=False)
        cls.pairs_df = pd.read_csv(PAIRS_PATH, low_memory=False)
        cls.sample_df = pd.read_csv(SAMPLE_PATH, low_memory=False)

    def test_no_duplicate_tweet_ids(self):
        """Verify that there are no duplicate tweet IDs in the processed tweets dataset."""
        self.assertTrue(self.tweets_df["tweet_id"].is_unique, "Found duplicate tweet_id in tweets.csv")

    def test_support_pairs_reference_valid_tweet_ids(self):
        """Verify that every support pair references valid tweet IDs existing in tweets.csv."""
        valid_ids = set(self.tweets_df["tweet_id"].values)
        cust_ids = set(self.pairs_df["customer_tweet_id"].values)
        resp_ids = set(self.pairs_df["response_tweet_id"].values)

        self.assertTrue(cust_ids.issubset(valid_ids), "Support pairs contain customer_tweet_ids not in tweets.csv")
        self.assertTrue(resp_ids.issubset(valid_ids), "Support pairs contain response_tweet_ids not in tweets.csv")

    def test_support_pairs_directionality(self):
        """Verify customer side is inbound and AppleSupport response side is outbound."""
        inbound_map = dict(zip(self.tweets_df["tweet_id"], self.tweets_df["inbound"]))
        author_map = dict(zip(self.tweets_df["tweet_id"], self.tweets_df["author_id"]))

        sample_pairs = self.pairs_df.sample(n=min(10000, len(self.pairs_df)), random_state=42)
        for _, row in sample_pairs.iterrows():
            c_id = row["customer_tweet_id"]
            r_id = row["response_tweet_id"]
            self.assertTrue(inbound_map[c_id], f"Customer tweet {c_id} is not inbound")
            self.assertFalse(inbound_map[r_id], f"Response tweet {r_id} is not outbound")
            self.assertEqual(author_map[r_id], "AppleSupport", f"Response tweet {r_id} author is not AppleSupport")

    def test_temporal_ordering(self):
        """Verify response timestamp does not precede customer timestamp."""
        invalid_delays = self.pairs_df[self.pairs_df["turn_delay_seconds"] < 0]
        self.assertEqual(len(invalid_delays), 0, f"Found {len(invalid_delays)} responses preceding customer tweets")

    def test_original_text_preserved(self):
        """Verify that customer and response original texts are preserved and non-empty."""
        self.assertTrue(self.pairs_df["customer_text"].notnull().all())
        self.assertTrue(self.pairs_df["response_text"].notnull().all())
        self.assertTrue((self.pairs_df["customer_text"].str.strip().str.len() > 0).all())
        self.assertTrue((self.pairs_df["response_text"].str.strip().str.len() > 0).all())

    def test_manual_sample_structure(self):
        """Verify reproducible manual sample dataset exists and has 250 valid rows."""
        self.assertEqual(len(self.sample_df), 250)
        self.assertIn("customer_text", self.sample_df.columns)
        self.assertIn("response_text", self.sample_df.columns)


if __name__ == "__main__":
    unittest.main()
