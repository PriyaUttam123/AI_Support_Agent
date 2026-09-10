"""
Tests for Phase 5 baseline classifiers and evaluation pipeline.

Covers:
  1. MajorityClassifier always predicts general_inquiry_other.
  2. RuleBasedIntentClassifier returns valid taxonomy labels.
  3. Evaluation counts match test-set size (16,298).
  4. Metrics are generated successfully.
"""

import os
import sys
import json
import unittest
from pathlib import Path

# Make src importable when run from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.classification.rule_based import RuleBasedIntentClassifier
from src.evaluation.evaluate_baselines import (
    MajorityClassifier,
    compute_metrics,
    INTENT_LABELS,
    MAJORITY_CLASS,
    EXPECTED_TEST_SIZE,
    TEXT_COL,
    LABEL_COL,
    TEST_PATH,
)

RESULTS_PATH = PROJECT_ROOT / "reports" / "baseline_results.json"


# ─────────────────────────────────────────────────────────────────────────────
class TestMajorityClassifier(unittest.TestCase):
    """Verify the majority-class baseline behaves trivially correctly."""

    def setUp(self):
        self.clf = MajorityClassifier(MAJORITY_CLASS)

    def test_always_predicts_majority_class(self):
        """Every prediction must equal general_inquiry_other."""
        texts = ["any text", "another tweet", ""]
        preds = self.clf.predict(texts)
        for p in preds:
            self.assertEqual(p, MAJORITY_CLASS)

    def test_output_length_matches_input(self):
        """Output list length must match input length."""
        texts = ["a", "b", "c", "d", "e"]
        preds = self.clf.predict(texts)
        self.assertEqual(len(preds), len(texts))

    def test_empty_input(self):
        """Empty input list should return empty output."""
        self.assertEqual(self.clf.predict([]), [])

    def test_majority_class_is_in_taxonomy(self):
        """The majority class must be a member of the official taxonomy."""
        self.assertIn(MAJORITY_CLASS, INTENT_LABELS)


# ─────────────────────────────────────────────────────────────────────────────
class TestRuleBasedClassifier(unittest.TestCase):
    """Verify the rule-based classifier returns valid taxonomy labels."""

    def setUp(self):
        self.clf = RuleBasedIntentClassifier()

    def test_predict_single_returns_dict_with_required_keys(self):
        """predict_single must return a dict with predicted_intent, confidence, matched_rule."""
        result = self.clf.predict_single("My battery is dying fast")
        self.assertIn("predicted_intent", result)
        self.assertIn("confidence", result)
        self.assertIn("matched_rule", result)

    def test_predict_single_returns_valid_intent(self):
        """predicted_intent must always be a member of the 8-class taxonomy."""
        queries = [
            "My battery is dying fast",
            "Cannot connect to wifi",
            "Update keeps failing",
            "Thank you so much",
            "Screen is cracked",
            "Apple ID password reset",
            "App keeps crashing",
            "keyboard types wrong letter",
        ]
        for q in queries:
            result = self.clf.predict_single(q)
            self.assertIn(
                result["predicted_intent"], INTENT_LABELS,
                f"Invalid intent '{result['predicted_intent']}' for query: {q}"
            )

    def test_predict_batch_returns_list_of_dicts(self):
        """predict() must return a list of dicts, one per input."""
        texts = ["wifi broken", "battery dead", "no match here"]
        results = self.clf.predict(texts)
        self.assertEqual(len(results), len(texts))
        for r in results:
            self.assertIsInstance(r, dict)
            self.assertIn(r["predicted_intent"], INTENT_LABELS)

    def test_fallback_on_unknown_text(self):
        """Text with no recognizable keywords should fall back to general_inquiry_other."""
        result = self.clf.predict_single("This is a completely generic sentence.")
        self.assertEqual(result["predicted_intent"], "general_inquiry_other")
        self.assertEqual(result["matched_rule"], "fallback_no_match")

    def test_known_battery_query(self):
        """Battery-related query should be classified as battery_power."""
        result = self.clf.predict_single("Why is my battery draining so fast?")
        self.assertEqual(result["predicted_intent"], "battery_power")

    def test_known_keyboard_query(self):
        """Autocorrect/keyboard query should be classified as keyboard_typing."""
        result = self.clf.predict_single("The autocorrect changes everything I type")
        self.assertEqual(result["predicted_intent"], "keyboard_typing")

    def test_known_connectivity_query(self):
        """WiFi query should be classified as connectivity_network."""
        result = self.clf.predict_single("My Wi-Fi keeps disconnecting")
        self.assertEqual(result["predicted_intent"], "connectivity_network")


# ─────────────────────────────────────────────────────────────────────────────
class TestEvaluationCounts(unittest.TestCase):
    """Verify that evaluation counts match the test-set size."""

    @classmethod
    def setUpClass(cls):
        """Load test data once for the whole class."""
        if not TEST_PATH.exists():
            raise unittest.SkipTest(f"Test CSV not found: {TEST_PATH}")
        cls.df = pd.read_csv(TEST_PATH)
        cls.texts  = cls.df[TEXT_COL].tolist()
        cls.y_true = cls.df[LABEL_COL].tolist()

    def test_test_set_has_expected_size(self):
        """Test CSV must contain exactly 16,298 rows."""
        self.assertEqual(len(self.df), EXPECTED_TEST_SIZE)

    def test_majority_predictions_match_test_size(self):
        """MajorityClassifier must produce exactly 16,298 predictions."""
        clf   = MajorityClassifier(MAJORITY_CLASS)
        preds = clf.predict(self.texts)
        self.assertEqual(len(preds), EXPECTED_TEST_SIZE)

    def test_rule_based_predictions_match_test_size(self):
        """RuleBasedIntentClassifier must produce exactly 16,298 predictions."""
        clf   = RuleBasedIntentClassifier()
        preds = clf.predict(self.texts)
        self.assertEqual(len(preds), EXPECTED_TEST_SIZE)

    def test_no_missing_majority_predictions(self):
        """No prediction in the majority baseline may be None or empty."""
        clf   = MajorityClassifier(MAJORITY_CLASS)
        preds = clf.predict(self.texts)
        for p in preds:
            self.assertIsNotNone(p)
            self.assertNotEqual(p, "")

    def test_no_missing_rule_based_predictions(self):
        """No prediction in the rule-based baseline may be None or empty."""
        clf   = RuleBasedIntentClassifier()
        preds = [r["predicted_intent"] for r in clf.predict(self.texts)]
        for p in preds:
            self.assertIsNotNone(p)
            self.assertNotEqual(p, "")

    def test_all_rule_based_predictions_in_taxonomy(self):
        """Every rule-based prediction must belong to the 8-class taxonomy."""
        clf   = RuleBasedIntentClassifier()
        preds = [r["predicted_intent"] for r in clf.predict(self.texts)]
        invalid = set(preds) - set(INTENT_LABELS)
        self.assertEqual(len(invalid), 0, f"Invalid predicted intents: {invalid}")

    def test_all_ground_truth_labels_in_taxonomy(self):
        """Every ground-truth label in the test set must belong to the 8-class taxonomy."""
        invalid = set(self.y_true) - set(INTENT_LABELS)
        self.assertEqual(len(invalid), 0, f"Unexpected labels: {invalid}")


# ─────────────────────────────────────────────────────────────────────────────
class TestMetricsGeneration(unittest.TestCase):
    """Verify that metric computation succeeds and returns sensible values."""

    def _make_dummy_data(self):
        """Return small representative y_true / y_pred lists."""
        y_true = INTENT_LABELS * 2          # 16 examples, 2 per class
        y_pred = INTENT_LABELS[::-1] * 2    # all wrong
        return y_true, y_pred

    def test_compute_metrics_returns_required_keys(self):
        """compute_metrics must return accuracy, macro_f1, weighted_f1, per_class, confusion_matrix."""
        y_true, y_pred = self._make_dummy_data()
        result = compute_metrics(y_true, y_pred, INTENT_LABELS)
        for key in ("accuracy", "macro_f1", "weighted_f1", "per_class", "confusion_matrix"):
            self.assertIn(key, result)

    def test_accuracy_in_valid_range(self):
        """Accuracy must be in [0, 1]."""
        y_true, y_pred = self._make_dummy_data()
        result = compute_metrics(y_true, y_pred, INTENT_LABELS)
        self.assertGreaterEqual(result["accuracy"], 0.0)
        self.assertLessEqual(result["accuracy"],   1.0)

    def test_f1_scores_in_valid_range(self):
        """Macro F1 and weighted F1 must be in [0, 1]."""
        y_true, y_pred = self._make_dummy_data()
        result = compute_metrics(y_true, y_pred, INTENT_LABELS)
        self.assertGreaterEqual(result["macro_f1"],    0.0)
        self.assertLessEqual(result["macro_f1"],       1.0)
        self.assertGreaterEqual(result["weighted_f1"], 0.0)
        self.assertLessEqual(result["weighted_f1"],    1.0)

    def test_per_class_contains_all_intents(self):
        """per_class dict must have an entry for every intent in the taxonomy."""
        y_true, y_pred = self._make_dummy_data()
        result = compute_metrics(y_true, y_pred, INTENT_LABELS)
        for label in INTENT_LABELS:
            self.assertIn(label, result["per_class"])

    def test_perfect_predictions_give_accuracy_one(self):
        """When predictions equal ground truth, accuracy must be 1.0."""
        y_true = INTENT_LABELS * 3
        y_pred = INTENT_LABELS * 3
        result = compute_metrics(y_true, y_pred, INTENT_LABELS)
        self.assertAlmostEqual(result["accuracy"], 1.0)

    def test_confusion_matrix_shape(self):
        """Confusion matrix must be n_classes x n_classes."""
        y_true, y_pred = self._make_dummy_data()
        result = compute_metrics(y_true, y_pred, INTENT_LABELS)
        n = len(INTENT_LABELS)
        self.assertEqual(len(result["confusion_matrix"]), n)
        for row in result["confusion_matrix"]:
            self.assertEqual(len(row), n)

    def test_saved_results_json_is_loadable(self):
        """If baseline_results.json exists, it must be valid JSON with required keys."""
        if not RESULTS_PATH.exists():
            self.skipTest("baseline_results.json not yet generated -- run evaluate_baselines.py first")
        with open(RESULTS_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        for key in ("test_size", "labels", "majority_baseline", "rule_based"):
            self.assertIn(key, data)
        self.assertEqual(data["test_size"], EXPECTED_TEST_SIZE)

    def test_saved_results_rule_based_beats_majority_macro_f1(self):
        """Rule-based macro F1 must exceed majority baseline macro F1."""
        if not RESULTS_PATH.exists():
            self.skipTest("baseline_results.json not yet generated -- run evaluate_baselines.py first")
        with open(RESULTS_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        maj_f1 = data["majority_baseline"]["macro_f1"]
        rb_f1  = data["rule_based"]["macro_f1"]
        self.assertGreater(rb_f1, maj_f1,
            f"Rule-based macro F1 ({rb_f1}) should exceed majority ({maj_f1})")


if __name__ == "__main__":
    unittest.main()
