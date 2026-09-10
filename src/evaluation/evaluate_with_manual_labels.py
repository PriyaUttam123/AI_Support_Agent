"""
Phase 5: Manual Label Evaluation (Placeholder)
===============================================
This script is a TEMPLATE for future evaluation using human-annotated labels.

Purpose:
--------
To properly evaluate classifiers (rule-based, LLM-based, etc.) against
ground truth labels assigned by human annotators, not regex patterns.

This addresses the data leakage issue in evaluate_baselines.py where
labels were generated using the same regex patterns as the classifier.

TODO: Implement when manual labels are available
=================================================

1. Load manual labels:
   - TODO: Define path to manually annotated CSV file
   - TODO: File should contain at minimum: customer_text, manual_intent_label
   - Example: MANUAL_LABELS_PATH = PROJECT_ROOT / "data" / "evaluation" / "human_annotated_test.csv"

2. Load test data:
   - TODO: Load the corresponding test set with customer_text
   - TODO: Ensure alignment between manual labels and test data (e.g., by pair_id or customer_tweet_id)

3. Generate predictions:
   - TODO: Instantiate RuleBasedIntentClassifier (or other classifier to evaluate)
   - TODO: Run predict() on customer_text from the manual label set
   - TODO: Extract predicted_intent from prediction results

4. Align predictions with manual labels:
   - TODO: Ensure predictions and manual labels are in the same order
   - TODO: Handle any missing or mismatched entries
   - TODO: Create aligned lists: y_true (manual labels), y_pred (classifier predictions)

5. Calculate metrics:
   - TODO: Use sklearn.metrics to compute:
     * accuracy_score(y_true, y_pred)
     * precision_recall_fscore_support(y_true, y_pred, average='macro')
     * precision_recall_fscore_support(y_true, y_pred, average='weighted')
     * classification_report(y_true, y_pred)
     * confusion_matrix(y_true, y_pred)
   - TODO: Calculate per-class precision, recall, F1 for all 8 intent classes

6. Generate reports:
   - TODO: Save results to JSON (e.g., reports/manual_label_evaluation.json)
   - TODO: Generate confusion matrix CSV and PNG
   - TODO: Create markdown report comparing:
     * Rule-based classifier performance on manual labels
     * Majority baseline on manual labels
     * Any other classifiers (e.g., LLM-based) on manual labels

7. Error analysis:
   - TODO: Sample mis-classified examples
   - TODO: Analyze common failure patterns
   - TODO: Compare against regex-based upper bound results

Implementation Notes:
----------------------
- Manual labels should be created by domain experts familiar with the intent taxonomy
- Recommended sample size: 200-500 diverse examples covering all 8 intent classes
- Ensure annotators have access to the intent taxonomy documentation (configs/intent_taxonomy.json)
- Consider inter-annotator agreement metrics if multiple annotators are used

Expected Output:
----------------
When implemented, this script will produce:
- True performance metrics without data leakage
- Comparison between rule-based and LLM-based classifiers
- Realistic expectations for production deployment
- Identification of classifier weaknesses not visible in upper-bound evaluation

Status: PENDING MANUAL ANNOTATION
==================================
This script is a placeholder. Do not run until manual labels are available.
See project documentation for manual annotation guidelines.
"""

import sys
from pathlib import Path

# ── Make src importable when run from project root ────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# TODO: Uncomment and implement when manual labels are available
# import pandas as pd
# from sklearn.metrics import (
#     accuracy_score,
#     precision_recall_fscore_support,
#     classification_report,
#     confusion_matrix,
# )
# from src.classification.rule_based import RuleBasedIntentClassifier


def run_manual_label_evaluation():
    """
    TODO: Implement this function when manual labels are available.
    
    Steps:
    1. Load manual label CSV
    2. Load corresponding test data
    3. Align labels and data
    4. Generate classifier predictions
    5. Compute metrics
    6. Save reports
    """
    raise NotImplementedError(
        "Manual label evaluation not yet implemented. "
        "This requires human-annotated intent labels. "
        "See script docstring for implementation TODOs."
    )


if __name__ == "__main__":
    print("=" * 70)
    print("Manual Label Evaluation - NOT YET IMPLEMENTED")
    print("=" * 70)
    print("\nThis script is a template for future evaluation using")
    print("human-annotated labels to avoid data leakage.")
    print("\nSee script docstring for detailed implementation TODOs.")
    print("\nStatus: PENDING MANUAL ANNOTATION")
    print("=" * 70)
