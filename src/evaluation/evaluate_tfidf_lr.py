"""
Phase 6A: TF-IDF + Logistic Regression Baseline Evaluation
==========================================================
Evaluates a traditional机器学习 baseline using TF-IDF features and
Logistic Regression for intent classification.

This provides a proper machine learning baseline that:
- Learns from data rather than using hand-crafted rules
- Uses only training data for feature extraction (no leakage)
- Can be compared against rule-based and majority baselines

Dataset:
  Train: data/processed/apple_support/intent/train_intents.csv
  Test:  data/processed/apple_support/intent/test_intents.csv

Outputs:
  reports/tfidf_lr_results.json
  reports/tfidf_lr_confusion_matrix.csv
  reports/tfidf_lr_confusion_matrix.png
  reports/baseline_comparison.md
"""

import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# ── Make src importable when run from project root ────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "apple_support" / "intent" / "train_intents.csv"
TEST_PATH  = PROJECT_ROOT / "data" / "processed" / "apple_support" / "intent" / "test_intents.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Taxonomy ──────────────────────────────────────────────────────────────────
INTENT_LABELS = [
    "general_inquiry_other",
    "software_update",
    "battery_power",
    "performance_system",
    "hardware_audio_display",
    "account_billing",
    "keyboard_typing",
    "connectivity_network",
]

# ── Text column used for classification ───────────────────────────────────────
TEXT_COL  = "customer_text_normalized"
LABEL_COL = "intent"

EXPECTED_TRAIN_SIZE = 73_700
EXPECTED_TEST_SIZE = 16_298


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: list, y_pred: list, labels: list) -> dict:
    """Compute accuracy, macro precision/recall/F1, weighted F1, and per-class metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    
    # Macro metrics (unweighted mean across classes)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    
    # Weighted metrics (weighted by support)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )

    # Per-class metrics
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    per_class = {}
    for label in labels:
        per_class[label] = {
            "precision": round(report[label]["precision"], 4),
            "recall":    round(report[label]["recall"],    4),
            "f1":        round(report[label]["f1-score"],  4),
            "support":   int(report[label]["support"]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy":           round(accuracy,           4),
        "macro_precision":    round(precision_macro,    4),
        "macro_recall":       round(recall_macro,       4),
        "macro_f1":           round(f1_macro,           4),
        "weighted_precision": round(precision_weighted, 4),
        "weighted_recall":    round(recall_weighted,    4),
        "weighted_f1":        round(f1_weighted,        4),
        "per_class":          per_class,
        "confusion_matrix":   cm.tolist(),
    }


def save_confusion_matrix_csv(cm: list, labels: list, path: Path) -> None:
    """Save confusion matrix as a labeled CSV."""
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    df_cm.index.name = "true \\ predicted"
    df_cm.to_csv(path)
    print(f"  Saved confusion matrix CSV -> {path.relative_to(PROJECT_ROOT)}")


def save_confusion_matrix_png(cm: list, labels: list, path: Path, title: str) -> None:
    """Save confusion matrix as a PNG heatmap (requires matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cm_arr = np.array(cm)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Predicted Intent", fontsize=11)
        ax.set_ylabel("True Intent",      fontsize=11)
        ax.set_title(title, fontsize=13, pad=12)

        thresh = cm_arr.max() / 2.0
        for i in range(cm_arr.shape[0]):
            for j in range(cm_arr.shape[1]):
                ax.text(
                    j, i, str(cm_arr[i, j]),
                    ha="center", va="center",
                    color="white" if cm_arr[i, j] > thresh else "black",
                    fontsize=8,
                )

        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved confusion matrix PNG  -> {path.relative_to(PROJECT_ROOT)}")
    except ImportError:
        warnings.warn("matplotlib not installed -- PNG confusion matrix skipped.")


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation() -> dict:
    """Run the TF-IDF + Logistic Regression evaluation pipeline."""
    print("=" * 70)
    print("Phase 6A: TF-IDF + Logistic Regression Baseline Evaluation")
    print("=" * 70)

    # 1. Load datasets
    print(f"\n[1/8] Loading datasets")
    print(f"  Train: {TRAIN_PATH.relative_to(PROJECT_ROOT)}")
    train_df = pd.read_csv(TRAIN_PATH)
    print(f"  Test:  {TEST_PATH.relative_to(PROJECT_ROOT)}")
    test_df = pd.read_csv(TEST_PATH)

    # Validation checks
    assert len(train_df) == EXPECTED_TRAIN_SIZE, (
        f"Expected {EXPECTED_TRAIN_SIZE} train examples, got {len(train_df)}"
    )
    assert len(test_df) == EXPECTED_TEST_SIZE, (
        f"Expected {EXPECTED_TEST_SIZE} test examples, got {len(test_df)}"
    )
    assert TEXT_COL  in train_df.columns, f"Missing column: {TEXT_COL}"
    assert LABEL_COL in train_df.columns, f"Missing column: {LABEL_COL}"
    assert TEXT_COL  in test_df.columns,  f"Missing column: {TEXT_COL}"
    assert LABEL_COL in test_df.columns,  f"Missing column: {LABEL_COL}"
    
    # Check for null values
    assert train_df[TEXT_COL].isnull().sum()  == 0, "Null values found in train customer_text_normalized"
    assert train_df[LABEL_COL].isnull().sum() == 0, "Null values found in train intent"
    assert test_df[TEXT_COL].isnull().sum()   == 0, "Null values found in test customer_text_normalized"
    assert test_df[LABEL_COL].isnull().sum()  == 0, "Null values found in test intent"

    print(f"  Train loaded: {len(train_df):,} rows  OK")
    print(f"  Test loaded:  {len(test_df):,} rows  OK")

    # 2. Extract features and labels
    print(f"\n[2/8] Extracting features and labels")
    X_train_text = train_df[TEXT_COL].tolist()
    y_train      = train_df[LABEL_COL].tolist()
    X_test_text  = test_df[TEXT_COL].tolist()
    y_test       = test_df[LABEL_COL].tolist()

    print(f"  Train features: {len(X_train_text):,}")
    print(f"  Train labels:   {len(y_train):,}")
    print(f"  Test features:  {len(X_test_text):,}")
    print(f"  Test labels:    {len(y_test):,}")

    # 3. Fit TF-IDF on training data ONLY (no leakage)
    print(f"\n[3/8] Fitting TF-IDF vectorizer on training data")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words='english',
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    print(f"  TF-IDF fitted: {X_train_tfidf.shape[1]:,} features")
    print(f"  Train TF-IDF shape: {X_train_tfidf.shape}")

    # 4. Transform test data using fitted vectorizer
    print(f"\n[4/8] Transforming test data using fitted TF-IDF")
    X_test_tfidf = vectorizer.transform(X_test_text)
    print(f"  Test TF-IDF shape: {X_test_tfidf.shape}")

    # 5. Train Logistic Regression
    print(f"\n[5/8] Training Logistic Regression")
    clf = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced',
    )
    clf.fit(X_train_tfidf, y_train)
    print(f"  Model trained successfully")

    # 6. Generate predictions
    print(f"\n[6/8] Generating predictions on test set")
    y_pred = clf.predict(X_test_tfidf)
    print(f"  Predictions generated: {len(y_pred):,}")

    # 7. Compute metrics
    print(f"\n[7/8] Computing metrics")
    results = compute_metrics(y_test, y_pred, INTENT_LABELS)

    print(f"\n  TF-IDF + Logistic Regression:")
    print(f"    Accuracy:           {results['accuracy']:.4f}")
    print(f"    Macro Precision:    {results['macro_precision']:.4f}")
    print(f"    Macro Recall:       {results['macro_recall']:.4f}")
    print(f"    Macro F1:           {results['macro_f1']:.4f}")
    print(f"    Weighted Precision: {results['weighted_precision']:.4f}")
    print(f"    Weighted Recall:    {results['weighted_recall']:.4f}")
    print(f"    Weighted F1:        {results['weighted_f1']:.4f}")

    # 8. Save results
    print(f"\n[8/8] Saving results")
    
    # Save JSON results
    results_path = REPORTS_DIR / "tfidf_lr_results.json"
    results_payload = {
        "model": "TF-IDF + Logistic Regression",
        "train_set": str(TRAIN_PATH.relative_to(PROJECT_ROOT)),
        "test_set": str(TEST_PATH.relative_to(PROJECT_ROOT)),
        "train_size": len(train_df),
        "test_size": len(test_df),
        "labels": INTENT_LABELS,
        "tfidf_params": {
            "max_features": 10000,
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 0.95,
            "stop_words": "english",
        },
        "logistic_regression_params": {
            "max_iter": 1000,
            "random_state": 42,
            "class_weight": "balanced",
        },
        "metrics": results,
    }
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(results_payload, fh, indent=2)
    print(f"  Saved results JSON -> {results_path.relative_to(PROJECT_ROOT)}")

    # Save confusion matrix
    cm_csv_path = REPORTS_DIR / "tfidf_lr_confusion_matrix.csv"
    cm_png_path = REPORTS_DIR / "tfidf_lr_confusion_matrix.png"
    save_confusion_matrix_csv(results["confusion_matrix"], INTENT_LABELS, cm_csv_path)
    save_confusion_matrix_png(
        results["confusion_matrix"],
        INTENT_LABELS,
        cm_png_path,
        "TF-IDF + Logistic Regression -- Confusion Matrix (Test Set)",
    )

    # Final validation
    print(f"\n" + "=" * 70)
    print("Evaluation complete.")
    print("=" * 70)
    print(f"  Train examples: {len(train_df):,}")
    print(f"  Test examples:  {len(test_df):,}")
    print(f"  TF-IDF features: {X_train_tfidf.shape[1]:,}")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Macro F1: {results['macro_f1']:.4f}")

    return results_payload


if __name__ == "__main__":
    run_evaluation()
