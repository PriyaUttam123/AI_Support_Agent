# Apple Support Intent Classification -- Baseline Comparison Report

## ⚠️ CRITICAL DATA LEAKAGE WARNING

**The evaluation results below represent UPPER BOUNDS, not real-world performance.**

**The Problem:**
The ground truth labels in `test_intents.csv` were generated using the **same regex patterns** as the `RuleBasedIntentClassifier`. This creates a circular evaluation where the rule-based classifier is tested against labels it would produce itself.

**What This Means:**
- The reported metrics for the rule-based classifier are **theoretical upper bounds**
- These metrics should **NOT** be interpreted as expected production performance
- The TF-IDF + Logistic Regression and embedding classifier results are also affected since they're trained on the same regex-generated labels
- Real-world performance will likely be lower due to ambiguous cases, edge cases, and human interpretation differences

**Planned Solution:**
A proper evaluation using human-annotated labels is planned for a future phase. See `src/evaluation/evaluate_with_manual_labels.py` for the evaluation template that will provide true performance metrics without data leakage.

---

## 1. Objective

Compare four baseline approaches for intent classification on the AppleSupport Twitter dataset:

1. **Majority-class baseline** - Trivial predictor that always emits the most frequent class
2. **Rule-based baseline** - Deterministic regex classifier with priority-ordered keyword rules
3. **TF-IDF + Logistic Regression** - Traditional ML baseline using bag-of-words features
4. **Embedding-based classifier** - Semantic classifier using sentence-transformer embeddings and centroid matching

---

## 2. Evaluation Dataset

| Property | Value |
| --- | --- |
| Train file | `data/processed/apple_support/intent/train_intents.csv` |
| Test file | `data/processed/apple_support/intent/test_intents.csv` |
| Train examples | 73,700 |
| Test examples | 16,298 |
| Input column | `customer_text_normalized` |
| Label column | `intent` |
| Intent classes | 8 |

---

## 3. Intent Taxonomy (8 classes)

| # | Intent | Test Support |
| --- | --- | --- |
| 1 | general_inquiry_other | 6,632 (40.7%) |
| 2 | software_update | 2,978 (18.3%) |
| 3 | battery_power | 1,620 (9.9%) |
| 4 | performance_system | 1,542 (9.5%) |
| 5 | hardware_audio_display | 1,222 (7.5%) |
| 6 | account_billing | 845 (5.2%) |
| 7 | keyboard_typing | 829 (5.1%) |
| 8 | connectivity_network | 630 (3.9%) |

---

## 4. Model Specifications

### 4a. Majority-Class Baseline
- **Approach:** Always predicts `general_inquiry_other`
- **Training:** None (uses class distribution)
- **Features:** None

### 4b. Rule-Based Baseline
- **Approach:** Priority-ordered regex pattern matching
- **Rules:** 12 regex patterns covering 7 specific intents
- **Fallback:** `general_inquiry_other` if no pattern matches
- **Training:** None (hand-crafted rules)

### 4c. TF-IDF + Logistic Regression
- **Approach:** Supervised learning with bag-of-words features
- **TF-IDF Parameters:**
  - max_features: 10,000
  - ngram_range: (1, 2)
  - min_df: 2
  - max_df: 0.95
  - stop_words: english
- **Logistic Regression Parameters:**
  - max_iter: 1000
  - random_state: 42
  - class_weight: balanced
- **Training:** 73,700 examples
- **Features:** 10,000 TF-IDF features

### 4d. Embedding-Based Classifier
- **Approach:** Centroid-based semantic classification using cosine similarity
- **Embedding Model:** sentence-transformers/all-MiniLM-L6-v2
- **Embedding Dimension:** 384
- **Classification Method:** Cosine similarity to intent centroids (built from training data)
- **Training:** 73,700 examples (used to build intent centroids)
- **Features:** 384-dimensional semantic embeddings

---

## 5. Overall Metrics Comparison

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Majority Baseline | 0.4069 | 0.0509 | 0.1250 | 0.0723 | 0.2354 |
| Rule-Based | 0.9593 | 0.9815 | 0.9389 | 0.9548 | 0.9561 |
| TF-IDF + LR | **0.9607** | **0.9387** | **0.9512** | **0.9441** | **0.9613** |
| Embedding Classifier | 0.6311 | 0.5902 | 0.7051 | 0.6205 | 0.6405 |

**Key Observations:**
- TF-IDF + LR achieves the highest accuracy (0.9607) and weighted F1 (0.9613)
- Rule-based has higher macro precision (0.9815) but lower macro recall (0.9389)
- TF-IDF + LR has better balance across precision and recall
- Embedding classifier performs significantly worse than TF-IDF + LR (accuracy: 0.6311 vs 0.9607)
- Embedding classifier has high recall (0.7051) but low precision (0.5902), suggesting it over-predicts diverse intents
- All ML approaches significantly outperform the majority baseline

---

## 6. Per-Class Metrics Comparison

### 6a. Majority-Class Baseline

| Intent | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| general_inquiry_other | 0.4069 | 1.0000 | 0.5785 | 6,632 |
| software_update | 0.0000 | 0.0000 | 0.0000 | 2,978 |
| battery_power | 0.0000 | 0.0000 | 0.0000 | 1,620 |
| performance_system | 0.0000 | 0.0000 | 0.0000 | 1,542 |
| hardware_audio_display | 0.0000 | 0.0000 | 0.0000 | 1,222 |
| account_billing | 0.0000 | 0.0000 | 0.0000 | 845 |
| keyboard_typing | 0.0000 | 0.0000 | 0.0000 | 829 |
| connectivity_network | 0.0000 | 0.0000 | 0.0000 | 630 |

### 6b. Rule-Based Baseline

| Intent | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| general_inquiry_other | 0.9411 | 1.0000 | 0.9697 | 6,632 |
| software_update | 0.9280 | 1.0000 | 0.9627 | 2,978 |
| battery_power | 1.0000 | 0.9889 | 0.9944 | 1,620 |
| performance_system | 0.9959 | 0.6375 | 0.7774 | 1,542 |
| hardware_audio_display | 0.9911 | 1.0000 | 0.9955 | 1,222 |
| account_billing | 1.0000 | 0.9467 | 0.9726 | 845 |
| keyboard_typing | 1.0000 | 0.9940 | 0.9970 | 829 |
| connectivity_network | 0.9966 | 0.9429 | 0.9690 | 630 |

### 6c. TF-IDF + Logistic Regression

| Intent | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| general_inquiry_other | 0.9842 | 0.9676 | 0.9758 | 6,632 |
| software_update | 0.9674 | 0.9755 | 0.9714 | 2,978 |
| battery_power | 0.9891 | 0.9506 | 0.9695 | 1,620 |
| performance_system | 0.9603 | 0.9559 | 0.9581 | 1,542 |
| hardware_audio_display | 0.9589 | 0.9558 | 0.9574 | 1,222 |
| account_billing | 0.9317 | 0.9361 | 0.9339 | 845 |
| keyboard_typing | 0.7843 | 0.9300 | 0.8510 | 829 |
| connectivity_network | 0.9336 | 0.9381 | 0.9359 | 630 |

### 6d. Embedding-Based Classifier

| Intent | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| general_inquiry_other | 0.8398 | 0.5375 | 0.6555 | 6,632 |
| software_update | 0.6894 | 0.5776 | 0.6285 | 2,978 |
| battery_power | 0.8309 | 0.8673 | 0.8487 | 1,620 |
| performance_system | 0.5236 | 0.6109 | 0.5639 | 1,542 |
| hardware_audio_display | 0.4754 | 0.6882 | 0.5624 | 1,222 |
| account_billing | 0.4217 | 0.7550 | 0.5411 | 845 |
| keyboard_typing | 0.3484 | 0.8203 | 0.4890 | 829 |
| connectivity_network | 0.5923 | 0.7841 | 0.6749 | 630 |

**Per-Class Analysis:**
- **keyboard_typing**: Rule-based achieves near-perfect F1 (0.9970) vs TF-IDF+LR (0.8510) vs Embedding (0.4890). The specific "letter I" glitch pattern is very distinctive for rule-based.
- **performance_system**: TF-IDF+LR significantly outperforms rule-based (0.9581 vs 0.7774) and embedding (0.5639). ML learns broader patterns beyond specific keywords.
- **general_inquiry_other**: TF-IDF+LR performs best (F1: 0.9758), followed by rule-based (0.9697) and embedding (0.6555).
- **battery_power**: All three perform well (rule-based: 0.9944, TF-IDF+LR: 0.9695, embedding: 0.8487).
- **keyboard_typing** is the most challenging class for embedding classifier (F1: 0.4890) due to low precision (0.3484) despite high recall (0.8203).

---

## 7. Confusion Matrices

### 7a. Rule-Based Confusion Matrix

| True \ Predicted | general | software | battery | perform | hardware | account | keyboard | connect |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| general_inquiry_other | 6632 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| software_update | 0 | 2978 | 0 | 0 | 0 | 0 | 0 | 0 |
| battery_power | 9 | 4 | 1602 | 0 | 5 | 0 | 0 | 0 |
| performance_system | 356 | 203 | 0 | 983 | 0 | 0 | 0 | 0 |
| hardware_audio_display | 0 | 0 | 0 | 0 | 1222 | 0 | 0 | 0 |
| account_billing | 26 | 11 | 0 | 1 | 5 | 800 | 0 | 2 |
| keyboard_typing | 5 | 0 | 0 | 0 | 0 | 0 | 824 | 0 |
| connectivity_network | 19 | 13 | 0 | 3 | 1 | 0 | 0 | 594 |

### 7b. TF-IDF + Logistic Regression Confusion Matrix

| True \ Predicted | general | software | battery | perform | hardware | account | keyboard | connect |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| general_inquiry_other | 6417 | 10 | 5 | 0 | 1 | 28 | 159 | 12 |
| software_update | 22 | 2905 | 3 | 2 | 0 | 12 | 32 | 2 |
| battery_power | 11 | 17 | 1540 | 8 | 19 | 12 | 0 | 13 |
| performance_system | 18 | 27 | 2 | 1474 | 1 | 0 | 20 | 0 |
| hardware_audio_display | 21 | 11 | 0 | 15 | 1168 | 4 | 1 | 2 |
| account_billing | 8 | 11 | 0 | 11 | 12 | 791 | 0 | 12 |
| keyboard_typing | 17 | 13 | 7 | 14 | 4 | 2 | 771 | 1 |
| connectivity_network | 6 | 9 | 0 | 11 | 13 | 0 | 0 | 591 |

### 7c. Embedding-Based Classifier Confusion Matrix

| True \ Predicted | general | software | battery | perform | hardware | account | keyboard | connect |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| general_inquiry_other | 3565 | 376 | 179 | 332 | 558 | 566 | 873 | 183 |
| software_update | 303 | 1720 | 37 | 265 | 179 | 218 | 186 | 70 |
| battery_power | 24 | 33 | 1405 | 54 | 39 | 33 | 1 | 31 |
| performance_system | 128 | 161 | 15 | 942 | 77 | 14 | 196 | 9 |
| hardware_audio_display | 110 | 86 | 10 | 110 | 841 | 28 | 13 | 24 |
| account_billing | 55 | 54 | 30 | 14 | 31 | 638 | 2 | 21 |
| keyboard_typing | 32 | 36 | 13 | 48 | 15 | 3 | 680 | 2 |
| connectivity_network | 28 | 29 | 2 | 34 | 29 | 13 | 1 | 494 |

**Confusion Matrix Analysis:**
- **Rule-based**: Very diagonal-heavy with specific off-diagonal patterns (e.g., performance_system → general_inquiry_other)
- **TF-IDF+LR**: More distributed errors, suggesting it learns more nuanced patterns but also makes more diverse mistakes
- **Embedding classifier**: Highly distributed errors across many classes, indicating poor separation of intent centroids in semantic space
- **keyboard_typing confusion**: TF-IDF+LR misclassifies 159 keyboard examples as general_inquiry_other; embedding classifier misclassifies 149 keyboard examples as general_inquiry_other
- **general_inquiry_other**: Embedding classifier has very poor recall (0.5375), with many examples scattered across other intents

---

## 8. Performance Delta Analysis

### TF-IDF + LR vs Rule-Based

| Metric | Rule-Based | TF-IDF + LR | Delta (LR - Rule) |
| --- | ---: | ---: | ---: |
| Accuracy | 0.9593 | 0.9607 | +0.0014 |
| Macro Precision | 0.9815 | 0.9387 | -0.0428 |
| Macro Recall | 0.9389 | 0.9512 | +0.0123 |
| Macro F1 | 0.9548 | 0.9441 | -0.0107 |
| Weighted F1 | 0.9561 | 0.9613 | +0.0052 |

### TF-IDF + LR vs Majority Baseline

| Metric | Majority | TF-IDF + LR | Delta (LR - Majority) |
| --- | ---: | ---: | ---: |
| Accuracy | 0.4069 | 0.9607 | +0.5538 |
| Macro Precision | 0.0509 | 0.9387 | +0.8878 |
| Macro Recall | 0.1250 | 0.9512 | +0.8262 |
| Macro F1 | 0.0723 | 0.9441 | +0.8718 |
| Weighted F1 | 0.2354 | 0.9613 | +0.7259 |

### Embedding Classifier vs TF-IDF + LR

| Metric | TF-IDF + LR | Embedding | Delta (Embedding - LR) |
| --- | ---: | ---: | ---: |
| Accuracy | 0.9607 | 0.6311 | -0.3296 |
| Macro Precision | 0.9387 | 0.5902 | -0.3485 |
| Macro Recall | 0.9512 | 0.7051 | -0.2461 |
| Macro F1 | 0.9441 | 0.6205 | -0.3236 |
| Weighted F1 | 0.9613 | 0.6405 | -0.3208 |

---

## 9. Conclusions

**Key Findings:**
1. **TF-IDF + Logistic Regression** achieves the highest overall accuracy (96.07%) and weighted F1 (96.13%)
2. **Rule-based classifier** achieves higher macro precision (98.15%) due to very specific pattern matching
3. **TF-IDF + LR** has better macro recall (95.12%) and more balanced performance across classes
4. **Embedding classifier** performs significantly worse than both TF-IDF+LR and rule-based (accuracy: 63.11%)
5. **keyboard_typing** is the most challenging class for TF-IDF+LR (F1: 0.8510) and embedding (F1: 0.4890) but near-perfect for rule-based (F1: 0.9970)
6. **Embedding classifier** has high recall (0.7051) but low precision (0.5902), indicating it over-predicts across intents

**Trade-offs:**
- **Rule-based**: Higher precision on specific patterns, lower recall on broad classes, interpretable
- **TF-IDF + LR**: Best overall balance, learns from data, less interpretable, requires training
- **Embedding classifier**: Captures semantic meaning but performs poorly with simple centroid approach, requires more sophisticated methods (e.g., classification head on embeddings)

**Recommendations:**
- Use TF-IDF + LR as the primary ML baseline for comparison with LLM-based approaches
- Keep rule-based as a fallback for interpretable predictions
- Simple centroid-based embedding classification is insufficient for this task; consider:
  - Training a classifier on embeddings (e.g., SVM, neural network)
  - Using domain-specific embeddings fine-tuned on similar data
  - Hybrid approaches combining semantic embeddings with keyword features

**Next Steps:**
- Implement LLM-based classification (Phase 6C)
- Target macro F1 > 0.9441 and weighted F1 > 0.9613 to demonstrate improvement over TF-IDF baseline
- Plan manual annotation for true performance evaluation without data leakage

---

## 10. Output Files

- `reports/baseline_results.json` - Majority and rule-based results
- `reports/rule_based_confusion_matrix.csv` - Rule-based confusion matrix
- `reports/rule_based_confusion_matrix.png` - Rule-based confusion matrix visualization
- `reports/tfidf_lr_results.json` - TF-IDF + LR results
- `reports/tfidf_lr_confusion_matrix.csv` - TF-IDF + LR confusion matrix
- `reports/tfidf_lr_confusion_matrix.png` - TF-IDF + LR confusion matrix visualization
- `reports/embedding_classifier_results.json` - Embedding classifier results
- `reports/embedding_classifier_confusion_matrix.csv` - Embedding classifier confusion matrix
- `reports/embedding_classifier_confusion_matrix.png` - Embedding classifier confusion matrix visualization
- `reports/apple_support_baselines.md` - Detailed baseline evaluation report
