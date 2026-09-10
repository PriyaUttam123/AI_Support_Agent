# AppleSupport Data Cleaning, Quality Filtering, and Leakage-Safe Splitting

## Executive Summary
This report details the execution and results of **Phase 4**: data quality audit, conservative text normalization, transparent quality filtering, response-type labeling, and conversation-level leakage-safe dataset partitioning for the **AppleSupport** corpus.

All operations were executed deterministically (seed `42`) on the real extracted dataset (`data/processed/apple_support/support_pairs.csv`), producing a cleaned dataset and three completely disjoint splits (`train.csv`, `validation.csv`, `test.csv`) along with an isolated historical retrieval corpus (`retrieval_train.csv`).

---

## A. Input Dataset Statistics

Before cleaning, the Phase 3 extraction pipeline produced:

| Metric | Raw Extracted Value |
| :--- | :--- |
| **Total Extracted Tweets** | `236,738` |
| **Total Conversation Threads** | `80,749` |
| **Raw Support Pairs** $(Q, R)$ | `106,646` |
| **Unique Customer Authors** | `76,365` |
| **Initial Missing Values in Key Fields** | `0` (0.00% across all text and ID columns) |
| **Duplicate (Customer ID, Response ID) Pairs** | `0` |

---

## B. Conservative Text Normalization Rules

To ensure intent classification and retrieval algorithms operate on clean text while retaining critical domain signals, text normalization was applied strictly to separate columns (`customer_text_normalized`, `response_text_normalized`), leaving the original raw text 100% unaltered.

### Normalization Pipeline:
1. **Whitespace Normalization**: Collapsed multiple spaces, tabs, and newline breaks into single spaces (`\s+` $\to$ `" "`), trimming leading/trailing whitespace.
2. **Synthetic User Mention Standardization**: Upstream Twitter anonymization replaced user handles with numeric strings (e.g., `@115854`). These were standardized to `@user` (`@\d+\b` $\to$ `@user`).
3. **Agent Signature Removal (Response Text Only)**: Trailing agent initials appended to responses (e.g., `^HP`, `/LS`) were stripped in normalized text (`\s*[\^/][A-Za-z]{1,3}\s*$` $\to$ `""`).
4. **Preservation of Domain Specifics**:
   - Product names (`iPhone 6s Plus`, `Apple Watch`, `MacBook Pro`) preserved intact.
   - OS versions and firmware updates (`iOS 11.1`, `macOS High Sierra`) preserved.
   - Error messages, punctuation, and technical terms preserved.
   - Punctuation (apostrophes, question marks, quotation marks) preserved.
5. **URL Tokenization**: An auxiliary column (`customer_text_url_norm`, `response_text_url_norm`) replaces raw URLs with `[URL]` tokens for models where URL string variance causes vocabulary bloat.

---

## C. Quality Filtering & Retention Breakdown

Support pairs were audited against explicit quality rules. Rather than silently deleting rows, each row in the raw dataset was evaluated and annotated with `quality_status` (`clean` vs. `flagged`) and `quality_reason`.

| Quality Category / Filter Reason | Count | % of Raw Pairs | Status | Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **Valid Clean Pair** | **105,739** | **99.15%** | **RETAINED** | Contains substantive customer text and valid AppleSupport response. |
| **URL or Mention Only (No Words)** | `707` | `0.66%` | **FLAGGED / REMOVED** | Customer message contained only an attached screenshot URL or an empty handle mention (e.g. `@AppleSupport https://t.co/...`) without text to classify intent. |
| **Ultra-Short Customer Message** | `177` | `0.17%` | **FLAGGED / REMOVED** | Customer text contained less than 3 characters of substantive text (e.g., `"?"`, `"ok"`, `"hi"`). |
| **Duplicate Customer Turn** | `23` | `0.02%` | **FLAGGED / REMOVED** | Customer tweet received multi-part responses; only primary response retained to prevent duplicate training targets. |
| **Total Flagged / Removed** | **907** | **0.85%** | — | Isolated from the cleaned modeling dataset. |
| **Total Clean Dataset** | **105,739** | **100.0%** | **SAVED** | Exported to `data/processed/apple_support/support_pairs_clean.csv`. |

---

## D. Response-Type Labeling Rules

AppleSupport responses were categorized using transparent rule-based logic to establish operational labels for auto-handle and escalation policy benchmarking:

| Response Category | Count (Clean Pairs) | Percentage | Operational Rule Definition |
| :--- | :---: | :---: | :--- |
| **Immediate Escalation** | `49,124` | **46.46%** | `(is_dm or is_genius) and not is_troubleshooting` — direct redirection to DM or Apple Store / Genius Bar. |
| **Pure Troubleshooting** | `18,004` | **17.03%** | `is_troubleshooting and not (is_dm or is_genius)` — provides actionable diagnostic instructions (Settings paths, reboots, toggles). |
| **General Inquiry / Clarification** | `17,156` | **16.22%** | Does not provide direct instructions or DM requests; asks diagnostic questions (e.g., `"Which iOS version is installed?"`). |
| **Documentation / Help Article URL** | `14,919` | **14.11%** | `is_url and not (is_dm or is_genius or is_troubleshooting)` — provides direct link to official Apple support documentation. |
| **Troubleshooting + Escalation** | `6,536` | **6.18%** | `is_troubleshooting and (is_dm or is_genius)` — provides an initial troubleshooting step and asks to DM if it fails. |

---

## E. Conversation-Level Splitting

To prevent intra-conversation data leakage, the dataset was partitioned strictly by `conversation_id`. If multiple turns or support pairs belong to the same conversation tree, they are guaranteed to stay together in the same split.

### Split Target vs. Actual Distribution
* Target: 70% Train / 15% Validation / 15% Test
* Total Clean Conversations: `80,330`

| Split | Conversation Count | Conversation % | Support Pair Count | Support Pair % | Total Tweets Represented |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | `56,231` | **70.00%** | `73,700` | **69.70%** | `164,135` |
| **Validation** | `12,050` | **15.00%** | `15,741` | **14.89%** | `35,055` |
| **Test** *(incl. Golden)* | `12,049` | **15.00%** | `16,298` | **15.41%** | `36,362` |
| **Total** | **80,330** | **100.0%** | **105,739** | **100.0%** | **235,552** |

---

## F. Leakage Checks & Verification

Rigorous programmatic tests were run across all partition boundaries:

1. **Train $\cap$ Validation Conversation Overlap**: `0` (Zero shared conversation IDs)
2. **Train $\cap$ Test Conversation Overlap**: `0` (Zero shared conversation IDs)
3. **Validation $\cap$ Test Conversation Overlap**: `0` (Zero shared conversation IDs)
4. **Support Pair Disjointness**: Every individual support pair belongs exclusively to exactly one split.

---

## G. Retrieval Corpus Construction

To support retrieval-augmented generation (RAG) and historical exemplar search without evaluation leakage, a dedicated retrieval corpus was constructed:

* **Corpus File**: `data/processed/apple_support/retrieval/retrieval_train.csv`
* **Corpus Size**: `73,700` clean support pairs
* **Conversation Trees**: `56,231` training conversations
* **Leakage Safeguard**:
  - `validation.csv` conversations in retrieval: `0`
  - `test.csv` conversations in retrieval: `0`
  - All test queries retrieve purely from historical training exemplars.

---

## H. Golden-Set Protection

The 250-example manual inspection sample created in Phase 3 (`data/evaluation/apple_support_manual_sample.csv`) was cross-referenced during splitting:

* **Sample Conversations**: 250 distinct conversation threads.
* **Cleaning Impact**: 249 sample conversations had valid substantive text; 1 sample conversation had URL/mention-only text and was flagged by quality filtering.
* **Protection Mechanism**: All 249 valid sample conversations were forcibly assigned to the `test` split (`test.csv`).
* **Verification Result**:
  - Sample conversations in `train.csv`: `0`
  - Sample conversations in `retrieval_train.csv`: `0`
  - Sample conversations in `validation.csv`: `0`
  - 100% of golden evaluation instances are isolated from training and retrieval.

---

## I. Temporal Analysis & Split Trade-Offs

### Temporal Distribution
* **Earliest Customer Timestamp**: `2016-03-04 01:19:41 UTC`
* **Latest Customer Timestamp**: `2017-12-03 23:03:13 UTC`
* **Yearly Breakdown**:
  * `2016`: 7 tweets
  * `2017`: 105,732 tweets
* **Monthly Breakdown (2017 Concentration)**:
  * January – September: `170` tweets
  * October: `42,660` tweets (40.35%)
  * November: `58,254` tweets (55.10%)
  * December (first 3 days): `4,648` tweets (4.40%)

### Chronological Cutoffs (70% / 15% / 15%)
* **70% Cutoff Date**: `2017-11-12 06:24:57 UTC`
* **85% Cutoff Date**: `2017-11-22 01:26:37 UTC`

### Architectural Trade-Off: Random Conversation Split vs. Temporal Split
| Dimension | Conversation-Level Random Split (Primary Choice) | Temporal / Chronological Split |
| :--- | :--- | :--- |
| **Intent Distribution** | Balances seasonal issues (e.g. iOS 11 launch bugs) proportionally across train, val, and test. | Severely skewed: early split misses later bugs; test split receives new bugs absent from train. |
| **Version Drift Risk** | Low: train and test share similar OS update environments (iOS 11.0 through 11.1). | High: test period (late Nov/Dec 2017) introduced iOS 11.1.2 patch notes not present in train. |
| **Leakage Prevention** | Strictly preserves complete conversation trees within each split. | Can fragment long threads spanning cutoff timestamps. |
| **Primary Utility** | Optimal for benchmarking intent classification and grounded retrieval capabilities. | Useful as a secondary stress test for temporal robustness. |

---

## J. Limitations and Remaining Risks

1. **Extreme Temporal Concentration**: Over 95% of the dataset is concentrated in October and November 2017, coinciding with the rocky release of iOS 11.0/11.1 and iPhone 8/X hardware launches. Intent definitions must accommodate this release context.
2. **Short Character Limits**: Many customer tweets are under 140 characters, resulting in abbreviated vocabulary that benefits significantly from our conservative normalization pipeline.
3. **Retrieval Sparsity for Rare Edge Cases**: While 73,700 training pairs provide massive coverage for common intents (battery, updates, screen freeze), niche hardware inquiries (e.g., Apple TV remote pairing) may have fewer historical exemplars in the retrieval corpus.
