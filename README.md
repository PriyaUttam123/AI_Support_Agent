# AI Customer Support Agent (Hiver SDE Intern Assignment)

## 📌 Project Objective
Build a production-minded AI customer-support agent based on the Customer Support on Twitter dataset. The agent classifies inbound customer messages into brand-specific intents, retrieves historically grounded resolution patterns from the selected brand, synthesizes context-aware replies, and autonomously decides whether an inquiry can be auto-handled or must be escalated to a human agent with justification.

> [!NOTE]
> **Incremental Implementation Notice**: This project is being developed incrementally step-by-step. Models, data ingestion, baseline evaluations, and reporting pipelines are intentionally structured as stubs and will be implemented iteratively with thorough review at each phase.

---

## 📋 Assignment Requirements
1. **Brand Selection**: Select one brand from the Customer Support on Twitter dataset.
2. **Intent Schema Definition**: Define a focused, meaningful set of customer-support intents tailored to that brand's operational domain.
3. **Intent Classification**: Classify incoming customer messages into the defined intent categories.
4. **Grounded Reply Generation**: Draft replies strictly grounded in how the selected brand historically resolved similar customer queries.
5. **Auto-Handle vs. Escalation Routing**: Implement explicit decision logic to determine whether an inquiry is auto-handled or routed to human agents, providing a structured reason.
6. **Golden Evaluation Dataset**: Construct a manually labeled 150–250 example golden evaluation benchmark.
7. **Automated Evaluation Harness**: Build an automated harness assessing classification, retrieval quality, escalation accuracy, and reply quality.
8. **LLM-as-Judge & Human Alignment**: Employ an LLM-as-judge setup for reply quality and quantify its correlation and agreement with human judgment.
9. **Baseline Comparison**: Benchmark system performance against at least two baselines:
   - One trivial baseline (e.g., majority class / rule-based / fixed response)
   - One simple baseline (e.g., TF-IDF + linear classifier / BM25)
10. **Failure Analysis**: Document the top 5 failure modes with real-world examples and root-cause breakdowns.
11. **Metric Critique**: Provide an in-depth analysis exposing what is misleading about headline metrics (e.g., raw accuracy or high auto-handle rates).
12. **Engineering Decision Log**: Maintain a comprehensive log of 10–15 non-obvious engineering decisions and trade-offs.
13. **Reproducibility**: Ensure headline evaluation results can be reproduced end-to-end in under 15 minutes.

---

## 🏗️ Planned Architecture

```
                                 [ Inbound Customer Tweet / Message ]
                                                  │
                                                  ▼
                                      ┌───────────────────────┐
                                      │ Preprocessing Engine  │
                                      │  - Noise cleaning     │
                                      │  - Thread resolution  │
                                      └───────────┬───────────┘
                                                  │
                                                  ▼
                                      ┌───────────────────────┐
                                      │ Intent Classifier     │
                                      │  - Classify intent    │
                                      │  - Confidence scoring │
                                      └───────────┬───────────┘
                                                  │
                                                  ▼
                        ┌──────────────────────────────────────────────────┐
                        │            Retrieval & Policy Engine             │
                        │  - Retrieve historical resolution exemplars      │
                        │  - Evaluate escalation criteria & risk triggers  │
                        └───────────┬──────────────────────────┬───────────┘
                                    │                          │
                 ┌──────────────────┴───────────────┐          │
                 │ Auto-Handle Approved             │ Escalation Required
                 ▼                                  ▼
   ┌───────────────────────────┐      ┌───────────────────────────────┐
   │ Grounded Reply Generation │      │ Human Escalation Packet       │
   │  - Synthesize resolution  │      │  - Route to human inbox       │
   │  - Strict brand grounding │      │  - Explicit reason & context  │
   └─────────────┬─────────────┘      └───────────────┬───────────────┘
                 │                                    │
                 └──────────────────┬─────────────────┘
                                    ▼
                      [ Support Action / Response ]
                                    │
       ┌────────────────────────────┴────────────────────────────┐
       ▼                                                         ▼
[ Automated Eval Harness ]                            [ LLM-as-Judge Suite ]
 - Intent Classification (Precision/Recall/F1)        - Reply groundedness & helpfulness
 - Routing Accuracy & Escalation Safety               - Human-judge agreement metrics
```

---

## 📂 Planned Project Structure

```
AI_Support_Agent/
├── .gitignore               # Ignored environments, raw datasets, caches, and secrets
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation and architectural overview
├── data/
│   ├── raw/                 # Original, immutable Twitter customer support dataset
│   ├── processed/           # Cleaned brand-filtered subsets and thread pairings
│   └── evaluation/          # 150-250 sample curated golden test set
├── notebooks/               # Exploratory data analysis, prototyping, and analysis
├── src/
│   ├── __init__.py
│   ├── data/                # Ingestion, brand extraction, and dataset loaders
│   ├── preprocessing/       # Cleaning, text normalization, and conversation threading
│   ├── classification/      # Intent classification models and pipelines
│   ├── retrieval/           # Historical resolution search and exemplar retrieval
│   ├── generation/          # Grounded reply prompting and synthesis
│   ├── escalation/          # Confidence gates, escalation policies, and decision logic
│   └── evaluation/          # Metrics, evaluation harness, and LLM-as-judge logic
├── baselines/               # Trivial and simple baselines for benchmarking
├── configs/                 # Model configs, prompts, thresholds, and brand parameters
├── tests/                   # Unit and integration test suites
├── reports/                 # Evaluation reports, failure analyses, and decision logs
└── app/                     # Interactive demonstration interface / API entry point
```

---

## 🎯 Target Brand Selection: AppleSupport

### Why AppleSupport Was Selected
Following extensive chunked profiling across all 108 brands in the Twitter Customer Support dataset (`reports/dataset_profile.md`), **AppleSupport** was selected as the optimal target brand for the following reasons:
1. **Balanced Escalation Boundary**: Unlike telecom carriers with >80% DM deflection (deflecting before troubleshooting) or platforms with <1% DM deflection, AppleSupport displays a balanced split: ~46.5% direct escalation (account/hardware/DM) vs. ~53.5% public diagnostic assistance/troubleshooting. This provides an ideal ground-truth distribution for training and evaluating auto-handle vs. escalation policies.
2. **Actionable Diagnostic Content**: Responses frequently include concrete diagnostic guidance (Settings navigation paths, restart sequences, update verification) and official support links (75.4% URL presence).
3. **High Volume & Clean English Text**: 236,738 extracted tweets across 80,749 conversation threads, with 100.0% English/ASCII language consistency.
4. **Distinct Natural Technical Intents**: Issues naturally cluster into well-defined domains (Battery & Power, iOS Updates, App Crashes, Apple ID / Security, Hardware / Display, Connectivity).

### How to Reproduce Extraction
The AppleSupport conversation extraction pipeline is fully automated and deterministic:
```powershell
python src/data/extract_apple_support.py
```
This script:
1. Filters outbound tweets from `author_id == "AppleSupport"` and traverses the conversation graph in `data/raw/twcs/twcs.csv` to capture all parent queries and follow-up turns.
2. Reconstructs full conversation threads and exports `data/processed/apple_support/tweets.csv` and `data/processed/apple_support/conversations.jsonl`.
3. Extracts clean customer $\to$ AppleSupport support pairs into `data/processed/apple_support/support_pairs.csv` with rich metadata (response delay, DM flags, URL presence, actionable keywords).
4. Generates a reproducible 250-pair manual inspection sample at `data/evaluation/apple_support_manual_sample.csv` (random seed `42`).

### Run Data Integrity Tests
```powershell
python -m unittest tests/test_apple_support_data.py
```

---

## 🧹 Data Cleaning, Quality Filtering & Leakage-Safe Splitting (Phase 4)

### Data Cleaning & Normalization
* **Conservative Normalization**: Original customer and response texts are strictly preserved intact in `customer_text` and `response_text`. Normalized columns standardize whitespace (`\s+`), convert synthetic numeric handles (`@123456` $\to$ `@user`), and strip agent initials (`/LS`, `^HP`), while preserving technical terms, product models (`iPhone 6s`), and iOS versions.
* **Transparent Quality Filtering**: Evaluated raw support pairs and flagged 907 unusable rows (707 empty/URL-only customer messages, 177 ultra-short messages <3 characters, 23 duplicate turns), yielding **105,739 clean support pairs** in `data/processed/apple_support/support_pairs_clean.csv`.

### Conversation-Level Splitting
* **Why Conversation-Level Splitting is Critical**: Splitting randomly by individual support pairs causes severe data leakage when multiple turns of the same customer thread appear in both training and test sets. Splitting strictly by `conversation_id` guarantees that complete conversational trees remain isolated.
* **Splits Created**:
  * **Train**: 56,231 conversations (70.0%) | 73,700 support pairs | 164,135 tweets
  * **Validation**: 12,050 conversations (15.0%) | 15,741 support pairs | 35,055 tweets
  * **Test**: 12,049 conversations (15.0%) | 16,298 support pairs | 36,362 tweets

### Retrieval Isolation & Golden-Set Protection
* **Retrieval Knowledge Base**: `data/processed/apple_support/retrieval/retrieval_train.csv` (73,700 pairs) is constructed strictly from `train.csv`. Validation and test conversations are 100% excluded to prevent retrieval memorization.
* **Golden-Set Protection**: All conversations present in the 250-example manual evaluation sample (`data/evaluation/apple_support_manual_sample.csv`) are forcibly assigned to the test split, guaranteeing zero presence in training or retrieval.

### How to Reproduce Cleaning & Splitting
```powershell
python src/preprocessing/clean_and_split.py
```

### Run Leakage and Split Validation Tests
```powershell
python -m unittest tests/test_data_cleaning_and_splits.py
```


