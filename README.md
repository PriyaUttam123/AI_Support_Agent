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

## 🚀 Execution & Implementation Roadmap
1. **Phase 1: Environment & Scaffolding** *(Current)*: Project skeleton, directory structure, and foundational configuration.
2. **Phase 2: Brand Exploration & Intent Schema**: Analyze candidate brands, select target brand, define grounded intent taxonomy.
3. **Phase 3: Data Ingestion & Thread Pairing**: Filter dataset, reconstruct customer-agent conversation turns, build training/retrieval corpora.
4. **Phase 4: Golden Evaluation Benchmark**: Curate and manually label 150–250 representative examples with ground-truth intent, escalation decision, and resolution.
5. **Phase 5: Baselines**: Implement trivial and simple benchmark models.
6. **Phase 6: Core Pipeline**: Implement classification, retrieval-augmented resolution, generation, and escalation routing.
7. **Phase 7: Evaluation Harness & LLM-as-Judge**: Build automated evaluation pipeline, measure human agreement, and run head-to-head benchmarking.
8. **Phase 8: Failure Analysis & Decision Log**: Document failure modes, dissect misleading metrics, and finalize documentation for <15 minute reproduction.
