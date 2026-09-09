# Dataset Profile: Customer Support on Twitter (`thoughtvector/customer-support-on-twitter`)

## 1. Executive Summary & Raw File Inventory

This report provides an empirical profile of the raw Twitter Customer Support dataset placed in `data/raw/`. All metrics and statistics were derived via chunked streaming over the complete 2.8+ million rows without sampling or synthetic generation.

### Raw Files Inventory
| File Path | Format | Size | Row Count | Description |
| :--- | :--- | :--- | :--- | :--- |
| `data/raw/sample.csv` | CSV | 17,357 bytes (~17.0 KB) | 93 data rows | 100-line excerpt provided by upstream repository for schema inspection. |
| `data/raw/twcs/twcs.csv` | CSV | 516,508,641 bytes (~492.6 MB) | 2,811,774 data rows | Full, immutable Twitter Customer Support dataset. |
| `data/raw/.gitkeep` | Plaintext | 88 bytes | N/A | Directory persistence marker for Git. |

---

## 2. Dataset-Wide Core Metrics

| Metric | Measured Value | Percentage / Notes |
| :--- | :--- | :--- |
| **Total Rows (Tweets)** | `2,811,774` | 100.0% |
| **Unique Tweet IDs** | `2,811,774` | 100.0% (Zero duplicate tweet IDs found) |
| **Duplicate Tweet IDs** | `0` | Clean primary key integrity |
| **Inbound Tweets (Customer)** | `1,537,843` | 54.69% of all tweets |
| **Outbound Tweets (Company Support)** | `1,273,931` | 45.31% of all tweets |
| **Total Unique Authors** | `702,777` | Combined user accounts |
| **Unique Customer Authors** | `702,669` | Anonymized numeric IDs (e.g., `105834`) |
| **Unique Brand Support Accounts** | `108` | Distinct verified corporate handles |
| **Earliest Timestamp** | `2008-05-08 20:13:59 UTC` | Earliest recorded tweet |
| **Latest Timestamp** | `2017-12-03 23:14:01 UTC` | Latest recorded tweet |
| **Overall Date Range Span** | `~9.57 years` | Vast majority concentrated between 2016 and late 2017 |

---

## 3. Schema, Column Definitions & Missing Value Analysis

Across all **2,811,774** rows, missing value analysis revealed zero missing values in primary metadata and text fields, with missingness observed only in structural graph pointers:

| Column Name | Inferred Data Type | Missing Count | Missing (%) | Operational Meaning |
| :--- | :--- | :--- | :--- | :--- |
| `tweet_id` | `int64` | `0` | `0.00%` | Globally unique identifier for every tweet in the dataset. |
| `author_id` | `object` (string) | `0` | `0.00%` | Identifier of the tweet author. Masked as an integer string for customers (e.g., `105834`); explicit text handle for brands (e.g., `AppleSupport`). |
| `inbound` | `bool` | `0` | `0.00%` | Direction of communication: `True` indicates incoming customer message; `False` indicates outbound brand response. |
| `created_at` | `object` (datetime) | `0` | `0.00%` | Publication timestamp in RFC 2822 format (`%a %b %d %H:%M:%S +0000 %Y`). |
| `text` | `object` (string) | `0` | `0.00%` | Raw content of the tweet (140 to 280 characters max). Contains mentions, URLs, hashtags, and emojis. |
| `response_tweet_id` | `object` (string) | `1,040,629` | `37.01%` | Comma-separated list of child tweet IDs that directly responded to this tweet. `NaN` indicates a terminal leaf node (no reply recorded). |
| `in_response_to_tweet_id` | `float64` / `int` | `794,335` | `28.25%` | ID of the parent tweet to which this tweet responds. `NaN` indicates the initiation of a new conversation thread. |

---

## 4. Conversation Structure & Reconstruction Logic

The dataset models customer-brand interactions as **directed conversation trees / DAGs** through two complementary pointer fields:

```
[Customer Initial Tweet]  ── (tweet_id: 119241, in_response_to: null, response_tweet_id: 119240)
           │
           ▼
[Brand Initial Response]  ── (tweet_id: 119240, in_response_to: 119241, response_tweet_id: 119242)
           │
           ▼
[Customer Follow-up]      ── (tweet_id: 119242, in_response_to: 119240, response_tweet_id: 119243)
           │
           ▼
[Brand Final Resolution]  ── (tweet_id: 119243, in_response_to: 119242, response_tweet_id: null)
```

### Key Structural Mechanics:
1. **Thread Starters**: An inbound customer tweet with `in_response_to_tweet_id == NaN` represents a new customer issue arriving at the support queue. There are **794,335** thread-starter tweets in total.
2. **Forward Chaining**: `response_tweet_id` points to the subsequent turn. When multiple IDs are present (e.g. `"119249,119251"`), it denotes branching (e.g., brand sent a two-part tweet or multiple agents/users replied).
3. **Backward Chaining**: Given any outbound brand resolution, following `in_response_to_tweet_id` backwards guarantees retrieval of the immediate customer query and prior context turns.
4. **Resolution Pairing**: For training and retrieval evaluation, a clean support pair $(Q, R)$ consists of an inbound customer question $Q$ directly responded to by outbound brand reply $R$.

---

## 5. Identification of Brands & Corporate Accounts

All corporate support accounts were reliably extracted by filtering for `inbound == False`. This produced **108 distinct brands**. Inbound tweets were mapped to these brands through:
1. **Response Linking**: Matching `tweet_id` of customer tweets against `in_response_to_tweet_id` of outbound brand tweets (**1,169,251** direct links).
2. **Handle Mention Fallback**: Parsing customer tweets that mention `@BrandHandle` (**307,805** additional links).
3. **Unmatched Inbound**: Only 60,787 inbound tweets (3.95%) could not be linked to one of the 108 brands (primarily orphaned customer replies to deleted tweets or other users).

---

## 6. Objective Candidate Brand Ranking & Comparative Statistics

To select the most suitable brand for building an intent classification, grounded retrieval, and escalation decision system, we examined the top candidate brands across volume, resolution actionability, escalation balance, and linguistic cleanliness:

| Rank | Brand Account | Industry Domain | Total Tweets | Outbound (Brand) | Inbound (Customer) | Initiating Threads | Unique Users | % Outbound DM | % Outbound URL | % English / ASCII | Date Range |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **AppleSupport** | Tech Hardware / OS / Services | 232,421 | 106,860 | 125,561 | 76,557 | 78,330 | **52.47%** | 75.35% | **100.00%** | 2016-03 to 2017-12 |
| **2** | **SpotifyCares** | Digital Streaming / SaaS | 90,014 | 43,265 | 46,749 | 26,828 | 28,140 | **30.79%** | 50.49% | **99.99%** | 2013-09 to 2017-12 |
| **3** | **AmazonHelp** | E-Commerce / Cloud / Devices | 366,780 | 169,840 | 196,940 | 77,497 | 72,348 | **0.64%** | 41.30% | **94.08%** *(10k+ non-En)* | 2015-06 to 2017-12 |
| **4** | **Uber_Support** | Ride-hailing / Gig Platform | 125,448 | 56,270 | 69,178 | 40,161 | 39,423 | **35.16%** | 51.28% | **100.00%** | 2014-12 to 2017-12 |
| **5** | **Delta** | Airline / Travel | 86,352 | 42,253 | 44,099 | 25,591 | 23,242 | **16.45%** | 15.34% | **99.92%** | 2012-11 to 2017-12 |
| **6** | **AmericanAir** | Airline / Travel | 85,525 | 36,764 | 48,761 | 25,528 | 22,954 | **16.79%** | 6.47% | **100.00%** | 2011-06 to 2017-12 |
| **7** | **VirginTrains** | Rail Transportation | 65,002 | 27,817 | 37,185 | 14,180 | 13,350 | **2.59%** | 13.20% | **99.96%** | 2012-03 to 2017-12 |
| **8** | **hulu_support** | Streaming / Media | 47,341 | 21,872 | 25,469 | 14,399 | 13,654 | **0.53%** | 54.84% | **99.99%** | 2014-08 to 2017-12 |
| **9** | **Tesco** | Retail / Supermarket | 71,945 | 38,573 | 33,372 | 16,378 | 16,720 | **26.81%** | 8.27% | **99.99%** | 2014-09 to 2017-12 |
| **10** | **Ask_Spectrum** | Telecom / ISP | 56,956 | 25,860 | 31,096 | 17,541 | 17,972 | **49.48%** | 39.40% | **99.98%** | 2015-04 to 2017-12 |
| **11** | **XboxSupport** | Gaming / Hardware / Live | 54,068 | 24,557 | 29,511 | 12,577 | 14,106 | **20.91%** | 40.35% | **99.97%** | 2012-06 to 2017-12 |
| **12** | **TMobileHelp** | Telecom / Mobile Carrier | 74,275 | 34,317 | 39,958 | 20,047 | 20,248 | **81.82%** | 48.20% | **99.95%** | 2013-05 to 2017-12 |
| **13** | **comcastcares** | Telecom / Cable | 69,102 | 33,031 | 36,071 | 22,394 | 22,541 | **71.46%** | 4.00% | **99.97%** | 2014-07 to 2017-12 |
| **14** | **UPSHelp** | Logistics / Package Delivery | 40,682 | 17,817 | 22,865 | 14,410 | 14,916 | **68.29%** | 85.06% | **99.98%** | 2012-12 to 2017-12 |

---

## 7. Qualitative Suitability Assessment of Top Contenders

### Contender A: `AppleSupport` (Strong Candidate)
* **Strengths**:
  * Massive, homogeneous dataset (232k tweets, 100.0% English/ASCII).
  * High diagnostic diversity: issues cleanly partition into well-defined intents (e.g., Battery & Performance, iOS Update Issues, Hardware/Display, Apple ID & iCloud Authentication, Audio/Microphone).
  * Rich technical content: responses include precise navigational steps (e.g. `Settings > General > About`), reboot combinations, and official support documentation links (75.35% URL rate).
  * **Balanced Escalation Boundary**: ~52.47% of responses invite the user to DM or visit a Genius Bar, while ~47.5% provide direct troubleshooting steps. This provides a natural ground-truth label for the Auto-Handle vs. Escalate policy.
* **Trade-offs**:
  * Highly structured format; tweets frequently include automated signature URLs (`https://t.co/GDrqU22YpT`).

### Contender B: `SpotifyCares` (Strong Candidate)
* **Strengths**:
  * Excellent digital software support domain (90k tweets, 99.99% English/ASCII).
  * Crisp technical troubleshooting: responses provide clear, self-contained procedures (e.g., offline cache clearing, clean reinstall, checking Bluetooth/crossfade settings, account tier validation).
  * **Realistic Escalation Boundary**: 30.79% DM deflection rate. Routine playback/app bugs are auto-handled publicly; billing disputes and account compromise are escalated to DMs.
  * Very high ratio of conversational dialogue and follow-up troubleshooting.
* **Trade-offs**:
  * Smaller overall volume than AppleSupport or AmazonHelp, though 90,000+ tweets is more than ample for retrieval and evaluation.

### Contender C: `AmazonHelp` (Mixed Feasibility)
* **Strengths**:
  * Highest raw volume in the dataset (366k tweets, 77k initiating threads).
  * Extremely low DM deflection rate (0.64%): Amazon support answers almost everything in public tweets or points directly to help portal URLs.
  * Diverse retail intents (Delivery status, Missing packages, Returns/Refunds, Prime Video, Kindle).
* **Trade-offs / Risks**:
  * **Multilingual Noise**: Over 10,000 tweets are in Japanese, German, French, Spanish, or Hindi (e.g., `@115770 こんにちは、アマゾン公式です...`). Filtering non-English tweets would be an extra preprocessing requirement.
  * Very low DM rate means historical escalation signals are weak in the tweet text, making ground-truth escalation labelling less naturally derived from tweet keywords.

### Telecom Brands (`TMobileHelp`, `comcastcares`, `sprintcare`) (Not Recommended)
* **Why Unsuitable**:
  * Extremely high DM deflection (>70–82%). Telecom support on Twitter consists almost entirely of: *"Please send us a DM with your account number, PIN, and billing address."*
  * Because the actual troubleshooting occurred off-platform, the dataset lacks grounded resolution steps for retrieval.

---

## 8. Data-Quality Problems Discovered

1. **Terminal "DM Black Box" Problem**:
   * For many complex inquiries, the final public tweet is a canned request: *"Please DM us your email address so we can take a closer look."*
   * **Impact**: The actual technical resolution was reached in private DMs and is not present in the public corpus. This requires an agent to recognize when an issue cannot be resolved publicly and must be escalated.
2. **Encoding & Multilingual Quirks**:
   * Emojis, smart quotes, Japanese script, and special punctuation cause `UnicodeEncodeError` on Windows systems using default `cp1252` encoding.
   * **Resolution**: All file reading, text processing, and evaluation pipelines must enforce `utf-8` encoding explicitly.
3. **Synthetic User Masking**:
   * Upstream curators replaced customer Twitter handles with numeric IDs (e.g., `@105834`). These appear frequently throughout the message text and must be cleaned or normalized during preprocessing.
4. **Agent Signatures & Boilerplate**:
   * Human agents append agent codes (e.g., `^RR`, `^MM`, `/LS`, `/CH`) to their tweets. These are stylistic artifacts that models could improperly memorize rather than focusing on the support semantics.
5. **Short-Form Ambiguity**:
   * Tweets capped at 140 characters often omit critical technical context (device model, error code, OS version), leading to ambiguous intent classification without clarifying questions.

---

## 9. Evaluation Risks to Guard Against

1. **Misleading Auto-Handle Accuracy**:
   * If a brand has an extreme escalation skew (e.g., 82% DM or 0.5% DM), a trivial model predicting a single class can achieve artificially high accuracy while failing catastrophically in real-world deployment.
2. **Data Leakage in Retrieval Evaluation**:
   * If the evaluation test set contains customer queries whose historical brand responses are present in the retrieval knowledge base, the retrieval system could simply memorize the exact match rather than retrieving generalizable resolutions. Strict train/test splits by conversation thread or time window are required.
3. **Temporal Policy Drift**:
   * The dataset spans 2008–2017. Policies, links, and software versions (e.g., iOS 10/11, old Amazon return policies) from 2016 are technically obsolete today. The evaluation harness must measure groundedness against *historical brand behavior*, not modern 2026 external reality.
