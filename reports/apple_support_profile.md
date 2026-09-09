# AppleSupport Brand Profile & Conversation Analysis

## Executive Summary
This report analyzes the extracted, cleaned, and structured **AppleSupport** dataset derived from `data/raw/twcs/twcs.csv`. Extraction was performed using strict conversation-graph closure (ancestor traversal and follow-up chaining). All statistics, distributions, and n-grams are computed directly from the processed dataset without synthetic data, approximations, or model inferences.

---

## A. Dataset Size & Structural Inventory

All extracted records are preserved with zero information loss in `data/processed/apple_support/`:

| File / Component | Format | Size on Disk | Row Count | Operational Description |
| :--- | :--- | :--- | :--- | :--- |
| **`data/processed/apple_support/tweets.csv`** | CSV | 47.5 MB | **236,738** | All tweets belonging to AppleSupport conversation trees, including thread roots and follow-ups. |
| **`data/processed/apple_support/conversations.jsonl`** | JSONL | 80.2 MB | **80,749** | Fully reconstructed, chronologically ordered conversation threads keyed by `conversation_id`. |
| **`data/processed/apple_support/support_pairs.csv`** | CSV | 67.1 MB | **106,646** | Canonical $(Q, R)$ pairs: incoming customer question directly answered by AppleSupport. |
| **`data/evaluation/apple_support_manual_sample.csv`** | CSV | 160 KB | **250** | Reproducible random sample (fixed seed `42`) reserved for manual inspection and labeling. |

### Tweet Direction & Author Composition
* **Total AppleSupport-Related Tweets**: `236,738`
  * **Outbound AppleSupport Tweets**: `106,860` (45.14%)
  * **Inbound Customer Tweets**: `129,722` (54.79%)
  * **Third-Party / Colleague Tweets**: `156` (0.07%)
* **Unique Customer Users**: `79,392`

---

## B. Conversation Thread Statistics

A conversation is defined as a connected tree of customer and support tweets rooted at the earliest initiating query (`conversation_id`).

* **Total Conversation Threads**: `80,749`
* **Mean Conversation Length**: `2.93` turns
* **Median Conversation Length**: `2.00` turns
* **Standard Deviation**: `2.57` turns
* **Min / Max Turns**: `2` to `282` turns

### Conversation Length Distribution
| Turns in Thread | Conversation Count | Percentage | Operational Meaning |
| :---: | :---: | :---: | :--- |
| **2 turns** | `52,573` | **65.11%** | Standard single-exchange resolution: Customer Question $\to$ Apple Support Reply. |
| **3 turns** | `8,036` | **9.95%** | Customer question $\to$ Apple clarifying question $\to$ Customer clarification. |
| **4 turns** | `10,998` | **13.62%** | Two full dialogue rounds: Question $\to$ Diagnosis $\to$ Feedback $\to$ Escalation/Resolution. |
| **5+ turns** | `9,142` | **11.32%** | Extended diagnostic sessions or persistent technical troubleshooting dialogues. |

---

## C. Support-Pair Statistics

A **support pair** is defined as an inbound customer tweet directly responded to by an outbound AppleSupport tweet ($R.\text{in\_response\_to\_tweet\_id} == Q.\text{tweet\_id}$).

* **Total Clean Support Pairs**: `106,646`
* **Unique Customers in Support Pairs**: `76,365`
* **Median Response Delay**: `71.0 minutes` (~1.18 hours)
* **Mean Response Delay**: `147.3 minutes` (~2.45 hours)

Both original raw customer/response text and non-destructive normalized versions are preserved in `support_pairs.csv`.

---

## D. Response-Type & Escalation Statistics

AppleSupport responses fall into five distinct structural categories based on whether they offer self-serve diagnostic instructions, direct URLs, or escalate to private Direct Messages (DMs) / Genius Bar visits:

| Response Category | Count | Percentage | Definition & Operational Impact |
| :--- | :---: | :---: | :--- |
| **Immediate Escalation (DM / Store)** | `49,581` | **46.49%** | Response immediately redirects the customer to a private DM or Apple Store Genius Bar without public troubleshooting. |
| **Pure Troubleshooting (Self-Serve)** | `18,161` | **17.03%** | Actionable, in-tweet instructions (e.g. restart combinations, Settings menu paths, toggles). |
| **General Inquiry / Clarification** | `17,294` | **16.22%** | Apple agent requests diagnostic details (iOS version, device model, symptoms). |
| **Documentation / Help Article URL** | `15,034` | **14.10%** | Provides a direct link to an official Apple Support knowledge-base article. |
| **Troubleshooting + Escalation** | `6,576` | **6.17%** | Provides initial diagnostic steps and conditionally asks to DM if the issue persists. |

### Feature Breakdown Across All 106,646 Responses:
* **Responses containing URLs**: `80,383` (**75.37%**)
* **Responses requesting DM**: `56,046` (**52.55%**)
* **Responses with actionable troubleshooting keywords**: `24,737` (**23.20%**)
* **Responses explicitly mentioning Genius Bar / Apple Store**: `221` (**0.21%**)

---

## E. Common Issue Patterns (Keyword & N-Gram Analysis)

Without using an LLM to hallucinate categories, analyzing frequent unigrams, bigrams, and trigrams from customer queries reveals the natural distribution of technical friction points:

### Top Customer Unigrams
| Rank | Term | Count | Context / Domain |
| :---: | :--- | :---: | :--- |
| 1 | `phone` / `iphone` | 42,424 | Hardware / Device identifier |
| 2 | `ios` / `update` / `updated` | 36,885 | Software updates (prominently iOS 11 rollouts) |
| 3 | `fix` | 14,517 | Frustration / Action request |
| 4 | `battery` | 8,529 | Rapid drain / Power / Health degradation |
| 5 | `app` / `apps` | 9,340 | App Store / Crashing / Freezing third-party apps |
| 6 | `screen` | 4,508 | Unresponsive touch, black screen, cracked display |

### Top Customer Bigrams & Trigrams
* **Software / Update issues**: `ios update` (2,211), `updated phone` (1,141), `latest ios update` (261), `since ios update` (240), `high sierra` (879)
* **Battery degradation**: `battery life` (1,746), `battery drain` (542)
* **Display / Performance glitch**: `phone keeps freezing` (220), `doesn work` (1,054), `question mark box` (408 - referring to the famous iOS 11 'A [?]' autocorrect bug)
* **Hardware models**: `iphone plus` (1,574), `iphone 6` (742), `iphone 7` (681)

---

## F. Representative Real Customer $\to$ AppleSupport Pairs

### 1. Pure Troubleshooting (Self-Serve - Battery/Performance)
* **Customer Tweet ID**: `1761`
  > *"iOS 11 is killing my battery @AppleSupport. Fix it."*
* **AppleSupport Reply ID**: `1760`
  > *"@user We're here to help. Which exact iOS version is your device running? This info can be found under Settings > General > About."*

### 2. Immediate Escalation (DM Request - Account/Authentication)
* **Customer Tweet ID**: `1768`
  > *"@AppleSupport it has, but not for some time now. i've tried with two different apple ids but it's still not showing up on my phone"*
* **AppleSupport Reply ID**: `1767`
  > *"@user Thanks for clarifying. Let's meet in DM so we can do more in-depth on this. https://t.co/GDrqU22YpT"*

### 3. Troubleshooting + Conditional Escalation
* **Customer Tweet ID**: `735`
  > *"@AppleSupport I have the iPhone 6s Plus and just did the most recent update."*
* **AppleSupport Reply ID**: `737`
  > *"@user To make sure, is iOS 11.1 installed on it currently? Also, any steps tried so far? DM us here: https://t.co/GDrqU22YpT"*

### 4. General Diagnostic Inquiry
* **Customer Tweet ID**: `698`
  > *"@AppleSupport [attached image of setting failure]"*
* **AppleSupport Reply ID**: `696`
  > *"@user We're here for you. Which version of the iOS are you running? Check from Settings > General > About."*

---

## G. Data-Quality Problems Discovered

1. **The iOS 11 Autocorrect Anomaly**:
   * During late 2017, a well-known iOS bug caused the letter "i" to autocorrect to an "A" followed by a question mark in a box (`question mark box`). This generated thousands of identical queries that could skew intent distributions if not recognized.
2. **Standardized Link Cloaking**:
   * Over 75% of AppleSupport responses include standard shortlinks (`https://t.co/GDrqU22YpT`), many of which redirect to a private Twitter DM prompt. Identifying DM intent cannot rely solely on the string `"DM"`; checking URL targets or DM keywords is necessary.
3. **Synthetic Handle Clutter**:
   * User handles replaced with `@115854` alter readability and can confuse semantic tokenizers. The normalized column replaces these with `@user`.
4. **Agent Signature Inconsistency**:
   * Some Apple agents sign tweets with initials while others do not. Normalization should ensure model responses do not overfit to specific agent initials.

---

## H. Potential Intent Categories Suggested by the Data

Based on the empirical keyword, n-gram, and dialogue analyses, the customer queries cluster into 6 natural intent categories:

1. **Battery & Power Performance**: Rapid battery drain, unexpected shutdowns, charging cable/port failures.
2. **Software Update & Installation**: Failures updating to iOS 11, stuck on Apple logo, iTunes restore errors.
3. **App Crashing & System Freezing**: Third-party app crashes, keyboard lag/autocorrect bugs, touchscreen unresponsiveness.
4. **Apple ID, iCloud & Account Security**: Password resets, two-factor authentication lockout, iCloud storage limits.
5. **Hardware, Audio & Display**: Cracked glass, camera blur, speaker/microphone distortion, water damage.
6. **Connectivity (Wi-Fi, Bluetooth & Cellular)**: Dropped calls, Bluetooth pairing with Apple Watch/AirPods, Wi-Fi grayed out.

---

## I. Potential Escalation Signals

The data reveals clear structural patterns governing when an issue is auto-handled vs. escalated:

* **Auto-Handle Candidate Signals**:
  * Routine configuration queries ("Where do I find battery health?", "How do I turn off autocorrect?").
  * Clear, non-destructive software troubleshooting (restart device, reset network settings, update iOS).
  * Requests for general documentation or feature explanation.
* **Human Escalation Candidate Signals**:
  * Inquiries involving Apple ID credentials, billing disputes, or subscription charges (security & PII risk).
  * Physical damage or hardware defects requiring Genius Bar or mail-in repair.
  * Customer expressing frustration after self-serve troubleshooting already failed ("I already restarted three times").
  * Multi-turn conversations exceeding 4 turns where public instructions failed to resolve the issue.

---

## J. Risks and Limitations

1. **DM Deflection Ground-Truth Ceiling**:
   * Because 52.55% of responses redirect to DM, public Twitter text does not show the ultimate technical fix for those sessions. The agent must be trained to recognize this boundary and route to a human agent with context rather than hallucinating private account resolutions.
2. **Historical Policy Drift**:
   * iOS 11 troubleshooting instructions (from 2017) are technically obsolete in 2026. The evaluation harness must measure fidelity against the *historical AppleSupport corpus*, not modern iOS versions.
3. **Short-Form Ambiguity**:
   * Many customer tweets are under 100 characters and omit device/OS information. The agent should emulate AppleSupport's strategy: asking for the iOS version before prescribing complex steps.
