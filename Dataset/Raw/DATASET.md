# ScamSense Datasets

This document describes all raw datasets used in the ScamSense project for ham and spam detection across multiple languages and message types.

---

## Dataset Overview
| Dataset | Records | Format | Language(s) | Label Type | Source |
|---|---|---|---|---|---|
| [Phishing Email Dataset](#1-phishing-email-dataset) | 82,486 | CSV | English | Ham / Scam | [Kaggle](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset) |
| [NUS SMS Corpus](#2-nus-sms-corpus) | 87,295 | JSON | Singlish, Mandarin | Ham only |  [Kaggle](https://www.kaggle.com/datasets/rtatman/the-national-university-of-singapore-sms-corpus) |
| [UCI SMS Spam Collection](#3-uci-sms-spam-collection) | 5,572 | CSV | English | Ham / Scam |  [UCI SMS](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) |
| [Real/Fake Job Postings](#4-realfake-job-postings) | 17,880 | CSV | English | Real / Fake |  [Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) |
| [Wikipedia MS/TA/ZH](#5-wikipedia-mstazh) | 15,000 | Parquet | Malay, Tamil, Mandarin | Ham only | [HuggingFace](https://huggingface.co/datasets/wikimedia/wikipedia) |
| [Synthetic Scam (SPF 2025)](#6-synthetic-scam-dataset-spf-2025) | 20,482 | CSV | English, Singlish, Malay, Tamil, Mandarin | Scam |  [SPF 2025 Report](https://isomer-user-content.by.gov.sg/537/0f81ce7a-8b96-4184-b4bb-89c45954bfcb/2025_annual_scams_and_cybercrime_brief%20(1).pdf) |

> **Label note:** "Ham" = legitimate messages. "Scam" = fraudulent messages.

---

## Dataset Breakdown (Scam vs Ham)

### Scam Data (Total: 64,986 rows)

* Phishing Email (scam only): 42,891
* Synthetic SPF 2025 scams: 20,482
* Fake job postings: 866
* UCI SMS spam: 747

⸻

### Ham Data (Total: 163,729 rows)

* NUS SMS Corpus: 87,295
* Phishing Email (ham portion): 39,595
* Real job postings: 17,014
* Wikipedia Tamil: 5,000
* Wikipedia Malay: 5,000
* Wikipedia Mandarin: 5,000
* UCI SMS ham: 4,825

⸻

## Key Dataset Insights

* Strong class imbalance: ham (163K) vs scam (64K)
* Largest ham source: NUS SMS Corpus (Singapore context)
* Largest scam source: Phishing Email dataset
* Synthetic dataset adds Singapore-specific multilingual scam patterns
* Wikipedia data improves non-English robustness (ms/ta/zh)

---

## Dataset Summary

| Metric | Value |
|---|---|
| Total raw records | ~228,715 rows |
| Scam rows | 64,986 |
| Ham rows | 163,729 |
| Languages | English, Singlish, Malay, Tamil, Mandarin |
| Formats | CSV, JSON, Parquet |
| Task type | Binary classification (Ham vs Scam) |

