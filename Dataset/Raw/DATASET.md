# ScamSense Datasets

This document describes all raw datasets used in the ScamSense project for ham and spam detection across multiple languages and message types.

---

## Dataset Overview
| Dataset | Records | Format | Language(s) | Label Type | Source |
|---|---|---|---|---|---|
| Phishing Email Dataset | 82,486 | CSV | English | Ham / Scam | [Kaggle](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset) |
| NUS SMS Corpus | 87,295 | JSON | Singlish, Mandarin | Ham only |  [Kaggle](https://www.kaggle.com/datasets/rtatman/the-national-university-of-singapore-sms-corpus) |
| UCI SMS Spam Collection | 5,572 | CSV | English | Ham / Scam |  [UCI SMS](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) |
| Real/Fake Job Postings | 17,880 | CSV | English | Real / Fake |  [Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) |
| Wikipedia MS/TA/ZH | 15,000 | Parquet | Malay, Tamil, Mandarin | Ham only | [HuggingFace](https://huggingface.co/datasets/wikimedia/wikipedia) |
| Synthetic Scam | 24,383 | CSV | English, Singlish, Malay, Tamil, Mandarin | Scam |  [Synthetic Scam](https://tinyurl.com/5ys6r6dn) |

> **Label note:** "Ham" = legitimate messages. "Scam" = fraudulent messages.

---

## Dataset Breakdown (Scam vs Ham)

### Scam Data (Total: 68,887 rows)

* Phishing Email (scam only): 42,891
* Synthetic scams(SPF 2025 as a base): 24,383
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

* Strong class imbalance: ham (163K) vs scam (68K)
* Largest ham source: NUS SMS Corpus (Singapore context)
* Largest scam source: Phishing Email dataset
* Synthetic dataset adds Singapore-specific multilingual scam patterns
* Wikipedia data improves non-English robustness (ms/ta/zh)

---

## Dataset Summary

| Metric | Value |
|---|---|
| Total raw records | ~232,616 rows |
| Scam rows | 68,887 |
| Ham rows | 163,729 |
| Languages | English, Singlish, Malay, Tamil, Mandarin |
| Formats | CSV, JSON, Parquet |
| Task type | Binary classification (Ham vs Scam) |

