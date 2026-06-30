# ScamSense Processed Dataset

This document describes the processed and balanced dataset used for training and evaluating the ScamSense binary classification model.

---

## Processed Dataset Overview

| Dataset | Records | Language(s) | Label Type | Split | Source |
|---|---|---|---|---|---|
| `scamsense_full_dataset.csv` | 136,930 | English, Singlish, Malay, Tamil, Mandarin | Ham / Scam | Full dataset |[Link](https://tinyurl.com/3y3tcc2t) |
| `train.csv` | 95,851 | English, Singlish, Malay, Tamil, Mandarin | Ham / Scam | 70% train split |[Link](https://tinyurl.com/4as75n6x)|
| `val.csv` | 20,539 | English, Singlish, Malay, Tamil, Mandarin | Ham / Scam | 15% val split |[Link](https://tinyurl.com/3jzk9ms9)|
| `test.csv` | 20,540 | English, Singlish, Malay, Tamil, Mandarin | Ham / Scam | 15% test split |[Link](https://tinyurl.com/2r8y5vfy)|

> **Label note:** "Ham" = legitimate messages (label 0). "Scam" = fraudulent messages (label 1). All splits are perfectly balanced at 50/50.


## Processing Pipeline Overview

| Stage | Records | Notes |
|---|---|---|
| Raw merged (scam + ham) | 232,616 | Combined all raw sources |
| After null removal | 232,616 | No nulls found |
| After short-text filter | 225,672 | Removed 6,944 rows |
| After deduplication | 210,929 | Removed 14,743 duplicates |
| After punctuation-only removal | 209,447 | Removed 1,482 rows |

---

## Label Distribution (Post-Cleaning, Pre-Balancing)

| Label | Count |
|---|---|
| 0 (Ham) | 140,982 |
| 1 (Scam) | 68,465 |
| **Total** | **209,447** |

> Strong class imbalance observed: ~2× more ham than scam before balancing.

---

## Per-Language Label Breakdown (Post-Cleaning, Pre-Balancing)

| Language | Ham (0) | Scam (1) |
|---|---|---|
| English (`en`) | 58,563 | 49,093 |
| Singlish (`singlish`) | 43,115 | 5,000 |
| Chinese (`zh`) | 29,333 | 4,962 |
| Tamil (`ta`) | 4,997 | 4,637 |
| Malay (`ms`) | 4,974 | 4,773 |

> Singlish and Chinese show the most severe imbalance, reflecting the scarcity of non-English scam samples in raw sources.

---

## Balancing Strategy

**Method:** Per-language random undersampling of the majority class (ham) to match the minority class (scam) count within each language group.

| Language | Scam Count | Ham (original) | Ham (undersampled) |
|---|---|---|---|
| English (`en`) | 49,093 | 58,563 | 49,093 |
| Singlish (`singlish`) | 5,000 | 43,115 | 5,000 |
| Chinese (`zh`) | 4,962 | 29,333 | 4,962 |
| Tamil (`ta`) | 4,637 | 4,997 | 4,637 |
| Malay (`ms`) | 4,773 | 4,974 | 4,773 |

---

## Dataset Summary

| Metric | Value |
|---|---|
| Final processed records | 136,930 |
| Ham rows | 68,465 |
| Scam rows | 68,465 |
| Languages | English, Singlish, Malay, Tamil, Mandarin |
| Class balance | Perfectly balanced (50/50) |
| Balancing method | Per-language ham undersampling |
| Task type | Binary classification (Ham = 0 vs Scam = 1) |

---

## Key Observations

- **Perfect balance** achieved (50/50) while preserving the natural language distribution of scam samples.
- **Singlish** had the most aggressive undersampling, reflecting a large NUS SMS corpus but limited Singlish-specific scam data.
- **Malay** was the most naturally balanced language pre-balancing, requiring minimal adjustment.
- **English** dominates the final dataset (76.1%), consistent with the largest raw scam source being the Phishing Email dataset.
