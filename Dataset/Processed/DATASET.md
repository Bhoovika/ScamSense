# ScamSense Processed Dataset

This document describes the processed and balanced dataset used for training and evaluating the ScamSense binary classification model.

---

## Processed Dataset Overview

| Dataset | Records | Language(s) | Label Type | Split | Source |
|---|---|---|---|---|---|
| `scamsense_full_dataset.csv` | 134,818 | English, Singlish, Malay, Tamil, Mandarin | Ham / Scam | Full dataset |[Link](https://tinyurl.com/3y3tcc2t) |
| `train.csv` | 94,372 | English, Singlish, Malay, Tamil, Mandarin | Ham / Scam | 70% train split |[Link](https://tinyurl.com/4as75n6x)|
| `val.csv` | 20,223 | English, Singlish, Malay, Tamil, Mandarin | Ham / Scam | 15% val split |[Link](https://tinyurl.com/3jzk9ms9)|
| `test.csv` | 20,223 | English, Singlish, Malay, Tamil, Mandarin | Ham / Scam | 15% test split |[Link](https://tinyurl.com/2r8y5vfy)|

> **Label note:** "Ham" = legitimate messages (label 0). "Scam" = fraudulent messages (label 1). All splits are perfectly balanced at 50/50.


## Processing Pipeline Overview

| Stage | Records | Notes |
|---|---|---|
| Raw merged (scam + ham) | 231,488 | Combined all raw sources |
| After null removal | 231,488 | No nulls found |
| After short-text filter | 224,544 | Removed 6,944 rows |
| After deduplication | 214,964 | Removed 9,580 duplicates |
| After per-language balancing | 134,818 | Undersampled ham per language |

---

## Label Distribution (Post-Cleaning, Pre-Balancing)

| Label | Count |
|---|---|
| 0 (Ham) | 147,555 |
| 1 (Scam) | 67,409 |
| **Total** | **214,964** |

> Strong class imbalance observed: ~2.1× more ham than scam before balancing.

---

## Per-Language Label Breakdown (Post-Cleaning, Pre-Balancing)

| Language | Ham (0) | Scam (1) | Ham:Scam Ratio |
|---|---|---|---|
| English (`en`) | 58,765 | 49,155 | 1.20:1 |
| Singlish (`singlish`) | 47,957 | 3,704 | 12.9:1 |
| Chinese (`zh`) | 30,862 | 4,991 | 6.1:1 |
| Tamil (`ta`) | 4,997 | 4,728 | 1.05:1 |
| Malay (`ms`) | 4,974 | 4,831 | 1.02:1 |

> Singlish and Chinese show the most severe imbalance, reflecting the scarcity of non-English scam samples in raw sources.

---

## Balancing Strategy

**Method:** Per-language random undersampling of the majority class (ham) to match the minority class (scam) count within each language group.

| Language | Scam Count | Ham (original) | Ham (undersampled) | Reduction |
|---|---|---|---|---|
| English (`en`) | 49,155 | 58,765 | 49,155 | −9,610 |
| Singlish (`singlish`) | 3,704 | 47,957 | 3,704 | −44,253 |
| Chinese (`zh`) | 4,991 | 30,862 | 4,991 | −25,871 |
| Tamil (`ta`) | 4,728 | 4,997 | 4,728 | −269 |
| Malay (`ms`) | 4,831 | 4,974 | 4,831 | −143 |

---

## Final Balanced Dataset

### Overall

| Metric | Value |
|---|---|
| Total rows | 134,818 |
| Ham (label 0) | 67,409 |
| Scam (label 1) | 67,409 |
| Balance ratio | 50% / 50% |

### Per-Language Balanced Counts

| Language | Ham (0) | Scam (1) | Total |
|---|---|---|---|
| English (`en`) | 49,155 | 49,155 | 98,310 |
| Tamil (`ta`) | 4,728 | 4,728 | 9,456 |
| Chinese (`zh`) | 4,991 | 4,991 | 9,982 |
| Singlish (`singlish`) | 3,704 | 3,704 | 7,408 |
| Malay (`ms`) | 4,831 | 4,831 | 9,662 |
| **Total** | **67,409** | **67,409** | **134,818** |

---

## Dataset Summary

| Metric | Value |
|---|---|
| Final processed records | 134,818 |
| Ham rows | 67,409 |
| Scam rows | 67,409 |
| Languages | English, Singlish, Malay, Tamil, Mandarin |
| Class balance | Perfectly balanced (50/50) |
| Balancing method | Per-language ham undersampling |
| Task type | Binary classification (Ham = 0 vs Scam = 1) |

---

## Key Observations

- **Perfect global balance** achieved (50/50) while preserving the natural language distribution of scam samples.
- **Singlish** had the most aggressive undersampling, reflecting a large NUS SMS corpus but limited Singlish-specific scam data.
- **Malay** was the most naturally balanced language pre-balancing (~1.02:1 ratio), requiring minimal adjustment.
- **English** dominates the final dataset (76.1%), consistent with the largest raw scam source being the Phishing Email dataset.
