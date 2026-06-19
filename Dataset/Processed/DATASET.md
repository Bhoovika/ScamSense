# ScamSense Processed Dataset

This document describes the processed and balanced dataset used for training and evaluating the ScamSense binary classification model.

---

## Processing Pipeline Overview

| Stage | Records | Notes |
|---|---|---|
| Raw merged (scam + ham) | 228,715 | Combined all raw sources |
| After null removal | 228,715 | No nulls found |
| After short-text filter | 221,771 | Removed 6,944 rows |
| After deduplication | 212,122 | Removed 9,649 duplicates |
| After per-language balancing | 129,134 | Undersampled ham per language |

---

## Label Distribution (Post-Cleaning, Pre-Balancing)

| Label | Count |
|---|---|
| 0 (Ham) | 147,555 |
| 1 (Scam) | 64,567 |
| **Total** | **212,122** |

> Strong class imbalance observed: ~2.3× more ham than scam before balancing.

---

## Per-Language Label Breakdown (Post-Cleaning, Pre-Balancing)

| Language | Ham (0) | Scam (1) | Ham:Scam Ratio |
|---|---|---|---|
| English (`en`) | 58,765 | 49,155 | 1.20:1 |
| Singlish (`singlish`) | 47,957 | 3,704 | 12.9:1 |
| Chinese (`zh`) | 30,862 | 3,823 | 8.1:1 |
| Tamil (`ta`) | 4,997 | 4,761 | 1.05:1 |
| Malay (`ms`) | 4,974 | 3,124 | 1.59:1 |

> Singlish and Chinese show the most severe imbalance, reflecting the scarcity of non-English scam samples in raw sources.

---

## Balancing Strategy

**Method:** Per-language random undersampling of the majority class (ham) to match the minority class (scam) count within each language group.

| Language | Scam Count | Ham (original) | Ham (undersampled) | Reduction |
|---|---|---|---|---|
| English (`en`) | 49,155 | 58,765 | 49,155 | −9,610 |
| Singlish (`singlish`) | 3,704 | 47,957 | 3,704 | −44,253 |
| Chinese (`zh`) | 3,823 | 30,862 | 3,823 | −27,039 |
| Tamil (`ta`) | 4,761 | 4,997 | 4,761 | −236 |
| Malay (`ms`) | 3,124 | 4,974 | 3,124 | −1,850 |

---

## Final Balanced Dataset

### Overall

| Metric | Value |
|---|---|
| Total rows | 129,134 |
| Ham (label 0) | 64,567 |
| Scam (label 1) | 64,567 |
| Balance ratio | 50% / 50% |

### Per-Language Balanced Counts

| Language | Ham (0) | Scam (1) | Total |
|---|---|---|---|
| English (`en`) | 49,155 | 49,155 | 98,310 |
| Tamil (`ta`) | 4,761 | 4,761 | 9,522 |
| Chinese (`zh`) | 3,823 | 3,823 | 7,646 |
| Singlish (`singlish`) | 3,704 | 3,704 | 7,408 |
| Malay (`ms`) | 3,124 | 3,124 | 6,248 |
| **Total** | **64,567** | **64,567** | **129,134** |

---

## Dataset Summary

| Metric | Value |
|---|---|
| Final processed records | 129,134 |
| Ham rows | 64,567 |
| Scam rows | 64,567 |
| Languages | English, Singlish, Malay, Tamil, Mandarin |
| Class balance | Perfectly balanced (50/50) |
| Balancing method | Per-language ham undersampling |
| Task type | Binary classification (Ham = 0 vs Scam = 1) |

---

## Key Observations

- **Perfect global balance** achieved (50/50) while preserving the natural language distribution of scam samples.
- **Singlish** had the most aggressive undersampling (ham reduced by ~92%), reflecting a large NUS SMS corpus but limited Singlish-specific scam data.
- **Tamil** was the most naturally balanced language pre-balancing (~1.05:1 ratio), requiring minimal adjustment.
- **English** dominates the final dataset (76.1%), consistent with the largest raw scam source being the Phishing Email dataset.
