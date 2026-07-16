# ScamSense

**Multilingual scam-message detection with explainable, agentic risk assessment.**

ScamSense classifies text messages as **scam or legitimate (ham)** across **5 languages** — English, Singlish, Malay, Tamil, and Mandarin — and goes beyond a simple label by producing a **risk score, scam category, and a grounded explanation**, referencing real Singapore Police Force (SPF) scam statistics.

The project is built as five sequential Kaggle notebooks (synthetic data → preprocessing → model training → agentic pipeline → deployment) plus a consolidated pipeline script, and ships with a working **FastAPI backend + Telegram bot**.

---

## Table of Contents

- [Project Pipeline](#project-pipeline)
- [Results Summary](#results-summary)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Setup & Running on Kaggle](#setup--running-on-kaggle)
- [Running the API + Telegram Bot](#running-the-api--telegram-bot)
- [API Reference](#api-reference)
- [Future Work](#future-work)
- [Credits & Data Sources](#credits--data-sources)

---
## Project Pipeline

| Stage | Notebook / Script                             | Purpose                                                                                  |
| ----- | --------------------------------------------- | ---------------------------------------------------------------------------------------- |
| 1     | `01_synthetic-data-generation.ipynb`          | Generates synthetic scam messages in 5 languages via templates + MT                      |
| 2     | `02_data-preparation-preprocessing-eda.ipynb` | Merges with real ham/scam datasets, cleans, balances, splits, runs EDA                   |
| 2a    | `02a_mnar_leakage_resolved.ipynb`              | Retrains Experiment 2 with tokenizer-only truncation + template-skeleton-grouped split, to verify the reported F1 isn't inflated by MNAR truncation bias or template leakage |
| 3     | `03_modelling.ipynb`                          | Trains & compares 6 XLM-R / mBERT model configurations                                   |
| 4     | `04_langgraph_rag.ipynb`                      | LangGraph 3-agent pipeline: detection → SHAP+RAG explanation → risk scoring              |
| 5     | `05_fastapi_backend.ipynb`                    | FastAPI REST backend + Telegram bot deployment (notebook version)                        |
| —     | `scamsense_pipeline.py`                       | Consolidated detection → explanation → risk pipeline module used by the deployed API/bot |

**Data flow:** `01` (synthetic scam only) → `02` (cleaned, balanced, split) → `03` (trained model) → `04`/`05` (inference pipelines using the trained model + SPF taxonomy). `02a` is a standalone validation branch off `02`, used only to confirm Experiment 2's F1 under a leakage-safe split.

All notebooks are designed to be run directly on **Kaggle** (see [Setup & Running on Kaggle](#setup--running-on-kaggle)).

---

## Results Summary

**Final model:** XLM-RoBERTa (base), fine-tuned — lr=2e-5, batch=32, 5 epochs.

| Metric                  | Value                          |
| ----------------------- | ------------------------------ |
| Test F1                 | **0.9929**                     |
| Test AUC                | **0.9990**                     |
| Train/Test accuracy gap | 0.0025–0.0073 (no overfitting) |
| Leakage-safe test F1 (`02a`, tokenizer-only truncation + skeleton-grouped split) | **0.9901** (Δ -0.0028) |

Full experiment comparison (6 configs including a frozen-mBERT baseline) is logged via MLflow/DagsHub and detailed in the training notebook.

Model checkpoint hosted on Hugging Face Hub: [`Bhoovika/scamsense-xlmroberta-new1`](https://huggingface.co/Bhoovika/scamsense-xlmroberta-new1).

---

## Repository Structure

```
ScamSense/
├── Dataset/
│   ├── Processed/                # NB02 output: train.csv, val.csv, test.csv, scamsense_full_dataset.csv
│   └── Raw/                       # Real ham/scam source datasets (phishing email, UCI SMS, etc.)
├── Source_code/
│   ├── 01_synthetic-data-generation.ipynb
│   ├── 02_data-preparation-preprocessing-eda.ipynb
│   ├── 03_modelling.ipynb
│   ├── 04_langgraph_rag.ipynb
│   ├── 05_fastapi_backend.ipynb
│   └── scamsense_pipeline.py      # Shared detection/explanation/risk pipeline used in deployment
└── README.md
```

---

## Dataset

The full dataset (raw + processed) plus shared utility code are hosted on **Kaggle Datasets** so notebooks can attach them directly instead of re-uploading files each session:

- **Raw dataset:** [kaggle.com/datasets/bhoovika/scamsense-raw-dataset](https://www.kaggle.com/datasets/bhoovika/scamscene-raw-dataset)
- **Processed dataset:** [kaggle.com/datasets/bhoovika/scamsense-processed-dataset](https://www.kaggle.com/datasets/bhoovika/scamscene-processed-dataset)
- **Pipeline utils:** [kaggle.com/datasets/bhoovika/scamsense-utils](https://www.kaggle.com/datasets/bhoovika/scamsense-utils)

- **Optional** (used by `04_langgraph_rag.ipynb` only, for faster offline loading): the trained classifier and sentence embedder are also mirrored as Kaggle Datasets — [`bhoovika/scamsense-xlmroberta-new1`](https://www.kaggle.com/datasets/bhoovika/scamsense-xlmroberta-new1) and [`bhoovika/scamsense-minilm-embedder`](https://www.kaggle.com/datasets/bhoovika/scamsense-minilm-embedder). Attaching them is not required — if omitted, `scamsense_pipeline.init()` falls back to pulling both directly from the Hugging Face Hub, as `05_fastapi_backend.ipynb` already does.

Structure inside each Kaggle dataset:
```
scamscene-raw-dataset/
├── SMSSpamCollection
├── synthetic_dataset.csv
├── wiki_ms.csv
├── wiki_ta.csv
└── wiki_zh.csv
scamscene-processed-dataset/
├── scamsense_full_dataset.csv
├── train.csv
├── train_norm.csv
├── val.csv
├── val_norm.csv
├── test.csv
└── test_norm.csv
scamsense-utils/
└── scamsense_pipeline.py
```

When creating a Kaggle notebook, click **Add Input → Datasets** and attach the relevant dataset(s) above — they'll mount read-only under `/kaggle/input/scamscene-raw-dataset/`, `/kaggle/input/scamscene-processed-dataset/`, and `/kaggle/input/scamsense-utils/` respectively. Notebooks read raw sources from the raw dataset, write/read normalized splits from the processed dataset, and import `scamsense_pipeline.py` from the utils dataset for shared detection/explanation/risk logic.

---

## Setup & Running on Kaggle

All five notebooks are built to run on **Kaggle Notebooks** (Tesla T4 / P100 GPU, free tier) rather than a local environment.

### 1. Create/open a Kaggle Notebook

- Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**.
- Or fork directly from the notebooks in this repo by uploading them via **File → Upload Notebook**.

### 2. Attach the dataset

- **Add Input → Datasets** → search for and attach the ScamSense Kaggle dataset (see [Dataset](#dataset)).

### 3. Enable GPU

- **Settings → Accelerator → GPU T4 x2** (or P100). Required for `01_synthetic-data-generation.ipynb` (translation) and `03_modelling.ipynb` (training). `04` and `05` run fine on CPU.

### 4. Install dependencies

Run this as the first cell in each notebook:

```bash
!pip install -q transformers torch sentencepiece datasets scikit-learn pandas numpy \
    langdetect shap faiss-cpu langgraph langchain fastapi uvicorn httpx \
    python-telegram-bot==20.8 supabase mlflow python-dotenv
```

### 5. Secrets (Telegram token, Supabase, MLflow/DagsHub)

Use **Kaggle Secrets** instead of a local `.env` file:

- **Add-ons → Secrets** in the Kaggle notebook editor, add:
  - `TELEGRAM_BOT_TOKEN`
  - `SUPABASE_URL`, `SUPABASE_KEY`
  - `HF_MODEL_REPO` (default: `Bhoovika/scamsense-xlmroberta-new1`)
  - `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`

Then load them in-notebook:

```python
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
TELEGRAM_BOT_TOKEN = secrets.get_secret("TELEGRAM_BOT_TOKEN")
```

### 6. Run notebooks in order

Each notebook consumes the previous notebook's output, so run sequentially:

1. `01_synthetic-data-generation.ipynb` → outputs `synthetic_dataset.csv` (seed=42, reproducible)
2. `02_data-preparation-preprocessing-eda.ipynb` → reads real datasets from `Dataset/Raw/` (attached Kaggle dataset), cleans/balances/splits → outputs `train_norm.csv` / `val_norm.csv` / `test_norm.csv` , `train.csv` / `val.csv` / `test.csv` and 'scamsense_full_dataset.csv' to `Dataset/Processed/`.
3. `03_modelling.ipynb` → trains 6 model variants via a `train_or_load()` wrapper . Best model auto-selected by validation F1, confirmed by test F1, and pushed to the Hugging Face Hub repo / Kaggle model output.
4. `04_langgraph_rag.ipynb` → builds `spf_taxonomy.json`, `spf_corpus.json`, and a FAISS index (`spf_faiss.index`) — run before `05` if starting fresh, since `05` rebuilds its own FAISS corpus at API startup but shares the same taxonomy source.
5. `05_fastapi_backend.ipynb` → can be run as a Kaggle notebook for a quick test, or exported alongside `scamsense_pipeline.py` for real deployment (see below).

> Kaggle notebooks are session-based — for a persistent FastAPI + Telegram bot deployment, export `scamsense_pipeline.py` and run it on a separate always-on host (e.g. a small VM or container), pointing it at the same model checkpoint and taxonomy files produced above.

---

## Running the API + Telegram Bot

Once you've exported `scamsense_pipeline.py` (and the model checkpoint / taxonomy files produced by the notebooks) to a deployment host:


### Start the Telegram bot

The bot can run in a background thread from the same process, or as a standalone script:

```bash
python scamsense_pipeline.py --mode telegram
```

In Telegram, message **@ScamSense_Scout_bot** (or your own bot username) with `/start`, then send any suspicious message text — it will run `/predict` under the hood and return a formatted risk card.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/predict` | POST | Classify a message → `{label, confidence, scam_type, tier, risk_score}` |
| `/stats` | GET | Cumulative prediction stats logged to Supabase (all runs, all time) |
| `/history` | GET | Recent prediction history |

**Example request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Congratulations! You have won $5000, click here to claim now"}'
```

**Example response:**

```json
{
  "label": "scam",
  "confidence": 0.998,
  "language": "en",
  "scam_type": "prize",
  "tier": "medium",
  "risk_score": 55
}
```

> ⚠️ `/stats` returns cumulative counts across **all** predictions ever logged, not per-session. For a clean live demo, filter by timestamp or truncate the Supabase table first.

---


## Future Work

- Expand Tamil and Malay keyword sets in the risk-scoring agent.
- Tune the Singlish particle-density threshold for better recall on short messages.
- Targeted data augmentation for English, the current weakest-performing language.
- Add per-session filtering to `/stats` for demo/production use.

---

## Credits & Data Sources

- **SPF scam statistics:** Singapore Police Force annual scam bulletin (2025 figures).
- **Translation model:** [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M).
- **Classification model:** XLM-RoBERTa (base), fine-tuned on merged synthetic + real datasets.
- **Real datasets used:** phishing email dataset, UCI SMS Spam Collection, fake job postings dataset, NUS SMS Corpus, Wikipedia multilingual corpus (for ham diversity).
- **Experiment tracking:** MLflow via [DagsHub](https://dagshub.com/Bhoovika/ScamSense).
