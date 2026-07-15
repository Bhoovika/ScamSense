"""
scamsense_pipeline.py
======================
Single source of truth for the ScamSense scam-detection pipeline.

Both NB04 (LangGraph agentic pipeline) and NB05 (FastAPI backend) import
their logic from here instead of each keeping their own copy. Previously
NB05's copy had quietly drifted from NB04's (fewer SPF categories, smaller
keyword lists, no bank_impersonation/parcel_delivery/rental/charity/prize
handling) — importing a shared module makes that class of bug impossible.

Usage
-----
    from scamsense_pipeline import init, classify, explain, get_risk, run_pipeline

    init(hf_token=HF_TOKEN, rag_dir=RAG_DIR)   # once per process

    det = classify("Your POSB account has been suspended...")
    exp = explain("Your POSB account has been suspended...", scam_type_hint=det["scam_type_guess"])
    risk = get_risk(exp["scam_type"], det["scam_prob"])

    # or, simplest:
    result = run_pipeline("Your POSB account has been suspended...")

Requires: torch, transformers, sentence-transformers, faiss-cpu, shap,
langdetect, numpy. (pandas not required by this module.)
"""

from __future__ import annotations  # allow modern type hints (e.g. list[dict]) on older Python

import re                                     # keyword/regex matching for scam-type detection
from pathlib import Path                      # filesystem paths for corpus/index files
from typing import Optional                   # optional-argument type hints

import numpy as np                            # arrays for embeddings and scores
import torch                                  # runs the XLM-RoBERTa classifier
import faiss                                  # vector similarity search over the SPF corpus
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # HF classifier + tokenizer
from sentence_transformers import SentenceTransformer  # embeds text for the RAG search
import shap                                   # explains which tokens drove the classifier's decision
from langdetect import detect, LangDetectException  # fallback language detector

# ---------------------------------------------------------------------------
# Config / lazily-loaded globals
# ---------------------------------------------------------------------------
HF_MODEL_ID = "Bhoovika/scamsense-xlmroberta-new1"       # HuggingFace repo for the trained classifier
EMBEDDER_MODEL_ID = "paraphrase-multilingual-MiniLM-L12-v2"  # sentence embedder for RAG retrieval
LABEL_MAP = {0: "ham", 1: "scam"}                    # classifier output index -> label name
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # use GPU if available, else CPU

_clf_tokenizer = None        # loaded by init(): tokenizer for the classifier
_clf_model = None            # loaded by init(): the XLM-RoBERTa classifier itself
_embedder = None             # loaded by init(): sentence embedder for RAG
_shap_explainer = None       # loaded by init(): SHAP explainer over the classifier
_faiss_index = None          # loaded by init(): FAISS index over the SPF corpus embeddings
_corpus_embeddings = None    # loaded by init(): raw embeddings backing the FAISS index
_ready = False                # flips to True once init() has finished loading everything

SPF_CORPUS: list[dict] = []  # populated by init()


def _is_ready() -> bool:
    return _ready                                    # simple getter for whether init() has run


def _require_ready():
    if not _ready:                                    # guard used at the top of every public function
        raise RuntimeError(
            "scamsense_pipeline.init(...) must be called once before "
            "classify()/explain()/get_risk()/run_pipeline() are used."
        )


# ---------------------------------------------------------------------------
# Init: load classifier, embedder, SHAP explainer, FAISS index + RAG corpus
# ---------------------------------------------------------------------------
def init(
    hf_token: Optional[str] = None,
    rag_dir: Optional[Path] = None,
    corpus_path: Optional[Path] = None,
    index_path: Optional[Path] = None,
    embeddings_path: Optional[Path] = None,
    build_index_if_missing: bool = True,
    model_path: Optional[str] = None,   # NEW: local path (e.g. Kaggle dataset dir) overrides HF_MODEL_ID
    embedder_path: Optional[str] = None,    # NEW: local path for the sentence embedder
):

    """
    Load all heavy resources once per process.

    - If `rag_dir` is given, corpus/index/embeddings default to
      rag_dir/spf_corpus.json, rag_dir/spf_faiss.index, rag_dir/spf_embeddings.npy
      (this is how NB05 loads a pre-built index shipped as a Kaggle dataset).
    - If those files don't exist and `build_index_if_missing` is True, the
      FAISS index + embeddings are built from SPF_CORPUS in this module and
      (if rag_dir was writable) cached there (this is how NB04 builds it
      fresh the first time).
    """
    global _clf_tokenizer, _clf_model, _embedder, _shap_explainer   # these get assigned below
    global _faiss_index, _corpus_embeddings, SPF_CORPUS, _ready

    if rag_dir is not None:                            # derive default file paths from rag_dir if given
        rag_dir = Path(rag_dir)
        corpus_path = corpus_path or (rag_dir / "spf_corpus.json")
        index_path = index_path or (rag_dir / "spf_faiss.index")
        embeddings_path = embeddings_path or (rag_dir / "spf_embeddings.npy")

    print(f"Device: {DEVICE}")

    if model_path is not None:
        # Local path (e.g. a Kaggle Dataset mounted at /kaggle/input/...) — no HF download, no token needed
        print(f"Loading tokenizer/model from local path {model_path} ...")
        _clf_tokenizer = AutoTokenizer.from_pretrained(model_path)
        _clf_model = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE)
    else:
        # Default: pull from HuggingFace Hub using HF_MODEL_ID
        print(f"Loading tokenizer/model from {HF_MODEL_ID} ...")
        _clf_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, token=hf_token)
        _clf_model = AutoModelForSequenceClassification.from_pretrained(
            HF_MODEL_ID, token=hf_token
        ).to(DEVICE)

    _clf_model.eval()                                   # inference mode: no dropout, no gradients
    print("Classifier loaded")

    if embedder_path is not None:
        print(f"Loading embedder from local path {embedder_path} ...")
        _embedder = SentenceTransformer(embedder_path)
    else:
        print(f"Loading embedder from {EMBEDDER_MODEL_ID} ...")
        _embedder = SentenceTransformer(EMBEDDER_MODEL_ID)
    print("Embedder loaded")

    # ── Load or build the RAG corpus ──────────────────────────────────────
    import json                                          # only needed here, for reading the corpus file

    if corpus_path is not None and Path(corpus_path).exists():   # prefer a pre-built corpus file on disk
        with open(corpus_path) as f:
            SPF_CORPUS = json.load(f)
        print(f"SPF corpus loaded from {corpus_path} ({len(SPF_CORPUS)} chunks)")
    else:
        SPF_CORPUS = list(_DEFAULT_SPF_CORPUS)            # fall back to the corpus baked into this module
        print(f"SPF corpus loaded from built-in default ({len(SPF_CORPUS)} chunks)")

    # ── Load or build the FAISS index ─────────────────────────────────────
    if index_path is not None and embeddings_path is not None \
            and Path(index_path).exists() and Path(embeddings_path).exists():
        _faiss_index = faiss.read_index(str(index_path))   # load a pre-built index from disk
        _corpus_embeddings = np.load(str(embeddings_path))  # load the matching embeddings
        print(f"FAISS index loaded from disk ({_faiss_index.ntotal} vectors)")
    elif build_index_if_missing:
        print("Building FAISS index from SPF corpus ...")
        corpus_texts = [c["text"] for c in SPF_CORPUS]      # texts to embed, one per corpus chunk
        _corpus_embeddings = _embedder.encode(
            corpus_texts, normalize_embeddings=True, show_progress_bar=True
        ).astype(np.float32)                                # embed all corpus chunks at once
        dim = _corpus_embeddings.shape[1]                   # embedding dimensionality
        _faiss_index = faiss.IndexFlatIP(dim)               # inner-product index (cosine sim, since normalized)
        _faiss_index.add(_corpus_embeddings)                # populate the index with corpus vectors
        print(f"FAISS index built ({_faiss_index.ntotal} vectors, dim={dim})")

        if index_path is not None:                          # try to cache the freshly built index to disk
            try:
                Path(index_path).parent.mkdir(parents=True, exist_ok=True)
                faiss.write_index(_faiss_index, str(index_path))
                np.save(str(embeddings_path), _corpus_embeddings)
                print(f"   Cached → {index_path}")
            except OSError:
                pass  # e.g. read-only Kaggle dataset dir — fine, just don't cache
    else:
        raise FileNotFoundError(
            "No FAISS index/embeddings found and build_index_if_missing=False"
        )

    # ── SHAP explainer ─────────────────────────────────────────────────────
    def _shap_predict(texts):
        if isinstance(texts, str):                       # SHAP may pass a single string or a list
            texts = [texts]
        inputs = _clf_tokenizer(
            list(texts), return_tensors="pt", truncation=True, max_length=128, padding=True
        ).to(DEVICE)
        with torch.no_grad():                             # no gradients needed, just a forward pass
            logits = _clf_model(**inputs).logits
        return torch.softmax(logits, dim=-1).cpu().numpy()  # SHAP needs class probabilities, not logits

    masker = shap.maskers.Text(_clf_tokenizer)             # masks/removes tokens to test their contribution
    _shap_explainer = shap.Explainer(_shap_predict, masker, output_names=["ham", "scam"])
    print("SHAP explainer ready")

    _ready = True                                          # unblocks classify()/explain()/get_risk() etc.
    print("scamsense_pipeline ready")


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------
_SINGLISH_MARKERS = [                                    # common Singlish particles/phrases, word-boundary matched
    r"\blah\b", r"\bleh\b", r"\bsia\b", r"\blor\b", r"\bwah\b",
    r"\baiyo\b", r"\bsiao\b", r"\bdun\b", r"\bwan\b", r"\bcan or not\b",
    r"\bgot meh\b", r"\bcan leh\b", r"\bone\b",
]
_singlish_re = re.compile("|".join(_SINGLISH_MARKERS), re.IGNORECASE)  # combined regex for all markers


def detect_language(text: str) -> str:
    """Detect en / zh / ta / ms / singlish."""
    hits = len(_singlish_re.findall(text))               # count Singlish marker occurrences
    words = len(text.split())                             # rough word count for the ratio below
    if words > 0 and (hits / words) >= 0.08:              # enough Singlish markers -> treat as Singlish
        return "singlish"
    try:
        d = detect(text)                                   # fall back to general language detection
        return {
            "en": "en", "ms": "ms", "id": "ms", "ta": "ta",   # collapse related codes to one label each
            "zh-cn": "zh", "zh-tw": "zh", "zh": "zh",
        }.get(d, "en")                                      # default to English for unrecognised codes
    except LangDetectException:
        return "en"                                          # default to English if detection fails outright


# ---------------------------------------------------------------------------
# SPF 2025 Risk Taxonomy (12 categories + unknown fallback)
# Source: Singapore Police Force, Annual Scam and Cybercrime Brief 2025
# ---------------------------------------------------------------------------
SPF_TAXONOMY = {
    # Investment scams — highest losses of any category in 2025
    "investment": {
        "spf_name": "Investment Scam", "2025_cases": 5462, "2025_losses": 336.2,
        "avg_loss": 61559, "risk_tier": "CRITICAL", "risk_score": 95,
        "keywords": [
            r"invest", r"return", r"profit", r"forex", r"crypto",
            r"trading", r"guaranteed", r"passive income", r"portfolio",
            r"capital", r"dividend", r"scheme", r"fund", r"roi",
            r"bitcoin", r"ethereum", r"tether", r"usdt", r"wallet",
            r"keuntungan", r"pelaburan", r"முதலீட்", r"லாபம்",
            r"投资", r"收益", r"理财", r"赚钱",
        ],
        "advice": (
            "Investment scams caused the HIGHEST losses in Singapore in 2025 — "
            "$336.2 million (SPF 2025). Never transfer money to strangers for "
            "'investments'. Verify at MAS (mas.gov.sg/investor-alert-list). "
            "Report to SPF at 1800-255-0000."
        ),
    },
    # Scammers posing as police/CPF/ICA/MAS/court officials
    "government_impersonation": {
        "spf_name": "Government Officials Impersonation Scam", "2025_cases": 3363,
        "2025_losses": 242.9, "avg_loss": 72229, "risk_tier": "CRITICAL", "risk_score": 93,
        "keywords": [
            r"police", r"spf", r"cpf", r"ica", r"mas", r"iras", r"hdb",
            r"officer", r"detective", r"warrant", r"arrest", r"investigation",
            r"money laundering", r"safety account", r"court", r"ministry",
            r"government", r"official", r"authority", r"polis",
            r"警察", r"公安", r"调查", r"安全账户", r"காவல்", r"அரசு",
        ],
        "advice": (
            "Cases MORE THAN DOUBLED in 2025 (+123.6%), $242.9M lost (SPF 2025). "
            "Singapore government officials will NEVER ask you to transfer money, "
            "disclose banking details, or install unofficial apps. "
            "Hang up and call ScamShield at 1799."
        ),
    },
    # Fake bank security alerts (tracked by SPF under Phishing)
    "bank_impersonation": {
        "spf_name": "Bank Impersonation Scam (SPF-tracked under Phishing Scam)",
        "2025_cases": 6264, "2025_losses": 39.9, "avg_loss": 6384,
        "risk_tier": "HIGH", "risk_score": 76,
        "keywords": [
            r"dbs", r"posb", r"ocbc", r"uob", r"standard chartered",
            r"citibank", r"hsbc", r"maybank", r"trust bank",
            r"security team", r"new device login", r"unauthorised access",
            r"account frozen", r"account suspended", r"security alert",
            r"kata laluan bank", r"akaun bank", r"银行", r"账户冻结", r"安全团队",
        ],
        "advice": (
            "SPF does not track bank impersonation as a standalone category — it "
            "is folded into Phishing Scam statistics ($39.9M lost, 6,264 cases, "
            "SPF 2025), where fake bank alerts push victims to fake login pages. "
            "Banks never ask you to verify your account via a link in an SMS or "
            "email. Go directly to your bank's official app. Report to "
            "report@scamalert.sg."
        ),
    },
    # Fake job offers demanding upfront fees
    "job_scam": {
        "spf_name": "Job Scam", "2025_cases": 5575, "2025_losses": 123.5,
        "avg_loss": 22163, "risk_tier": "HIGH", "risk_score": 80,
        "keywords": [
            r"job", r"work from home", r"part.?time", r"hiring", r"salary",
            r"task", r"commission", r"earn", r"vacancy", r"recruit",
            r"registration fee", r"training fee", r"deposit", r"agent fee",
            r"kerja", r"gaji", r"pendapatan", r"வேலை", r"சம்பளம்",
            r"工作", r"兼职", r"佣金", r"招聘",
        ],
        "advice": (
            "Job scams cost Singaporeans $123.5M in 2025 (SPF 2025). "
            "Legitimate employers never ask for upfront fees. "
            "Verify at mom.gov.sg. Report to MOM or call 1800-255-0000."
        ),
    },
    # Fake links/pages harvesting OTPs and credentials
    "phishing": {
        "spf_name": "Phishing Scam", "2025_cases": 6264, "2025_losses": 39.9,
        "avg_loss": 6384, "risk_tier": "HIGH", "risk_score": 78,
        "keywords": [
            r"click", r"link", r"verify", r"otp", r"credential", r"login",
            r"password", r"http", r"www\.", r"\.xyz", r"\.top", r"\.club",
            r"kata laluan", r"akaun", r"கணக்கு", r"கடவுச்சொல்",
            r"账户", r"密码", r"验证", r"点击",
        ],
        "advice": (
            "Phishing is the 2nd most common scam in Singapore (6,264 cases, SPF 2025). "
            "Never click unsolicited links or enter OTP/password from an SMS or email. "
            "Go directly to your bank's official website. "
            "Report to report@scamalert.sg."
        ),
    },
    # Fake online marketplace listings, most common scam by case count
    "ecommerce": {
        "spf_name": "E-commerce Scam", "2025_cases": 6703, "2025_losses": 16.7,
        "avg_loss": 2503, "risk_tier": "MEDIUM", "risk_score": 60,
        "keywords": [
            r"sell", r"selling", r"buy", r"cheap", r"deal", r"item",
            r"carousell", r"shopee", r"lazada", r"facebook marketplace",
            r"ship", r"delivery", r"transfer first", r"paynow first",
            r"deposit", r"pokemon", r"brand new", r"legit",
            r"jual", r"beli", r"murah", r"விற்பனை", r"வாங்க",
            r"出售", r"购买", r"便宜", r"转账",
        ],
        "advice": (
            "E-commerce scams are the most common type in Singapore (6,703 cases, SPF 2025). "
            "Never PayNow before receiving goods. "
            "Meet in person for high-value items or use Carousell protected payment. "
            "Report at go.gov.sg/scamalert."
        ),
    },
    # Impersonating a friend/family member with a "new number"
    "fake_friend": {
        "spf_name": "Fake Friend Call Scam", "2025_cases": 1551, "2025_losses": 4.7,
        "avg_loss": 3056, "risk_tier": "MEDIUM", "risk_score": 62,
        "keywords": [
            r"new number", r"changed number", r"it's me", r"lost my phone",
            r"lost phone", r"stuck", r"stranded", r"emergency",
            r"transfer.{0,15}(now|urgent|first)", r"pay you back",
            r"hi mum", r"hi dad", r"hi mom", r"妈妈", r"爸爸", r"新号码",
            r"அம்மா", r"அப்பா",
        ],
        "advice": (
            "Fake friend call scams fell sharply in 2025 but still cost $4.7M "
            "across 1,551 cases (SPF 2025). Always verify a claimed new number by "
            "calling the person's old number or contacting them another way "
            "before sending money."
        ),
    },
    # Fake courier/customs-fee messages
    "parcel_delivery": {
        "spf_name": "Parcel Delivery Scam", "2025_cases": None, "2025_losses": None,
        "avg_loss": None, "risk_tier": "MEDIUM", "risk_score": 55,
        "keywords": [
            r"parcel", r"delivery", r"courier", r"singpost", r"ninja van",
            r"dhl", r"fedex", r"j&t", r"tracking", r"customs",
            r"clearance fee", r"shipping fee", r"cannot deliver",
            r"包裹", r"快递", r"清关费", r"பார்சல்", r"விநியோகம்",
        ],
        "advice": (
            "SPF does not publish separate 2025 statistics for parcel delivery "
            "scams — a related advisory (Sept 2024) recorded 338+ cases and "
            "$616K+ in losses, largely impersonating SingPost. Legitimate couriers "
            "never request payment via SMS link for customs clearance. Verify "
            "directly with the courier's official app or hotline."
        ),
    },
    # Fake rental listings pressuring an upfront deposit
    "rental": {
        "spf_name": "Rental Scam", "2025_cases": None, "2025_losses": None,
        "avg_loss": None, "risk_tier": "MEDIUM", "risk_score": 55,
        "keywords": [
            r"rent", r"deposit", r"landlord", r"room", r"apartment",
            r"condo", r"hdb.{0,10}(rent|room|unit)", r"lease", r"reserve.{0,10}unit",
            r"租房", r"房租", r"押金", r"sewa", r"வாடகை",
        ],
        "advice": (
            "SPF does not publish separate 2025 statistics for rental scams; they "
            "are grouped within the SPF's broader 'Other Scams' bucket. Never "
            "transfer a deposit before viewing the property in person and "
            "verifying the landlord owns the unit (check against HDB/URA records)."
        ),
    },
    # Fake moneylenders charging upfront "processing" fees
    "loan": {
        "spf_name": "Loan Scam", "2025_cases": 935, "2025_losses": 7.0,
        "avg_loss": 7515, "risk_tier": "MEDIUM", "risk_score": 64,
        "keywords": [
            r"loan", r"credit", r"approve", r"interest", r"fast cash",
            r"money lender", r"licensed moneylender", r"no credit check",
            r"emergency cash", r"disbursement", r"pinjaman", r"贷款", r"கடன்",
        ],
        "advice": (
            "Loan scams: 935 cases, $7.0M lost in 2025 (SPF 2025) — one of the few "
            "scam types where average loss rose despite fewer cases. Never pay "
            "upfront 'processing fees' before receiving a loan. Verify licensed "
            "moneylenders at Ministry of Law's registry (mlaw.gov.sg)."
        ),
    },
    # Fake donation requests
    "charity": {
        "spf_name": "Charity Scam", "2025_cases": None, "2025_losses": None,
        "avg_loss": None, "risk_tier": "MEDIUM", "risk_score": 55,
        "keywords": [
            r"donate", r"donation", r"charity", r"fundraising",
            r"disaster relief", r"food bank", r"treatment fund",
            r"捐款", r"慈善", r"derma", r"நன்கொடை",
        ],
        "advice": (
            "SPF does not publish separate 2025 statistics for charity scams; "
            "they fall within the 'Other Scams' bucket. Donate only through "
            "registered charities listed on the Charity Portal (charities.gov.sg) "
            "and avoid urgent, unsolicited donation requests via messaging apps."
        ),
    },
    # Fake lucky-draw / prize-claim messages
    "prize": {
        "spf_name": "Prize Scam", "2025_cases": None, "2025_losses": None,
        "avg_loss": None, "risk_tier": "MEDIUM", "risk_score": 55,
        "keywords": [
            r"winner", r"won", r"congratulations", r"lucky draw",
            r"claim.{0,10}prize", r"cash prize", r"ntuc voucher", r"reward",
            r"中奖", r"奖金", r"hadiah", r"பரிசு",
        ],
        "advice": (
            "SPF does not publish separate 2025 statistics for prize scams; they "
            "fall within the 'Other Scams' bucket. Legitimate prizes never "
            "require payment, personal banking details, or an OTP to claim."
        ),
    },
    # Fallback category when no other keywords match
    "unknown": {
        "spf_name": "Other Scam", "2025_cases": 9941, "2025_losses": 135.1,
        "avg_loss": 13590, "risk_tier": "MEDIUM", "risk_score": 55,
        "keywords": [],
        "advice": (
            "This message shows scam indicators. Do not send money, share personal "
            "details or OTPs, or click unknown links. "
            "Verify via ScamShield app or call 1799."
        ),
    },
}


# ---------------------------------------------------------------------------
# classify_scam_type(text) — keyword-based scam-type guesser
# ---------------------------------------------------------------------------
def classify_scam_type(text: str) -> tuple:
    """Match text against SPF taxonomy keywords.
    Returns (scam_type, risk_score, risk_tier, advice)."""
    text_lower = text.lower()                             # keyword regexes are matched case-insensitively
    scores = {}                                            # scam_type -> number of matched keywords
    for stype, info in SPF_TAXONOMY.items():
        if stype == "unknown":                              # "unknown" has no keywords, only a fallback
            continue
        hits = sum(1 for kw in info["keywords"] if re.search(kw, text_lower))
        if hits > 0:
            scores[stype] = hits
    if not scores:                                          # no category matched -> use the fallback category
        t = SPF_TAXONOMY["unknown"]
        return "unknown", t["risk_score"], t["risk_tier"], t["advice"]
    best = max(scores, key=scores.get)                      # category with the most keyword hits wins
    t = SPF_TAXONOMY[best]
    return best, t["risk_score"], t["risk_tier"], t["advice"]


# ---------------------------------------------------------------------------
# classify(text) — Detection agent
# ---------------------------------------------------------------------------
def classify(text: str) -> dict:
    """
    Detection step: language + scam/ham classification via XLM-RoBERTa.
    Returns: label, confidence, scam_prob, language, logits.
    """
    _require_ready()                                        # ensure init() has already loaded the model
    inputs = _clf_tokenizer(
        text, return_tensors="pt", truncation=True, max_length=128, padding=True
    ).to(DEVICE)                                            # tokenise and move to the model's device
    with torch.no_grad():                                   # no gradients needed for inference
        logits = _clf_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]  # convert logits to class probabilities
    pred_idx = int(probs.argmax())                          # index of the predicted class (0=ham, 1=scam)

    return {
        "label": LABEL_MAP[pred_idx],                       # "ham" or "scam"
        "confidence": round(float(probs[pred_idx]), 4),     # probability of the predicted class
        "scam_prob": round(float(probs[1]), 4),             # probability of the "scam" class specifically
        "language": detect_language(text),                  # detected message language
        "logits": logits.cpu().numpy()[0].tolist(),         # raw logits, useful for debugging/logging
    }


# ---------------------------------------------------------------------------
# get_risk(scam_type, scam_prob) — Risk agent
# ---------------------------------------------------------------------------
def get_risk(scam_type: str, scam_prob: float = 1.0) -> dict:
    """
    Risk scoring: maps a scam_type to its SPF advisory + a confidence-adjusted
    risk score/tier. `scam_prob` should be the classifier's scam probability
    (0-1); pass 1.0 to just get the taxonomy's base score.
    """
    info = SPF_TAXONOMY.get(scam_type, SPF_TAXONOMY["unknown"])   # fall back to "unknown" if type not found
    base_score = info["risk_score"]                          # the taxonomy's fixed base risk score

    adjusted_score = max(10, min(100, int(base_score * scam_prob)))   # scale by confidence, clamp to 10-100
    if adjusted_score >= 85:
        risk_tier = "CRITICAL"
    elif adjusted_score >= 65:
        risk_tier = "HIGH"
    elif adjusted_score >= 40:
        risk_tier = "MEDIUM"
    else:
        risk_tier = "LOW"

    return {
        "scam_type": scam_type,
        "spf_name": info["spf_name"],                       # official SPF category name
        "spf_cases_2025": info["2025_cases"],                # SPF-reported case count for 2025
        "spf_losses_2025": info["2025_losses"],              # SPF-reported losses (SGD millions) for 2025
        "avg_loss_sgd": info["avg_loss"],                    # average loss per victim (SGD)
        "risk_score": adjusted_score,                        # confidence-adjusted 10-100 risk score
        "risk_tier": risk_tier,                              # CRITICAL / HIGH / MEDIUM / LOW
        "advice": info["advice"],                            # human-readable SPF-grounded advice
    }


# ---------------------------------------------------------------------------
# explain(text) — Explanation agent (SHAP + RAG)
# ---------------------------------------------------------------------------
def get_top_shap_features(text: str, top_n: int = 5) -> list[dict]:
    """Top tokens (by SHAP value) pushing the model toward 'scam'."""
    _require_ready()                                        # ensure init() has already loaded the explainer
    shap_values = _shap_explainer([text])                    # compute SHAP values for this one text
    token_names = shap_values.data[0]                        # the tokens the text was split into
    token_shaps = shap_values.values[0, :, 1]                # each token's contribution toward "scam"
    pairs = [
        {"token": tok, "shap_value": round(float(val), 4)}
        for tok, val in zip(token_names, token_shaps)
        if tok not in ["", "▁", "<s>", "</s>", "<pad>"]      # skip empty/special tokenizer artifacts
    ]
    return sorted(pairs, key=lambda x: x["shap_value"], reverse=True)[:top_n]  # highest-impact tokens first


def retrieve_spf_passages(message: str, scam_type: Optional[str] = None, top_k: int = 3) -> list[dict]:
    """
    Retrieve top-k relevant SPF advisory passages for a message.
    If scam_type is given, combines scam-type-specific results with general
    top results (deduplicated). Otherwise does a plain top-k search.
    """
    _require_ready()                                        # ensure init() has already loaded the FAISS index
    q_embed = _embedder.encode([message], normalize_embeddings=True).astype(np.float32)  # embed the query

    if not scam_type:                                        # no hint given: plain top-k search over everything
        scores, idxs = _faiss_index.search(q_embed, top_k)
        return [
            {**SPF_CORPUS[i], "score": round(float(s), 4)}
            for s, i in zip(scores[0], idxs[0]) if i < len(SPF_CORPUS)
        ]

    type_indices = [i for i, c in enumerate(SPF_CORPUS) if c["scam_type"] == scam_type]  # rows for this type
    if not type_indices:                                      # no chunks tagged with this type: search everything
        type_indices = list(range(len(SPF_CORPUS)))

    type_embeds = _corpus_embeddings[type_indices]            # embeddings restricted to this scam type
    type_scores = (q_embed @ type_embeds.T)[0]                # cosine similarity (embeddings are normalized)
    top_type = np.argsort(type_scores)[::-1][:top_k]          # indices of the top-k type-specific matches

    results, seen = [], set()                                 # seen tracks ids already added, to avoid duplicates
    for local_i in top_type:
        chunk = SPF_CORPUS[type_indices[local_i]].copy()
        chunk["score"] = round(float(type_scores[local_i]), 4)
        if chunk["id"] not in seen:
            seen.add(chunk["id"])
            results.append(chunk)

    gen_scores, gen_idx = _faiss_index.search(q_embed, 2)      # also pull in a couple of general top matches
    for s, i in zip(gen_scores[0], gen_idx[0]):
        chunk = SPF_CORPUS[i].copy()
        chunk["score"] = round(float(s), 4)
        if chunk["id"] not in seen:
            seen.add(chunk["id"])
            results.append(chunk)

    return results[: top_k + 1]                                # cap the combined result list


_RISK_ADVICE_BY_TIER = {                                    # generic fallback advice, keyed by risk tier
    "CRITICAL": "Do NOT transfer any money. Report immediately to SPF at www.police.gov.sg/iwitness or call 1800-255-0000.",
    "HIGH": "Exercise extreme caution. Verify independently before taking any action. Report suspected scams to SPF.",
    "MEDIUM": "Be cautious. Do not share personal information or make payments without verification.",
    "LOW": "Stay alert. If something feels off, trust your instincts and verify through official channels.",
    "NONE": "This message appears legitimate. Always stay vigilant against scams.",   # used for confident "ham"
}


def explain(
    text: str,
    scam_type_hint: Optional[str] = None,
    top_n: int = 5,
    top_k: int = 3,
) -> dict:
    """
    Explanation step: SHAP token attribution + FAISS/RAG retrieval over the
    SPF advisory corpus, plus a scam_type guess (from keyword matching) and a
    human-readable, RAG-grounded explanation string.

    If scam_type_hint is not given, the scam type is guessed via
    classify_scam_type() (keyword match) purely to steer retrieval.
    """
    _require_ready()                                        # ensure init() has already loaded everything
    top_features = get_top_shap_features(text, top_n=top_n)  # tokens driving the "scam" prediction

    scam_type = scam_type_hint or classify_scam_type(text)[0]   # use given hint, else guess via keywords
    rag_chunks = retrieve_spf_passages(text, scam_type=scam_type, top_k=top_k)  # relevant SPF passages

    key_tokens = ", ".join(f"'{f['token']}'" for f in top_features[:3]) if top_features else "—"
    best_chunk = rag_chunks[0] if rag_chunks else None       # most relevant retrieved passage, if any

    if best_chunk:
        page_label = f"p.{best_chunk['source_page']}" if best_chunk.get("source_page") else "advisory (page n/a)"
        rag_summary = (                                       # human-readable summary combining SHAP + RAG
            f"Key suspicious tokens: {key_tokens}. "
            f"SPF 2025 ({page_label}): {best_chunk['text'][:220].rstrip()}..."
        )
    else:
        rag_summary = f"Key suspicious tokens: {key_tokens}. No SPF passage retrieved."

    sources = [                                               # citation strings for the UI, one per chunk
        f"SPF 2025 Annual Scams Brief — {c['topic']} (p.{c.get('source_page', '?')})"
        for c in rag_chunks
    ]

    return {
        "scam_type": scam_type,
        "top_features": top_features,
        "rag_chunks": rag_chunks,
        "rag_summary": rag_summary,
        "sources": sources,
    }


# ---------------------------------------------------------------------------
# run_pipeline(text) — the full classify -> explain -> risk flow in one call
# ---------------------------------------------------------------------------
def run_pipeline(text: str, ham_safe_exit_confidence: float = 0.80) -> dict:
    """
    Full pipeline: classify -> (skip if clearly ham) -> explain -> get_risk.
    Mirrors the LangGraph routing in NB04 without requiring LangGraph itself,
    so it's usable as a plain function call (e.g. from FastAPI in NB05).
    """
    _require_ready()                                        # ensure init() has already loaded everything
    det = classify(text)                                     # Step 1: language + scam/ham classification

    if det["label"] == "ham" and det["confidence"] >= ham_safe_exit_confidence:
        # Confidently "ham": skip the expensive explain/risk steps and return a minimal safe result
        return {
            "message": text,
            "language": det["language"],
            "label": "ham",
            "confidence": det["confidence"],
            "scam_prob": det["scam_prob"],
            "top_features": [],
            "rag_chunks": [],
            "rag_summary": "",
            "sources": [],
            "scam_type": None,
            "spf_name": None,
            "spf_cases_2025": None,
            "spf_losses_2025": None,
            "avg_loss_sgd": None,
            "risk_score": 0,
            "risk_tier": "NONE",
            "advice": _RISK_ADVICE_BY_TIER["NONE"],
        }

    exp = explain(text)                                       # Step 2: SHAP + RAG explanation
    risk = get_risk(exp["scam_type"], det["scam_prob"])        # Step 3: confidence-adjusted risk scoring

    return {
        "message": text,
        "language": det["language"],
        "label": det["label"],
        "confidence": det["confidence"],
        "scam_prob": det["scam_prob"],
        "top_features": exp["top_features"],
        "rag_chunks": exp["rag_chunks"],
        "rag_summary": exp["rag_summary"],
        "sources": exp["sources"],
        **risk,                                                # merge in scam_type/spf_name/risk_score/etc.
    }


# ---------------------------------------------------------------------------
# Default SPF advisory corpus (used only if init() isn't given a corpus_path)
# 40 passages extracted from the SPF Annual Scam and Cybercrime Brief 2025.
# ---------------------------------------------------------------------------
_DEFAULT_SPF_CORPUS = [
    {"id": "spf_001", "scam_type": "overview", "topic": "Overall scam situation 2025", "source_page": 1,
     "text": ("In 2025, scam and cybercrime cases decreased by 24.8% to 41,974 cases. "
              "Scam cases fell by 27.6% to 37,308 cases. Total losses fell 17.9% to $913.1 million. "
              "Despite the decrease, the situation remains very concerning.")},
    {"id": "spf_002", "scam_type": "overview", "topic": "Self-effected transfers 2025", "source_page": 4,
     "text": ("81.8% of scams involved self-effected transfers — scammers manipulated victims into "
              "performing monetary transactions through deception and social engineering, "
              "without gaining direct control of victims' accounts.")},
    {"id": "spf_003", "scam_type": "overview", "topic": "Cryptocurrency losses 2025", "source_page": 3,
     "text": ("Cryptocurrency losses accounted for $182.2 million (about 20% of total scam losses) "
              "in 2025. Tether, Ethereum, and Bitcoin accounted for 91.7% of cryptocurrency losses. "
              "Crypto transactions are irreversible, making recovery very challenging.")},
    {"id": "spf_004", "scam_type": "overview", "topic": "Top scam types by cases 2025", "source_page": 4,
     "text": ("Top 5 scam types by cases in 2025: e-commerce (6,703 cases, 18.0%), "
              "phishing (6,264 cases, 16.8%), job scams (5,575 cases, 14.9%), "
              "investment scams (5,462 cases, 14.6%), government impersonation (3,363 cases, 9.0%).")},
    {"id": "spf_005", "scam_type": "overview", "topic": "Top scam types by losses 2025", "source_page": 5,
     "text": ("Top 5 scam types by losses in 2025: investment scams ($336.2M, 36.8%), "
              "government impersonation ($242.9M, 26.6%), job scams ($123.5M, 13.5%), "
              "phishing ($39.9M, 4.4%), business email compromise ($35.3M, 3.9%).")},
    {"id": "spf_006", "scam_type": "investment", "topic": "Investment scam statistics 2025", "source_page": 7,
     "text": ("Investment scams recorded the highest losses in 2025: $336.2 million (+4.8% from 2024). "
              "There were 5,462 cases. Average loss per victim: $61,559 — highest of all scam types.")},
    {"id": "spf_007", "scam_type": "investment", "topic": "Investment scam tactics — platforms", "source_page": 18,
     "text": ("Victims encountered investment opportunities via Telegram and Facebook chat groups, "
              "online ads, and recommendations from online contacts. They were shown false testimonies "
              "and instructed to transfer money to bank accounts or cryptocurrency wallets.")},
    {"id": "spf_008", "scam_type": "investment", "topic": "Investment scam — crypto tactics", "source_page": 18,
     "text": ("Investment scams accounted for 38.4% of all crypto-related scam cases in 2025. "
              "Victims were directed to open crypto accounts, fund them from bank accounts, "
              "then transfer cryptocurrency to scammer-controlled wallets.")},
    {"id": "spf_009", "scam_type": "investment", "topic": "Investment scam — fake apps", "source_page": 19,
     "text": ("Scammers directed victims to download fake investment apps showing fictitious profits. "
              "Victims only discovered the scam when they attempted to withdraw returns and "
              "were asked to pay increasingly large 'fees' or 'taxes'.")},
    {"id": "spf_010", "scam_type": "government_impersonation", "topic": "Government impersonation statistics 2025", "source_page": 6,
     "text": ("Government impersonation scams more than doubled in 2025 (+123.6% to 3,363 cases). "
              "Losses were $242.9 million (+60.5%). Average loss per victim: $72,229 — highest of all types.")},
    {"id": "spf_011", "scam_type": "government_impersonation", "topic": "Government impersonation — bank transfer tactic", "source_page": 17,
     "text": ("91.7% of government impersonation cases: victims received unsolicited calls from scammers "
              "posing as bank representatives, then were transferred to fake government officials who "
              "accused them of money laundering and told them to transfer funds to 'safety accounts'.")},
    {"id": "spf_012", "scam_type": "government_impersonation", "topic": "What Singapore government officials will never do", "source_page": 18,
     "text": ("Singapore Government officials will NEVER ask you over a phone call to: transfer money, "
              "disclose banking login details, install apps from unofficial stores, or transfer "
              "your call to the Police. Never hand money or valuables to any unknown person.")},
    {"id": "spf_013", "scam_type": "government_impersonation", "topic": "Government impersonation — PayNow and crypto", "source_page": 17,
     "text": ("New 2025 tactics: scammers instructed victims to transfer funds via PayNow to "
              "Payment Service Provider accounts (e.g. YouTrip), or to open new crypto accounts "
              "and transfer cryptocurrency directly to scammer-controlled wallets.")},
    {"id": "spf_031", "scam_type": "bank_impersonation", "topic": "Bank impersonation as a phishing sub-pattern", "source_page": 7,
     "text": ("SPF does not track bank impersonation as a standalone scam type; it is captured within "
              "Phishing Scam statistics. Victims receive fake alerts claiming their bank account is "
              "frozen, suspended, or flagged, and are directed to a fraudulent login page to 'verify' "
              "their identity, resulting in credential or card theft.")},
    {"id": "spf_032", "scam_type": "bank_impersonation", "topic": "Bank impersonation — never share OTP", "source_page": 7,
     "text": ("Banks will never ask a customer to verify their account, provide an OTP, or confirm login "
              "credentials via a link sent by SMS or email. Genuine account issues are addressed by "
              "logging in directly through the bank's official app, not through a message link.")},
    {"id": "spf_014", "scam_type": "job_scam", "topic": "Job scam statistics 2025", "source_page": 16,
     "text": ("Job scams: 5,575 cases in 2025 (down 38.4% from 2024). Total losses: $123.5 million. "
              "Average loss per victim: $22,163. Third highest by both cases and losses.")},
    {"id": "spf_015", "scam_type": "job_scam", "topic": "Job scam tactics — upfront fees and fake tasks", "source_page": 16,
     "text": ("Job scammers advertise easy work-from-home jobs with high pay on social media. "
              "Victims are asked to pay registration fees, training fees, or deposits before starting. "
              "Some involve fake online tasks (liking posts, boosting products) with small initial payments "
              "to build trust, before large sums are requested and never returned.")},
    {"id": "spf_016", "scam_type": "job_scam", "topic": "Job scam — victim demographics", "source_page": 9,
     "text": ("Job scams disproportionately targeted young adults aged 20–29 (19.9% of all victims) "
              "and adults aged 30–49 (17.6% falling prey to job scams). "
              "Legitimate employers do not ask for upfront fees before employment.")},
    {"id": "spf_017", "scam_type": "phishing", "topic": "Phishing scam statistics 2025", "source_page": 7,
     "text": ("Phishing scams: 6,264 cases in 2025 (down 26.8% from 2024). "
              "Total losses: $39.9 million (down 32.8%). Average loss per victim: $6,384.")},
    {"id": "spf_018", "scam_type": "phishing", "topic": "Phishing tactics — card details and OTP theft", "source_page": 7,
     "text": ("Phishing scams predominantly involved victims submitting card details and OTPs "
              "on fake websites impersonating banks, government agencies, or e-commerce platforms, "
              "resulting in unauthorised card transactions.")},
    {"id": "spf_019", "scam_type": "phishing", "topic": "Phishing contact methods", "source_page": 11,
     "text": ("Phishing scammers contact victims via SMS, email, and messaging platforms with links "
              "to fraudulent websites. Major retail banks have phased out SMS OTPs for digital token users. "
              "Never click links in unsolicited messages — go directly to your bank's official app or website.")},
    {"id": "spf_020", "scam_type": "ecommerce", "topic": "E-commerce scam statistics 2025", "source_page": 6,
     "text": ("E-commerce scams: 6,703 cases in 2025 (down 42.5% from 2024, but still the most cases). "
              "Total losses: $16.7 million. Average loss per victim: $2,503.")},
    {"id": "spf_021", "scam_type": "ecommerce", "topic": "E-commerce scam platforms and payment", "source_page": 6,
     "text": ("Carousell (29.0%) and Facebook Marketplace (22.2%) were the most used platforms. "
              "Victims were asked to pay via PayNow or bank transfer upfront. "
              "They discovered the scam when goods were not delivered and sellers became uncontactable.")},
    {"id": "spf_022", "scam_type": "ecommerce", "topic": "E-commerce scam — most common items", "source_page": 6,
     "text": ("Pokemon trading cards were the most commonly scammed item (13.6% of e-commerce cases). "
              "Other items: electronics, concert tickets, luxury goods. "
              "Use platforms' protected payment and meet sellers in person for high-value items.")},
    {"id": "spf_033", "scam_type": "fake_friend", "topic": "Fake friend call scam statistics 2025", "source_page": 16,
     "text": ("Fake friend call scams: 1,551 cases in 2025, down sharply from 4,179 cases in 2024. "
              "Total losses fell to $4.7 million, from $13.6 million in 2024. "
              "Average loss per victim: $3,056. One of the scam types with the largest year-on-year decrease.")},
    {"id": "spf_034", "scam_type": "fake_friend", "topic": "Fake friend call scam — verify before transferring", "source_page": 16,
     "text": ("Scammers claiming to be a friend or family member with a 'new number' typically request "
              "urgent money transfers citing an emergency, without giving the victim time to verify. "
              "Always call the person's known number, or contact them through another channel, before "
              "transferring any money.")},
    {"id": "spf_035", "scam_type": "loan", "topic": "Loan scam statistics 2025", "source_page": 16,
     "text": ("Loan scams: 935 cases in 2025, down from 1,154 cases in 2024. "
              "Total losses rose to $7.0 million, from $6.0 million in 2024 — one of the few scam types "
              "where losses increased despite fewer cases. Average loss per victim: $7,515.")},
    {"id": "spf_036", "scam_type": "loan", "topic": "Loan scam — verify licensed moneylenders", "source_page": 16,
     "text": ("Legitimate licensed moneylenders never guarantee approval without checks, and never "
              "request an upfront 'processing' or 'unlocking' fee before disbursing a loan. Verify any "
              "moneylender against the Ministry of Law's registry before proceeding.")},
    {"id": "spf_037", "scam_type": "parcel_delivery", "topic": "Parcel delivery phishing — not separately tracked in 2025 Brief", "source_page": None,
     "text": ("Parcel delivery scams are not broken out separately in the SPF Annual Scam and Cybercrime "
              "Brief 2025; they fall within the broader 'Other Scams' bucket. A dedicated SPF advisory "
              "(September 2024) recorded 338+ cases and $616,000+ in losses since January 2024, mostly "
              "impersonating SingPost with fake failed-delivery messages.")},
    {"id": "spf_038", "scam_type": "parcel_delivery", "topic": "Parcel delivery scam tactic", "source_page": None,
     "text": ("Victims receive a message claiming a parcel delivery failed and are asked to click a link "
              "to 'confirm their address' or pay a small clearance fee. The link leads to a phishing page "
              "that harvests card details rather than a legitimate courier site.")},
    {"id": "spf_039", "scam_type": "rental", "topic": "Rental scam — not separately tracked in 2025 Brief", "source_page": None,
     "text": ("Rental scams are not broken out separately in the SPF Annual Scam and Cybercrime Brief "
              "2025; they fall within the broader 'Other Scams' bucket. SPF enforcement operations in "
              "late 2025 noted rental scams among the mix of case types investigated alongside "
              "e-commerce and impersonation scams.")},
    {"id": "spf_040", "scam_type": "rental", "topic": "Rental scam tactic", "source_page": None,
     "text": ("Scammers advertise attractively priced rooms or units and pressure prospective tenants "
              "to transfer a deposit quickly, citing high demand, before any viewing takes place. "
              "Never transfer a deposit before viewing the property and verifying ownership.")},
    {"id": "spf_041", "scam_type": "charity", "topic": "Charity scam — not separately tracked in 2025 Brief", "source_page": None,
     "text": ("Charity scams are not broken out separately in the SPF Annual Scam and Cybercrime Brief "
              "2025; they fall within the broader 'Other Scams' bucket. Donate only through registered "
              "charities listed on the Charity Portal, and be wary of urgent, unsolicited donation "
              "requests received via messaging apps.")},
    {"id": "spf_042", "scam_type": "prize", "topic": "Prize scam — not separately tracked in 2025 Brief", "source_page": None,
     "text": ("Prize scams are not broken out separately in the SPF Annual Scam and Cybercrime Brief "
              "2025; they fall within the broader 'Other Scams' bucket. Legitimate lucky draws and "
              "prizes never require payment, banking details, or an OTP to release winnings.")},
    {"id": "spf_025", "scam_type": "overview", "topic": "Online platforms used by scammers 2025", "source_page": 8,
     "text": ("Scammers used online platforms in 84.1% of all cases. Meta platforms: 35.4% of cases. "
              "Top contact methods: social media (10,448 cases), messaging platforms (9,355), "
              "phone calls (5,477), online shopping platforms (3,804).")},
    {"id": "spf_026", "scam_type": "overview", "topic": "WhatsApp and Telegram most used for scam contact", "source_page": 8,
     "text": ("Among messaging platforms: WhatsApp 53.5%, Telegram 37.9%, Facebook Messenger 4.2%. "
              "Among social media: Facebook 51.9%, TikTok 26.0%, Instagram 14.2% of scam contact cases.")},
    {"id": "spf_027", "scam_type": "overview", "topic": "Scam victim age profile 2025", "source_page": 9,
     "text": ("85.2% of scam victims were aged below 65. Adults aged 30–49: 36.1% of victims ($22,283 avg loss). "
              "Elderly aged 65+: 14.8% with the highest average loss of $37,053, mainly from investment, "
              "impersonation, and phishing scams.")},
    {"id": "spf_028", "scam_type": "overview", "topic": "ScamShield helpline and reporting channels", "source_page": 13,
     "text": ("When in doubt, call ScamShield Helpline at 1799 or use the ScamShield app. "
              "Report scams to SPF at 1800-255-0000 or police.gov.sg/i-witness. "
              "Report phishing links to report@scamalert.sg. "
              "Check investment legitimacy at MAS Investor Alert List: mas.gov.sg.")},
    {"id": "spf_029", "scam_type": "overview", "topic": "Anti-scam laws — caning and Protection from Scams Act", "source_page": 10,
     "text": ("From 30 December 2025, caning for scam offences was operationalised (6–24 strokes). "
              "The Protection from Scams Act (1 July 2025) allows SPF to restrict banking facilities "
              "of scam victims who continue to believe scammers despite police engagement.")},
    {"id": "spf_030", "scam_type": "overview", "topic": "Money mules — do not share accounts or Singpass", "source_page": 13,
     "text": ("Do not allow anyone to use your financial accounts to transfer funds, share Singpass "
              "credentials, or supply local SIM cards. In 2025, 7,000+ money mules were investigated "
              "and 940+ charged. Penalties include substantial prison terms and caning.")},
]
