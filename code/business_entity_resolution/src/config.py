"""
Configuration module — all hyperparameters, paths, and constants.
"""
import os
from pathlib import Path

# ──────────────────────────────────────────────
#  Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # .../src/config.py -> Amazon_ML_Challenge
STUDENT_RESOURCE = PROJECT_ROOT / "student_resource"
DATASET_DIR = STUDENT_RESOURCE / "dataset"

TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

TRAIN_S1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_S2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_S3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GT = TRAIN_DIR / "train_ground_truth.tsv"

TEST_S1 = TEST_DIR / "test_source1.tsv"
TEST_S2 = TEST_DIR / "test_source2.tsv"
TEST_S3 = TEST_DIR / "test_source3.tsv"

OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
#  GPU Detection
# ──────────────────────────────────────────────
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
    DEVICE = "cuda" if HAS_CUDA else "cpu"
    GPU_NAME = torch.cuda.get_device_name(0) if HAS_CUDA else "CPU"
except ImportError:
    HAS_CUDA = False
    DEVICE = "cpu"
    GPU_NAME = "CPU (torch not installed)"

# ──────────────────────────────────────────────
#  Embedding Model Config
# ──────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, fast, reliable
EMBEDDING_DIM = 384
EMBEDDING_BATCH_SIZE = 1024 if HAS_CUDA else 256
FAISS_TOP_K = 10                     # top-K from dense retrieval (top 10 dense candidates per entity)
FAISS_NPROBE = 32                    # IVF search granularity (fast CPU search)
FAISS_MIN_SIMILARITY = 0.50          # empirical 98.3% coverage of true matches; prunes noise pairs
BLOCKING_MAX_CANDS_PER_ENTITY = 50   # max candidates per S1 entity across all blockers
INVERTED_INDEX_MAX_BUCKET = 200      # max bucket size to prevent non-selective combinatorial explosion

# ──────────────────────────────────────────────
#  Legal suffixes vs generic business tokens
# ──────────────────────────────────────────────
LEGAL_SUFFIXES = {
    # US
    'llc', 'inc', 'corp', 'corporation', 'co', 'lp', 'llp',
    # India
    'pvt', 'private', 'ltd', 'limited', 'opc',
    # France
    'sarl', 'sas', 'sasu', 'sa', 'eurl', 'sci', 'snc', 'selarl', 'ste',
    # Other
    'dba', 'company',
}

GENERIC_BUSINESS_TOKENS = {
    'group', 'holdings', 'enterprises', 'foundation', 'associates',
    'partners', 'solutions', 'services', 'international', 'industries',
    'technologies', 'consultants', 'agency', 'studio', 'ventures',
}

# ──────────────────────────────────────────────
#  Address abbreviation mappings
# ──────────────────────────────────────────────
ADDRESS_ABBREVIATIONS = {
    # English
    'rd': 'road', 'st': 'street', 'ave': 'avenue', 'blvd': 'boulevard',
    'ct': 'court', 'dr': 'drive', 'apt': 'apartment', 'ste': 'suite',
    'hwy': 'highway', 'ln': 'lane', 'pl': 'place', 'pkwy': 'parkway',
    'cir': 'circle', 'trl': 'trail', 'ter': 'terrace', 'sq': 'square',
    'mt': 'mount', 'ft': 'fort', 'twp': 'township',
    # French
    'r': 'rue', 'av': 'avenue', 'bd': 'boulevard', 'bvd': 'boulevard',
    'imp': 'impasse', 'rte': 'route', 'chem': 'chemin',
}

US_STATE_ABBREV = {
    'al': 'alabama', 'ak': 'alaska', 'az': 'arizona', 'ar': 'arkansas',
    'ca': 'california', 'co': 'colorado', 'ct': 'connecticut', 'de': 'delaware',
    'fl': 'florida', 'ga': 'georgia', 'hi': 'hawaii', 'id': 'idaho',
    'il': 'illinois', 'in': 'indiana', 'ia': 'iowa', 'ks': 'kansas',
    'ky': 'kentucky', 'la': 'louisiana', 'me': 'maine', 'md': 'maryland',
    'ma': 'massachusetts', 'mi': 'michigan', 'mn': 'minnesota', 'ms': 'mississippi',
    'mo': 'missouri', 'mt': 'montana', 'ne': 'nebraska', 'nv': 'nevada',
    'nh': 'new hampshire', 'nj': 'new jersey', 'nm': 'new mexico', 'ny': 'new york',
    'nc': 'north carolina', 'nd': 'north dakota', 'oh': 'ohio', 'ok': 'oklahoma',
    'or': 'oregon', 'pa': 'pennsylvania', 'ri': 'rhode island', 'sc': 'south carolina',
    'sd': 'south dakota', 'tn': 'tennessee', 'tx': 'texas', 'ut': 'utah',
    'vt': 'vermont', 'va': 'virginia', 'wa': 'washington', 'wv': 'west virginia',
    'wi': 'wisconsin', 'wy': 'wyoming', 'dc': 'district of columbia',
}

# ──────────────────────────────────────────────
#  TF-IDF Blocking (secondary retrieval)
# ──────────────────────────────────────────────
TFIDF_NAME_NGRAM_RANGE = (3, 5)
TFIDF_NAME_MAX_FEATURES = 500_000
TFIDF_ADDR_MAX_FEATURES = 300_000
TFIDF_MIN_SIMILARITY = 0.3
TFIDF_REVERSE_TOPK = 20

DEFAULT_K_CONFIG = {
    'unique_idf_thresh': 8.0,
    'normal_idf_thresh': 5.0,
    'common_idf_thresh': 3.0,
    'unique_k': 15,
    'normal_k': 30,
    'common_k': 50,
    'max_k': 100,
}

# ──────────────────────────────────────────────
#  Model configuration — GPU LightGBM
# ──────────────────────────────────────────────
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 127,
    'max_depth': 8,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'n_estimators': 2000,
    'verbose': -1,
    'n_jobs': -1,
    'random_state': 42,
}

# Enable GPU for LightGBM only if the GPU build is available
if HAS_CUDA:
    try:
        import lightgbm as _lgb
        _test_params = {'device': 'gpu', 'n_estimators': 1, 'num_leaves': 4, 'verbose': -1}
        _test_model = _lgb.LGBMClassifier(**_test_params)
        _test_model.fit([[1], [0]], [1, 0])
        LGBM_PARAMS['device'] = 'gpu'
        LGBM_PARAMS['gpu_use_dp'] = False
        del _test_model
    except Exception:
        pass  # LightGBM GPU not available, use CPU

VAL_SPLIT = 0.15
RANDOM_SEED = 42
N_FOLDS_OOF = 5

# ──────────────────────────────────────────────
#  IDF rarity thresholds
# ──────────────────────────────────────────────
RARE_TOKEN_IDF_THRESHOLD = 6.0
GENERIC_TOKEN_IDF_THRESHOLD = 3.0
