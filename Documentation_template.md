# Amazon ML Challenge 2026 — Methodology Document

**Team Name**: CodeCrafter  
**Members**: Samradh Sahni  
**Final Leaderboard Score**: _(to be filled after submission)_

---

## 1. Problem Understanding

### 1.1 Task Description
Given business records from 3 independent data sources with noisy and inconsistent fields, determine which records across Source 2 and Source 3 refer to the same real-world business entity as records in Source 1 (the deduplicated reference source).

### 1.2 Key Challenges
- **Scale**: S1=2.2M, S2=5M, S3=5.3M records (train); ~11.7M test records
- **Noise**: Typos, abbreviations, reorderings, DBA names, mixed scripts
- **Multi-lingual**: Latin, Devanagari, Tamil, Kannada, Gurmukhi scripts + French diacritics
- **Unseen country**: France appears only in test set (not train)
- **Null addresses**: ~170K records per source with missing addresses
- **URL-as-name**: Some S3 records use domains as business names
- **Singletons**: 5.6% of S1 entities have zero matches

### 1.3 Evaluation Metric
Macro F0.5 — precision-weighted harmonic mean. Correctly predicting singletons (empty set) scores 1.0 for that entity.

---

## 2. Architecture Overview

```
Phase 1: Preprocessing     → Multi-view normalization (6 views per record)
Phase 2: Embedding (GPU)   → Sentence-Transformer dense vectors (384-dim)
Phase 3: Blocking          → FAISS + TF-IDF + Phonetic + Rare-token + Postal + City
Phase 4: Features          → 74 features across 9 categories
Phase 5: Training          → LightGBM two-stage with OOF hard negatives
Phase 6: Post-processing   → Constraint resolution + per-country thresholds
```

---

## 3. Preprocessing (Phase 1)

### 3.1 Multi-View Design
We follow a **non-destructive** principle: never destroy information. Each record produces 6 parallel name views and 4 address views.

| View | Description | Used For |
|---|---|---|
| `name_raw` | Original string | Unicode features |
| `name_unicode` | NFKC-normalized, encoding fixed (ftfy) | Unicode-level comparisons |
| `name_transliterated` | Indic → Latin via unidecode | Cross-script matching |
| `name_clean` | Lowercased, punct normalized, URL-aware | Primary similarity |
| `name_core` | Legal suffixes removed | Core identity |
| `name_tokens_sorted` | Sorted token set | Blocking keys |

### 3.2 Key Normalization Steps
- **URL-as-name**: Domain stem extracted (e.g., `maurewilliamscolombier.com` → `maurewilliamscolombier`)
- **Legal vs. Generic separation**: Legal suffixes (LLC, Pvt, SARL) stripped; generic tokens (Holdings, Group) preserved in core but tracked separately
- **Period abbreviation**: `L.L.C` → `LLC`
- **Script detection**: Using `regex` Unicode properties (not `unicodedata.script`)
- **Phonetic key**: Pre-computed Metaphone for blocking speedup
- **Address components**: Postal code, city, state, street number extracted via regex heuristics only (no external databases)

### 3.3 Performance
- Joblib parallel processing (`n_jobs=-1`) for large datasets
- Process-and-cache to parquet for memory efficiency

---

## 4. Blocking (Phase 3)

### 4.1 Hybrid Strategy
We use a **union of 7 blocking strategies** to maximize recall while maintaining tractable pair counts.

| Strategy | Description | Purpose |
|---|---|---|
| **FAISS Dense** | Sentence-Transformer embeddings + IVF index | Semantic similarity |
| **TF-IDF Name** | Sparse cosine on name, adaptive top-K | Character-level name matches |
| **TF-IDF Address** | Sparse cosine on address | Address-based retrieval |
| **Phonetic** | Metaphone inverted index | Sound-alike matching |
| **Rare Token** | High-IDF tokens as blocking keys | Distinctive word overlap |
| **Postal+Prefix** | Postal code + 3-char name prefix | Geographic + name filter |
| **City+Bigram** | City + name character bigrams | City-aware matching |

### 4.2 Country Partitioning
Blocking runs per-country to prevent cross-country matches (verified <0.01% cross-country rate in training ground truth).

### 4.3 Independent Reciprocal Retrieval
Both forward (S1→S2/S3) and reverse (S2/S3→S1) retrievals run independently to produce legitimate reciprocal rank features.

---

## 5. Feature Engineering (Phase 4)

### 5.1 Feature Categories (74 total)

| Category | Count | Examples |
|---|---|---|
| **Name Similarity** | 14 | Jaro-Winkler, Levenshtein, token sort/set ratio, Jaccard, 3-gram overlap |
| **Address Similarity** | 15 | Three-state evidence (match/missing/conflict), postal/city/state relations |
| **Blocker Provenance** | 6 | Which blockers fired, how many, boolean hits |
| **Reciprocal Retrieval** | 5 | Forward rank, reverse rank, mutual top-K, reciprocal rank fusion |
| **Embedding Features** | 7 | Cosine similarity, embedding rank, mutual top-K in dense space |
| **IDF-Weighted** | 5 | Weighted Jaccard, max shared IDF, distinctive token overlap |
| **Record Quality** | 7 | Null address flags, URL-as-name, script compatibility, length ratios |
| **Suffix/Generic** | 3 | Suffix compatibility, generic token overlap/count |
| **Character N-gram** | 2 | 3-gram Jaccard on name, 4-gram on address |
| **Density (Stage 2)** | 10 | OOF Stage-1 score statistics per S1 entity |

### 5.2 Three-State Address Evidence
Address components use +1 (match) / 0 (missing) / −1 (conflict) scoring. Conflicts are strong negative signals — two businesses at different postal codes are almost certainly different entities.

---

## 6. Model Training (Phase 5)

### 6.1 LightGBM Configuration
- 1500 estimators, 256 leaves, learning rate 0.05
- GPU acceleration when available
- scale_pos_weight auto-computed from class imbalance

### 6.2 Two-Stage Architecture
1. **Stage 1**: LightGBM trained on raw features → produces OOF probability scores
2. **Density features**: Per-entity statistics (mean, std, max, gap, entropy) computed from Stage-1 OOF scores
3. **Stage 2**: LightGBM trained on features + density features → final predictions

### 6.3 Structured Negative Mining
- OOF hard negatives: High-scoring false positives from out-of-fold predictions re-weighted
- Augmented training with hard negatives for Pass 2

### 6.4 Threshold Optimization
- Global threshold optimized on macro F0.5 via grid search
- Per-country thresholds for US and India
- Fallback global threshold for unseen countries (France)

---

## 7. Post-Processing (Phase 6)

### 7.1 Constraint Resolution
- Each S2/S3 record maps to at most one S1 entity (verified from ground truth)
- Conflicts resolved by keeping the highest-probability assignment

### 7.2 Singleton Handling
- S1 entities with no candidates above threshold predicted as singletons (empty set)
- Correct singleton prediction contributes F0.5 = 1.0

---

## 8. Reproducibility

### 8.1 Hardware Used
- **GPU**: NVIDIA H200 (lab) / RTX 4050 (development)
- **RAM**: 128 GB (lab) / 16 GB (development)
- **Storage**: 100 GB

### 8.2 Run Instructions
```bash
git clone https://github.com/SamradhSahni/CodeCrafter_ML.git
cd CodeCrafter_ML
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r code/business_entity_resolution/requirements.txt
# Place dataset in student_resource/dataset/
cd code/business_entity_resolution
python -m src.pipeline --phase full
# Output: ../../output/matching_results.tsv
```

### 8.3 Runtime
| Phase | Time (H200) | Time (RTX 4050) |
|---|---|---|
| Preprocessing | ~20 min | ~2 hrs |
| Embeddings | ~10 min | ~2 hrs |
| Blocking | ~15 min | ~45 min |
| Features | ~30 min | ~90 min |
| Training | ~5 min | ~5 min |
| Inference | ~30 min | ~90 min |
| **Total** | **~2 hrs** | **~8 hrs** |

---

## 9. Key Design Decisions

1. **Hybrid blocking over pure dense**: Dense retrieval alone misses character-level matches; sparse retrieval alone misses semantic matches. Union gives highest recall.
2. **Non-destructive preprocessing**: Multiple normalization views prevent information loss — features pick the right view for each comparison type.
3. **Three-state evidence**: Distinguishing "missing" from "conflicting" address components provides critical negative signal.
4. **No external databases**: All city/state extraction uses regex heuristics and dictionaries built from training data only (compliance).
5. **Country-agnostic design**: No hardcoded country logic — French suffixes and address formats handled generically.
6. **Precision-oriented**: F0.5 rewards precision over recall — our threshold optimization and constraint resolution prioritize high-confidence matches.

---

## 10. Libraries & Versions

| Library | Version | Purpose |
|---|---|---|
| torch | ≥2.0 | GPU tensor operations |
| sentence-transformers | ≥2.7 | Dense embedding generation |
| faiss-cpu/gpu | ≥1.7 | Approximate nearest neighbor search |
| lightgbm | ≥4.5 | Gradient boosted decision tree classifier |
| rapidfuzz | ≥3.0 | Fast string similarity metrics |
| jellyfish | ≥1.0 | Phonetic algorithms (Metaphone) |
| pandas | ≥2.0 | Data manipulation |
| scikit-learn | ≥1.3 | TF-IDF vectorization, train/test split |
| unidecode | ≥1.3 | Transliteration (Indic → Latin) |
| ftfy | ≥6.0 | Unicode text fixing |
| regex | ≥2023 | Unicode-aware regex (script detection) |
| sparse_dot_topn | ≥1.0 | Efficient sparse matrix top-N |
| joblib | ≥1.3 | Parallel processing |
