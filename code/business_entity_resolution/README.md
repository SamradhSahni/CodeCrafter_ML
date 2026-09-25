# 🏢 Business Entity Resolution — Amazon ML Challenge 2026

**Team CodeCrafter** | GPU-accelerated entity resolution pipeline using Sentence-Transformers + FAISS + LightGBM.

## Architecture

```
Phase 1: Preprocessing     → Multi-view normalization (name/address/URL/script)
Phase 2: Embedding (GPU)   → Sentence-Transformer (all-MiniLM-L6-v2) dense vectors
Phase 3: Blocking          → FAISS top-K + TF-IDF + Phonetic + Rare-token + Postal + City
Phase 4: Features          → 74 features (name sim, address, blocker provenance, embedding, IDF, quality)
Phase 5: Training          → LightGBM (GPU) + OOF hard negatives + two-stage density refinement
Phase 6: Post-processing   → GT-derived constraint resolution + per-country thresholds
```

## Quick Start (GPU Lab Machine)

### 1. Clone & Setup

```bash
git clone https://github.com/SamradhSahni/CodeCrafter_ML.git
cd CodeCrafter_ML
```

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

# Install PyTorch with CUDA (for H200/A100/etc.)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install all other dependencies
pip install -r code/business_entity_resolution/requirements.txt
```

### 3. Place Dataset

```
CodeCrafter_ML/
└── student_resource/
    └── dataset/
        ├── train/
        │   ├── train_source1.tsv
        │   ├── train_source2.tsv
        │   ├── train_source3.tsv
        │   └── train_ground_truth.tsv
        └── test/
            ├── test_source1.tsv
            ├── test_source2.tsv
            └── test_source3.tsv
```

### 4. Run Full Pipeline

```bash
cd code/business_entity_resolution
python -m src.pipeline --phase full
```

### 5. Get Results

```
CodeCrafter_ML/output/matching_results.tsv      ← Submit this
CodeCrafter_ML/output/candidate_pairs.tsv        ← Blocking candidates
```

## Run Individual Phases

```bash
# Phase 1: Preprocess all sources (saves to cache/)
python -m src.pipeline --phase preprocess

# Phase 2: Generate embeddings on GPU (saves .npy to cache/)
python -m src.pipeline --phase embed

# Phase 3: Run blocking (FAISS + TF-IDF hybrid)
python -m src.pipeline --phase blocking

# Phase 4: Compute features
python -m src.pipeline --phase features

# Phase 5: Train model
python -m src.pipeline --phase train

# Phase 6: Run test inference
python -m src.pipeline --phase inference
```

## Debug Mode (small subset)

```bash
# Run with 5000 rows per source for quick testing
python -m src.pipeline --phase full --nrows 5000
```

## One-Liner (Full Pipeline)

```bash
git clone https://github.com/SamradhSahni/CodeCrafter_ML.git && cd CodeCrafter_ML && pip install torch --index-url https://download.pytorch.org/whl/cu124 && pip install -r code/business_entity_resolution/requirements.txt && cd code/business_entity_resolution && python -m src.pipeline --phase full
```

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU       | NVIDIA T4 (16GB) | H200/A100 (80GB) |
| RAM       | 32 GB   | 128 GB |
| Storage   | 50 GB   | 100 GB |
| Python    | 3.10+   | 3.12 |
| CUDA      | 12.0+   | 12.4+ |

## Project Structure

```
code/business_entity_resolution/
├── requirements.txt
└── src/
    ├── __init__.py
    ├── config.py           # Paths, hyperparameters, GPU detection
    ├── preprocessing.py    # Multi-view text normalization
    ├── embeddings.py       # Sentence-Transformer + FAISS (GPU)
    ├── blocking.py         # TF-IDF + phonetic + rare token blocking
    ├── features.py         # 74 features across 9 categories
    ├── model.py            # LightGBM (GPU) + two-stage refinement
    ├── postprocessing.py   # Constraint resolution + thresholds
    ├── pipeline.py         # End-to-end orchestrator
    └── utils.py            # I/O, F0.5 evaluation, helpers
```

## Key Technical Decisions

1. **Hybrid blocking**: Dense (FAISS) captures semantic similarity; sparse (TF-IDF) catches character-level matches — union gives highest recall
2. **Independent reciprocal retrieval**: Both S1→S2S3 and S2S3→S1 searches run independently for legitimate reciprocal rank features
3. **OOF density features**: Stage-2 uses out-of-fold Stage-1 scores to prevent information leakage
4. **URL-as-name preservation**: URL business names (e.g., `maurewilliamscolombier.com`) extract domain stem for matching
5. **France generalization**: No hardcoded country logic; French suffixes (SARL, SAS) and address formats handled generically
6. **Three-state evidence**: Address components scored as match(+1)/missing(0)/conflict(-1) — conflicts are strong negatives

## Metric

**Macro F0.5** — precision-weighted. Correctly predicting singletons (empty set) = 1.0 for that entity.
