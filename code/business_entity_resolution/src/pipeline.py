"""
End-to-end pipeline orchestrator for Business Entity Resolution.
GPU-accelerated with sentence-transformers + FAISS + LightGBM.

Usage:
    python -m src.pipeline                           # Full pipeline
    python -m src.pipeline --phase preprocess         # Only preprocessing
    python -m src.pipeline --phase embed              # Only embedding
    python -m src.pipeline --phase blocking           # Only blocking
    python -m src.pipeline --phase features           # Only features
    python -m src.pipeline --phase train              # Only training
    python -m src.pipeline --phase inference           # Only inference
    python -m src.pipeline --nrows 10000              # Debug subset
"""
import argparse
import gc
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config
from . import preprocessing as prep
from . import embeddings as emb
from . import blocking as block
from . import features as feat
from . import model as mdl
from . import postprocessing as post
from .utils import (
    logger, timed, load_source, load_ground_truth,
    compute_macro_f05, verify_country_gate,
    generate_output_files, free_memory,
)


# ──────────────────────────────────────────────
#  Phase 1: Preprocessing
# ──────────────────────────────────────────────
@timed
def run_preprocess(nrows=None):
    """Load and preprocess all source files — one at a time to save memory."""
    logger.info("=" * 60)
    logger.info("PHASE 1: PREPROCESSING")
    logger.info("=" * 60)

    cache = config.CACHE_DIR
    sources = [
        ("train_S1", config.TRAIN_S1, "pp_train_s1.parquet"),
        ("train_S2", config.TRAIN_S2, "pp_train_s2.parquet"),
        ("train_S3", config.TRAIN_S3, "pp_train_s3.parquet"),
        ("test_S1",  config.TEST_S1,  "pp_test_s1.parquet"),
        ("test_S2",  config.TEST_S2,  "pp_test_s2.parquet"),
        ("test_S3",  config.TEST_S3,  "pp_test_s3.parquet"),
    ]

    for label, path, parquet_name in sources:
        out_path = cache / parquet_name
        if out_path.exists() and nrows is None:
            logger.info(f"  {label}: cached, skipping")
            continue
        raw = load_source(path, nrows=nrows)
        pp = prep.preprocess_dataframe(raw, label)
        del raw
        free_memory()
        pp.to_parquet(out_path)
        logger.info(f"  Saved {out_path.name} ({len(pp):,} rows)")
        del pp
        free_memory()

    logger.info(f"All preprocessed data saved to {cache}")


# ──────────────────────────────────────────────
#  Phase 2: Embedding generation (GPU)
# ──────────────────────────────────────────────
@timed
def run_embedding_phase():
    """Compute sentence-transformer embeddings for all preprocessed sources.
    Memory-safe: encodes in 500K-row chunks, saves directly to .npy on disk.
    """
    logger.info("=" * 60)
    logger.info(f"PHASE 2: EMBEDDING GENERATION ({config.GPU_NAME})")
    logger.info("=" * 60)

    cache = config.CACHE_DIR
    CHUNK_SIZE = 500_000  # encode this many texts at a time

    # Check if all embeddings already cached
    emb_files = [
        "emb_train_s1.npy", "emb_train_s2.npy", "emb_train_s3.npy",
        "emb_test_s1.npy", "emb_test_s2.npy", "emb_test_s3.npy",
    ]
    if all((cache / f).exists() for f in emb_files):
        logger.info("All embeddings cached, skipping.")
        return

    # Load model once
    model = emb.load_embedding_model()

    pp_files = [
        ("pp_train_s1.parquet", "emb_train_s1.npy", "train_S1"),
        ("pp_train_s2.parquet", "emb_train_s2.npy", "train_S2"),
        ("pp_train_s3.parquet", "emb_train_s3.npy", "train_S3"),
        ("pp_test_s1.parquet",  "emb_test_s1.npy",  "test_S1"),
        ("pp_test_s2.parquet",  "emb_test_s2.npy",  "test_S2"),
        ("pp_test_s3.parquet",  "emb_test_s3.npy",  "test_S3"),
    ]

    for pp_file, emb_file, label in pp_files:
        emb_path = cache / emb_file
        if emb_path.exists():
            logger.info(f"  {label}: embedding cached, skipping")
            continue

        # Only load the columns we need
        df = pd.read_parquet(cache / pp_file, columns=["name_clean", "addr_clean"])
        n = len(df)
        texts = (df["name_clean"].fillna("") + " " + df["addr_clean"].fillna("")).tolist()
        del df
        free_memory()

        logger.info(f"  {label}: encoding {n:,} texts in chunks of {CHUNK_SIZE:,} ...")

        # Encode in chunks and save to temp files
        chunk_paths = []
        dim = None
        for start in range(0, n, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, n)
            chunk_texts = texts[start:end]
            logger.info(f"    Chunk {start:,}-{end:,} ...")

            chunk_emb = model.encode(
                [t if t and t.strip() else " " for t in chunk_texts],
                batch_size=config.EMBEDDING_BATCH_SIZE,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype(np.float32)

            if dim is None:
                dim = chunk_emb.shape[1]

            chunk_path = cache / f"_emb_chunk_{label}_{start}.npy"
            np.save(chunk_path, chunk_emb)
            chunk_paths.append(chunk_path)
            del chunk_emb, chunk_texts
            free_memory()

        del texts
        free_memory()

        # Merge chunks into final file using memory-mapped output
        logger.info(f"  Merging {len(chunk_paths)} chunks -> {emb_file} ...")
        # Pre-allocate output file as memory-mapped
        fp = np.lib.format.open_memmap(
            str(emb_path), mode='w+', dtype=np.float32, shape=(n, dim)
        )
        offset = 0
        for cp in chunk_paths:
            chunk = np.load(cp)
            fp[offset:offset + len(chunk)] = chunk
            offset += len(chunk)
            del chunk
            cp.unlink()
        fp.flush()
        del fp
        free_memory()
        logger.info(f"  -> {label}: saved {emb_path.name} ({n:,} x {dim})")

    del model
    free_memory()
    logger.info("All embeddings saved.")


# ──────────────────────────────────────────────
#  Phase 3: Blocking (FAISS + TF-IDF hybrid)
# ──────────────────────────────────────────────
@timed
def run_faiss_blocking():
    """Run FAISS embedding blocking with minimal memory footprint.
    Only loads entity IDs and embeddings — no full DataFrames.
    Saves results to cache for later use.
    """
    cache = config.CACHE_DIR
    faiss_cache = cache / "faiss_blocking.pkl"
    if faiss_cache.exists():
        logger.info("FAISS blocking cached, loading ...")
        import pickle
        with open(faiss_cache, "rb") as f:
            return pickle.load(f)

    logger.info("\n--- FAISS Embedding Blocking ---")

    # Load only entity IDs from parquet (not full DataFrames!)
    s1_ids = pd.read_parquet(cache / "pp_train_s1.parquet", columns=[]).index.values
    s2_ids = pd.read_parquet(cache / "pp_train_s2.parquet", columns=[]).index.values
    s3_ids = pd.read_parquet(cache / "pp_train_s3.parquet", columns=[]).index.values
    cand_ids = np.concatenate([s2_ids, s3_ids])
    del s2_ids, s3_ids

    logger.info(f"S1: {len(s1_ids):,}, Candidates: {len(cand_ids):,}")

    # Load embeddings via memory-mapping
    emb_s1 = np.load(cache / "emb_train_s1.npy", mmap_mode="r")
    emb_s2 = np.load(cache / "emb_train_s2.npy", mmap_mode="r")
    emb_s3 = np.load(cache / "emb_train_s3.npy", mmap_mode="r")
    n_s2s3 = len(emb_s2) + len(emb_s3)
    dim = emb_s1.shape[1]

    # Build combined S2+S3 embeddings via memmap file
    emb_s2s3_path = cache / "_emb_train_s2s3.npy"
    if not emb_s2s3_path.exists():
        logger.info(f"Building combined S2+S3 embedding ({n_s2s3:,} x {dim}) ...")
        fp = np.lib.format.open_memmap(
            str(emb_s2s3_path), mode='w+', dtype=np.float32, shape=(n_s2s3, dim)
        )
        fp[:len(emb_s2)] = emb_s2[:]
        fp[len(emb_s2):] = emb_s3[:]
        fp.flush()
        del fp
    del emb_s2, emb_s3
    free_memory()
    emb_s2s3 = np.load(str(emb_s2s3_path), mmap_mode="r")

    # FAISS blocking
    emb_candidates, emb_provenance = emb.run_embedding_blocking(
        s1_ids, cand_ids, emb_s1, emb_s2s3, topk=config.FAISS_TOP_K
    )

    # Reciprocal retrieval
    emb_reciprocal = emb.run_embedding_reciprocal(
        s1_ids, cand_ids, emb_s1, emb_s2s3, emb_provenance, topk=20
    )

    del emb_s1, emb_s2s3
    free_memory()

    # Cache FAISS results to disk
    import pickle
    result = (emb_candidates, emb_provenance, emb_reciprocal)
    with open(faiss_cache, "wb") as f:
        pickle.dump(result, f)
    logger.info(f"FAISS blocking saved: {len(emb_candidates):,} S1 entities with candidates")

    return result


@timed
def run_blocking_phase(pp_train_s1, pp_train_s2s3, ground_truth,
                       emb_candidates, emb_provenance, emb_reciprocal):
    """Run TF-IDF + inverted index blocking, merging with pre-computed FAISS results."""
    logger.info("=" * 60)
    logger.info("PHASE 3: BLOCKING (TF-IDF + INVERTED INDEX)")
    logger.info("=" * 60)

    # Country gate
    cross_rate = verify_country_gate(ground_truth, pp_train_s1, pp_train_s2s3)
    if cross_rate > 0:
        logger.warning(f"Cross-country rate = {cross_rate:.4%}")

    # Build token IDF
    all_names = pd.concat([pp_train_s1["name_core"], pp_train_s2s3["name_core"]])
    token_idf = block.build_token_idf(all_names)
    pd.Series(token_idf).to_frame("idf").to_parquet(config.CACHE_DIR / "token_idf.parquet")
    logger.info(f"Token IDF: {len(token_idf):,} tokens")

    # ── TF-IDF + inverted index blocking (per country) ──
    all_candidates = defaultdict(set)
    all_provenance = defaultdict(dict)
    all_reciprocal = {}

    # Merge embedding candidates into all_candidates
    for s1_id, cands in emb_candidates.items():
        all_candidates[s1_id].update(cands)
    logger.info(f"After FAISS: {sum(len(v) for v in all_candidates.values()):,} pairs")

    countries = pp_train_s1["country"].unique()
    logger.info(f"Countries: {list(countries)}")

    for country in countries:
        if not country:
            continue
        s1_country = pp_train_s1[pp_train_s1["country"] == country]
        s2s3_country = pp_train_s2s3[pp_train_s2s3["country"] == country]
        if len(s1_country) == 0 or len(s2s3_country) == 0:
            continue

        cands, prov, fwd_cands, nb, sv, sv2 = block.run_blocking(
            s1_country, s2s3_country, country, token_idf
        )

        # TF-IDF reciprocal
        recip = block.run_reciprocal_retrieval(
            sv, sv2, s1_country.index.values, s2s3_country.index.values,
            fwd_cands, topk=config.TFIDF_REVERSE_TOPK,
        )

        for s1_id, c_set in cands.items():
            all_candidates[s1_id].update(c_set)
        all_provenance.update(prov)
        all_reciprocal.update(recip)

        del sv, sv2, nb
        free_memory()

    # Handle missing country
    s1_no_country = pp_train_s1[
        pp_train_s1["country"].isin(["", "nan"]) | pp_train_s1["country"].isna()
    ]
    if len(s1_no_country) > 0:
        logger.info(f"Processing {len(s1_no_country):,} S1 without country ...")
        c, p, _, _, _, _ = block.run_blocking(s1_no_country, pp_train_s2s3, "UNKNOWN", token_idf)
        for s1_id, c_set in c.items():
            all_candidates[s1_id].update(c_set)
        all_provenance.update(p)
        free_memory()

    total = sum(len(v) for v in all_candidates.values())
    logger.info(f"\nTotal blocking pairs: {total:,}")
    block.evaluate_blocking_recall(dict(all_candidates), ground_truth)

    # Save blocking results
    recs = [{"s1_id": s1, "cand_id": c} for s1, cs in all_candidates.items() for c in cs]
    pd.DataFrame(recs).to_parquet(cache / "blocking_candidates.parquet")

    return (dict(all_candidates), dict(all_provenance), all_reciprocal,
            emb_provenance, emb_reciprocal, token_idf)


# ──────────────────────────────────────────────
#  Phase 4: Feature Engineering
# ──────────────────────────────────────────────
@timed
def run_features_phase(pp_train_s1, pp_train_s2s3, candidates, provenance,
                       reciprocal, emb_provenance, emb_reciprocal,
                       token_idf, ground_truth):
    """Compute features for all candidate pairs."""
    logger.info("=" * 60)
    logger.info("PHASE 4: FEATURE ENGINEERING")
    logger.info("=" * 60)

    pairs = [(s1, c) for s1, cs in candidates.items() for c in cs]
    logger.info(f"Total candidate pairs: {len(pairs):,}")

    feature_df = feat.compute_features_batch(
        pairs, pp_train_s1, pp_train_s2s3,
        provenance, reciprocal, token_idf,
        embedding_provenance=emb_provenance,
        embedding_reciprocal=emb_reciprocal,
    )

    # Labels
    labels = np.zeros(len(pairs), dtype=np.int32)
    for i, (s1_id, cand_id) in enumerate(pairs):
        if cand_id in ground_truth.get(s1_id, set()):
            labels[i] = 1
    feature_df["label"] = labels

    feature_df.to_parquet(config.CACHE_DIR / "train_features.parquet")
    pos = (feature_df["label"] == 1).sum()
    neg = (feature_df["label"] == 0).sum()
    logger.info(f"Features saved: {feature_df.shape}, pos={pos:,}, neg={neg:,}, ratio=1:{neg/max(pos,1):.1f}")

    return feature_df


# ──────────────────────────────────────────────
#  Phase 5: Training
# ──────────────────────────────────────────────
@timed
def run_training_phase(feature_df, ground_truth):
    """Train LightGBM (GPU) with structured negatives and threshold optimization."""
    logger.info("=" * 60)
    logger.info(f"PHASE 5: MODEL TRAINING (device={config.DEVICE})")
    logger.info("=" * 60)

    meta_cols = ["s1_id", "cand_id", "label"]
    feature_cols = [c for c in feature_df.columns if c not in meta_cols]

    X = feature_df[feature_cols].values.astype(np.float32)
    y = feature_df["label"].values.astype(np.int32)
    s1_ids = feature_df["s1_id"].values
    cand_ids = feature_df["cand_id"].values

    # Stratified entity-level split
    unique_s1 = np.unique(s1_ids)
    train_s1, val_s1 = train_test_split(
        unique_s1, test_size=config.VAL_SPLIT, random_state=config.RANDOM_SEED,
    )
    train_s1_set, val_s1_set = set(train_s1), set(val_s1)
    train_mask = np.array([s in train_s1_set for s in s1_ids])
    val_mask = np.array([s in val_s1_set for s in s1_ids])

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    s1_train, s1_val = s1_ids[train_mask], s1_ids[val_mask]
    cand_train, cand_val = cand_ids[train_mask], cand_ids[val_mask]

    logger.info(f"Train: {len(X_train):,} ({y_train.sum():,} pos)")
    logger.info(f"Val:   {len(X_val):,} ({y_val.sum():,} pos)")

    # --- Pass 1: Initial model ---
    logger.info("\n--- Pass 1: Initial Training ---")
    stage1_model = mdl.train_model(X_train, y_train, X_val, y_val)
    mdl.log_feature_importance(stage1_model, feature_cols)

    # --- OOF hard negative mining ---
    train_pairs = [(s1_ids[i], cand_ids[i], y[i]) for i in range(len(y)) if train_mask[i]]
    hard_negs = mdl.mine_hard_negatives_oof(X_train, y_train, train_pairs)

    if hard_negs:
        logger.info(f"\n--- Pass 2: +{len(hard_negs):,} hard negatives ---")
        hn_indices = []
        for hn_s1, hn_cand, _ in hard_negs:
            mask = (s1_ids == hn_s1) & (cand_ids == hn_cand) & train_mask
            idx = np.where(mask)[0]
            if len(idx) > 0:
                hn_indices.append(idx[0])
        if hn_indices:
            X_train_aug = np.vstack([X_train, X[hn_indices]])
            y_train_aug = np.concatenate([y_train, y[hn_indices]])
            stage1_model = mdl.train_model(X_train_aug, y_train_aug, X_val, y_val)
            del X_train_aug, y_train_aug
            free_memory()

    del X
    free_memory()

    # --- Stage 2: density features ---
    logger.info("\n--- Stage 2: Two-stage with OOF density ---")
    oof_scores = mdl.generate_oof_stage1_scores(X_train, y_train)
    train_density = feat.compute_density_features(oof_scores, s1_train)

    val_probs = stage1_model.predict_proba(X_val)[:, 1]
    val_density = feat.compute_density_features(val_probs, s1_val)

    X_train_s2 = np.hstack([X_train, train_density])
    X_val_s2 = np.hstack([X_val, val_density])
    all_feature_cols = feature_cols + feat.DENSITY_FEATURE_NAMES

    stage2_model = mdl.train_model(X_train_s2, y_train, X_val_s2, y_val)

    # --- Threshold optimization ---
    val_probs_s2 = stage2_model.predict_proba(X_val_s2)[:, 1]
    val_gt = {s: gt for s, gt in ground_truth.items() if s in val_s1_set}
    thresholds = mdl.find_best_threshold(val_probs_s2, s1_val, cand_val, val_gt)

    # Validation eval
    best_thresh = thresholds["global"]
    pred_dict = defaultdict(set)
    for i in range(len(val_probs_s2)):
        if val_probs_s2[i] >= best_thresh:
            pred_dict[s1_val[i]].add(cand_val[i])
    for s1_id in val_gt:
        if s1_id not in pred_dict:
            pred_dict[s1_id] = set()
    val_f05 = compute_macro_f05(dict(pred_dict), val_gt)
    logger.info(f"\n*** Validation F0.5: {val_f05:.4f} ***")

    # Save
    mdl.save_model(stage1_model, config.MODEL_DIR / "stage1_model.pkl")
    mdl.save_model(stage2_model, config.MODEL_DIR / "stage2_model.pkl")
    with open(config.MODEL_DIR / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    with open(config.MODEL_DIR / "feature_cols.json", "w") as f:
        json.dump(all_feature_cols, f)

    return stage1_model, stage2_model, thresholds, feature_cols


# ──────────────────────────────────────────────
#  Phase 6: Test Inference
# ──────────────────────────────────────────────
@timed
def run_inference_phase(pp_test_s1, pp_test_s2, pp_test_s3,
                        stage1_model, stage2_model, thresholds,
                        feature_cols, ground_truth_train=None):
    """Run inference on test data."""
    logger.info("=" * 60)
    logger.info("PHASE 6: TEST INFERENCE")
    logger.info("=" * 60)

    pp_test_s2s3 = pd.concat([pp_test_s2, pp_test_s3])
    cache = config.CACHE_DIR

    # Load token IDF
    token_idf = pd.read_parquet(cache / "token_idf.parquet")["idf"].to_dict()

    # Load embeddings using memory-mapping
    emb_s1 = np.load(cache / "emb_test_s1.npy", mmap_mode="r")
    emb_s2 = np.load(cache / "emb_test_s2.npy", mmap_mode="r")
    emb_s3 = np.load(cache / "emb_test_s3.npy", mmap_mode="r")
    n_s2s3 = len(emb_s2) + len(emb_s3)
    dim = emb_s1.shape[1]

    emb_s2s3_path = cache / "_emb_test_s2s3.npy"
    if not emb_s2s3_path.exists():
        logger.info(f"Building combined test S2+S3 embedding ({n_s2s3:,} x {dim}) ...")
        fp = np.lib.format.open_memmap(
            str(emb_s2s3_path), mode='w+', dtype=np.float32, shape=(n_s2s3, dim)
        )
        fp[:len(emb_s2)] = emb_s2[:]
        fp[len(emb_s2):] = emb_s3[:]
        fp.flush()
        del fp
    del emb_s2, emb_s3
    free_memory()
    emb_s2s3 = np.load(str(emb_s2s3_path), mmap_mode="r")

    s1_ids = pp_test_s1.index.values
    cand_ids = pp_test_s2s3.index.values

    # FAISS embedding blocking
    emb_candidates, emb_provenance = emb.run_embedding_blocking(
        s1_ids, cand_ids, emb_s1, emb_s2s3
    )
    emb_reciprocal = emb.run_embedding_reciprocal(
        s1_ids, cand_ids, emb_s1, emb_s2s3, emb_provenance, topk=20
    )
    del emb_s1, emb_s2s3
    free_memory()

    # TF-IDF blocking per country
    all_candidates = defaultdict(set)
    all_provenance = {}
    all_reciprocal = {}

    for s1_id, cs in emb_candidates.items():
        all_candidates[s1_id].update(cs)

    countries = pp_test_s1["country"].unique()
    for country in countries:
        if not country:
            continue
        s1_c = pp_test_s1[pp_test_s1["country"] == country]
        s2s3_c = pp_test_s2s3[pp_test_s2s3["country"] == country]
        if len(s1_c) == 0 or len(s2s3_c) == 0:
            continue

        c, p, fc, nb, sv, sv2 = block.run_blocking(s1_c, s2s3_c, country, token_idf)
        r = block.run_reciprocal_retrieval(
            sv, sv2, s1_c.index.values, s2s3_c.index.values,
            fc, topk=config.TFIDF_REVERSE_TOPK,
        )
        for s1_id, cs in c.items():
            all_candidates[s1_id].update(cs)
        all_provenance.update(p)
        all_reciprocal.update(r)
        del sv, sv2, nb
        free_memory()

    # Features
    pairs = [(s1, c) for s1, cs in all_candidates.items() for c in cs]
    logger.info(f"Test candidate pairs: {len(pairs):,}")

    feature_df = feat.compute_features_batch(
        pairs, pp_test_s1, pp_test_s2s3,
        all_provenance, all_reciprocal, token_idf,
        embedding_provenance=emb_provenance,
        embedding_reciprocal=emb_reciprocal,
    )

    # Align feature columns
    for c in feature_cols:
        if c not in feature_df.columns:
            feature_df[c] = 0.0

    X_test = feature_df[feature_cols].values.astype(np.float32)
    s1_arr = feature_df["s1_id"].values
    cand_arr = feature_df["cand_id"].values

    # Stage 1 → density → Stage 2
    s1_probs = stage1_model.predict_proba(X_test)[:, 1]
    test_density = feat.compute_density_features(s1_probs, s1_arr)
    X_test_s2 = np.hstack([X_test, test_density])
    s2_probs = stage2_model.predict_proba(X_test_s2)[:, 1]

    # Per-country threshold
    s1_country_map = pp_test_s1["country"].to_dict()
    countries_arr = np.array([
        s1_country_map.get(s, "")
        for s in s1_arr
    ])

    predictions = post.postprocess(
        s2_probs, s1_arr, cand_arr, thresholds,
        ground_truth=ground_truth_train,
        all_s1_ids=list(pp_test_s1.index),
        countries=countries_arr,
    )

    # Output
    cand_dict = {s1: list(cs) for s1, cs in all_candidates.items()}
    pred_list = {s1: list(v) for s1, v in predictions.items()}
    generate_output_files(pred_list, cand_dict, list(pp_test_s1.index))

    return predictions


# ──────────────────────────────────────────────
#  Full pipeline
# ──────────────────────────────────────────────
@timed
def run_full_pipeline(nrows=None):
    """Run the complete entity resolution pipeline."""
    t_start = time.time()

    logger.info("=" * 60)
    logger.info("BUSINESS ENTITY RESOLUTION — GPU PIPELINE")
    logger.info(f"Device: {config.GPU_NAME} | Embedding: {config.EMBEDDING_MODEL}")
    logger.info("=" * 60)

    # Phase 1: Preprocessing
    run_preprocess(nrows=nrows)
    free_memory()

    # Phase 2: Embeddings (GPU)
    run_embedding_phase()
    free_memory()

    # Phase 3a: FAISS blocking (minimal memory - no DataFrames loaded)
    emb_candidates, emb_provenance, emb_reciprocal = run_faiss_blocking()
    free_memory()

    # Now load train data for TF-IDF blocking + features
    logger.info("Loading train preprocessed data ...")
    # Load only blocking-relevant columns first
    blocking_cols = ["name_core", "name_clean", "addr_clean", "country",
                     "phonetic_key", "postal_code", "city"]
    pp_train_s1_block = pd.read_parquet(config.CACHE_DIR / "pp_train_s1.parquet",
                                         columns=blocking_cols)
    pp_train_s2_block = pd.read_parquet(config.CACHE_DIR / "pp_train_s2.parquet",
                                         columns=blocking_cols)
    pp_train_s3_block = pd.read_parquet(config.CACHE_DIR / "pp_train_s3.parquet",
                                         columns=blocking_cols)
    pp_train_s2s3_block = pd.concat([pp_train_s2_block, pp_train_s3_block])
    del pp_train_s2_block, pp_train_s3_block
    free_memory()

    s1_filter = list(pp_train_s1_block.index) if nrows else None
    ground_truth = load_ground_truth(config.TRAIN_GT, s1_ids_filter=s1_filter)

    # Phase 3b: TF-IDF + inverted index blocking (uses only blocking columns)
    (candidates, provenance, reciprocal,
     _emb_prov, _emb_recip, token_idf) = run_blocking_phase(
        pp_train_s1_block, pp_train_s2s3_block, ground_truth,
        emb_candidates, emb_provenance, emb_reciprocal
    )
    del pp_train_s1_block, pp_train_s2s3_block
    free_memory()

    # Phase 4: Features (load only needed feature columns)
    logger.info("Loading train data for feature computation (needed columns only) ...")
    feat_cols = ["name_clean", "name_core", "addr_clean", "country", "postal_code",
                 "city", "state", "street_number", "landmark_tokens", "is_url_name",
                 "generic_tokens", "url_stem", "script_type", "legal_suffix", "name_unicode"]
    pp_train_s1 = pd.read_parquet(config.CACHE_DIR / "pp_train_s1.parquet", columns=feat_cols)
    pp_train_s2 = pd.read_parquet(config.CACHE_DIR / "pp_train_s2.parquet", columns=feat_cols)
    pp_train_s3 = pd.read_parquet(config.CACHE_DIR / "pp_train_s3.parquet", columns=feat_cols)
    pp_train_s2s3 = pd.concat([pp_train_s2, pp_train_s3])
    del pp_train_s2, pp_train_s3
    free_memory()

    feature_df = run_features_phase(
        pp_train_s1, pp_train_s2s3, candidates, provenance,
        reciprocal, emb_provenance, emb_reciprocal,
        token_idf, ground_truth,
    )
    del candidates, provenance, reciprocal, emb_provenance, emb_reciprocal, pp_train_s2s3
    free_memory()

    # Phase 5: Training
    stage1_model, stage2_model, thresholds, feature_cols = run_training_phase(
        feature_df, ground_truth
    )
    del feature_df, pp_train_s1
    free_memory()

    # Phase 6: Test inference
    logger.info("Loading test preprocessed data ...")
    pp_test_s1 = pd.read_parquet(config.CACHE_DIR / "pp_test_s1.parquet")
    pp_test_s2 = pd.read_parquet(config.CACHE_DIR / "pp_test_s2.parquet")
    pp_test_s3 = pd.read_parquet(config.CACHE_DIR / "pp_test_s3.parquet")

    run_inference_phase(
        pp_test_s1, pp_test_s2, pp_test_s3,
        stage1_model, stage2_model, thresholds, feature_cols,
        ground_truth_train=ground_truth,
    )

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info(f"PIPELINE COMPLETE — {elapsed / 60:.1f} minutes")
    logger.info(f"Output: {config.OUTPUT_DIR}")
    logger.info("=" * 60)


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Business Entity Resolution — GPU Pipeline")
    parser.add_argument("--phase", default="full",
                        choices=["full", "preprocess", "embed", "blocking",
                                 "features", "train", "inference"])
    parser.add_argument("--nrows", type=int, default=None, help="Subset rows for debugging")
    args = parser.parse_args()

    if args.phase == "full":
        run_full_pipeline(nrows=args.nrows)
    elif args.phase == "preprocess":
        run_preprocess(nrows=args.nrows)
    elif args.phase == "embed":
        run_embedding_phase()
    elif args.phase in ("blocking", "features", "train", "inference"):
        pp_s1 = pd.read_parquet(config.CACHE_DIR / "pp_train_s1.parquet")
        pp_s2 = pd.read_parquet(config.CACHE_DIR / "pp_train_s2.parquet")
        pp_s3 = pd.read_parquet(config.CACHE_DIR / "pp_train_s3.parquet")
        gt = load_ground_truth(config.TRAIN_GT)

        if args.phase == "blocking":
            run_blocking_phase(pp_s1, pp_s2, pp_s3, gt)
        elif args.phase == "features":
            cdf = pd.read_parquet(config.CACHE_DIR / "blocking_candidates.parquet")
            cands = defaultdict(set)
            for _, r in cdf.iterrows():
                cands[r["s1_id"]].add(r["cand_id"])
            idf = pd.read_parquet(config.CACHE_DIR / "token_idf.parquet")["idf"].to_dict()
            s2s3 = pd.concat([pp_s2, pp_s3])
            run_features_phase(pp_s1, s2s3, dict(cands), {}, {}, {}, {}, idf, gt)
        elif args.phase == "train":
            fdf = pd.read_parquet(config.CACHE_DIR / "train_features.parquet")
            run_training_phase(fdf, gt)
        elif args.phase == "inference":
            s1m = mdl.load_model(config.MODEL_DIR / "stage1_model.pkl")
            s2m = mdl.load_model(config.MODEL_DIR / "stage2_model.pkl")
            with open(config.MODEL_DIR / "thresholds.json") as f:
                th = json.load(f)
            with open(config.MODEL_DIR / "feature_cols.json") as f:
                fc = json.load(f)
            ts1 = pd.read_parquet(config.CACHE_DIR / "pp_test_s1.parquet")
            ts2 = pd.read_parquet(config.CACHE_DIR / "pp_test_s2.parquet")
            ts3 = pd.read_parquet(config.CACHE_DIR / "pp_test_s3.parquet")
            run_inference_phase(ts1, ts2, ts3, s1m, s2m, th, fc, ground_truth_train=gt)


if __name__ == "__main__":
    main()
