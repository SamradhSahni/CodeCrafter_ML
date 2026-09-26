"""
Dense Embedding Module — Sentence-Transformer encoding + FAISS similarity search.

Uses GPU-accelerated sentence-transformers for encoding business names/addresses
into dense vectors, and FAISS for fast top-K similarity retrieval.
"""
import gc
from collections import defaultdict

import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from . import config
from .utils import logger, timed


# ──────────────────────────────────────────────
#  Model loading
# ──────────────────────────────────────────────
@timed
def load_embedding_model(model_name: str = None) -> SentenceTransformer:
    """Load sentence-transformer model onto the best available device."""
    model_name = model_name or config.EMBEDDING_MODEL
    device = config.DEVICE
    logger.info(f"Loading embedding model: {model_name} on {device} ({config.GPU_NAME})")
    model = SentenceTransformer(model_name, device=device)
    return model


# ──────────────────────────────────────────────
#  Text encoding
# ──────────────────────────────────────────────
@timed
def encode_texts(model: SentenceTransformer, texts, batch_size: int = None,
                 desc: str = "Encoding", chunk_size: int = 500_000) -> np.ndarray:
    """
    Encode texts to L2-normalized dense vectors.
    Processes in sub-chunks to avoid OOM on large datasets (5M+ texts).
    """
    batch_size = batch_size or config.EMBEDDING_BATCH_SIZE

    # Convert pandas Series to list
    if hasattr(texts, "tolist"):
        texts = texts.tolist()

    # Replace empty strings with a space (model needs non-empty input)
    texts = [t if t and str(t).strip() else " " for t in texts]

    n = len(texts)
    logger.info(f"Encoding {n:,} texts (batch_size={batch_size}, "
                f"device={model.device}) ...")

    if n <= chunk_size:
        # Small enough to encode in one shot
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        logger.info(f"  -> Embeddings shape: {embeddings.shape}, dtype: {embeddings.dtype}")
        return embeddings.astype(np.float32)

    # Large dataset: encode in sub-chunks and concatenate on disk
    import tempfile, os
    chunk_files = []
    dim = None

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_texts = texts[start:end]
        logger.info(f"  Encoding chunk {start:,}-{end:,} ({len(chunk_texts):,} texts) ...")

        chunk_emb = model.encode(
            chunk_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        if dim is None:
            dim = chunk_emb.shape[1]

        # Save chunk to temp file
        tmp_path = config.CACHE_DIR / f"_emb_chunk_{start}.npy"
        np.save(tmp_path, chunk_emb)
        chunk_files.append(tmp_path)
        del chunk_emb, chunk_texts
        gc.collect()
        if config.HAS_CUDA:
            torch.cuda.empty_cache()

    # Concatenate from disk using memory-mapped reads
    logger.info(f"  Merging {len(chunk_files)} embedding chunks ...")
    all_emb = np.empty((n, dim), dtype=np.float32)
    offset = 0
    for cf in chunk_files:
        chunk = np.load(cf)
        all_emb[offset:offset + len(chunk)] = chunk
        offset += len(chunk)
        del chunk
        cf.unlink()  # cleanup temp file

    logger.info(f"  -> Embeddings shape: {all_emb.shape}, dtype: {all_emb.dtype}")
    return all_emb


# ──────────────────────────────────────────────
#  FAISS Index
# ──────────────────────────────────────────────
@timed
def build_faiss_index(vectors: np.ndarray, use_gpu: bool = None,
                      add_batch_size: int = 500_000) -> faiss.Index:
    """
    Build a FAISS index for fast inner-product (cosine) similarity search.
    Uses IVF-PQ for very large datasets (>2M) to compress stored vectors,
    IVFFlat for medium datasets, and flat for small datasets.
    """
    if use_gpu is None:
        use_gpu = config.HAS_CUDA

    dim = vectors.shape[1]
    n = vectors.shape[0]

    # Train subsample (shared by IVFFlat and IVFPQ)
    def _get_train_vectors(max_train=500_000):
        train_size = min(max_train, n)
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=train_size, replace=False)
        idx.sort()
        tv = np.ascontiguousarray(vectors[idx].astype(np.float32))
        logger.info(f"  Training on {train_size:,} subsample ...")
        return tv

    if n > 2_000_000:
        # IVF-PQ for very large datasets — compressed storage
        # Each 384-dim vector stored as 48 bytes instead of 1536 bytes
        # 10M vectors: ~480MB instead of ~15GB
        nlist = min(int(np.sqrt(n)), 4096)
        m = 48  # number of sub-quantizers (dim must be divisible by m)
        nbits = 8  # bits per sub-quantizer
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits,
                                 faiss.METRIC_INNER_PRODUCT)
        logger.info(f"Building IVF-PQ index: {n:,} vectors, dim={dim}, "
                    f"nlist={nlist}, m={m}, nbits={nbits}")

        train_vectors = _get_train_vectors()
        index.train(train_vectors)
        del train_vectors
        gc.collect()

        # Add vectors in batches
        logger.info(f"  Adding {n:,} vectors in batches of {add_batch_size:,} ...")
        for start in range(0, n, add_batch_size):
            end = min(start + add_batch_size, n)
            batch = np.ascontiguousarray(vectors[start:end].astype(np.float32))
            index.add(batch)
            del batch
        gc.collect()
        index.nprobe = config.FAISS_NPROBE
        logger.info(f"  Index built: ~{index.ntotal * m / 1e6:.0f} MB compressed storage")

    elif n > 500_000:
        # IVFFlat for medium datasets — exact stored vectors
        nlist = min(int(np.sqrt(n)), 4096)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        logger.info(f"Building IVF-Flat index: {n:,} vectors, dim={dim}, nlist={nlist}")

        train_vectors = _get_train_vectors()
        index.train(train_vectors)
        del train_vectors
        gc.collect()

        logger.info(f"  Adding {n:,} vectors in batches of {add_batch_size:,} ...")
        for start in range(0, n, add_batch_size):
            end = min(start + add_batch_size, n)
            batch = np.ascontiguousarray(vectors[start:end].astype(np.float32))
            index.add(batch)
            del batch
        gc.collect()
        index.nprobe = config.FAISS_NPROBE

    else:
        # Flat index for small datasets — exact search
        vecs = np.ascontiguousarray(vectors[:].astype(np.float32))
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)
        del vecs
        logger.info(f"Building flat index: {n:,} vectors, dim={dim}")

    # Move to GPU if available
    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            logger.info("  FAISS index moved to GPU")
        except (AttributeError, RuntimeError) as e:
            logger.warning(f"  FAISS GPU not available ({e}), using CPU")

    return index


@timed
def search_faiss(query_vectors: np.ndarray, index: faiss.Index,
                 k: int = 50, batch_size: int = 10_000):
    """
    Batch-search FAISS index for top-K neighbors.
    Returns (distances, indices) arrays of shape (n_queries, k).
    Memory-safe: copies only one batch at a time to contiguous array.
    """
    n = query_vectors.shape[0]
    logger.info(f"FAISS search: {n:,} queries, top-{k} ...")

    all_distances = np.zeros((n, k), dtype=np.float32)
    all_indices = np.full((n, k), -1, dtype=np.int64)

    for start in tqdm(range(0, n, batch_size), desc="FAISS search"):
        end = min(start + batch_size, n)
        batch = np.ascontiguousarray(
            query_vectors[start:end].astype(np.float32)
        )
        distances, indices = index.search(batch, k)
        all_distances[start:end] = distances
        all_indices[start:end] = indices
        del batch

    logger.info(f"  -> Search complete: {n * k:,} results")
    return all_distances, all_indices


# ──────────────────────────────────────────────
#  Embedding-based blocking
# ──────────────────────────────────────────────
@timed
def run_embedding_blocking(s1_ids: np.ndarray, cand_ids: np.ndarray,
                           s1_embeddings: np.ndarray, cand_embeddings: np.ndarray,
                           topk: int = None):
    """
    Run FAISS-based blocking using pre-computed embeddings.
    Returns candidates dict and provenance dict.
    """
    topk = topk or config.FAISS_TOP_K

    # Build index on candidates (S2+S3)
    index = build_faiss_index(cand_embeddings)

    # Search
    distances, indices = search_faiss(s1_embeddings, index, k=topk)

    # Extract candidates with provenance
    candidates = defaultdict(set)
    provenance = {}

    for i in range(len(s1_ids)):
        s1_id = s1_ids[i]
        for rank in range(topk):
            j = int(indices[i, rank])
            if j < 0 or j >= len(cand_ids):
                continue
            cand_id = cand_ids[j]
            score = float(distances[i, rank])
            candidates[s1_id].add(cand_id)
            provenance[(s1_id, cand_id)] = {
                "embedding_rank": rank,
                "embedding_score": score,
            }

    avg_cands = np.mean([len(v) for v in candidates.values()]) if candidates else 0
    logger.info(f"Embedding blocking: {sum(len(v) for v in candidates.values()):,} pairs, "
                f"avg {avg_cands:.1f} per S1")

    # Clean up FAISS index
    del index
    gc.collect()
    if config.HAS_CUDA:
        torch.cuda.empty_cache()

    return dict(candidates), provenance


# ──────────────────────────────────────────────
#  Reciprocal embedding retrieval
# ──────────────────────────────────────────────
@timed
def run_embedding_reciprocal(s1_ids: np.ndarray, cand_ids: np.ndarray,
                             s1_embeddings: np.ndarray, cand_embeddings: np.ndarray,
                             forward_provenance: dict, topk: int = 20):
    """
    Independent reverse retrieval: S2S3 → S1 using embeddings.
    Joins with forward to produce reciprocal rank features.
    """
    # Build index on S1
    index_s1 = build_faiss_index(s1_embeddings)

    # Search: S2S3 queries against S1 corpus
    distances, indices = search_faiss(cand_embeddings, index_s1, k=topk)

    # Build reverse lookup: cand_idx -> {s1_idx: rank}
    s1_id_to_idx = {eid: i for i, eid in enumerate(s1_ids)}
    cand_id_to_idx = {eid: i for i, eid in enumerate(cand_ids)}

    reverse_ranks = {}
    for cand_idx in range(len(cand_ids)):
        rank_map = {}
        for rank in range(topk):
            s1_local_idx = int(indices[cand_idx, rank])
            if s1_local_idx < 0 or s1_local_idx >= len(s1_ids):
                continue
            rank_map[s1_local_idx] = rank
        if rank_map:
            reverse_ranks[cand_idx] = rank_map

    # Join forward + reverse
    reciprocal = {}
    for (s1_id, cand_id), prov in forward_provenance.items():
        s1_idx = s1_id_to_idx.get(s1_id)
        cand_idx = cand_id_to_idx.get(cand_id)
        if s1_idx is None or cand_idx is None:
            continue
        fwd_rank = prov.get("embedding_rank", 999)
        rev_rank_map = reverse_ranks.get(cand_idx, {})
        rev_rank = rev_rank_map.get(s1_idx, -1)

        reciprocal[(s1_id, cand_id)] = {
            "emb_fwd_rank": fwd_rank,
            "emb_rev_rank": rev_rank,
            "emb_mutual_top1": (fwd_rank == 0 and rev_rank == 0),
            "emb_mutual_top5": (fwd_rank < 5 and 0 <= rev_rank < 5),
            "emb_reciprocal_product": (
                (1.0 / (fwd_rank + 1)) * (1.0 / (rev_rank + 1))
                if rev_rank >= 0 else 0.0
            ),
        }

    # Clean up
    del index_s1
    gc.collect()
    if config.HAS_CUDA:
        torch.cuda.empty_cache()

    logger.info(f"Embedding reciprocal: {len(reciprocal):,} pairs with features")
    return reciprocal


# ──────────────────────────────────────────────
#  Pre-compute embeddings for a DataFrame
# ──────────────────────────────────────────────
@timed
def compute_embeddings_for_df(model: SentenceTransformer, df, text_col: str = "name_clean",
                              desc: str = "source"):
    """Compute embeddings for a preprocessed DataFrame's text column."""
    texts = df[text_col].fillna(" ").tolist()
    embeddings = encode_texts(model, texts, desc=f"{desc} ({text_col})")
    return embeddings
