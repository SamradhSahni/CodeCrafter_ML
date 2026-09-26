"""
Phase 2: Adaptive blocking with provenance tracking and reciprocal retrieval.

Uses multi-strategy blocking (TF-IDF name, TF-IDF address, phonetic, rare token,
postal+prefix, city+bigram), with per-entity adaptive K, integer-encoded pair storage,
and independent reverse retrieval for reciprocal rank features.
"""
import gc
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

try:
    from sparse_dot_topn import awesome_cossim_topn
except ImportError:
    from sparse_dot_topn import sp_matmul_topn as awesome_cossim_topn

import jellyfish

from . import config
from .utils import logger, timed, free_memory


# ──────────────────────────────────────────────
#  Adaptive K
# ──────────────────────────────────────────────
def get_adaptive_k(name_tokens: list, idf_dict: dict, has_address: bool,
                   k_config: dict = None) -> int:
    """Compute adaptive K based on entity uniqueness."""
    k_config = k_config or config.DEFAULT_K_CONFIG
    tokens_with_idf = [t for t in name_tokens if t in idf_dict]
    if not tokens_with_idf:
        return k_config["max_k"]

    max_idf = max(idf_dict[t] for t in tokens_with_idf)
    n_tokens = len(tokens_with_idf)

    if max_idf > k_config["unique_idf_thresh"] and n_tokens >= 3:
        return k_config["unique_k"]
    elif max_idf > k_config["normal_idf_thresh"] and has_address:
        return k_config["normal_k"]
    elif max_idf > k_config["common_idf_thresh"]:
        return k_config["common_k"]
    else:
        return k_config["max_k"]


# ──────────────────────────────────────────────
#  Phonetic key generation
# ──────────────────────────────────────────────
def get_phonetic_key(name_core: str) -> str:
    """Get double metaphone of the first meaningful token."""
    tokens = name_core.split()
    if not tokens:
        return ""
    # Use first non-trivial token
    for t in tokens:
        if len(t) >= 3:
            try:
                return jellyfish.metaphone(t)
            except Exception:
                return ""
    return ""


# ──────────────────────────────────────────────
#  TF-IDF blocking (name + address)
# ──────────────────────────────────────────────
class TFIDFBlocker:
    """TF-IDF based candidate retrieval using sparse_dot_topn."""

    def __init__(self, field: str, analyzer: str = "char_wb",
                 ngram_range=(3, 5), max_features=500_000):
        self.field = field
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.is_fitted = False

    def fit(self, all_texts: pd.Series):
        """Fit on all texts (S1 + S2 + S3) for consistent vocabulary."""
        logger.info(f"Fitting TF-IDF ({self.field}): {len(all_texts):,} texts ...")
        # Filter empty strings
        valid = all_texts[all_texts.str.len() > 0]
        self.vectorizer.fit(valid)
        self.is_fitted = True
        self.idf_ = dict(zip(
            self.vectorizer.get_feature_names_out(),
            self.vectorizer.idf_
        ))
        logger.info(f"  -> vocabulary size: {len(self.vectorizer.vocabulary_):,}")

    def transform(self, texts: pd.Series) -> csr_matrix:
        """Transform texts to TF-IDF vectors."""
        return self.vectorizer.transform(texts.fillna(""))

    def retrieve_topk(self, query_vecs: csr_matrix, corpus_vecs: csr_matrix,
                      topk: int, min_sim: float = 0.3, n_jobs: int = 4):
        """
        Retrieve top-K candidates for each query from corpus.
        Returns sparse matrix (n_queries x n_corpus) with similarities.
        """
        logger.info(f"TF-IDF retrieval ({self.field}): {query_vecs.shape[0]:,} queries x "
                     f"{corpus_vecs.shape[0]:,} corpus, top-{topk} ...")
        try:
            result = awesome_cossim_topn(
                query_vecs, corpus_vecs.T,
                ntop=topk, lower_bound=min_sim,
                use_threads=True, n_jobs=n_jobs,
            )
        except TypeError:
            # Older sparse_dot_topn API
            result = awesome_cossim_topn(
                query_vecs, corpus_vecs.T,
                ntop=topk, lower_bound=min_sim,
            )
        logger.info(f"  -> {result.nnz:,} candidate pairs")
        return result


# ──────────────────────────────────────────────
#  Extract candidates from sparse similarity matrix
# ──────────────────────────────────────────────
def sparse_to_candidates(sim_matrix, s1_ids, cand_ids):
    """
    Convert a sparse similarity matrix to a dict of candidates with ranks.
    Returns: {s1_id: [(cand_id, rank, score), ...]}
    """
    candidates = defaultdict(list)
    for i in range(sim_matrix.shape[0]):
        row = sim_matrix.getrow(i)
        if row.nnz == 0:
            continue
        indices = row.indices
        scores = row.data
        # Sort by score descending
        order = np.argsort(-scores)
        s1_id = s1_ids[i]
        for rank, j in enumerate(order):
            cand_id = cand_ids[indices[j]]
            candidates[s1_id].append((cand_id, rank, float(scores[j])))
    return candidates


# ──────────────────────────────────────────────
#  Build IDF dictionary from name tokens
# ──────────────────────────────────────────────
def build_token_idf(all_name_cores: pd.Series) -> dict:
    """Build word-level IDF from all name_core values."""
    doc_freq = defaultdict(int)
    n_docs = 0
    for name in all_name_cores:
        if not name:
            continue
        tokens = set(name.split())
        for t in tokens:
            doc_freq[t] += 1
        n_docs += 1
    idf = {}
    for t, df in doc_freq.items():
        idf[t] = np.log(n_docs / (1 + df))
    return idf


# ──────────────────────────────────────────────
#  Multi-strategy blocking orchestrator
# ──────────────────────────────────────────────
@timed
def run_blocking(s1_df: pd.DataFrame, s2s3_df: pd.DataFrame,
                 country: str, token_idf: dict,
                 k_config: dict = None):
    """
    Run all blocking strategies for a single country partition.

    Returns:
        candidates: dict {s1_id: list of cand_ids}
        provenance: dict {(s1_id, cand_id): provenance_dict}
        forward_candidates: TF-IDF forward retrieval results (for reciprocal)
        name_blocker: fitted TF-IDF blocker (for reciprocal)
    """
    k_config = k_config or config.DEFAULT_K_CONFIG
    s1_ids = s1_df.index.values
    cand_ids = s2s3_df.index.values

    logger.info(f"Blocking [{country}]: {len(s1_ids):,} S1 x {len(cand_ids):,} S2/S3")

    # Track provenance per pair
    provenance = defaultdict(lambda: {
        "name_tfidf_hit": False, "name_tfidf_rank": -1,
        "addr_tfidf_hit": False, "addr_tfidf_rank": -1,
        "phonetic_hit": False, "rare_token_hit": False,
        "postal_hit": False, "city_hit": False,
    })

    all_candidates = defaultdict(set)

    # ── Strategy 1: TF-IDF name blocking ──
    logger.info(f"  Strategy 1: TF-IDF Name blocking ...")
    name_blocker = TFIDFBlocker(
        "name", analyzer="char_wb",
        ngram_range=config.TFIDF_NAME_NGRAM_RANGE,
        max_features=config.TFIDF_NAME_MAX_FEATURES,
    )
    all_names = pd.concat([s1_df["name_clean"], s2s3_df["name_clean"]])
    name_blocker.fit(all_names)

    # Compute median adaptive K for this country
    median_k = int(np.median([
        get_adaptive_k(
            row["name_core"].split() if row["name_core"] else [],
            token_idf,
            bool(row["addr_clean"]),
            k_config,
        )
        for _, row in s1_df.iterrows()
    ]))
    logger.info(f"  Median adaptive K: {median_k}")

    s1_name_vecs = name_blocker.transform(s1_df["name_clean"])
    s2s3_name_vecs = name_blocker.transform(s2s3_df["name_clean"])

    name_sim = name_blocker.retrieve_topk(
        s1_name_vecs, s2s3_name_vecs,
        topk=median_k, min_sim=config.TFIDF_MIN_SIMILARITY,
    )
    name_candidates = sparse_to_candidates(name_sim, s1_ids, cand_ids)

    for s1_id, cands in name_candidates.items():
        for cand_id, rank, score in cands:
            all_candidates[s1_id].add(cand_id)
            prov = provenance[(s1_id, cand_id)]
            prov["name_tfidf_hit"] = True
            prov["name_tfidf_rank"] = rank

    logger.info(f"  After name TF-IDF: {sum(len(v) for v in all_candidates.values()):,} pairs")

    # ── Strategy 2: TF-IDF address blocking ──
    logger.info(f"  Strategy 2: TF-IDF Address blocking ...")
    addr_blocker = TFIDFBlocker(
        "addr", analyzer="word", ngram_range=(1, 2),
        max_features=config.TFIDF_ADDR_MAX_FEATURES,
    )
    all_addrs = pd.concat([s1_df["addr_clean"], s2s3_df["addr_clean"]])
    # Only fit on non-empty addresses
    addr_blocker.fit(all_addrs[all_addrs.str.len() > 0])

    s1_addr_vecs = addr_blocker.transform(s1_df["addr_clean"])
    s2s3_addr_vecs = addr_blocker.transform(s2s3_df["addr_clean"])

    addr_k = max(20, median_k // 2)
    addr_sim = addr_blocker.retrieve_topk(
        s1_addr_vecs, s2s3_addr_vecs,
        topk=addr_k, min_sim=config.TFIDF_MIN_SIMILARITY,
    )
    addr_candidates = sparse_to_candidates(addr_sim, s1_ids, cand_ids)

    for s1_id, cands in addr_candidates.items():
        for cand_id, rank, score in cands:
            all_candidates[s1_id].add(cand_id)
            prov = provenance[(s1_id, cand_id)]
            prov["addr_tfidf_hit"] = True
            prov["addr_tfidf_rank"] = rank

    del addr_sim, addr_candidates
    gc.collect()
    logger.info(f"  After addr TF-IDF: {sum(len(v) for v in all_candidates.values()):,} pairs")

    max_bucket = getattr(config, "INVERTED_INDEX_MAX_BUCKET", 200)
    max_cands = getattr(config, "BLOCKING_MAX_CANDS_PER_ENTITY", 50)

    # ── Strategy 3: Phonetic blocking ──
    logger.info(f"  Strategy 3: Phonetic blocking ...")
    # Use pre-computed phonetic_key if available, otherwise compute on the fly
    if "phonetic_key" in s1_df.columns:
        s1_phonetic = s1_df["phonetic_key"].to_dict()
    else:
        s1_phonetic = {eid: get_phonetic_key(row["name_core"])
                       for eid, row in s1_df.iterrows()}
    # Build inverted index: phonetic_key -> [cand_ids]
    phonetic_index = defaultdict(list)
    if "phonetic_key" in s2s3_df.columns:
        for eid, key in s2s3_df["phonetic_key"].items():
            if key:
                phonetic_index[key].append(eid)
    else:
        for eid, row in s2s3_df.iterrows():
            key = get_phonetic_key(row["name_core"])
            if key:
                phonetic_index[key].append(eid)

    phonetic_pairs = 0
    for s1_id, pkey in s1_phonetic.items():
        if len(all_candidates[s1_id]) >= max_cands:
            continue
        if pkey and pkey in phonetic_index:
            bucket = phonetic_index[pkey]
            if len(bucket) <= max_bucket:
                for cand_id in bucket:
                    if len(all_candidates[s1_id]) >= max_cands:
                        break
                    all_candidates[s1_id].add(cand_id)
                    provenance[(s1_id, cand_id)]["phonetic_hit"] = True
                    phonetic_pairs += 1

    del phonetic_index
    gc.collect()
    logger.info(f"  After phonetic: +{phonetic_pairs:,} pairs, "
                f"total {sum(len(v) for v in all_candidates.values()):,}")

    # ── Strategy 4: Rare token blocking ──
    logger.info(f"  Strategy 4: Rare token blocking ...")
    # Build inverted index: rare_token -> [cand_ids]
    rare_token_index = defaultdict(list)
    for eid, row in s2s3_df.iterrows():
        if not row["name_core"]:
            continue
        for t in row["name_core"].split():
            if token_idf.get(t, 0) > config.RARE_TOKEN_IDF_THRESHOLD:
                rare_token_index[t].append(eid)

    rare_pairs = 0
    for s1_id, row in s1_df.iterrows():
        if len(all_candidates[s1_id]) >= max_cands:
            continue
        if not row["name_core"]:
            continue
        for t in row["name_core"].split():
            if len(all_candidates[s1_id]) >= max_cands:
                break
            if token_idf.get(t, 0) > config.RARE_TOKEN_IDF_THRESHOLD:
                if t in rare_token_index:
                    bucket = rare_token_index[t]
                    if len(bucket) <= max_bucket:
                        for cand_id in bucket:
                            if len(all_candidates[s1_id]) >= max_cands:
                                break
                            all_candidates[s1_id].add(cand_id)
                            provenance[(s1_id, cand_id)]["rare_token_hit"] = True
                            rare_pairs += 1

    del rare_token_index
    gc.collect()
    logger.info(f"  After rare token: +{rare_pairs:,} pairs, "
                f"total {sum(len(v) for v in all_candidates.values()):,}")

    # ── Strategy 5: Postal code + name prefix ──
    logger.info(f"  Strategy 5: Postal+prefix blocking ...")
    postal_index = defaultdict(list)
    for eid, row in s2s3_df.iterrows():
        if row["postal_code"] and row["name_core"]:
            key = f"{row['postal_code']}_{row['name_core'][:3]}"
            postal_index[key].append(eid)

    postal_pairs = 0
    for s1_id, row in s1_df.iterrows():
        if len(all_candidates[s1_id]) >= max_cands:
            continue
        if row["postal_code"] and row["name_core"]:
            key = f"{row['postal_code']}_{row['name_core'][:3]}"
            if key in postal_index:
                bucket = postal_index[key]
                if len(bucket) <= max_bucket:
                    for cand_id in bucket:
                        if len(all_candidates[s1_id]) >= max_cands:
                            break
                        all_candidates[s1_id].add(cand_id)
                        provenance[(s1_id, cand_id)]["postal_hit"] = True
                        postal_pairs += 1

    del postal_index
    gc.collect()
    logger.info(f"  After postal+prefix: +{postal_pairs:,} pairs, "
                f"total {sum(len(v) for v in all_candidates.values()):,}")

    # ── Strategy 6: City + name bigram ──
    logger.info(f"  Strategy 6: City+bigram blocking ...")
    city_index = defaultdict(list)
    for eid, row in s2s3_df.iterrows():
        if row["city"] and row["name_core"] and len(row["name_core"]) >= 2:
            bigrams = {row["name_core"][i:i+2] for i in range(len(row["name_core"]) - 1)
                       if row["name_core"][i:i+2].strip()}
            for bg in bigrams:
                key = f"{row['city']}_{bg}"
                city_index[key].append(eid)

    city_pairs = 0
    for s1_id, row in s1_df.iterrows():
        if len(all_candidates[s1_id]) >= max_cands:
            continue
        if row["city"] and row["name_core"] and len(row["name_core"]) >= 2:
            bigrams = {row["name_core"][i:i+2] for i in range(len(row["name_core"]) - 1)
                       if row["name_core"][i:i+2].strip()}
            for bg in bigrams:
                if len(all_candidates[s1_id]) >= max_cands:
                    break
                key = f"{row['city']}_{bg}"
                if key in city_index:
                    bucket = city_index[key]
                    if len(bucket) <= max_bucket:
                        for cand_id in bucket:
                            if len(all_candidates[s1_id]) >= max_cands:
                                break
                            all_candidates[s1_id].add(cand_id)
                            provenance[(s1_id, cand_id)]["city_hit"] = True
                            city_pairs += 1

    del city_index
    gc.collect()
    logger.info(f"  After city+bigram: +{city_pairs:,} pairs, "
                f"total {sum(len(v) for v in all_candidates.values()):,}")

    # ── Compute blocker provenance summary ──
    for key, prov in provenance.items():
        hits = sum([
            prov["name_tfidf_hit"], prov["addr_tfidf_hit"],
            prov["phonetic_hit"], prov["rare_token_hit"],
            prov["postal_hit"], prov["city_hit"],
        ])
        prov["num_blockers_hit"] = hits
        ranks = []
        if prov["name_tfidf_rank"] >= 0:
            ranks.append(prov["name_tfidf_rank"])
        if prov["addr_tfidf_rank"] >= 0:
            ranks.append(prov["addr_tfidf_rank"])
        prov["best_block_rank"] = min(ranks) if ranks else 999

    avg_cands = np.mean([len(v) for v in all_candidates.values()]) if all_candidates else 0
    logger.info(f"  Blocking complete [{country}]: "
                f"{sum(len(v) for v in all_candidates.values()):,} total pairs, "
                f"avg {avg_cands:.1f} per S1")

    return dict(all_candidates), dict(provenance), name_candidates, name_blocker, s1_name_vecs, s2s3_name_vecs


# ──────────────────────────────────────────────
#  Reciprocal retrieval (independent reverse)
# ──────────────────────────────────────────────
@timed
def run_reciprocal_retrieval(s1_name_vecs, s2s3_name_vecs, s1_ids, cand_ids,
                             forward_candidates, topk=20):
    """
    Run INDEPENDENT reverse retrieval: S2S3 -> S1.
    Joins with forward retrieval to produce reciprocal rank features.
    Optimized: queries ONLY unique candidates that appear in forward_candidates,
    drastically lowering memory and CPU time.
    """
    if not forward_candidates:
        return {}

    cand_idx_map = {eid: i for i, eid in enumerate(cand_ids)}
    unique_cand_ids = list({cand_id for cands in forward_candidates.values() for cand_id, _, _ in cands if cand_id in cand_idx_map})
    if not unique_cand_ids:
        return {}

    sub_cand_indices = [cand_idx_map[cid] for cid in unique_cand_ids]
    sub_s2s3_vecs = s2s3_name_vecs[sub_cand_indices]

    logger.info(f"Running reciprocal retrieval: {len(unique_cand_ids):,} unique S2S3 -> "
                f"{s1_name_vecs.shape[0]:,} S1, top-{topk} ...")

    # Independent reverse: unique S2S3 queries against S1 corpus
    try:
        reverse_sim = awesome_cossim_topn(
            sub_s2s3_vecs, s1_name_vecs.T,
            ntop=topk, lower_bound=config.TFIDF_MIN_SIMILARITY,
            use_threads=True, n_jobs=4,
        )
    except TypeError:
        reverse_sim = awesome_cossim_topn(
            sub_s2s3_vecs, s1_name_vecs.T,
            ntop=topk, lower_bound=config.TFIDF_MIN_SIMILARITY,
        )

    logger.info(f"  Reverse retrieval: {reverse_sim.nnz:,} pairs")

    # Build reverse index directly using CSR arrays (fast & memory-efficient)
    indptr = reverse_sim.indptr
    indices = reverse_sim.indices
    data = reverse_sim.data
    s1_idx_map = {eid: i for i, eid in enumerate(s1_ids)}

    reverse_ranks = {}  # cand_id -> {s1_local_idx: rank}
    for local_idx, cid in enumerate(unique_cand_ids):
        start, end = indptr[local_idx], indptr[local_idx + 1]
        if start == end:
            continue
        row_indices = indices[start:end]
        row_data = data[start:end]
        order = np.argsort(-row_data)
        rank_map = {}
        for rank, pos in enumerate(order):
            s1_local_idx = row_indices[pos]
            rank_map[s1_local_idx] = rank
        reverse_ranks[cid] = rank_map

    del reverse_sim, sub_s2s3_vecs
    gc.collect()

    # Join forward + reverse ranks
    reciprocal_features = {}
    for s1_id, cands in forward_candidates.items():
        s1_idx = s1_idx_map.get(s1_id)
        if s1_idx is None:
            continue
        for cand_id, fwd_rank, fwd_score in cands:
            rev_rank_map = reverse_ranks.get(cand_id, {})
            rev_rank = rev_rank_map.get(s1_idx, -1)

            reciprocal_features[(s1_id, cand_id)] = {
                "s1_to_cand_rank": fwd_rank,
                "cand_to_s1_rank": rev_rank,
                "is_mutual_top1": (fwd_rank == 0 and rev_rank == 0),
                "is_mutual_top5": (fwd_rank < 5 and 0 <= rev_rank < 5),
                "reciprocal_rank_product": (
                    (1.0 / (fwd_rank + 1)) * (1.0 / (rev_rank + 1))
                    if rev_rank >= 0 else 0.0
                ),
            }

    del reverse_ranks, s1_idx_map, cand_idx_map, unique_cand_ids
    gc.collect()

    logger.info(f"  Reciprocal features computed for {len(reciprocal_features):,} pairs")
    return reciprocal_features


# ──────────────────────────────────────────────
#  Blocking recall evaluation
# ──────────────────────────────────────────────
def evaluate_blocking_recall(candidates: dict, ground_truth: dict) -> dict:
    """Evaluate blocking recall against ground truth."""
    true_positives = 0
    total_true = 0

    for s1_id, true_matches in ground_truth.items():
        if not true_matches:
            continue
        cands = set(candidates.get(s1_id, []))
        for m in true_matches:
            total_true += 1
            if m in cands:
                true_positives += 1

    recall = true_positives / total_true if total_true > 0 else 0.0
    avg_cands = np.mean([len(v) for v in candidates.values()]) if candidates else 0

    logger.info(f"  Blocking recall: {recall:.4f} ({true_positives:,}/{total_true:,}), "
                f"avg candidates: {avg_cands:.1f}")
    return {"recall": recall, "tp": true_positives, "total": total_true, "avg_cands": avg_cands}
