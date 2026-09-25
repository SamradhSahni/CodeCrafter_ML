"""
Phase 3: Feature engineering — 60+ features across 8 categories.

All features are computed ONLY for candidate pairs from blocking (Phase 2).
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
import jellyfish
from tqdm import tqdm

from . import config
from .utils import logger, timed


# ──────────────────────────────────────────────
#  Helper: three-state evidence
# ──────────────────────────────────────────────
def three_state(val_a: str, val_b: str) -> int:
    """
    +1 = both present and matching
     0 = one or both missing (neutral)
    -1 = both present but conflicting (strong negative)
    """
    a_empty = (not val_a or val_a in ("", "nan", "none"))
    b_empty = (not val_b or val_b in ("", "nan", "none"))
    if a_empty or b_empty:
        return 0
    if val_a == val_b:
        return 1
    return -1


# ──────────────────────────────────────────────
#  Helper: Jaccard similarity
# ──────────────────────────────────────────────
def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def overlap_coeff(set_a: set, set_b: set) -> float:
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    return inter / min(len(set_a), len(set_b))


def dice_coeff(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 0.0
    inter = len(set_a & set_b)
    return (2 * inter) / (len(set_a) + len(set_b)) if (len(set_a) + len(set_b)) > 0 else 0.0


def char_ngrams(text: str, n: int = 3) -> set:
    if not text or len(text) < n:
        return set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


# ──────────────────────────────────────────────
#  Feature computation for a single pair
# ──────────────────────────────────────────────
def compute_pair_features(s1_rec: dict, cand_rec: dict,
                          provenance: dict, reciprocal: dict,
                          token_idf: dict,
                          embedding_provenance: dict = None,
                          embedding_reciprocal: dict = None) -> dict:
    """
    Compute all features for one (S1, candidate) pair.
    s1_rec / cand_rec: preprocessed record dicts.
    provenance: blocker provenance dict for this pair.
    reciprocal: reciprocal retrieval dict for this pair.
    token_idf: word-level IDF dictionary.
    """
    feats = {}

    # Shorthand
    n1 = s1_rec.get("name_clean", "")
    n2 = cand_rec.get("name_clean", "")
    nc1 = s1_rec.get("name_core", "")
    nc2 = cand_rec.get("name_core", "")
    a1 = s1_rec.get("addr_clean", "")
    a2 = cand_rec.get("addr_clean", "")

    t1 = set(nc1.split()) if nc1 else set()
    t2 = set(nc2.split()) if nc2 else set()
    at1 = set(a1.split()) if a1 else set()
    at2 = set(a2.split()) if a2 else set()

    # ── 3.1 Name similarity features (14) ──
    feats["name_jaro_winkler"] = jellyfish.jaro_winkler_similarity(n1, n2) if n1 and n2 else 0.0
    feats["name_levenshtein_ratio"] = fuzz.ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
    feats["name_partial_ratio"] = fuzz.partial_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
    feats["name_token_sort_ratio"] = fuzz.token_sort_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
    feats["name_token_set_ratio"] = fuzz.token_set_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
    feats["name_jaccard_tokens"] = jaccard(t1, t2)
    feats["name_jaccard_3grams"] = jaccard(char_ngrams(n1, 3), char_ngrams(n2, 3))
    feats["name_overlap_coeff"] = overlap_coeff(t1, t2)
    feats["name_dice_coeff"] = dice_coeff(t1, t2)
    feats["name_core_jaro"] = jellyfish.jaro_winkler_similarity(nc1, nc2) if nc1 and nc2 else 0.0
    feats["name_core_exact"] = 1.0 if nc1 and nc2 and nc1 == nc2 else 0.0
    feats["name_unicode_similarity"] = (
        fuzz.token_set_ratio(
            s1_rec.get("name_unicode", ""),
            cand_rec.get("name_unicode", "")
        ) / 100.0
    ) if s1_rec.get("name_unicode") and cand_rec.get("name_unicode") else 0.0
    feats["name_len_ratio"] = (
        min(len(n1), len(n2)) / max(len(n1), len(n2))
        if n1 and n2 else 0.0
    )

    # Suffix compatibility
    ls1 = set(s1_rec.get("legal_suffix", "").split("|")) - {""}
    ls2 = set(cand_rec.get("legal_suffix", "").split("|")) - {""}
    if ls1 and ls2:
        feats["suffix_compatible"] = 1.0 if ls1 & ls2 else 0.0
    else:
        feats["suffix_compatible"] = 0.5  # neutral when one is missing

    # ── 3.2 Address similarity with three-state evidence (15) ──
    feats["addr_jaccard_tokens"] = jaccard(at1, at2)
    feats["addr_token_sort_ratio"] = fuzz.token_sort_ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
    feats["addr_token_set_ratio"] = fuzz.token_set_ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
    feats["addr_levenshtein"] = fuzz.ratio(a1, a2) / 100.0 if a1 and a2 else 0.0

    postal_rel = three_state(s1_rec.get("postal_code", ""), cand_rec.get("postal_code", ""))
    city_rel = three_state(s1_rec.get("city", ""), cand_rec.get("city", ""))
    state_rel = three_state(s1_rec.get("state", ""), cand_rec.get("state", ""))
    house_rel = three_state(s1_rec.get("street_number", ""), cand_rec.get("street_number", ""))

    feats["postal_relation"] = postal_rel
    feats["city_relation"] = city_rel
    feats["state_relation"] = state_rel
    feats["house_number_relation"] = house_rel
    feats["num_address_agreements"] = sum(1 for r in [postal_rel, city_rel, state_rel, house_rel] if r == 1)
    feats["num_address_conflicts"] = sum(1 for r in [postal_rel, city_rel, state_rel, house_rel] if r == -1)

    # Numeric token overlap
    nums1 = {t for t in at1 if t.isdigit()}
    nums2 = {t for t in at2 if t.isdigit()}
    feats["addr_numeric_overlap"] = jaccard(nums1, nums2)
    feats["addr_has_null"] = 1.0 if (not a1 or not a2) else 0.0

    # Landmark overlap
    lm1 = set(s1_rec.get("landmark_tokens", "").split()) - {""}
    lm2 = set(cand_rec.get("landmark_tokens", "").split()) - {""}
    feats["landmark_overlap"] = jaccard(lm1, lm2) if lm1 or lm2 else 0.0

    # Address component overlap
    components = ["postal_code", "city", "state", "street_number"]
    matching = sum(1 for c in components
                   if s1_rec.get(c) and cand_rec.get(c) and s1_rec[c] == cand_rec[c])
    present = sum(1 for c in components if s1_rec.get(c) and cand_rec.get(c))
    feats["addr_component_overlap"] = matching / present if present > 0 else 0.0

    # ── 3.3 Blocker provenance (12) ──
    prov = provenance or {}
    feats["name_block_hit"] = 1.0 if prov.get("name_tfidf_hit", False) else 0.0
    feats["addr_block_hit"] = 1.0 if prov.get("addr_tfidf_hit", False) else 0.0
    feats["phonetic_block_hit"] = 1.0 if prov.get("phonetic_hit", False) else 0.0
    feats["rare_token_block_hit"] = 1.0 if prov.get("rare_token_hit", False) else 0.0
    feats["postal_block_hit"] = 1.0 if prov.get("postal_hit", False) else 0.0
    feats["city_block_hit"] = 1.0 if prov.get("city_hit", False) else 0.0
    feats["num_blockers_hit"] = prov.get("num_blockers_hit", 0)
    feats["best_block_rank"] = prov.get("best_block_rank", 999)
    feats["name_block_rank"] = prov.get("name_tfidf_rank", -1)
    feats["addr_block_rank"] = prov.get("addr_tfidf_rank", -1)

    recip = reciprocal or {}
    feats["is_mutual_top5"] = 1.0 if recip.get("is_mutual_top5", False) else 0.0
    feats["reciprocal_rank_product"] = recip.get("reciprocal_rank_product", 0.0)

    # ── 3.4 Rarity / IDF features (6) ──
    shared_tokens = t1 & t2
    all_t1_idf = [token_idf.get(t, 0) for t in t1] if t1 else [0]
    feats["name_token_idf_sum"] = sum(all_t1_idf)
    feats["name_max_token_idf"] = max(all_t1_idf) if all_t1_idf else 0.0

    if shared_tokens:
        shared_idfs = [token_idf.get(t, 0) for t in shared_tokens]
        feats["matched_token_idf_sum"] = sum(shared_idfs)
        feats["rare_token_exact_match"] = 1.0 if any(
            token_idf.get(t, 0) > config.RARE_TOKEN_IDF_THRESHOLD for t in shared_tokens
        ) else 0.0
        feats["rare_token_count"] = sum(
            1 for t in shared_tokens if token_idf.get(t, 0) > config.RARE_TOKEN_IDF_THRESHOLD
        )
    else:
        feats["matched_token_idf_sum"] = 0.0
        feats["rare_token_exact_match"] = 0.0
        feats["rare_token_count"] = 0

    shared_addr = at1 & at2
    feats["addr_token_idf_sum"] = sum(token_idf.get(t, 0) for t in shared_addr) if shared_addr else 0.0

    # ── 3.5 Record quality features (12) ──
    feats["s1_name_length"] = len(n1)
    feats["cand_name_length"] = len(n2)
    feats["s1_name_token_count"] = len(t1)
    feats["cand_name_token_count"] = len(t2)
    feats["s1_addr_length"] = len(a1)
    feats["cand_addr_length"] = len(a2)
    feats["name_is_url"] = 1.0 if cand_rec.get("is_url_name", False) else 0.0
    feats["name_is_too_short"] = 1.0 if len(n2) < 3 else 0.0

    # Generic name score
    if t2:
        generic_count = sum(1 for t in t2 if token_idf.get(t, 0) < config.GENERIC_TOKEN_IDF_THRESHOLD)
        feats["generic_name_score"] = generic_count / len(t2)
    else:
        feats["generic_name_score"] = 1.0

    feats["both_names_high_quality"] = 1.0 if (len(n1) > 10 and len(t1) > 2 and
                                                len(n2) > 10 and len(t2) > 2) else 0.0
    feats["both_addresses_present"] = 1.0 if (a1 and a2) else 0.0
    feats["one_side_low_quality"] = 1.0 if (len(n1) < 3 or len(n2) < 3 or
                                             not a1 or not a2) else 0.0

    # ── 3.6 Generic token features ──
    gt1 = set(s1_rec.get("generic_tokens", "").split("|")) - {""}
    gt2 = set(cand_rec.get("generic_tokens", "").split("|")) - {""}
    feats["generic_token_overlap"] = jaccard(gt1, gt2) if gt1 or gt2 else 0.5
    feats["generic_token_count"] = len(gt1 & gt2)

    # ── 3.8 Meta features (6) ──
    feats["source_type"] = 1.0 if cand_rec.get("entity_id_prefix", "").startswith("S3") else 0.0
    feats["name_length_ratio"] = (
        min(len(n1), len(n2)) / max(len(n1), len(n2)) if n1 and n2 else 0.0
    )
    feats["token_count_difference"] = abs(len(t1) - len(t2))

    # URL stem vs name similarity
    if s1_rec.get("is_url_name") or cand_rec.get("is_url_name"):
        url_stem = s1_rec.get("url_stem", "") or cand_rec.get("url_stem", "")
        other_name = n2 if s1_rec.get("is_url_name") else n1
        if url_stem and other_name:
            # Compare domain stem (e.g. "maurewilliamscolombier") to business name
            other_concat = other_name.replace(" ", "")
            feats["url_stem_vs_name"] = fuzz.ratio(url_stem, other_concat) / 100.0
        else:
            feats["url_stem_vs_name"] = 0.0
    else:
        feats["url_stem_vs_name"] = 0.0

    # Script features
    feats["same_script"] = 1.0 if s1_rec.get("script_type") == cand_rec.get("script_type") else 0.0

    # ── 3.9 Embedding features (7) ──
    emb_prov = embedding_provenance or {}
    emb_recip = embedding_reciprocal or {}
    feats["embedding_cosine_sim"] = emb_prov.get("embedding_score", 0.0)
    feats["embedding_rank"] = emb_prov.get("embedding_rank", 999)
    feats["emb_fwd_rank"] = emb_recip.get("emb_fwd_rank", 999)
    feats["emb_rev_rank"] = emb_recip.get("emb_rev_rank", -1)
    feats["emb_mutual_top1"] = 1.0 if emb_recip.get("emb_mutual_top1", False) else 0.0
    feats["emb_mutual_top5"] = 1.0 if emb_recip.get("emb_mutual_top5", False) else 0.0
    feats["emb_reciprocal_product"] = emb_recip.get("emb_reciprocal_product", 0.0)

    return feats


# ──────────────────────────────────────────────
#  Batch feature computation
# ──────────────────────────────────────────────
@timed
def compute_features_batch(pairs: list, s1_df: pd.DataFrame, s2s3_df: pd.DataFrame,
                           provenance: dict, reciprocal: dict, token_idf: dict,
                           embedding_provenance: dict = None,
                           embedding_reciprocal: dict = None) -> pd.DataFrame:
    """
    Compute features for all candidate pairs.
    pairs: list of (s1_id, cand_id) tuples.
    Returns DataFrame of features.
    """
    logger.info(f"Computing features for {len(pairs):,} pairs ...")
    embedding_provenance = embedding_provenance or {}
    embedding_reciprocal = embedding_reciprocal or {}

    feature_rows = []
    for s1_id, cand_id in tqdm(pairs, desc="Features"):
        s1_rec = s1_df.loc[s1_id].to_dict() if s1_id in s1_df.index else {}
        cand_rec = s2s3_df.loc[cand_id].to_dict() if cand_id in s2s3_df.index else {}

        # Add entity_id prefix for source_type feature
        cand_rec["entity_id_prefix"] = cand_id[:2] if cand_id else ""

        prov = provenance.get((s1_id, cand_id), {})
        recip = reciprocal.get((s1_id, cand_id), {})
        emb_prov = embedding_provenance.get((s1_id, cand_id), {})
        emb_recip = embedding_reciprocal.get((s1_id, cand_id), {})

        feats = compute_pair_features(
            s1_rec, cand_rec, prov, recip, token_idf,
            embedding_provenance=emb_prov,
            embedding_reciprocal=emb_recip,
        )
        feats["s1_id"] = s1_id
        feats["cand_id"] = cand_id
        feature_rows.append(feats)

    df = pd.DataFrame(feature_rows)
    logger.info(f"  -> Feature matrix: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


# ──────────────────────────────────────────────
#  Density / margin features (Stage 2)
# ──────────────────────────────────────────────
def compute_density_features(scores: np.ndarray, s1_ids: np.ndarray) -> np.ndarray:
    """
    Compute per-entity density/margin features from model scores.
    Returns array of shape (len(scores), 8).
    """
    # Group scores by S1 entity
    entity_scores = {}
    for i, (s1_id, score) in enumerate(zip(s1_ids, scores)):
        if s1_id not in entity_scores:
            entity_scores[s1_id] = []
        entity_scores[s1_id].append((i, score))

    density = np.zeros((len(scores), 8), dtype=np.float32)
    for s1_id, items in entity_scores.items():
        sorted_scores = sorted([s for _, s in items], reverse=True)
        top1 = sorted_scores[0] if len(sorted_scores) > 0 else 0.0
        top2 = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        top3 = sorted_scores[2] if len(sorted_scores) > 2 else 0.0
        n_cands = len(sorted_scores)
        n_high = sum(1 for s in sorted_scores if s > 0.5)

        # Entropy
        probs = np.array(sorted_scores)
        probs = probs / (probs.sum() + 1e-10)
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        for idx, score in items:
            density[idx, 0] = top1
            density[idx, 1] = top2
            density[idx, 2] = top3
            density[idx, 3] = top1 - top2
            density[idx, 4] = top1 - top3
            density[idx, 5] = n_cands
            density[idx, 6] = n_high
            density[idx, 7] = entropy

    return density


DENSITY_FEATURE_NAMES = [
    "top1_score", "top2_score", "top3_score",
    "top1_minus_top2", "top1_minus_top3",
    "candidate_count", "high_score_candidate_count", "score_entropy",
]
