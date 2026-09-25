"""
Shared utility functions: I/O, evaluation, and data helpers.
"""
import gc
import time
import logging
import functools
from collections import defaultdict

import numpy as np
import pandas as pd

from . import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ER")


# ──────────────────────────────────────────────
#  Timing decorator
# ──────────────────────────────────────────────
def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        logger.info(f"{func.__name__} completed in {elapsed:.1f}s")
        return result
    return wrapper


# ──────────────────────────────────────────────
#  Data I/O
# ──────────────────────────────────────────────
@timed
def load_source(path, nrows=None):
    """Load a TSV source file."""
    logger.info(f"Loading {path.name} ...")
    df = pd.read_csv(path, sep="\t", dtype=str, nrows=nrows)
    df = df.fillna("")
    logger.info(f"  -> {len(df):,} rows, columns: {list(df.columns)}")
    return df


@timed
def load_ground_truth(path, s1_ids_filter=None):
    """Load ground truth TSV and parse matched IDs into sets (vectorized)."""
    logger.info(f"Loading ground truth from {path.name} ...")
    df = pd.read_csv(path, sep="\t", dtype=str)

    # Optional: filter to only relevant S1 IDs (for subsetted runs)
    if s1_ids_filter is not None:
        s1_set = set(s1_ids_filter)
        df = df[df["source1_entity_id"].isin(s1_set)]
        logger.info(f"  Filtered to {len(df):,} rows matching {len(s1_set):,} S1 IDs")

    # Vectorized parsing — avoid iterrows
    gt = {}
    s1_col = df["source1_entity_id"].values
    match_col = df["matched_entity_ids"].values

    for i in range(len(s1_col)):
        s1_id = s1_col[i]
        matched = match_col[i]
        if pd.isna(matched) or (isinstance(matched, str) and matched.strip() == ""):
            gt[s1_id] = set()
        else:
            gt[s1_id] = set(m.strip() for m in str(matched).split(",") if m.strip())

    n_with = sum(1 for v in gt.values() if v)
    logger.info(f"  -> {len(gt):,} S1 entities, {n_with:,} with matches")
    return gt


# ──────────────────────────────────────────────
#  Evaluation: Macro F0.5
# ──────────────────────────────────────────────
def compute_macro_f05(predictions: dict, ground_truth: dict) -> float:
    """
    Compute macro-averaged F_beta (beta=0.5) across all S1 entities.
    predictions: {s1_id: set(matched_ids)}
    ground_truth: {s1_id: set(matched_ids)}
    """
    scores = []
    for s1_id in ground_truth:
        true = ground_truth[s1_id]
        pred = predictions.get(s1_id, set())

        if len(true) == 0 and len(pred) == 0:
            scores.append(1.0)   # correct singleton
        elif len(pred) == 0:
            scores.append(0.0)   # missed all matches
        elif len(true) == 0:
            scores.append(0.0)   # false positives on singleton
        else:
            tp = len(true & pred)
            precision = tp / len(pred) if len(pred) > 0 else 0.0
            recall = tp / len(true) if len(true) > 0 else 0.0
            if precision + recall == 0:
                scores.append(0.0)
            else:
                f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
                scores.append(f05)

    return float(np.mean(scores))


# ──────────────────────────────────────────────
#  Ground truth helpers
# ──────────────────────────────────────────────
def discover_constraints(ground_truth: dict) -> dict:
    """Check whether S2/S3 entities can match multiple S1 entities."""
    reverse_map = defaultdict(set)
    for s1_id, matched in ground_truth.items():
        for mid in matched:
            reverse_map[mid].add(s1_id)

    s2_counts = [len(v) for k, v in reverse_map.items() if k.startswith("S2")]
    s3_counts = [len(v) for k, v in reverse_map.items() if k.startswith("S3")]

    max_s2 = max(s2_counts) if s2_counts else 0
    max_s3 = max(s3_counts) if s3_counts else 0

    logger.info(f"Constraint check: max S1 per S2 = {max_s2}, max S1 per S3 = {max_s3}")

    return {
        "s2_unique": (max_s2 <= 1),
        "s3_unique": (max_s3 <= 1),
    }


def verify_country_gate(ground_truth: dict, s1_df: pd.DataFrame, s2s3_df: pd.DataFrame) -> float:
    """Check if any true match pairs have different country labels."""
    # DataFrames are indexed by entity_id
    s1_country = s1_df["country"].to_dict()
    s2s3_country = s2s3_df["country"].to_dict()

    cross_country = 0
    total = 0
    for s1_id, matched in ground_truth.items():
        c1 = s1_country.get(s1_id, "")
        for mid in matched:
            c2 = s2s3_country.get(mid, "")
            total += 1
            if c1 != c2 and c1 and c2:
                cross_country += 1

    rate = cross_country / total if total else 0.0
    logger.info(f"Cross-country true pairs: {cross_country}/{total} ({rate:.4%})")
    return rate


# ──────────────────────────────────────────────
#  Output generation
# ──────────────────────────────────────────────
def generate_output_files(predictions: dict, candidates: dict, all_s1_ids: list):
    """
    Write matching_results.tsv and candidate_pairs.tsv.
    predictions: {s1_id: list[str]}
    candidates:  {s1_id: list[str]}
    """
    out_dir = config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # matching_results.tsv
    rows_match = []
    for s1_id in all_s1_ids:
        matched = predictions.get(s1_id, [])
        rows_match.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(matched) if matched else "",
        })
    df_match = pd.DataFrame(rows_match)
    match_path = out_dir / "matching_results.tsv"
    df_match.to_csv(match_path, sep="\t", index=False)
    logger.info(f"Wrote {match_path}  ({len(df_match):,} rows)")

    # candidate_pairs.tsv
    rows_cand = []
    for s1_id in all_s1_ids:
        cands = candidates.get(s1_id, [])
        rows_cand.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(cands) if cands else "",
        })
    df_cand = pd.DataFrame(rows_cand)
    cand_path = out_dir / "candidate_pairs.tsv"
    df_cand.to_csv(cand_path, sep="\t", index=False)
    logger.info(f"Wrote {cand_path}  ({len(df_cand):,} rows)")

    # Sanity: every match should be in candidates
    orphans_total = 0
    for s1_id in all_s1_ids:
        matched_set = set(predictions.get(s1_id, []))
        cand_set = set(candidates.get(s1_id, []))
        orphans = matched_set - cand_set
        if orphans:
            orphans_total += len(orphans)
    if orphans_total:
        logger.warning(f"WARNING: {orphans_total} matched IDs are NOT in candidates!")
    else:
        logger.info("All matched IDs are present in candidate pairs.")


def free_memory():
    """Force garbage collection."""
    gc.collect()
