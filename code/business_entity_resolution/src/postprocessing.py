"""
Phase 5: Post-processing — GT-derived constraint validation, conflict resolution,
singleton handling, and output generation.
"""
from collections import defaultdict

import numpy as np
import pandas as pd

from .utils import logger, timed, discover_constraints


# ──────────────────────────────────────────────
#  Resolve conflicts (if GT shows uniqueness)
# ──────────────────────────────────────────────
@timed
def resolve_conflicts(predictions: dict, confidences: dict,
                      constraints: dict) -> dict:
    """
    If GT shows S2/S3 uniqueness constraint, resolve multi-claimant conflicts
    by keeping highest-confidence assignment.

    predictions: {s1_id: set(cand_ids)}
    confidences: {(s1_id, cand_id): float probability}
    constraints: from discover_constraints()
    """
    if not constraints.get("s2_unique") and not constraints.get("s3_unique"):
        logger.info("No uniqueness constraints to enforce.")
        return predictions

    # Build reverse index: cand_id -> [(s1_id, confidence)]
    reverse = defaultdict(list)
    for s1_id, cand_set in predictions.items():
        for cand_id in cand_set:
            conf = confidences.get((s1_id, cand_id), 0.0)
            reverse[cand_id].append((s1_id, conf))

    # Resolve conflicts
    conflicts_resolved = 0
    resolved = defaultdict(set)

    for cand_id, claimants in reverse.items():
        source = "s2" if cand_id.startswith("S2") else "s3"
        unique_key = f"{source}_unique"

        if constraints.get(unique_key, False) and len(claimants) > 1:
            # Conflict: multiple S1 entities claim the same S2/S3 record
            best_s1, best_conf = max(claimants, key=lambda x: x[1])
            resolved[best_s1].add(cand_id)
            conflicts_resolved += 1
        else:
            for s1_id, conf in claimants:
                resolved[s1_id].add(cand_id)

    logger.info(f"Conflict resolution: {conflicts_resolved:,} conflicts resolved")
    return dict(resolved)


# ──────────────────────────────────────────────
#  Apply threshold and build predictions
# ──────────────────────────────────────────────
@timed
def apply_threshold(probabilities: np.ndarray, s1_ids: np.ndarray,
                    cand_ids: np.ndarray, thresholds: dict,
                    countries: np.ndarray = None) -> tuple:
    """
    Apply threshold to probabilities, returning predictions and confidences.

    Returns:
        predictions: {s1_id: set(cand_ids)}
        confidences: {(s1_id, cand_id): float}
    """
    global_thresh = thresholds.get("global", 0.5)
    predictions = defaultdict(set)
    confidences = {}

    for i in range(len(probabilities)):
        s1_id = s1_ids[i]
        cand_id = cand_ids[i]
        prob = probabilities[i]

        # Determine threshold for this entity
        if countries is not None and countries[i] in thresholds:
            thresh = thresholds[countries[i]]
        else:
            thresh = global_thresh

        if prob >= thresh:
            predictions[s1_id].add(cand_id)
            confidences[(s1_id, cand_id)] = float(prob)

    logger.info(f"Applied threshold: {sum(len(v) for v in predictions.values()):,} matches "
                f"across {len(predictions):,} S1 entities")
    return dict(predictions), confidences


# ──────────────────────────────────────────────
#  Full post-processing pipeline
# ──────────────────────────────────────────────
@timed
def postprocess(probabilities: np.ndarray, s1_ids: np.ndarray,
                cand_ids: np.ndarray, thresholds: dict,
                ground_truth: dict = None, all_s1_ids: list = None,
                countries: np.ndarray = None) -> dict:
    """
    Full post-processing: threshold -> constraint resolution -> ensure all S1 present.
    """
    # Step 1: Apply threshold
    predictions, confidences = apply_threshold(
        probabilities, s1_ids, cand_ids, thresholds, countries
    )

    # Step 2: Discover and apply constraints (only if GT available)
    if ground_truth is not None:
        constraints = discover_constraints(ground_truth)
        predictions = resolve_conflicts(predictions, confidences, constraints)

    # Step 3: Ensure all S1 entities are present (singletons get empty set)
    if all_s1_ids is not None:
        for s1_id in all_s1_ids:
            if s1_id not in predictions:
                predictions[s1_id] = set()

    n_with_matches = sum(1 for v in predictions.values() if v)
    n_singletons = sum(1 for v in predictions.values() if not v)
    logger.info(f"Post-processing complete: {n_with_matches:,} with matches, "
                f"{n_singletons:,} singletons")

    return predictions
