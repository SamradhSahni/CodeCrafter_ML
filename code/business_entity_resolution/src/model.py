"""
Phase 4: ML matching model — LightGBM with structured negative mining,
OOF hard-negative mining, threshold optimization, and two-stage refinement.
"""
import gc
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm

from . import config
from .utils import logger, timed, compute_macro_f05
from .features import compute_density_features, DENSITY_FEATURE_NAMES


# ──────────────────────────────────────────────
#  Training pair construction
# ──────────────────────────────────────────────
@timed
def construct_training_pairs(ground_truth: dict, candidates: dict,
                             s1_df: pd.DataFrame, s2s3_df: pd.DataFrame,
                             token_idf: dict):
    """
    Build structured positive/negative pairs from GT + blocking candidates.

    Returns: list of (s1_id, cand_id, label) tuples.
    """
    logger.info("Constructing training pairs ...")

    positives = []
    for s1_id, true_matches in ground_truth.items():
        for m in true_matches:
            positives.append((s1_id, m, 1))

    logger.info(f"  Positives: {len(positives):,}")

    # Collect all negative candidates (blocking candidates that are NOT true matches)
    all_negatives = []
    for s1_id, cand_set in candidates.items():
        true_set = ground_truth.get(s1_id, set())
        for cand_id in cand_set:
            if cand_id not in true_set:
                all_negatives.append((s1_id, cand_id, 0))

    logger.info(f"  All available negatives: {len(all_negatives):,}")

    # Structured sampling: up to 4x positives
    target_neg = min(len(all_negatives), len(positives) * 4)
    rng = np.random.RandomState(config.RANDOM_SEED)

    if len(all_negatives) <= target_neg:
        sampled_negatives = all_negatives
    else:
        # Structured negative sampling
        neg_by_type = {"random": [], "name": [], "city": [], "other": []}

        for s1_id, cand_id, label in all_negatives:
            s1_rec = s1_df.loc[s1_id] if s1_id in s1_df.index else None
            cand_rec = s2s3_df.loc[cand_id] if cand_id in s2s3_df.index else None

            if s1_rec is None or cand_rec is None:
                neg_by_type["random"].append((s1_id, cand_id, 0))
                continue

            # Categorize by similarity type
            s1_city = s1_rec.get("city", "")
            cand_city = cand_rec.get("city", "")

            s1_name = s1_rec.get("name_core", "")
            cand_name = cand_rec.get("name_core", "")

            if s1_name and cand_name and s1_name.split()[0] == cand_name.split()[0]:
                neg_by_type["name"].append((s1_id, cand_id, 0))
            elif s1_city and cand_city and s1_city == cand_city:
                neg_by_type["city"].append((s1_id, cand_id, 0))
            else:
                neg_by_type["random"].append((s1_id, cand_id, 0))

        # Sample ~25% from each category
        quarter = target_neg // 4
        sampled_negatives = []
        for cat_name, cat_negs in neg_by_type.items():
            n_sample = min(quarter, len(cat_negs))
            if n_sample > 0:
                indices = rng.choice(len(cat_negs), n_sample, replace=False)
                sampled_negatives.extend([cat_negs[i] for i in indices])

        # Fill remainder with random
        remaining = target_neg - len(sampled_negatives)
        if remaining > 0 and neg_by_type["random"]:
            extra = min(remaining, len(neg_by_type["random"]))
            indices = rng.choice(len(neg_by_type["random"]), extra, replace=False)
            sampled_negatives.extend([neg_by_type["random"][i] for i in indices])

    logger.info(f"  Sampled negatives: {len(sampled_negatives):,}")

    all_pairs = positives + sampled_negatives
    rng.shuffle(all_pairs)

    logger.info(f"  Total training pairs: {len(all_pairs):,} "
                f"(pos={len(positives):,}, neg={len(sampled_negatives):,}, "
                f"ratio=1:{len(sampled_negatives)/max(len(positives),1):.1f})")

    return all_pairs


# ──────────────────────────────────────────────
#  OOF hard negative mining
# ──────────────────────────────────────────────
@timed
def mine_hard_negatives_oof(X_train: np.ndarray, y_train: np.ndarray,
                            train_pairs: list, n_folds: int = 5,
                            threshold: float = 0.5) -> list:
    """
    Mine hard negatives using out-of-fold predictions.
    Returns list of (s1_id, cand_id, 0) for hard negatives.
    NEVER uses validation data.
    """
    logger.info(f"Mining hard negatives via OOF ({n_folds} folds) ...")

    n_pos = (y_train == 1).sum()
    if n_pos < n_folds:
        logger.warning(f"  Too few positives ({n_pos}) for {n_folds}-fold OOF, skipping hard neg mining")
        return []

    oof_scores = np.zeros(len(X_train))

    # Use stratified KFold to ensure each fold has positives
    from sklearn.model_selection import StratifiedKFold
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_SEED)

    for fold, (train_idx, oof_idx) in enumerate(kf.split(X_train, y_train)):
        try:
            fold_model = lgb.LGBMClassifier(**config.LGBM_PARAMS)
            fold_model.fit(
                X_train[train_idx], y_train[train_idx],
                eval_set=[(X_train[oof_idx], y_train[oof_idx])],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
            )
            oof_scores[oof_idx] = fold_model.predict_proba(X_train[oof_idx])[:, 1]
        except (ValueError, Exception) as e:
            logger.warning(f"  Fold {fold + 1} failed: {e}")
            continue

    # Hard negatives: true negatives with high predicted probability
    hard_mask = (y_train == 0) & (oof_scores > threshold)
    hard_count = hard_mask.sum()
    hard_pairs = [train_pairs[i] for i in range(len(train_pairs)) if hard_mask[i]]

    logger.info(f"  Found {hard_count:,} hard negatives "
                f"(neg with OOF score > {threshold})")
    return hard_pairs


# ──────────────────────────────────────────────
#  Model training
# ──────────────────────────────────────────────
@timed
def train_model(X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray = None, y_val: np.ndarray = None,
                params: dict = None) -> lgb.LGBMClassifier:
    """Train a LightGBM classifier."""
    params = params or config.LGBM_PARAMS.copy()

    # Set class weight
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    if n_pos > 0:
        params["scale_pos_weight"] = n_neg / n_pos
        logger.info(f"  scale_pos_weight: {params['scale_pos_weight']:.2f}")

    model = lgb.LGBMClassifier(**params)

    fit_kwargs = {}
    if X_val is not None and y_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["callbacks"] = [
            lgb.early_stopping(50, verbose=True),
            lgb.log_evaluation(100),
        ]

    model.fit(X_train, y_train, **fit_kwargs)

    if X_val is not None:
        val_probs = model.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(y_val, val_preds)
        f1 = f1_score(y_val, val_preds)
        logger.info(f"  Validation accuracy: {acc:.4f}, F1: {f1:.4f}")

    return model


# ──────────────────────────────────────────────
#  OOF Stage-1 scores for Stage-2 training
# ──────────────────────────────────────────────
@timed
def generate_oof_stage1_scores(X_train: np.ndarray, y_train: np.ndarray,
                               n_folds: int = 5) -> np.ndarray:
    """Generate Stage-1 scores via OOF to avoid information leakage into Stage 2."""
    logger.info(f"Generating OOF Stage-1 scores ({n_folds} folds) ...")

    n_pos = (y_train == 1).sum()
    if n_pos < n_folds:
        logger.warning(f"  Too few positives ({n_pos}), using direct prediction instead of OOF")
        model = lgb.LGBMClassifier(**config.LGBM_PARAMS)
        model.fit(X_train, y_train)
        return model.predict_proba(X_train)[:, 1].astype(np.float32)

    oof_scores = np.zeros(len(X_train), dtype=np.float32)

    from sklearn.model_selection import StratifiedKFold
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_SEED)

    for fold, (train_idx, oof_idx) in enumerate(kf.split(X_train, y_train)):
        logger.info(f"  Fold {fold + 1}/{n_folds} ...")
        try:
            fold_model = lgb.LGBMClassifier(**config.LGBM_PARAMS)
            fold_model.fit(
                X_train[train_idx], y_train[train_idx],
                eval_set=[(X_train[oof_idx], y_train[oof_idx])],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
            )
            oof_scores[oof_idx] = fold_model.predict_proba(X_train[oof_idx])[:, 1]
        except (ValueError, Exception) as e:
            logger.warning(f"  Fold {fold + 1} failed: {e}")
            continue

    return oof_scores


# ──────────────────────────────────────────────
#  Threshold optimization
# ──────────────────────────────────────────────
@timed
def optimize_threshold(probabilities: np.ndarray, y_true: np.ndarray,
                       pair_s1_ids: np.ndarray, ground_truth: dict,
                       countries: np.ndarray = None) -> dict:
    """
    Sweep thresholds to maximize macro F0.5.
    Returns dict of thresholds (global + per-country if provided).
    """
    logger.info("Optimizing threshold for F0.5 ...")

    def _eval_threshold(probs, labels, s1_ids, gt, thresh):
        """Build predictions dict at given threshold and evaluate."""
        preds = defaultdict(set)
        for i, (s1_id, prob) in enumerate(zip(s1_ids, probs)):
            if prob >= thresh and labels[i] >= 0:  # labels[i] can be -1 for unlabeled
                # We need cand_id here, but for simplicity we use a different approach
                pass

        # Actually build predictions from pairs
        pred_dict = defaultdict(set)
        for i in range(len(probs)):
            if probs[i] >= thresh:
                s1_id = s1_ids[i]
                # We need the cand_id — passed via separate array
                pass

        return 0.0

    # Simpler approach: build predictions from probabilities
    def sweep_threshold(probs, s1_ids, cand_ids, gt):
        best_f05, best_thresh = 0.0, 0.5
        for thresh in np.arange(0.30, 0.95, 0.005):
            pred_dict = defaultdict(set)
            for i in range(len(probs)):
                if probs[i] >= thresh:
                    pred_dict[s1_ids[i]].add(cand_ids[i])
            # Ensure all GT s1_ids are in pred_dict
            for s1_id in gt:
                if s1_id not in pred_dict:
                    pred_dict[s1_id] = set()
            f05 = compute_macro_f05(dict(pred_dict), gt)
            if f05 > best_f05:
                best_f05, best_thresh = f05, thresh
        return best_thresh, best_f05

    return sweep_threshold


@timed
def find_best_threshold(probs, s1_ids, cand_ids, ground_truth, countries=None):
    """Find the best threshold(s) for F0.5."""
    logger.info(f"Sweeping thresholds across {len(probs):,} predictions ...")

    best_f05, best_thresh = 0.0, 0.5
    for thresh in np.arange(0.30, 0.95, 0.005):
        pred_dict = defaultdict(set)
        for i in range(len(probs)):
            if probs[i] >= thresh:
                pred_dict[s1_ids[i]].add(cand_ids[i])
        # Add singletons
        for s1_id in ground_truth:
            if s1_id not in pred_dict:
                pred_dict[s1_id] = set()
        f05 = compute_macro_f05(dict(pred_dict), ground_truth)
        if f05 > best_f05:
            best_f05, best_thresh = f05, thresh

    logger.info(f"  Best threshold: {best_thresh:.3f}, F0.5: {best_f05:.4f}")

    result = {"global": best_thresh, "global_f05": best_f05}

    # Per-country if provided
    if countries is not None:
        unique_countries = set(countries)
        for country in unique_countries:
            mask = np.array(countries) == country
            c_probs = probs[mask]
            c_s1 = s1_ids[mask]
            c_cands = cand_ids[mask]

            # Build per-country GT
            c_gt = {s1: gt for s1, gt in ground_truth.items()
                    if s1 in set(c_s1)}
            if not c_gt:
                continue

            c_best_f05, c_best_thresh = 0.0, best_thresh
            for thresh in np.arange(0.30, 0.95, 0.005):
                pred_dict = defaultdict(set)
                for i in range(len(c_probs)):
                    if c_probs[i] >= thresh:
                        pred_dict[c_s1[i]].add(c_cands[i])
                for s1_id in c_gt:
                    if s1_id not in pred_dict:
                        pred_dict[s1_id] = set()
                f05 = compute_macro_f05(dict(pred_dict), c_gt)
                if f05 > c_best_f05:
                    c_best_f05, c_best_thresh = f05, thresh

            result[country] = c_best_thresh
            result[f"{country}_f05"] = c_best_f05
            logger.info(f"  {country}: threshold={c_best_thresh:.3f}, F0.5={c_best_f05:.4f}")

    return result


# ──────────────────────────────────────────────
#  Model save / load
# ──────────────────────────────────────────────
def save_model(model, path: Path):
    """Save model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {path}")


def load_model(path: Path):
    """Load model from disk."""
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Model loaded from {path}")
    return model


# ──────────────────────────────────────────────
#  Feature importance
# ──────────────────────────────────────────────
def log_feature_importance(model, feature_names: list, top_n: int = 20):
    """Log top-N feature importances."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    logger.info(f"\nTop-{top_n} Feature Importances:")
    logger.info("-" * 50)
    for i in range(min(top_n, len(feature_names))):
        idx = indices[i]
        logger.info(f"  {i + 1:2d}. {feature_names[idx]:35s} {importances[idx]:8d}")
