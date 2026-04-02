"""Correlation analysis: Spearman, Kendall, per-performer breakdown."""

from typing import Dict, List, Optional

import numpy as np
from scipy import stats


def spearman_correlation(predicted: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
    """Compute Spearman rank correlation.

    Args:
        predicted: [N] or [N, D] predicted scores
        ground_truth: [N] or [N, D] ground truth scores

    Returns dict with rho and p-value (per-dimension if multi-dimensional).
    """
    if predicted.ndim == 1:
        rho, p = stats.spearmanr(predicted, ground_truth)
        return {"rho": rho, "p_value": p}

    results = {}
    dim_names = ["rhythm", "pitch", "completeness", "flow", "overall"]
    for i in range(predicted.shape[1]):
        name = dim_names[i] if i < len(dim_names) else f"dim_{i}"
        rho, p = stats.spearmanr(predicted[:, i], ground_truth[:, i])
        results[f"{name}_rho"] = rho
        results[f"{name}_p"] = p

    return results


def kendall_tau(predicted: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
    """Compute Kendall's tau for ranking accuracy."""
    if predicted.ndim == 1:
        tau, p = stats.kendalltau(predicted, ground_truth)
        return {"tau": tau, "p_value": p}

    results = {}
    dim_names = ["rhythm", "pitch", "completeness", "flow", "overall"]
    for i in range(predicted.shape[1]):
        name = dim_names[i] if i < len(dim_names) else f"dim_{i}"
        tau, p = stats.kendalltau(predicted[:, i], ground_truth[:, i])
        results[f"{name}_tau"] = tau
        results[f"{name}_p"] = p

    return results


def per_performer_correlation(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    performer_ids: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute Spearman correlation separately for each performer.

    Returns {performer_id: {rho, p_value}} for each unique performer.
    """
    unique_performers = sorted(set(performer_ids))
    results = {}

    for pid in unique_performers:
        mask = np.array([p == pid for p in performer_ids])
        if mask.sum() < 3:
            results[pid] = {"rho": float('nan'), "p_value": float('nan'), "n_samples": int(mask.sum())}
            continue

        pred_sub = predicted[mask]
        gt_sub = ground_truth[mask]

        if pred_sub.ndim > 1:
            pred_sub = pred_sub[:, -1]  # overall
        if gt_sub.ndim > 1:
            gt_sub = gt_sub[:, -1]

        rho, p = stats.spearmanr(pred_sub, gt_sub)
        results[pid] = {"rho": rho, "p_value": p, "n_samples": int(mask.sum())}

    return results
