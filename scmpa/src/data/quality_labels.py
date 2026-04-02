"""Compute ground truth quality scores from note-level alignment."""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.data.score_parser import NoteEvent


@dataclass
class QualityLabels:
    rhythm: float       # [0, 1] — timing accuracy
    pitch: float        # [0, 1] — pitch accuracy
    completeness: float # [0, 1] — note coverage
    flow: float         # [0, 1] — smoothness / consistency
    overall: float      # [0, 1] — weighted combination


# Notes with cost > threshold are considered unmatched.
# Set high to handle sight-reading at very slow tempos.
# A user playing at 1/4 speed means notes could be 4x later than expected.
# The pitch penalty in the cost function ensures correct matching
# even with this wide time window.
MATCH_COST_THRESHOLD = 10.0

# Pitch mismatch is heavily penalized to ensure correct pitch matching
# even when onset times are far apart due to slow playing.
PITCH_MISMATCH_WEIGHT = 100.0

# Onset distance weight — lowered so pitch dominates matching decisions.
ONSET_DISTANCE_WEIGHT = 0.1


def align_notes(
    original: List[NoteEvent],
    degraded: List[NoteEvent],
) -> Tuple[
    List[Tuple[NoteEvent, NoteEvent]],  # matched pairs
    List[NoteEvent],                     # unmatched originals
    List[NoteEvent],                     # unmatched degraded
]:
    """Match notes using the Hungarian algorithm on (onset, pitch) distance.

    Cost(i,j) = |onset_i - onset_j| + PITCH_MISMATCH_WEIGHT * (pitch_i != pitch_j)
    Pairs with cost > MATCH_COST_THRESHOLD are treated as unmatched.
    """
    n_orig = len(original)
    n_deg = len(degraded)

    if n_orig == 0 or n_deg == 0:
        return [], list(original), list(degraded)

    # Build cost matrix [n_orig, n_deg]
    # Pitch mismatch is heavily penalized so the algorithm strongly prefers
    # matching notes by pitch first, timing second.
    cost = np.zeros((n_orig, n_deg), dtype=np.float64)
    for i, o in enumerate(original):
        for j, d in enumerate(degraded):
            onset_dist = abs(o.onset - d.onset) * ONSET_DISTANCE_WEIGHT
            pitch_penalty = PITCH_MISMATCH_WEIGHT * (o.midi_pitch != d.midi_pitch)
            cost[i, j] = onset_dist + pitch_penalty

    row_ind, col_ind = linear_sum_assignment(cost)

    matched = []
    matched_orig_idx = set()
    matched_deg_idx = set()

    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= MATCH_COST_THRESHOLD:
            matched.append((original[r], degraded[c]))
            matched_orig_idx.add(r)
            matched_deg_idx.add(c)

    unmatched_orig = [original[i] for i in range(n_orig) if i not in matched_orig_idx]
    unmatched_deg = [degraded[j] for j in range(n_deg) if j not in matched_deg_idx]

    return matched, unmatched_orig, unmatched_deg


def compute_labels(
    original: List[NoteEvent],
    degraded: List[NoteEvent],
) -> QualityLabels:
    """Compute quality scores by comparing original and degraded note lists.

    Uses note-level alignment (Hungarian matching on onset+pitch),
    NOT audio-level comparison. Labels are deterministic given the degradation.
    """
    if len(original) == 0:
        return QualityLabels(rhythm=0.0, pitch=0.0, completeness=0.0, flow=0.0, overall=0.0)

    matched, unmatched_orig, unmatched_deg = align_notes(original, degraded)

    # Pitch: fraction of matched notes with correct pitch
    n_correct_pitch = sum(1 for (o, d) in matched if o.midi_pitch == d.midi_pitch)
    pitch_score = n_correct_pitch / max(len(original), 1)

    # Rhythm: compare proportional timing between notes, not absolute timing.
    # This makes the score tempo-independent — playing at any speed is fine
    # as long as the rhythmic ratios are correct.
    if len(matched) >= 3:
        matched_sorted = sorted(matched, key=lambda pair: pair[0].onset)

        expected_iois = []
        detected_iois = []
        for i in range(len(matched_sorted) - 1):
            e_ioi = matched_sorted[i + 1][0].onset - matched_sorted[i][0].onset
            d_ioi = matched_sorted[i + 1][1].onset - matched_sorted[i][1].onset
            if e_ioi > 0.02 and d_ioi > 0.02:
                expected_iois.append(e_ioi)
                detected_iois.append(d_ioi)

        if len(expected_iois) >= 2:
            e_median = np.median(expected_iois)
            d_median = np.median(detected_iois)

            e_ratios = [ioi / e_median for ioi in expected_iois]
            d_ratios = [ioi / d_median for ioi in detected_iois]

            ratio_errors = []
            for e_r, d_r in zip(e_ratios, d_ratios):
                if e_r > 0.01:
                    ratio_errors.append(abs(e_r - d_r) / e_r)

            if ratio_errors:
                mean_ratio_error = np.mean(ratio_errors)
                rhythm_score = float(np.clip(1.0 - mean_ratio_error, 0, 1))
            else:
                rhythm_score = 1.0
        else:
            rhythm_score = 0.8 if len(matched) >= 2 else 1.0

    elif len(matched) == 2:
        rhythm_score = 0.85

    elif len(matched) == 1:
        rhythm_score = 1.0

    else:
        rhythm_score = 0.0

    # Completeness: fraction of original notes that were matched
    completeness_score = len(matched) / max(len(original), 1)

    # Flow: consistency of rhythmic ratios.
    # High flow = the user maintained steady proportions throughout.
    # Low flow = some intervals were rushed and others dragged.
    if len(matched) >= 3:
        matched_sorted = sorted(matched, key=lambda pair: pair[0].onset)

        tempo_ratios = []
        for i in range(len(matched_sorted) - 1):
            e_ioi = matched_sorted[i + 1][0].onset - matched_sorted[i][0].onset
            d_ioi = matched_sorted[i + 1][1].onset - matched_sorted[i][1].onset
            if e_ioi > 0.02 and d_ioi > 0.02:
                tempo_ratios.append(d_ioi / e_ioi)

        if len(tempo_ratios) >= 2:
            mean_ratio = np.mean(tempo_ratios)
            std_ratio = np.std(tempo_ratios)
            if mean_ratio > 0:
                cv = std_ratio / mean_ratio
                flow_score = float(np.clip(1.0 - cv / 0.5, 0, 1))
            else:
                flow_score = 1.0
        else:
            flow_score = 0.8
    else:
        flow_score = 1.0 if len(matched) >= 1 else 0.0

    overall = (0.4 * rhythm_score + 0.3 * pitch_score +
               0.2 * completeness_score + 0.1 * flow_score)

    return QualityLabels(
        rhythm=rhythm_score,
        pitch=pitch_score,
        completeness=completeness_score,
        flow=flow_score,
        overall=overall,
    )
