"""Core scoring pipeline: transcribe audio with basic-pitch, DTW alignment, calibrated scoring."""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

from src.data.score_parser import NoteEvent, parse_midi


@dataclass
class NoteResult:
    """Per-note feedback for the user."""
    score_pitch: int
    detected_pitch: int
    expected_onset: float
    actual_onset: float
    timing_error: float
    status: str  # "correct", "wrong_octave", "wrong_pitch", "missed", "extra"


@dataclass
class ScoringResult:
    """Complete scoring output for a performance."""
    rhythm: float
    pitch: float
    completeness: float
    overall: float
    note_details: List[NoteResult]
    n_correct: int
    n_wrong_pitch: int
    n_wrong_octave: int
    n_missed: int
    n_extra: int


def transcribe_audio_informed(
    audio_path: str,
    expected_notes: List[NoteEvent],
    high_onset: float = 0.5,
    high_frame: float = 0.3,
    low_onset: float = 0.2,
    low_frame: float = 0.15,
    minimum_note_length: float = 58,
) -> List[NoteEvent]:
    """Score-informed two-tier transcription.

    Tier 1 (high thresholds): keep all detections unconditionally.
    Tier 2 (low thresholds):  keep only if the pitch matches an expected pitch.
    This rescues soft or uncertain notes the player actually intended.
    """
    expected_pitches = {n.midi_pitch for n in expected_notes}

    _mo_hi, _md_hi, hi_events = predict(
        audio_path,
        model_or_model_path=ICASSP_2022_MODEL_PATH,
        onset_threshold=high_onset,
        frame_threshold=high_frame,
        minimum_note_length=minimum_note_length,
    )

    hi_set = set()
    detected = []
    for start, end, pitch, velocity, _pb in hi_events:
        key = (round(start, 3), int(pitch))
        hi_set.add(key)
        detected.append(NoteEvent(
            onset=float(start), offset=float(end),
            midi_pitch=int(pitch), velocity=int(velocity),
            voice=0 if pitch >= 60 else 1, is_chord=False,
        ))

    _mo_lo, _md_lo, lo_events = predict(
        audio_path,
        model_or_model_path=ICASSP_2022_MODEL_PATH,
        onset_threshold=low_onset,
        frame_threshold=low_frame,
        minimum_note_length=minimum_note_length,
    )

    for start, end, pitch, velocity, _pb in lo_events:
        key = (round(start, 3), int(pitch))
        if key in hi_set:
            continue
        if int(pitch) in expected_pitches:
            detected.append(NoteEvent(
                onset=float(start), offset=float(end),
                midi_pitch=int(pitch), velocity=int(velocity),
                voice=0 if pitch >= 60 else 1, is_chord=False,
            ))

    detected.sort(key=lambda n: n.onset)
    return detected


# ============================================================
# ONSET DEDUPLICATION
# ============================================================

def deduplicate_onsets(
    notes: List[NoteEvent],
    onset_window: float = 0.05,
) -> List[NoteEvent]:
    """Merge near-simultaneous duplicate detections of the same pitch.

    Keeps the higher-velocity instance. Chord notes (different pitches
    at the same onset) are preserved.
    """
    if not notes:
        return []

    notes = sorted(notes, key=lambda n: (n.onset, n.midi_pitch))
    result: List[NoteEvent] = []

    for note in notes:
        merged = False
        for i in range(len(result) - 1, -1, -1):
            existing = result[i]
            if note.onset - existing.onset > onset_window:
                break
            if existing.midi_pitch == note.midi_pitch:
                if note.velocity > existing.velocity:
                    result[i] = note
                merged = True
                break
        if not merged:
            result.append(note)

    return result


# ============================================================
# DTW ALIGNMENT
# ============================================================

def dtw_align(
    expected_notes: List[NoteEvent],
    detected_notes: List[NoteEvent],
    chroma_match_bonus: float = 2.0,
    exact_pitch_bonus: float = 1.0,
    skip_cost: float = 0.5,
) -> List[Tuple[int, int]]:
    """Align detected notes to expected notes using DTW on pitch sequences.

    DTW finds the optimal monotonic alignment between two sequences,
    naturally handling tempo differences, pauses, and missing notes.

    The cost function compares chroma (pitch class) primarily,
    with a bonus for exact pitch match.
    """
    N = len(expected_notes)
    M = len(detected_notes)

    if N == 0 or M == 0:
        return []

    cost_matrix = np.full((N, M), dtype=np.float64, fill_value=1.0)

    for i in range(N):
        e_chroma = expected_notes[i].midi_pitch % 12
        e_pitch = expected_notes[i].midi_pitch
        for j in range(M):
            d_chroma = detected_notes[j].midi_pitch % 12
            if e_chroma == d_chroma:
                cost_matrix[i, j] -= chroma_match_bonus
                if e_pitch == detected_notes[j].midi_pitch:
                    cost_matrix[i, j] -= exact_pitch_bonus

    D = np.full((N + 1, M + 1), fill_value=np.inf)
    D[0, 0] = 0.0

    for i in range(1, N + 1):
        D[i, 0] = i * skip_cost
    for j in range(1, M + 1):
        D[0, j] = j * skip_cost

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            match = D[i - 1, j - 1] + cost_matrix[i - 1, j - 1]
            skip_exp = D[i - 1, j] + skip_cost
            skip_det = D[i, j - 1] + skip_cost
            D[i, j] = min(match, skip_exp, skip_det)

    path = []
    i, j = N, M
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
            continue
        if j == 0:
            i -= 1
            continue

        match = D[i - 1, j - 1] + cost_matrix[i - 1, j - 1]
        skip_exp = D[i - 1, j] + skip_cost
        current = D[i, j]

        if abs(current - match) < 1e-9:
            path.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif abs(current - skip_exp) < 1e-9:
            i -= 1
        else:
            j -= 1

    path.reverse()
    return path


def extract_matches(
    path: List[Tuple[int, int]],
    expected_notes: List[NoteEvent],
    detected_notes: List[NoteEvent],
) -> Tuple[list, list, list]:
    """From the DTW path, extract matched pairs, missed notes, and extras."""
    matched_exp_idx = set()
    matched_det_idx = set()
    matched_pairs = []

    for ei, di in path:
        matched_pairs.append((expected_notes[ei], detected_notes[di]))
        matched_exp_idx.add(ei)
        matched_det_idx.add(di)

    unmatched_expected = [
        expected_notes[i] for i in range(len(expected_notes))
        if i not in matched_exp_idx
    ]
    unmatched_detected = [
        detected_notes[j] for j in range(len(detected_notes))
        if j not in matched_det_idx
    ]

    return matched_pairs, unmatched_expected, unmatched_detected


# ============================================================
# MATCH VALIDATION
# ============================================================

def validate_matches(
    matched_pairs: list,
    unmatched_expected: list,
    unmatched_detected: list,
    max_semitone_distance: int = 12,
) -> Tuple[list, list, list]:
    """Reject DTW matches with implausibly large pitch distance.

    Pairs where |expected_pitch - detected_pitch| > max_semitone_distance
    are reclassified: the expected note becomes missed, the detected
    note becomes extra.
    """
    valid_pairs = []
    for exp, det in matched_pairs:
        if abs(exp.midi_pitch - det.midi_pitch) > max_semitone_distance:
            unmatched_expected.append(exp)
            unmatched_detected.append(det)
        else:
            valid_pairs.append((exp, det))

    return valid_pairs, unmatched_expected, unmatched_detected


# ============================================================
# SIGHT-READING CALIBRATED SCORING
# ============================================================

def compute_pitch_score(matched_pairs: list, total_expected: int) -> float:
    """Pitch accuracy with partial credit for octave errors.

    Correct pitch: 1.0, correct chroma wrong octave: 0.5, wrong note: 0.0.
    Denominator is total expected notes so missed notes reduce the score.
    """
    if total_expected == 0:
        return 1.0

    credit = 0.0
    for expected, detected in matched_pairs:
        if expected.midi_pitch == detected.midi_pitch:
            credit += 1.0
        elif expected.midi_pitch % 12 == detected.midi_pitch % 12:
            credit += 0.5

    return credit / total_expected


def compute_rhythm_score(matched_pairs: list) -> float:
    """Ratio-based rhythm scoring with a 30% deadzone.

    Compares proportional timing between consecutive matched notes.
    Deviations under 30% incur no penalty.
    """
    if len(matched_pairs) < 3:
        return 0.9 if len(matched_pairs) >= 1 else 0.0

    expected_iois = []
    detected_iois = []
    for i in range(len(matched_pairs) - 1):
        e_ioi = matched_pairs[i + 1][0].onset - matched_pairs[i][0].onset
        d_ioi = matched_pairs[i + 1][1].onset - matched_pairs[i][1].onset
        if e_ioi > 0.05 and d_ioi > 0.05:
            expected_iois.append(e_ioi)
            detected_iois.append(d_ioi)

    if len(expected_iois) < 2 or len(detected_iois) < 2:
        return 0.85

    e_arr = np.array(expected_iois)
    d_arr = np.array(detected_iois)
    e_median = float(np.median(e_arr))
    d_median = float(np.median(d_arr))

    if e_median < 0.01 or d_median < 0.01:
        return 0.85

    e_ratios = e_arr / e_median
    d_ratios = d_arr / d_median

    DEADZONE = 0.3
    ratio_errors = []
    for e_r, d_r in zip(e_ratios, d_ratios):
        if e_r > 0.01:
            relative_error = abs(e_r - d_r) / e_r
            ratio_errors.append(max(0.0, relative_error - DEADZONE))

    if not ratio_errors:
        return 0.85

    mean_error = float(np.mean(ratio_errors))
    return float(np.clip(1.0 - mean_error / 0.7, 0, 1))


def compute_completeness_score(n_matched: int, total_expected: int) -> float:
    """Non-linear completeness: sqrt curve, generous at the top."""
    if total_expected == 0:
        return 1.0
    raw = n_matched / total_expected
    return float(np.sqrt(np.clip(raw, 0, 1)))


# ============================================================
# MAIN SCORING FUNCTION
# ============================================================

def score_performance(
    audio_path: str,
    segment_midi_path: str,
) -> ScoringResult:
    """Score a user's sight-reading performance.

    Pipeline:
    1. Load ground truth from segment MIDI
    2. Score-informed transcription (two-tier filtering)
    3. Onset deduplication
    4. DTW alignment on chroma sequences
    5. Extract matched/missed/extra notes
    6. Validate matches (reject >12 semitone distance)
    7. Compute calibrated quality scores
    8. Generate per-note feedback
    """
    try:
        return _score_performance_inner(audio_path, segment_midi_path)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return _empty_result()


def _score_performance_inner(audio_path: str, segment_midi_path: str) -> ScoringResult:
    score_data = parse_midi(segment_midi_path)
    expected_notes = sorted(score_data.notes, key=lambda n: (n.onset, n.midi_pitch))

    if not expected_notes:
        return _empty_result()

    detected_notes = transcribe_audio_informed(audio_path, expected_notes)
    detected_notes = deduplicate_onsets(detected_notes)

    if not detected_notes:
        return _no_notes_detected_result(expected_notes)

    dtw_path = dtw_align(expected_notes, detected_notes)

    matched_pairs, unmatched_expected, unmatched_detected = extract_matches(
        dtw_path, expected_notes, detected_notes
    )

    matched_pairs, unmatched_expected, unmatched_detected = validate_matches(
        matched_pairs, unmatched_expected, unmatched_detected
    )

    total_expected = len(expected_notes)

    pitch_score = compute_pitch_score(matched_pairs, total_expected)
    rhythm_score = compute_rhythm_score(matched_pairs)
    completeness_score = compute_completeness_score(len(matched_pairs), total_expected)

    overall = (0.45 * pitch_score +
               0.35 * completeness_score +
               0.20 * rhythm_score)

    note_details: List[NoteResult] = []
    n_correct = 0
    n_wrong_pitch = 0
    n_wrong_octave = 0

    for expected, detected in matched_pairs:
        if expected.midi_pitch == detected.midi_pitch:
            status = "correct"
            n_correct += 1
        elif expected.midi_pitch % 12 == detected.midi_pitch % 12:
            status = "wrong_octave"
            n_wrong_octave += 1
        else:
            status = "wrong_pitch"
            n_wrong_pitch += 1

        note_details.append(NoteResult(
            score_pitch=expected.midi_pitch,
            detected_pitch=detected.midi_pitch,
            expected_onset=expected.onset,
            actual_onset=detected.onset,
            timing_error=detected.onset - expected.onset,
            status=status,
        ))

    for expected in unmatched_expected:
        note_details.append(NoteResult(
            score_pitch=expected.midi_pitch,
            detected_pitch=-1,
            expected_onset=expected.onset,
            actual_onset=-1,
            timing_error=0,
            status="missed",
        ))

    for detected in unmatched_detected:
        note_details.append(NoteResult(
            score_pitch=-1,
            detected_pitch=detected.midi_pitch,
            expected_onset=-1,
            actual_onset=detected.onset,
            timing_error=0,
            status="extra",
        ))

    note_details.sort(
        key=lambda n: n.expected_onset if n.expected_onset >= 0 else n.actual_onset
    )

    return ScoringResult(
        rhythm=round(rhythm_score, 3),
        pitch=round(pitch_score, 3),
        completeness=round(completeness_score, 3),
        overall=round(overall, 3),
        note_details=note_details,
        n_correct=n_correct,
        n_wrong_pitch=n_wrong_pitch,
        n_wrong_octave=n_wrong_octave,
        n_missed=len(unmatched_expected),
        n_extra=len(unmatched_detected),
    )


def _empty_result():
    return ScoringResult(
        rhythm=0, pitch=0, completeness=0, overall=0,
        note_details=[], n_correct=0, n_wrong_pitch=0,
        n_wrong_octave=0, n_missed=0, n_extra=0,
    )


def _no_notes_detected_result(expected_notes):
    note_details = [
        NoteResult(
            score_pitch=n.midi_pitch, detected_pitch=-1,
            expected_onset=n.onset, actual_onset=-1,
            timing_error=0, status="missed",
        )
        for n in expected_notes
    ]
    return ScoringResult(
        rhythm=0, pitch=0, completeness=0, overall=0,
        note_details=note_details, n_correct=0, n_wrong_pitch=0,
        n_wrong_octave=0, n_missed=len(expected_notes), n_extra=0,
    )
