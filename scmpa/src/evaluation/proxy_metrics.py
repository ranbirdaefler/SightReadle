"""Proxy ground truth metrics for evaluating on real recordings."""

from typing import Dict, List, Optional

import numpy as np

from src.data.score_parser import NoteEvent


def transcribe_with_basic_pitch(audio_path: str) -> List[NoteEvent]:
    """Transcribe audio to NoteEvents using Spotify's basic-pitch.

    Returns a list of NoteEvents extracted from the audio.
    """
    from basic_pitch.inference import predict

    model_output, midi_data, note_events = predict(audio_path)

    notes = []
    for onset, offset, pitch, velocity, _ in note_events:
        notes.append(NoteEvent(
            onset=onset,
            offset=offset,
            midi_pitch=int(pitch),
            velocity=int(velocity * 127),
            voice=0 if pitch >= 60 else 1,
            is_chord=False,
        ))

    notes.sort(key=lambda n: n.onset)
    return notes


def compute_note_metrics(
    reference_notes: List[NoteEvent],
    transcribed_notes: List[NoteEvent],
    onset_tolerance: float = 0.05,
    pitch_tolerance: float = 50.0,  # cents
) -> Dict[str, float]:
    """Compute mir_eval note-level metrics.

    Returns dict with precision, recall, F1 for note-level and onset-only matching.
    """
    import mir_eval

    if not reference_notes or not transcribed_notes:
        return {
            "note_precision": 0.0, "note_recall": 0.0, "note_f1": 0.0,
            "onset_precision": 0.0, "onset_recall": 0.0, "onset_f1": 0.0,
        }

    # Convert to mir_eval format: (intervals [N, 2], pitches [N])
    ref_intervals = np.array([[n.onset, n.offset] for n in reference_notes])
    ref_pitches = np.array([n.midi_pitch for n in reference_notes], dtype=float)
    ref_pitches_hz = 440.0 * 2 ** ((ref_pitches - 69) / 12)

    est_intervals = np.array([[n.onset, n.offset] for n in transcribed_notes])
    est_pitches = np.array([n.midi_pitch for n in transcribed_notes], dtype=float)
    est_pitches_hz = 440.0 * 2 ** ((est_pitches - 69) / 12)

    # Note-level (onset + pitch)
    p, r, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches_hz,
        est_intervals, est_pitches_hz,
        onset_tolerance=onset_tolerance,
        pitch_tolerance=pitch_tolerance,
    )

    # Onset-only (rhythm proxy)
    op, orc, of1 = mir_eval.onset.f_measure(
        ref_intervals[:, 0], est_intervals[:, 0],
        window=onset_tolerance,
    )

    return {
        "note_precision": p,
        "note_recall": r,
        "note_f1": f1,
        "onset_precision": op,
        "onset_recall": orc,
        "onset_f1": of1,
    }


def compute_dtw_cost(
    ref_audio_path: str,
    perf_audio_path: str,
    sr: int = 24000,
) -> float:
    """Compute DTW alignment cost between reference and performance chroma features.

    Lower cost = better alignment.
    """
    import librosa

    ref, _ = librosa.load(ref_audio_path, sr=sr)
    perf, _ = librosa.load(perf_audio_path, sr=sr)

    ref_chroma = librosa.feature.chroma_cqt(y=ref, sr=sr)
    perf_chroma = librosa.feature.chroma_cqt(y=perf, sr=sr)

    D, wp = librosa.sequence.dtw(ref_chroma, perf_chroma, metric='cosine')

    # Normalized cost: total DTW cost / path length
    cost = D[-1, -1] / len(wp)
    return float(cost)


def compute_all_proxy_metrics(
    ref_audio_path: str,
    perf_audio_path: str,
    reference_notes: List[NoteEvent],
) -> Dict[str, float]:
    """Run all proxy metrics on a (reference, performance) pair.

    Returns a combined dict of all metrics.
    """
    # Transcribe performance
    transcribed = transcribe_with_basic_pitch(perf_audio_path)

    # Note-level metrics
    note_metrics = compute_note_metrics(reference_notes, transcribed)

    # DTW cost
    dtw_cost = compute_dtw_cost(ref_audio_path, perf_audio_path)

    return {
        **note_metrics,
        "dtw_cost": dtw_cost,
        "n_transcribed_notes": len(transcribed),
    }
