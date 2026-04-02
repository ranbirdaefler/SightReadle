"""Apply controlled degradations to note event lists for synthetic data generation."""

import copy
from dataclasses import dataclass, field
from typing import List

import numpy as np

from src.data.score_parser import NoteEvent


@dataclass
class DegradationConfig:
    pitch_error_rate: float = 0.0       # Fraction of notes with wrong pitch [0, 0.4]
    pitch_error_range: int = 2          # Max semitone deviation for wrong notes
    onset_jitter_std: float = 0.0       # Std dev of onset time noise in seconds [0, 0.3]
    duration_jitter_std: float = 0.0    # Std dev of duration noise [0, 0.2]
    omission_rate: float = 0.0          # Fraction of notes dropped [0, 0.4]
    insertion_rate: float = 0.0         # Fraction of extra random notes [0, 0.2]
    tempo_drift_rate: float = 0.0       # Linear tempo change over piece [0, 0.2]
    tempo_fluctuation_std: float = 0.0  # Random local tempo variation [0, 0.15]
    velocity_noise_std: float = 0.0     # Noise on MIDI velocity [0, 30]
    tempo_scale: float = 1.0            # Overall tempo multiplier [0.7, 1.3]


DEGRADATION_PRESETS = {
    "perfect":   DegradationConfig(),
    "excellent": DegradationConfig(onset_jitter_std=0.03, velocity_noise_std=10),
    "good":      DegradationConfig(pitch_error_rate=0.05, onset_jitter_std=0.06, omission_rate=0.02),
    "mediocre":  DegradationConfig(pitch_error_rate=0.15, onset_jitter_std=0.12,
                                   omission_rate=0.08, tempo_fluctuation_std=0.05),
    "poor":      DegradationConfig(pitch_error_rate=0.25, onset_jitter_std=0.20,
                                   omission_rate=0.15, insertion_rate=0.05, tempo_drift_rate=0.1),
    "very_poor": DegradationConfig(pitch_error_rate=0.35, onset_jitter_std=0.30,
                                   omission_rate=0.25, insertion_rate=0.10,
                                   tempo_drift_rate=0.15, tempo_fluctuation_std=0.1),
}


def random_degradation_config(rng: np.random.Generator) -> DegradationConfig:
    """Sample a random degradation config with each parameter uniformly in its valid range."""
    return DegradationConfig(
        pitch_error_rate=rng.uniform(0, 0.4),
        pitch_error_range=rng.integers(1, 4),
        onset_jitter_std=rng.uniform(0, 0.3),
        duration_jitter_std=rng.uniform(0, 0.2),
        omission_rate=rng.uniform(0, 0.4),
        insertion_rate=rng.uniform(0, 0.2),
        tempo_drift_rate=rng.uniform(0, 0.2),
        tempo_fluctuation_std=rng.uniform(0, 0.15),
        velocity_noise_std=rng.uniform(0, 30),
        tempo_scale=rng.uniform(0.7, 1.3),
    )


def degrade_score(
    notes: List[NoteEvent],
    config: DegradationConfig,
    rng: np.random.Generator,
) -> List[NoteEvent]:
    """Apply controlled degradations to a note event list.

    Returns a new degraded note list (does not modify the original).
    """
    degraded = copy.deepcopy(notes)

    # 1. Pitch errors
    for note in degraded:
        if rng.random() < config.pitch_error_rate:
            shift = rng.integers(-config.pitch_error_range, config.pitch_error_range + 1)
            note.midi_pitch = int(np.clip(note.midi_pitch + shift, 21, 108))

    # 2. Note omissions
    if config.omission_rate > 0:
        mask = rng.random(len(degraded)) > config.omission_rate
        degraded = [n for n, keep in zip(degraded, mask) if keep]

    # 3. Timing jitter
    for note in degraded:
        note.onset += rng.normal(0, config.onset_jitter_std)
        note.onset = max(0, note.onset)
        note.offset = note.onset + max(
            0.05,
            (note.offset - note.onset) + rng.normal(0, config.duration_jitter_std)
        )

    # 4. Extra note insertions
    if config.insertion_rate > 0 and len(degraded) > 0:
        n_insert = int(len(degraded) * config.insertion_rate)
        max_time = max(n.offset for n in degraded)
        for _ in range(n_insert):
            pos = rng.uniform(0, max_time)
            pitch = int(rng.integers(48, 84))
            degraded.append(NoteEvent(
                onset=pos, offset=pos + 0.3,
                midi_pitch=pitch, velocity=64, voice=0, is_chord=False,
            ))

    # 5. Tempo drift and fluctuation
    if config.tempo_drift_rate > 0 or config.tempo_fluctuation_std > 0:
        total_dur = max((n.offset for n in degraded), default=1.0)
        for note in degraded:
            drift = 1.0 + config.tempo_drift_rate * (note.onset / total_dur - 0.5)
            fluct = 1.0 + rng.normal(0, config.tempo_fluctuation_std)
            scale = drift * fluct
            note.onset *= scale
            note.offset *= scale

    # 6. Global tempo scaling
    if config.tempo_scale != 1.0:
        inv_scale = 1.0 / config.tempo_scale
        for note in degraded:
            note.onset *= inv_scale
            note.offset *= inv_scale

    # 7. Velocity noise
    for note in degraded:
        note.velocity = int(np.clip(
            note.velocity + rng.normal(0, config.velocity_noise_std), 1, 127
        ))

    degraded.sort(key=lambda n: n.onset)
    return degraded
