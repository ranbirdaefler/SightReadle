"""Audio loading, resampling, and chunking utilities."""

from pathlib import Path
from typing import Tuple

import numpy as np


def load_audio(path: str, target_sr: int = 24000) -> Tuple[np.ndarray, int]:
    """Load an audio file, convert to mono, resample to target_sr.

    Returns (audio_array, sample_rate) where audio_array is 1-D float32.
    """
    import soundfile as sf

    audio, sr = sf.read(path, dtype='float32', always_2d=True)
    # Convert to mono by averaging channels
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio[:, 0]

    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return audio, sr


def load_audio_torch(path: str, target_sr: int = 24000):
    """Load audio as a torch tensor. Returns (waveform [1, T], sample_rate)."""
    import torch

    audio, sr = load_audio(path, target_sr=target_sr)
    waveform = torch.from_numpy(audio).unsqueeze(0)  # [1, T]
    return waveform, target_sr


def save_audio(path: str, audio: np.ndarray, sr: int = 24000) -> None:
    """Save a 1-D float32 audio array to a WAV file."""
    import soundfile as sf
    sf.write(path, audio, sr, subtype='PCM_16')


def truncate_or_pad(audio: np.ndarray, max_samples: int) -> np.ndarray:
    """Truncate to max_samples or zero-pad if shorter."""
    if len(audio) > max_samples:
        return audio[:max_samples]
    elif len(audio) < max_samples:
        return np.pad(audio, (0, max_samples - len(audio)))
    return audio
