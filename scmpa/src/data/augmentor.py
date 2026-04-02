"""Domain augmentation for synthetic-to-real transfer.

Applies room impulse responses, background noise, mic simulation,
gain variation, and speed perturbation to synthesized audio.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import scipy.signal


@dataclass
class AugmentationConfig:
    apply_rir: bool = True
    rir_dir: str = "data/augmentation/rir/"
    apply_noise: bool = True
    noise_dir: str = "data/augmentation/noise/"
    snr_range: Tuple[float, float] = (15.0, 40.0)
    apply_mic_response: bool = True
    mic_highpass_hz: float = 80.0
    mic_lowpass_hz: float = 16000.0
    gain_range_db: Tuple[float, float] = (-6.0, 6.0)
    speed_perturbation: Tuple[float, float] = (0.97, 1.03)


_wav_cache: dict = {}

def _list_wav_files(directory: str) -> List[str]:
    """List all .wav files in a directory recursively (cached)."""
    if directory in _wav_cache:
        return _wav_cache[directory]
    d = Path(directory)
    if not d.exists():
        _wav_cache[directory] = []
        return []
    files = [str(f) for f in d.rglob("*.wav")]
    _wav_cache[directory] = files
    return files


def load_random_rir(rir_dir: str, rng: np.random.Generator) -> Optional[Tuple[np.ndarray, int]]:
    """Load a random RIR from the directory. Returns (rir_array, sample_rate) or None."""
    files = _list_wav_files(rir_dir)
    if not files:
        return None

    import soundfile as sf
    chosen = files[rng.integers(0, len(files))]
    rir, sr = sf.read(chosen, dtype='float32')
    if rir.ndim > 1:
        rir = rir[:, 0]
    return rir, sr


def load_random_noise(
    noise_dir: str,
    target_length: int,
    sr: int,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """Load random noise, loop/truncate to target_length samples."""
    files = _list_wav_files(noise_dir)
    if not files:
        return None

    import soundfile as sf
    chosen = files[rng.integers(0, len(files))]
    noise, noise_sr = sf.read(chosen, dtype='float32')
    if noise.ndim > 1:
        noise = noise[:, 0]

    # Resample if needed
    if noise_sr != sr:
        import librosa
        noise = librosa.resample(noise, orig_sr=noise_sr, target_sr=sr)

    # Loop or truncate
    if len(noise) < target_length:
        repeats = (target_length // len(noise)) + 1
        noise = np.tile(noise, repeats)
    noise = noise[:target_length]

    return noise


def compute_noise_scale(signal: np.ndarray, noise: np.ndarray, target_snr_db: float) -> float:
    """Compute scale factor for noise to achieve the desired SNR."""
    signal_power = np.mean(signal ** 2) + 1e-10
    noise_power = np.mean(noise ** 2) + 1e-10
    target_noise_power = signal_power / (10 ** (target_snr_db / 10))
    return float(np.sqrt(target_noise_power / noise_power))


def _resample_rir(rir: np.ndarray, rir_sr: int, target_sr: int) -> np.ndarray:
    """Resample RIR to match target sample rate."""
    if rir_sr == target_sr:
        return rir
    ratio = target_sr / rir_sr
    new_length = int(len(rir) * ratio)
    return scipy.signal.resample(rir, new_length)


def augment_audio(
    audio: np.ndarray,
    sr: int,
    config: AugmentationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply domain augmentation to synthesized audio.

    Transformations applied in order:
    1. Room impulse response convolution
    2. Background noise addition
    3. Microphone frequency response (highpass + lowpass)
    4. Random gain
    5. Speed perturbation
    6. Final normalization
    """
    audio = audio.copy().astype(np.float32)

    # 1. RIR convolution
    if config.apply_rir:
        result = load_random_rir(config.rir_dir, rng)
        if result is not None:
            rir, rir_sr = result
            rir = _resample_rir(rir, rir_sr, sr)
            audio = scipy.signal.fftconvolve(audio, rir, mode='full')[:len(audio)]

    # 2. Background noise
    if config.apply_noise:
        noise = load_random_noise(config.noise_dir, len(audio), sr, rng)
        if noise is not None:
            snr = rng.uniform(*config.snr_range)
            scale = compute_noise_scale(audio, noise, snr)
            audio = audio + scale * noise

    # 3. Microphone frequency response
    if config.apply_mic_response:
        nyquist = sr / 2.0
        if config.mic_highpass_hz < nyquist:
            sos_hp = scipy.signal.butter(
                4, config.mic_highpass_hz, btype='high', fs=sr, output='sos'
            )
            audio = scipy.signal.sosfilt(sos_hp, audio)
        if config.mic_lowpass_hz < nyquist:
            sos_lp = scipy.signal.butter(
                4, config.mic_lowpass_hz, btype='low', fs=sr, output='sos'
            )
            audio = scipy.signal.sosfilt(sos_lp, audio)

    # 4. Random gain
    gain_db = rng.uniform(*config.gain_range_db)
    audio = audio * (10 ** (gain_db / 20))

    # 5. Speed perturbation
    speed = rng.uniform(*config.speed_perturbation)
    if abs(speed - 1.0) > 0.001:
        new_length = int(len(audio) / speed)
        if new_length > 0:
            audio = scipy.signal.resample(audio, new_length)

    # 6. Normalize
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.95

    return audio.astype(np.float32)
