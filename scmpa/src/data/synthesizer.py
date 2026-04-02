"""Convert NoteEvents to MIDI and render to audio via FluidSynth."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

# Suppress FluidSynth SDL3 warnings at the C library level
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import ctypes
import sys

def _suppress_fluidsynth_logs():
    """Silence FluidSynth warning/info/debug log output via its C API."""
    try:
        if sys.platform == 'win32':
            _lib = ctypes.cdll.LoadLibrary(r'C:\tools\fluidsynth\bin\libfluidsynth.dll')
        else:
            _lib = ctypes.cdll.LoadLibrary('libfluidsynth.so')
        FLUID_WARN, FLUID_INFO, FLUID_DBG = 2, 3, 4
        _lib.fluid_set_log_function(FLUID_WARN, None, None)
        _lib.fluid_set_log_function(FLUID_INFO, None, None)
        _lib.fluid_set_log_function(FLUID_DBG, None, None)
    except (OSError, AttributeError):
        pass

_suppress_fluidsynth_logs()

import numpy as np

from src.data.score_parser import NoteEvent


def notes_to_midi(
    notes: List[NoteEvent],
    output_path: str,
    tempo: float = 120.0,
) -> None:
    """Write a list of NoteEvents to a standard MIDI file.

    Creates two tracks: voice 0 (right hand) and voice 1 (left hand).
    """
    import mido

    ticks_per_beat = 480
    microseconds_per_beat = int(60_000_000 / tempo)

    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)

    # Separate notes by voice
    voices = {0: [], 1: []}
    for note in notes:
        v = note.voice if note.voice in voices else 0
        voices[v].append(note)

    for voice_id in sorted(voices.keys()):
        track = mido.MidiTrack()
        mid.tracks.append(track)

        if voice_id == 0:
            track.append(mido.MetaMessage('set_tempo', tempo=microseconds_per_beat, time=0))
            track.append(mido.MetaMessage('track_name', name='Right Hand', time=0))
        else:
            track.append(mido.MetaMessage('track_name', name='Left Hand', time=0))

        # Build a list of (absolute_tick, event_type, pitch, velocity) events
        events = []
        for note in voices[voice_id]:
            onset_ticks = int(note.onset / (60.0 / tempo) * ticks_per_beat)
            offset_ticks = int(note.offset / (60.0 / tempo) * ticks_per_beat)
            onset_ticks = max(0, onset_ticks)
            offset_ticks = max(onset_ticks + 1, offset_ticks)

            vel = max(1, min(127, note.velocity))
            events.append((onset_ticks, 'note_on', note.midi_pitch, vel))
            events.append((offset_ticks, 'note_off', note.midi_pitch, 0))

        events.sort(key=lambda e: (e[0], 0 if e[1] == 'note_off' else 1))

        # Convert absolute ticks to delta ticks
        prev_tick = 0
        for abs_tick, etype, pitch, vel in events:
            delta = abs_tick - prev_tick
            delta = max(0, delta)
            track.append(mido.Message(etype, note=pitch, velocity=vel, time=delta))
            prev_tick = abs_tick

    mid.save(output_path)


def render_midi_to_audio(
    midi_path: str,
    output_path: str,
    soundfont_path: str,
    sample_rate: int = 24000,
) -> None:
    """Render a MIDI file to audio using FluidSynth.

    Tries pyfluidsynth first (more reliable on Windows), falls back to CLI.
    """
    midi_path = str(Path(midi_path).resolve())
    output_path = str(Path(output_path).resolve())
    soundfont_path = str(Path(soundfont_path).resolve())

    if not Path(soundfont_path).exists():
        raise FileNotFoundError(f"Soundfont not found: {soundfont_path}")
    if not Path(midi_path).exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    # Try pyfluidsynth first (more reliable cross-platform)
    try:
        _render_with_pyfluidsynth(midi_path, output_path, soundfont_path, sample_rate)
        return
    except Exception as e:
        pass  # Fall through to CLI

    # Fallback: CLI
    fluidsynth_bin = shutil.which("fluidsynth")
    if fluidsynth_bin:
        cmd = [
            fluidsynth_bin, "-ni",
            soundfont_path, midi_path,
            "-F", output_path,
            "-r", str(sample_rate),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and Path(output_path).exists() and Path(output_path).stat().st_size > 0:
            return

    raise RuntimeError("Both pyfluidsynth and FluidSynth CLI failed")


def _render_with_pyfluidsynth(
    midi_path: str,
    output_path: str,
    soundfont_path: str,
    sample_rate: int = 24000,
) -> None:
    """Render MIDI to audio using pyfluidsynth Python bindings."""
    import fluidsynth
    import mido
    import soundfile as sf

    fs = fluidsynth.Synth(samplerate=float(sample_rate))
    sfid = fs.sfload(soundfont_path)
    fs.program_select(0, sfid, 0, 0)
    fs.program_select(1, sfid, 0, 0)

    mid = mido.MidiFile(midi_path)

    # Collect all audio samples
    audio_chunks = []
    for msg in mid:
        if msg.time > 0:
            n_samples = int(msg.time * sample_rate)
            if n_samples > 0:
                chunk = fs.get_samples(n_samples)
                # chunk is interleaved stereo int16; convert to mono float
                chunk = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                # Deinterleave stereo to mono
                if len(chunk) >= 2:
                    left = chunk[0::2]
                    right = chunk[1::2]
                    mono = (left + right) / 2.0
                    audio_chunks.append(mono)

        if msg.type == 'note_on':
            channel = msg.channel if hasattr(msg, 'channel') else 0
            fs.noteon(channel, msg.note, msg.velocity)
        elif msg.type == 'note_off':
            channel = msg.channel if hasattr(msg, 'channel') else 0
            fs.noteoff(channel, msg.note)

    # Render a tail for note decay
    tail_samples = int(1.0 * sample_rate)
    tail = fs.get_samples(tail_samples)
    tail = np.frombuffer(tail, dtype=np.int16).astype(np.float32) / 32768.0
    if len(tail) >= 2:
        audio_chunks.append((tail[0::2] + tail[1::2]) / 2.0)

    fs.delete()

    if not audio_chunks:
        raise RuntimeError("FluidSynth produced no audio")

    audio = np.concatenate(audio_chunks)

    # Normalize
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.95

    sf.write(output_path, audio, sample_rate, subtype='PCM_16')


def find_soundfont() -> Optional[str]:
    """Try to find a usable soundfont on the system."""
    candidates = [
        # Linux
        "/usr/share/sounds/sf2/FluidR3_GM.sf2",
        "/usr/share/soundfonts/FluidR3_GM.sf2",
        "/usr/share/sounds/sf2/default-GM.sf2",
        # Project-local
        "data/augmentation/soundfonts/FluidR3_GM.sf2",
        # Windows common locations
        r"C:\tools\fluidsynth\share\soundfonts\default.sf2",
    ]

    # Also check the project's soundfont directory
    project_sf_dir = Path(__file__).resolve().parent.parent.parent / "data" / "augmentation" / "soundfonts"
    if project_sf_dir.exists():
        for sf in project_sf_dir.glob("*.sf2"):
            return str(sf)

    for c in candidates:
        if Path(c).exists():
            return c

    return None


def synthesize_score(
    notes: List[NoteEvent],
    output_path: str,
    soundfont_path: Optional[str] = None,
    tempo: float = 120.0,
    sample_rate: int = 24000,
) -> np.ndarray:
    """Convenience: NoteEvents -> MIDI -> audio -> numpy array.

    Returns the audio as a 1-D float32 numpy array.
    """
    if soundfont_path is None:
        soundfont_path = find_soundfont()
        if soundfont_path is None:
            raise RuntimeError(
                "No soundfont found. Install one with:\n"
                "  Ubuntu: sudo apt-get install fluid-soundfont-gm\n"
                "  Or download FluidR3_GM.sf2 to data/augmentation/soundfonts/"
            )

    with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp:
        midi_path = tmp.name

    try:
        notes_to_midi(notes, midi_path, tempo=tempo)
        render_midi_to_audio(midi_path, output_path, soundfont_path, sample_rate)
    finally:
        if Path(midi_path).exists():
            os.unlink(midi_path)

    from src.utils.audio import load_audio
    audio, _ = load_audio(output_path, target_sr=sample_rate)
    return audio
