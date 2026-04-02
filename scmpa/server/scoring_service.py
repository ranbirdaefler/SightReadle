"""FastAPI scoring service for SightReadle — basic-pitch transcription pipeline.

Accepts audio + segment_id, transcribes with basic-pitch, compares to
ground truth MIDI via Hungarian matching, returns quality scores and
per-note feedback.

Endpoints:
    POST /score   — multipart form: segment_id + audio file
    GET  /health  — service health check

Usage:
    cd scmpa/
    python -m uvicorn server.scoring_service:app --port 8001
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import mido
import pretty_midi
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.scoring import score_performance  # noqa: E402

app = FastAPI(title="SightReadle Scoring Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MANIFEST: dict = {}
SEGMENT_LOOKUP: dict = {}


@app.on_event("startup")
def load_manifest():
    global MANIFEST, SEGMENT_LOOKUP
    manifest_path = PROJECT_ROOT / "data" / "segments" / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            MANIFEST = json.load(f)
        SEGMENT_LOOKUP = {s["id"]: s for s in MANIFEST.get("segments", [])}
        print(f"Loaded {len(SEGMENT_LOOKUP)} segments from manifest")
    else:
        print(f"WARNING: manifest not found at {manifest_path}")


def get_segment_midi_path(segment_id: str) -> str:
    """Look up the MIDI path for a segment from the manifest."""
    seg = SEGMENT_LOOKUP.get(segment_id)
    if seg is None:
        raise ValueError(f"Unknown segment: {segment_id}")
    midi_path = PROJECT_ROOT / seg["midi_path"]
    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")
    return str(midi_path)


TEMPO_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "tempo_renders"
TEMPO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _get_cached_render(segment_id: str, bpm: int) -> str | None:
    cache_path = TEMPO_CACHE_DIR / f"{segment_id}_{bpm}.mp3"
    return str(cache_path) if cache_path.exists() else None


def _wav_to_mp3(wav_path: str, mp3_path: str):
    """Convert WAV to MP3 using ffmpeg."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path, "-b:a", "128k", "-q:a", "2", mp3_path],
        capture_output=True, timeout=15,
        check=True,
    )


def _cache_render(segment_id: str, bpm: int, mp3_path: str) -> str:
    cache_path = TEMPO_CACHE_DIR / f"{segment_id}_{bpm}.mp3"
    shutil.copy2(mp3_path, cache_path)
    return str(cache_path)


def _render_midi_at_tempo(midi_path: str, bpm: float, out_wav: str):
    """Rewrite MIDI tempo then synthesise with FluidSynth via pretty_midi."""
    mid = mido.MidiFile(midi_path)
    tempo_us = mido.bpm2tempo(bpm)
    for track in mid.tracks:
        track[:] = [msg for msg in track if msg.type != "set_tempo"]
    mid.tracks[0].insert(
        0, mido.MetaMessage("set_tempo", tempo=tempo_us, time=0)
    )

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        mid.save(tmp.name)
        tmp_midi = tmp.name

    try:
        pm = pretty_midi.PrettyMIDI(tmp_midi)
        audio = pm.fluidsynth(fs=44100)
        sf.write(out_wav, audio, 44100)
    finally:
        os.unlink(tmp_midi)


@app.post("/render-tempo")
async def render_at_tempo(
    segment_id: str = Form(...),
    bpm: float = Form(...),
):
    """Re-render a segment's reference audio at a user-chosen tempo."""
    bpm_int = int(round(bpm))
    try:
        midi_path = get_segment_midi_path(segment_id)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e))

    cached = _get_cached_render(segment_id, bpm_int)
    if cached:
        return FileResponse(cached, media_type="audio/mpeg")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name
    tmp_mp3 = tmp_wav.replace(".wav", ".mp3")

    try:
        _render_midi_at_tempo(midi_path, bpm, tmp_wav)
        _wav_to_mp3(tmp_wav, tmp_mp3)
    except Exception as e:
        for p in (tmp_wav, tmp_mp3):
            if os.path.exists(p):
                os.unlink(p)
        raise HTTPException(status_code=500, detail=f"Render failed: {e}")
    finally:
        if os.path.exists(tmp_wav):
            os.unlink(tmp_wav)

    cached_path = _cache_render(segment_id, bpm_int, tmp_mp3)
    os.unlink(tmp_mp3)
    return FileResponse(cached_path, media_type="audio/mpeg")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "pipeline": "basic-pitch-transcription",
        "segments_loaded": len(SEGMENT_LOOKUP),
    }


@app.post("/score")
async def score(
    segment_id: str = Form(...),
    audio: UploadFile = File(...),
):
    """Score a user's performance.

    Accepts multipart form data with segment_id and audio WAV file.
    Returns JSON with scores, summary, and per-note feedback.
    """
    try:
        segment_midi = get_segment_midi_path(segment_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        content = await audio.read()
        f.write(content)
        temp_path = f.name

    try:
        result = score_performance(temp_path, segment_midi)

        return {
            "scores": {
                "rhythm": round(result.rhythm, 3),
                "pitch": round(result.pitch, 3),
                "completeness": round(result.completeness, 3),
                "overall": round(result.overall, 3),
            },
            "summary": {
                "correct": result.n_correct,
                "wrong_pitch": result.n_wrong_pitch,
                "wrong_octave": result.n_wrong_octave,
                "missed": result.n_missed,
                "extra": result.n_extra,
                "total_expected": (result.n_correct + result.n_wrong_pitch
                                   + result.n_wrong_octave + result.n_missed),
            },
            "notes": [
                {
                    "expected_pitch": n.score_pitch,
                    "detected_pitch": n.detected_pitch,
                    "expected_onset": round(n.expected_onset, 3) if n.expected_onset >= 0 else None,
                    "actual_onset": round(n.actual_onset, 3) if n.actual_onset >= 0 else None,
                    "timing_error": round(n.timing_error, 3) if n.status not in ("missed", "extra") else None,
                    "status": n.status,
                }
                for n in result.note_details
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {e}")
    finally:
        os.unlink(temp_path)


@app.get("/segments")
def list_segments():
    """List all available segment IDs."""
    return {
        "segments": sorted(SEGMENT_LOOKUP.keys()),
        "count": len(SEGMENT_LOOKUP),
    }
