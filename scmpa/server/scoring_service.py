"""FastAPI scoring service for SightReadle.

Handles on-the-fly segmentation, audio transcription with basic-pitch,
and performance scoring.

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
import time
from pathlib import Path

import mido
import pretty_midi
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

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

segment_service = None

TEMPO_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "tempo_renders"
TEMPO_CACHE_DIR.mkdir(parents=True, exist_ok=True)


@app.on_event("startup")
def startup():
    import threading

    def _parse_in_background():
        global segment_service
        from server.segment_service import SegmentService
        source_dir = str(PROJECT_ROOT / "data" / "source_musicxml")
        segment_service = SegmentService(source_dir=source_dir)

    t = threading.Thread(target=_parse_in_background, daemon=True)
    t.start()


# ── Segment endpoints ──

@app.get("/segment/random")
async def get_random_segment(
    difficulty: str = "intermediate",
    bars: int = 4,
    recent_pieces: str = "",
    recent_segs: str = "",
):
    if segment_service is None:
        return JSONResponse(status_code=503, content={"error": "Service starting up"})
    if bars < 2 or bars > 12:
        return JSONResponse(status_code=400, content={"error": "bars must be 2-12"})
    if difficulty not in ("easy", "intermediate", "advanced"):
        return JSONResponse(status_code=400, content={"error": "invalid difficulty"})

    exclude_pieces = []
    if recent_pieces:
        try:
            exclude_pieces = json.loads(recent_pieces)
        except Exception:
            pass

    exclude_segments = []
    if recent_segs:
        try:
            exclude_segments = json.loads(recent_segs)
        except Exception:
            pass

    segment = segment_service.get_random_segment(
        difficulty, bars,
        exclude_pieces=exclude_pieces,
        exclude_segments=exclude_segments,
    )
    if not segment:
        return JSONResponse(status_code=404, content={"error": "No valid segment found"})

    return segment


@app.get("/segment/daily")
async def get_daily_segment():
    if segment_service is None:
        return JSONResponse(status_code=503, content={"error": "Service starting up"})
    day_number = int(time.time() // 86400)
    segment = segment_service.get_daily_segment(day_number)
    if not segment:
        return JSONResponse(status_code=404, content={"error": "No daily segment available"})
    segment["challenge_number"] = day_number
    return segment


@app.get("/segment/difficulties")
async def get_difficulties():
    if segment_service is None:
        return JSONResponse(status_code=503, content={"error": "Service starting up"})
    return segment_service.get_available_difficulties()


@app.get("/segment/pieces")
async def list_pieces():
    """List all loaded pieces with bar counts and note totals."""
    if segment_service is None:
        return JSONResponse(status_code=503, content={"error": "Service starting up"})
    result = {}
    for diff, pieces in segment_service.pieces.items():
        result[diff] = [
            {
                "name": p["name"],
                "total_bars": p["total_bars"],
                "total_notes": sum(p["bar_note_counts"]),
                "tempo": p["tempo"],
                "time_signature": p["time_signature"],
                "key_signature": p["key_signature"],
            }
            for p in pieces
        ]
    return result


@app.get("/segment/specific")
async def get_specific_segment(
    difficulty: str,
    piece_name: str,
    start_bar: int = 0,
    n_bars: int = 4,
):
    """Extract a specific segment from a named piece."""
    if segment_service is None:
        return JSONResponse(status_code=503, content={"error": "Service starting up"})

    pieces = segment_service.pieces.get(difficulty, [])
    piece = next((p for p in pieces if p["name"] == piece_name), None)
    if not piece:
        return JSONResponse(status_code=404, content={"error": f"Piece '{piece_name}' not found in {difficulty}"})

    actual_bars = min(n_bars, piece["total_bars"] - start_bar)
    if actual_bars < 1:
        return JSONResponse(status_code=400, content={"error": "Invalid bar range"})

    end_bar = start_bar + actual_bars
    note_count = sum(piece["bar_note_counts"][start_bar:end_bar])

    import hashlib
    seg_id = f"admin_{piece_name}_b{start_bar}-{end_bar}"
    cache_key = hashlib.md5(seg_id.encode()).hexdigest()[:12]
    seg_id_safe = f"admin_{cache_key}"

    try:
        segment_score = piece["score"].measures(start_bar + 1, end_bar)

        xml_path = segment_service.cache_dir / "musicxml" / f"{seg_id_safe}.musicxml"
        if not xml_path.exists():
            segment_score.write('musicxml', fp=str(xml_path))

        midi_path = segment_service.cache_dir / "midi" / f"{seg_id_safe}.mid"
        if not midi_path.exists():
            segment_score.write('midi', fp=str(midi_path))

        bar_duration = (60.0 / piece["tempo"]) * piece["time_signature"][0]

        return {
            "id": seg_id_safe,
            "source_piece": piece["name"],
            "difficulty": difficulty,
            "start_bar": start_bar,
            "n_bars": actual_bars,
            "n_notes": note_count,
            "tempo": piece["tempo"],
            "time_signature": piece["time_signature"],
            "key_signature": piece["key_signature"],
            "duration_sec": round(actual_bars * bar_duration, 2),
            "musicxml_path": str(xml_path),
            "midi_path": str(midi_path),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Extraction failed: {e}"})


@app.get("/segment/musicxml/{segment_id}")
async def get_segment_musicxml(segment_id: str):
    if segment_service is None:
        return JSONResponse(status_code=503, content={"error": "Service starting up"})
    xml_path = segment_service.cache_dir / "musicxml" / f"{segment_id}.musicxml"
    if not xml_path.exists():
        return JSONResponse(status_code=404, content={"error": "Segment not found"})
    return FileResponse(str(xml_path), media_type="application/xml")


# ── MIDI path resolution ──

def _resolve_midi_path(segment_id: str) -> str:
    """Find the MIDI file for a segment — check on-the-fly cache first."""
    if segment_service:
        cached_midi = segment_service.cache_dir / "midi" / f"{segment_id}.mid"
        if cached_midi.exists():
            return str(cached_midi)
    raise FileNotFoundError(f"MIDI file not found for segment: {segment_id}")


# ── Tempo rendering ──

def _get_cached_render(segment_id: str, bpm: int):
    cache_path = TEMPO_CACHE_DIR / f"{segment_id}_{bpm}.mp3"
    return str(cache_path) if cache_path.exists() else None


def _wav_to_mp3(wav_path: str, mp3_path: str):
    subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path, "-b:a", "128k", "-q:a", "2", mp3_path],
        capture_output=True, timeout=15, check=True,
    )


def _cache_render(segment_id: str, bpm: int, mp3_path: str) -> str:
    cache_path = TEMPO_CACHE_DIR / f"{segment_id}_{bpm}.mp3"
    shutil.copy2(mp3_path, cache_path)
    return str(cache_path)


def _render_midi_at_tempo(midi_path: str, bpm: float, out_wav: str):
    mid = mido.MidiFile(midi_path)
    tempo_us = mido.bpm2tempo(bpm)
    for track in mid.tracks:
        track[:] = [msg for msg in track if msg.type != "set_tempo"]
    mid.tracks[0].insert(0, mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))

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
    bpm_int = int(round(bpm))
    try:
        midi_path = _resolve_midi_path(segment_id)
    except FileNotFoundError as e:
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


# ── Scoring ──

@app.post("/score")
async def score_endpoint(
    segment_id: str = Form(...),
    audio: UploadFile = File(...),
):
    try:
        midi_path = _resolve_midi_path(segment_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        content = await audio.read()
        f.write(content)
        temp_path = f.name

    try:
        result = score_performance(temp_path, midi_path)
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
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {e}")
    finally:
        os.unlink(temp_path)


# ── Health ──

@app.get("/health")
def health():
    pieces_loaded = 0
    if segment_service:
        pieces_loaded = sum(len(v) for v in segment_service.pieces.values())
    return {
        "status": "ready" if segment_service else "starting",
        "pieces_loaded": pieces_loaded,
    }


@app.get("/debug")
async def debug():
    checks = {
        "cwd": os.getcwd(),
        "project_root": str(PROJECT_ROOT),
        "segment_service_ready": segment_service is not None,
    }
    if segment_service:
        checks["difficulties"] = segment_service.get_available_difficulties()
        checks["cache_dir"] = str(segment_service.cache_dir)
    return checks
