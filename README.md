# SightReadle

**Wordle meets sight-reading.** A daily challenge app for pianists — read a short excerpt of sheet music, play it on your piano, and get scored on how well you did.

Live at [playsightreadle.com](https://playsightreadle.com)

## How It Works

1. **See** — A 4-bar excerpt of classical piano music is displayed as sheet music
2. **Listen** (optional) — Hear a synthesized reference at an adjustable tempo with an optional metronome
3. **Play** — Record yourself playing the excerpt through your browser's microphone
4. **Score** — Get instant feedback on pitch accuracy, rhythm, and completeness

## Game Modes

- **Daily Challenge** — Everyone gets the same excerpt each day. Scores go on a leaderboard with rank, percentile, and a score distribution histogram. One attempt per day (persisted in localStorage).
- **Random Practice** — Unlimited random excerpts for practice. No leaderboard, replay as many times as you want.

## Scoring System

Scoring uses a **transcribe-then-compare** approach. Your recorded audio is transcribed into MIDI notes using a neural network, then those notes are aligned against the ground truth score using Dynamic Time Warping.

### Pipeline

```
Recorded Audio (WAV)
    │
    ▼
Score-Informed Transcription (basic-pitch, two-tier)
    │
    ▼
Onset Deduplication (merge near-simultaneous duplicates)
    │
    ▼
DTW Alignment (match detected notes to expected notes)
    │
    ▼
Match Validation (reject >1 octave mismatches)
    │
    ▼
Calibrated Scoring (pitch, rhythm, completeness → overall)
```

### Score-Informed Transcription

[basic-pitch](https://github.com/spotify/basic-pitch) (Spotify's polyphonic audio-to-MIDI model) runs twice on the recording:

- **Tier 1** (high confidence thresholds) — All detected notes are kept unconditionally
- **Tier 2** (low confidence thresholds) — Notes are only kept if their pitch matches a pitch that appears in the score

This rescues soft or uncertain notes the player actually intended to play, while filtering out noise.

### DTW Alignment

Dynamic Time Warping aligns the detected note sequence to the expected note sequence. This is tempo-independent — you can play faster or slower than the reference and still get a fair score. The cost function prioritizes chroma (pitch class) matching with a bonus for exact pitch matches.

### Score Components

**Pitch (45% of overall)**

Measures how accurately you played the right notes. Each matched note pair earns credit:
- Exact pitch match: 1.0
- Correct note name, wrong octave: 0.5
- Wrong note: 0.0

The denominator is total expected notes, so missed notes reduce your pitch score.

**Completeness (35% of overall)**

Measures how many notes you played out of the total expected. Uses a square root curve — playing 80% of the notes gives you ~89% completeness, so you're not heavily penalized for missing a few.

**Rhythm (20% of overall)**

Measures proportional timing between consecutive notes. Compares the ratio of inter-onset intervals (IOIs) between your performance and the score. A 30% deadzone means small timing deviations don't cost anything — only large rhythmic errors are penalized. This is fully tempo-independent.

**Overall**

```
overall = 0.45 × pitch + 0.35 × completeness + 0.20 × rhythm
```

### Robustness Features

- **Onset deduplication** — Merges near-simultaneous (within 50ms) duplicate detections of the same pitch, keeping the higher-velocity instance
- **Match validation** — DTW pairs where the pitch distance exceeds one octave (12 semitones) are reclassified as missed/extra notes rather than wrong matches
- **Empty audio handling** — If no notes are detected (silence, background noise), a safe zero-score result is returned

## Architecture

Two services deployed on Railway from a single repo:

```
Browser
  │
  ▼
Express Server (Node.js)          ← public-facing, serves frontend + static assets
  │  proxies /api/score
  │  proxies /api/render-tempo
  ▼
FastAPI Scoring Service (Python)  ← private, runs basic-pitch inference
```

### Frontend

- **Sheet music rendering** — [OpenSheetMusicDisplay](https://opensheetmusicdisplay.org/) renders MusicXML to SVG in the browser
- **Audio recording** — MediaRecorder API captures audio as WebM, converted to WAV client-side before upload
- **Tempo control** — Adjustable BPM slider re-renders reference audio server-side via FluidSynth
- **Metronome** — Web Audio API click track synced to selected tempo

### Backend

- **Express** (`server.js`) — Serves the SPA, static MusicXML/MP3 files, manages the daily challenge rotation, leaderboard, and proxies scoring requests to the Python service
- **FastAPI** (`scmpa/server/scoring_service.py`) — Accepts audio + segment ID, runs the scoring pipeline, synthesizes tempo-adjusted reference audio

### Data

- **473 segments** of 4-bar classical piano music (Bach, Mozart, Beethoven, Chopin, etc.)
- Each segment has: MusicXML (rendering), MIDI (ground truth), MP3 (reference audio)
- Segments sourced from public domain scores, programmatically split using music21

## Tech Stack

| Component | Technology |
|---|---|
| Frontend | Vanilla JS, HTML/CSS |
| Sheet music | OpenSheetMusicDisplay (OSMD) |
| Web server | Express (Node.js) |
| Scoring service | FastAPI (Python) |
| Audio transcription | basic-pitch (Spotify) |
| Note alignment | Dynamic Time Warping |
| Audio synthesis | FluidSynth via pretty-midi |
| MusicXML processing | music21 |
| Deployment | Railway (Docker) |

## Local Development

### Prerequisites

- Node.js 18+
- Python 3.11+
- ffmpeg
- FluidSynth

### Setup

```bash
# Install Node dependencies
npm install

# Install Python dependencies
cd scmpa
pip install -r requirements.txt
cd ..

# Start the scoring service
cd scmpa
uvicorn server.scoring_service:app --port 8001
cd ..

# Start the web server (in another terminal)
npm run dev
```

Visit `http://localhost:3000`.

## Deployment

Both services deploy to Railway from this repo using Dockerfiles:

- `Dockerfile.web` — Express server (public, serves frontend)
- `Dockerfile.scorer` — FastAPI scoring service (private, internal networking)

Set `SCORING_URL` on the Express service to point to the scorer's private Railway hostname.

Auto-deploys on every push to `main`.

