"""Parse MusicXML and MIDI files into a unified NoteEvent representation."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np


@dataclass
class NoteEvent:
    onset: float        # seconds from start
    offset: float       # seconds from start
    midi_pitch: int     # 0-127
    velocity: int       # 0-127 (default 80 if not specified)
    voice: int          # 0 = right hand, 1 = left hand
    is_chord: bool      # True if simultaneous with other notes (within 30ms)


@dataclass
class ScoreData:
    notes: List[NoteEvent]
    tempo: float
    time_signature: Tuple[int, int]
    key_signature: str
    duration: float     # total duration in seconds


CHORD_ONSET_TOLERANCE = 0.03  # 30ms


def _assign_voice(midi_pitch: int) -> int:
    """Heuristic: notes at or above middle C (60) -> right hand (0), below -> left hand (1)."""
    return 0 if midi_pitch >= 60 else 1


def _mark_chords(notes: List[NoteEvent], tolerance: float = CHORD_ONSET_TOLERANCE) -> None:
    """Mark notes as chords if they have near-simultaneous onsets."""
    for i, note in enumerate(notes):
        for j in range(i + 1, len(notes)):
            if notes[j].onset - note.onset > tolerance:
                break
            note.is_chord = True
            notes[j].is_chord = True


def parse_musicxml(path: str) -> ScoreData:
    """Parse a MusicXML (.mxl, .xml, .musicxml) file using music21.

    Handles ties, chords, multi-voice, and multi-staff scores.
    Returns a ScoreData with notes sorted by onset time.
    """
    import music21

    score = music21.converter.parse(path)

    flat_score = score.flatten()

    # Extract tempo (first tempo marking, or default 120)
    tempo = 120.0
    for mm in flat_score.getElementsByClass(music21.tempo.MetronomeMark):
        tempo = mm.number
        break

    # Extract time signature
    time_sig = (4, 4)
    for ts in flat_score.getElementsByClass(music21.meter.TimeSignature):
        time_sig = (ts.numerator, ts.denominator)
        break

    # Extract key signature
    key_sig = "C major"
    for ks in flat_score.getElementsByClass(music21.key.KeySignature):
        key_sig = str(ks)
        break
    for k in flat_score.getElementsByClass(music21.key.Key):
        key_sig = str(k)
        break

    notes: List[NoteEvent] = []

    # music21 offsets are in quarter-note lengths; convert to seconds
    quarter_duration = 60.0 / tempo

    for element in flat_score.notesAndRests:
        if isinstance(element, music21.note.Note):
            if element.tie is not None and element.tie.type == 'stop':
                # Tied continuation — extend the previous matching note
                for prev in reversed(notes):
                    if prev.midi_pitch == element.pitch.midi:
                        prev.offset = (element.offset + element.quarterLength) * quarter_duration
                        break
                continue
            if element.tie is not None and element.tie.type == 'continue':
                for prev in reversed(notes):
                    if prev.midi_pitch == element.pitch.midi:
                        prev.offset = (element.offset + element.quarterLength) * quarter_duration
                        break
                continue

            onset = element.offset * quarter_duration
            offset = (element.offset + element.quarterLength) * quarter_duration
            pitch = element.pitch.midi
            vel = element.volume.velocity if element.volume.velocity is not None else 80

            notes.append(NoteEvent(
                onset=onset,
                offset=offset,
                midi_pitch=pitch,
                velocity=int(np.clip(vel, 1, 127)),
                voice=_assign_voice(pitch),
                is_chord=False,
            ))

        elif isinstance(element, music21.chord.Chord):
            for p in element.pitches:
                onset = element.offset * quarter_duration
                offset = (element.offset + element.quarterLength) * quarter_duration
                pitch = p.midi
                vel = element.volume.velocity if element.volume.velocity is not None else 80

                notes.append(NoteEvent(
                    onset=onset,
                    offset=offset,
                    midi_pitch=pitch,
                    velocity=int(np.clip(vel, 1, 127)),
                    voice=_assign_voice(pitch),
                    is_chord=True,
                ))

    notes.sort(key=lambda n: (n.onset, n.midi_pitch))
    _mark_chords(notes)

    duration = max((n.offset for n in notes), default=0.0)

    return ScoreData(
        notes=notes,
        tempo=tempo,
        time_signature=time_sig,
        key_signature=key_sig,
        duration=duration,
    )


def parse_midi(path: str) -> ScoreData:
    """Parse a MIDI (.mid) file using pretty_midi.

    No voice/staff info is available from MIDI — uses the pitch heuristic.
    """
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(path)

    # Estimate tempo from MIDI tempo changes
    tempos = pm.get_tempo_changes()
    tempo = float(tempos[1][0]) if len(tempos[1]) > 0 else 120.0

    # Estimate time signature
    time_sig = (4, 4)
    if pm.time_signature_changes:
        ts = pm.time_signature_changes[0]
        time_sig = (ts.numerator, ts.denominator)

    notes: List[NoteEvent] = []

    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            notes.append(NoteEvent(
                onset=note.start,
                offset=note.end,
                midi_pitch=note.pitch,
                velocity=note.velocity,
                voice=_assign_voice(note.pitch),
                is_chord=False,
            ))

    notes.sort(key=lambda n: (n.onset, n.midi_pitch))
    _mark_chords(notes)

    duration = max((n.offset for n in notes), default=0.0)

    return ScoreData(
        notes=notes,
        tempo=tempo,
        time_signature=time_sig,
        key_signature="unknown",
        duration=duration,
    )


def parse_score(path: str) -> ScoreData:
    """Auto-detect format and parse."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in ('.mxl', '.xml', '.musicxml'):
        return parse_musicxml(path)
    elif suffix in ('.mid', '.midi'):
        return parse_midi(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
