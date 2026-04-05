"""On-the-fly segmentation service.

Parses all source MusicXML files into memory at startup.
Extracts bar ranges on request — fast because parsing is already done.
"""

import hashlib
import random
from pathlib import Path
from typing import Dict, List, Optional

import music21


class SegmentService:
    def __init__(self, source_dir: str, cache_dir: str = "/tmp/sightreadle_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "musicxml").mkdir(exist_ok=True)
        (self.cache_dir / "midi").mkdir(exist_ok=True)

        self.pieces: Dict[str, List[dict]] = {
            "easy": [],
            "intermediate": [],
            "advanced": [],
        }

        source_path = Path(source_dir)
        dir_map = {"easy": "easy", "inter": "intermediate", "advanced": "advanced"}

        print("Parsing source MusicXML files into memory...", flush=True)

        for dir_name, difficulty in dir_map.items():
            dir_path = source_path / dir_name
            if not dir_path.exists():
                print(f"  Warning: {dir_path} does not exist, skipping", flush=True)
                continue

            xml_files = []
            for ext in ["*.musicxml", "*.mxl", "*.xml"]:
                xml_files.extend(dir_path.glob(ext))

            for xml_file in sorted(xml_files):
                try:
                    print(f"  Parsing: {xml_file.name} ({difficulty})", flush=True)
                    score = music21.converter.parse(str(xml_file))

                    parts = score.parts
                    if not parts:
                        print(f"    Skipped: no parts found", flush=True)
                        continue

                    primary_part = parts[0]
                    measures = list(primary_part.getElementsByClass('Measure'))
                    total_bars = len(measures)

                    if total_bars < 4:
                        print(f"    Skipped: only {total_bars} bars", flush=True)
                        continue

                    bar_note_counts = []
                    for bar_idx in range(total_bars):
                        try:
                            bar_score = score.measures(bar_idx + 1, bar_idx + 1)
                            all_notes = list(bar_score.recurse().notes)
                            n = sum(
                                len(note.pitches) if hasattr(note, 'pitches') else 1
                                for note in all_notes
                            )
                            bar_note_counts.append(n)
                        except Exception:
                            bar_note_counts.append(0)

                    tempo = 120.0
                    for mm in score.recurse().getElementsByClass('MetronomeMark'):
                        if mm.number is not None:
                            tempo = mm.number
                        break

                    time_sig = (4, 4)
                    for ts in score.recurse().getElementsByClass('TimeSignature'):
                        time_sig = (ts.numerator, ts.denominator)
                        break

                    key_sig = "C major"
                    for ks in score.recurse().getElementsByClass('KeySignature'):
                        try:
                            key_sig = str(ks.asKey())
                        except Exception:
                            pass
                        break

                    self.pieces[difficulty].append({
                        "name": xml_file.stem,
                        "score": score,
                        "total_bars": total_bars,
                        "bar_note_counts": bar_note_counts,
                        "tempo": tempo,
                        "time_signature": list(time_sig),
                        "key_signature": key_sig,
                    })

                    print(f"    OK: {total_bars} bars, {sum(bar_note_counts)} total notes", flush=True)

                except Exception as e:
                    print(f"    ERROR parsing {xml_file.name}: {e}", flush=True)

        total = sum(len(v) for v in self.pieces.values())
        print(f"\nStartup complete: {total} pieces loaded", flush=True)
        for diff, pieces in self.pieces.items():
            print(f"  {diff}: {len(pieces)} pieces", flush=True)

    def get_random_segment(
        self,
        difficulty: str,
        n_bars: int,
        min_notes: int = 8,
        max_notes: int = 200,
        max_attempts: int = 50,
        exclude_pieces: Optional[List[str]] = None,
        exclude_segments: Optional[List[str]] = None,
    ) -> Optional[Dict]:
        """Extract a random segment from a random piece at the given difficulty."""
        pieces = self.pieces.get(difficulty, [])
        if not pieces:
            return None

        if exclude_pieces:
            available = [p for p in pieces if p["name"] not in exclude_pieces]
            if not available:
                available = pieces
        else:
            available = pieces

        for _ in range(max_attempts):
            piece = random.choice(available)

            if piece["total_bars"] < n_bars:
                continue

            max_start = piece["total_bars"] - n_bars
            start_bar = random.randint(0, max_start)
            end_bar = start_bar + n_bars

            note_count = sum(piece["bar_note_counts"][start_bar:end_bar])

            if note_count < min_notes or note_count > max_notes:
                continue

            seg_id = f"{piece['name']}_b{start_bar}-{end_bar}"
            cache_key = hashlib.md5(seg_id.encode()).hexdigest()[:12]
            seg_id_safe = f"{piece['name']}_{cache_key}"

            if exclude_segments and seg_id_safe in exclude_segments:
                continue

            try:
                segment_score = piece["score"].measures(start_bar + 1, end_bar)

                xml_path = self.cache_dir / "musicxml" / f"{seg_id_safe}.musicxml"
                if not xml_path.exists():
                    segment_score.write('musicxml', fp=str(xml_path))

                midi_path = self.cache_dir / "midi" / f"{seg_id_safe}.mid"
                if not midi_path.exists():
                    try:
                        segment_score.write('midi', fp=str(midi_path))
                    except Exception:
                        cleaned = segment_score.stripTies()
                        for el in cleaned.recurse():
                            if isinstance(el, (music21.bar.Repeat, music21.repeat.RepeatExpression)):
                                cleaned.remove(el, recurse=True)
                        cleaned.write('midi', fp=str(midi_path))

                bar_duration = (60.0 / piece["tempo"]) * piece["time_signature"][0]
                duration_sec = n_bars * bar_duration

                return {
                    "id": seg_id_safe,
                    "source_piece": piece["name"],
                    "difficulty": difficulty,
                    "start_bar": start_bar,
                    "n_bars": n_bars,
                    "n_notes": note_count,
                    "tempo": piece["tempo"],
                    "time_signature": piece["time_signature"],
                    "key_signature": piece["key_signature"],
                    "duration_sec": round(duration_sec, 2),
                    "musicxml_path": str(xml_path),
                    "midi_path": str(midi_path),
                }
            except Exception as e:
                print(f"  Segment extraction failed: {e}", flush=True)
                continue

        return None

    def get_daily_segment(self, day_number: int) -> Optional[Dict]:
        """Get a deterministic segment for the daily challenge.
        
        Bar count varies 4-8 deterministically per day.
        Cycles through all pieces across all difficulties.
        """
        all_pieces = []
        for diff in ["easy", "intermediate", "advanced"]:
            for piece in self.pieces[diff]:
                all_pieces.append((diff, piece))

        if not all_pieces:
            return None

        # Offset so today's challenge is different from old scheme
        adjusted_day = day_number + 7

        rng = random.Random(42)
        shuffled = list(range(len(all_pieces)))
        rng.shuffle(shuffled)

        piece_idx = shuffled[adjusted_day % len(shuffled)]
        difficulty, piece = all_pieces[piece_idx]

        day_rng = random.Random(adjusted_day)
        n_bars = day_rng.randint(4, 8)

        actual_bars = min(n_bars, piece["total_bars"])
        if actual_bars < 2:
            actual_bars = 2

        max_start = max(0, piece["total_bars"] - actual_bars)
        start_bar = day_rng.randint(0, max_start)
        end_bar = start_bar + actual_bars

        note_count = sum(piece["bar_note_counts"][start_bar:end_bar])

        attempts = 0
        while (note_count < 8 or note_count > 200) and attempts < 10:
            start_bar = day_rng.randint(0, max_start)
            end_bar = start_bar + actual_bars
            note_count = sum(piece["bar_note_counts"][start_bar:end_bar])
            attempts += 1

        try:
            segment_score = piece["score"].measures(start_bar + 1, end_bar)

            seg_id = f"daily_{day_number}_{piece['name']}"
            cache_key = hashlib.md5(seg_id.encode()).hexdigest()[:12]
            seg_id_safe = f"daily_{cache_key}"

            xml_path = self.cache_dir / "musicxml" / f"{seg_id_safe}.musicxml"
            if not xml_path.exists():
                segment_score.write('musicxml', fp=str(xml_path))

            midi_path = self.cache_dir / "midi" / f"{seg_id_safe}.mid"
            if not midi_path.exists():
                try:
                    segment_score.write('midi', fp=str(midi_path))
                except Exception:
                    cleaned = segment_score.stripTies()
                    cleaned.write('midi', fp=str(midi_path))

            bar_duration = (60.0 / piece["tempo"]) * piece["time_signature"][0]
            duration_sec = actual_bars * bar_duration

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
                "duration_sec": round(duration_sec, 2),
                "musicxml_path": str(xml_path),
                "midi_path": str(midi_path),
            }
        except Exception as e:
            print(f"  Daily segment extraction failed: {e}", flush=True)
            return None

    def get_available_difficulties(self) -> Dict:
        return {diff: len(pieces) for diff, pieces in self.pieces.items()}
