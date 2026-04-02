"""PyTorch Dataset for SCMPA training.

Supports two modes:
  Mode 1 (audio pairs): Returns raw audio waveforms for both ref and perf.
      Used when training with a live MERT backbone (LoRA fine-tuning).

  Mode 2 (cached ref): Returns cached MERT embeddings for ref, raw audio for perf.
      Used when training with frozen MERT (Rounds 1-2) to skip redundant ref encoding.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class SCMPADataset(Dataset):
    """Dataset of (reference, performance, quality_labels) triples."""

    def __init__(
        self,
        metadata_path: str,
        split: str,
        sample_rate: int = 24000,
        max_length: float = 20.0,
        embeddings_dir: Optional[str] = None,
        layer_indices: Optional[List[int]] = None,
    ):
        """
        Args:
            metadata_path: path to data/synthetic/metadata.json
            split: "train", "val", or "test_synth"
            sample_rate: target sample rate (24000 for MERT)
            max_length: max audio length in seconds
            embeddings_dir: if provided, load cached ref embeddings (Mode 2)
            layer_indices: which MERT layers to use from cached embeddings.
                          None = all 13 averaged.
        """
        self.sample_rate = sample_rate
        self.max_samples = int(max_length * sample_rate)
        self.embeddings_dir = Path(embeddings_dir) if embeddings_dir else None
        self.layer_indices = layer_indices

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        if split not in metadata:
            raise ValueError(f"Split '{split}' not found. Available: {list(metadata.keys())}")

        self.samples = metadata[split]

        if self.embeddings_dir:
            missing = []
            for s in self.samples:
                seg_id = s["segment_id"]
                if not (self.embeddings_dir / f"{seg_id}.pt").exists():
                    missing.append(seg_id)
            if missing:
                print(f"Warning: {len(missing)} segments missing cached embeddings")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        perf_audio = self._load_audio(sample["perf_audio"])

        labels = sample["labels"]
        label_tensor = torch.tensor([
            labels["rhythm"],
            labels["pitch"],
            labels["completeness"],
            labels["flow"],
            labels["overall"],
        ], dtype=torch.float32)

        if self.embeddings_dir:
            seg_id = sample["segment_id"]
            emb_path = self.embeddings_dir / f"{seg_id}.pt"
            emb_data = torch.load(emb_path, weights_only=False)

            if self.layer_indices:
                ref_emb = torch.stack(
                    [emb_data["hidden_states"][i] for i in self.layer_indices]
                ).mean(dim=0).float()
            else:
                ref_emb = torch.stack(
                    emb_data["hidden_states"]
                ).mean(dim=0).float()

            return {
                "ref_embedding": ref_emb,
                "perf_audio": perf_audio,
                "labels": label_tensor,
                "segment_id": seg_id,
            }
        else:
            ref_audio = self._load_audio(sample["ref_audio"])
            return {
                "ref_audio": ref_audio,
                "perf_audio": perf_audio,
                "labels": label_tensor,
                "segment_id": sample.get("segment_id", ""),
            }

    def _load_audio(self, path: str) -> torch.Tensor:
        """Load, resample, and truncate audio to a fixed length."""
        import soundfile as sf

        audio, sr = sf.read(path, dtype='float32', always_2d=True)

        if audio.shape[1] > 1:
            audio = audio.mean(axis=1, keepdims=True)

        if sr != self.sample_rate:
            import librosa
            audio_mono = audio[:, 0]
            audio_mono = librosa.resample(audio_mono, orig_sr=sr, target_sr=self.sample_rate)
            audio = audio_mono.reshape(-1, 1)

        waveform = torch.from_numpy(audio.T)  # [1, T]

        if waveform.shape[1] > self.max_samples:
            waveform = waveform[:, :self.max_samples]

        return waveform


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate variable-length data by padding to the longest in the batch."""
    has_cached_emb = "ref_embedding" in batch[0]
    labels = torch.stack([item["labels"] for item in batch])
    segment_ids = [item["segment_id"] for item in batch]

    perf_audios = [item["perf_audio"] for item in batch]
    perf_padded, perf_mask = _pad_sequences(perf_audios)

    result = {
        "perf_audio": perf_padded,
        "perf_mask": perf_mask,
        "labels": labels,
        "segment_ids": segment_ids,
    }

    if has_cached_emb:
        ref_embs = [item["ref_embedding"] for item in batch]
        ref_padded, ref_mask = _pad_embeddings(ref_embs)
        result["ref_embedding"] = ref_padded
        result["ref_mask"] = ref_mask
    else:
        ref_audios = [item["ref_audio"] for item in batch]
        ref_padded, ref_mask = _pad_sequences(ref_audios)
        result["ref_audio"] = ref_padded
        result["ref_mask"] = ref_mask

    return result


def _pad_sequences(tensors: List[torch.Tensor]) -> tuple:
    """Pad a list of [1, T_i] tensors to [B, 1, T_max], return masks."""
    lengths = [t.shape[1] for t in tensors]
    max_len = max(lengths)

    padded = torch.zeros(len(tensors), 1, max_len, dtype=tensors[0].dtype)
    mask = torch.zeros(len(tensors), max_len, dtype=torch.bool)

    for i, (t, length) in enumerate(zip(tensors, lengths)):
        padded[i, :, :length] = t
        mask[i, :length] = True

    return padded, mask


def _pad_embeddings(tensors: List[torch.Tensor]) -> tuple:
    """Pad a list of [T_i, D] embedding tensors to [B, T_max, D], return masks."""
    lengths = [t.shape[0] for t in tensors]
    max_len = max(lengths)
    d_model = tensors[0].shape[1]

    padded = torch.zeros(len(tensors), max_len, d_model, dtype=tensors[0].dtype)
    mask = torch.zeros(len(tensors), max_len, dtype=torch.bool)

    for i, (t, length) in enumerate(zip(tensors, lengths)):
        padded[i, :length] = t
        mask[i, :length] = True

    return padded, mask
