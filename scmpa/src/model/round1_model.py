"""Round 1 Model: Frozen MERT + Linear Probe.

Training:
    Reference embeddings loaded from cache (pre-computed, no MERT pass needed).
    Performance audio goes through frozen MERT to get embeddings.
    Both are mean-pooled and fed to a linear head.

For maximum training speed, precompute ALL embeddings (both ref and perf)
before training so the linear head trains on cached vectors in seconds.

Inference:
    Reference embeddings loaded from cache.
    User audio goes through frozen MERT.
    Linear head outputs 4 scores (rhythm, pitch, completeness, flow).
    Overall = 0.4*rhythm + 0.3*pitch + 0.2*completeness + 0.1*flow.
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, Wav2Vec2FeatureExtractor


class Round1ScoringHead(nn.Module):
    """Linear probe: concatenated mean-pooled embeddings -> 4 quality scores."""

    def __init__(self, d_model=768, n_scores=4):
        super().__init__()
        self.linear = nn.Linear(d_model * 2, n_scores)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ref_pooled, perf_pooled):
        x = torch.cat([ref_pooled, perf_pooled], dim=-1)
        return self.sigmoid(self.linear(x))


class Round2ScoringHead(nn.Module):
    """MLP head with rich pooling (mean+std+max)."""

    def __init__(self, d_pool=768*3, hidden=512, dropout=0.1, n_scores=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_pool * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_scores),
            nn.Sigmoid(),
        )

    def forward(self, ref_pooled, perf_pooled):
        x = torch.cat([ref_pooled, perf_pooled], dim=-1)
        return self.mlp(x)


def pool_embeddings(e):
    """Rich pooling: mean+std+max from frame-level embeddings [T, 768] -> [2304]."""
    mean = e.mean(dim=0)
    std = e.std(dim=0) if e.shape[0] > 1 else torch.zeros_like(mean)
    max_val = e.max(dim=0)[0]
    return torch.cat([mean, std, max_val], dim=-1)


class Round1Model(nn.Module):
    """Full inference model: frozen MERT + scoring head (linear or MLP)."""

    def __init__(self, model_name="m-a-p/MERT-v1-95M", layer_indices=None,
                 n_scores=4, head_type="linear", hidden=512, dropout=0.1):
        super().__init__()
        self.mert = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            model_name, trust_remote_code=True
        )

        for param in self.mert.parameters():
            param.requires_grad = False
        self.mert.eval()

        self.layer_indices = layer_indices or list(range(13))
        self.head_type = head_type

        if head_type == "mlp_rich":
            self.head = Round2ScoringHead(
                d_pool=768*3, hidden=hidden, dropout=dropout, n_scores=n_scores
            )
        elif head_type == "mlp":
            self.head = Round2ScoringHead(
                d_pool=768, hidden=hidden, dropout=dropout, n_scores=n_scores
            )
        else:
            self.head = Round1ScoringHead(d_model=768, n_scores=n_scores)

    @property
    def device(self):
        return next(self.mert.parameters()).device

    def encode_audio(self, audio_numpy, sr=24000):
        """Run numpy audio through frozen MERT, return mean-pooled [768] embedding."""
        inputs = self.processor(audio_numpy, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.mert(**inputs, output_hidden_states=True)

        selected = [outputs.hidden_states[i] for i in self.layer_indices]
        combined = torch.stack(selected).mean(dim=0)  # [1, T, 768]
        return combined.squeeze(0)  # [T, 768]

    def forward(self, ref_pooled, perf_pooled):
        """Score from pre-computed mean-pooled embeddings.

        Args:
            ref_pooled:  [B, 768]
            perf_pooled: [B, 768]
        Returns:
            scores: [B, 4]
        """
        return self.head(ref_pooled, perf_pooled)

    def _pool(self, frame_emb):
        """Pool frame-level embeddings based on head type."""
        if self.head_type == "mlp_rich":
            return pool_embeddings(frame_emb)
        return frame_emb.mean(dim=0)

    def score_audio(self, user_audio_numpy, ref_embedding_pooled):
        """End-to-end scoring for inference.

        Args:
            user_audio_numpy: 1-D numpy array at 24kHz
            ref_embedding_pooled: tensor (pre-pooled to match head type)
        Returns:
            dict with rhythm, pitch, completeness, flow, overall
        """
        perf_emb = self.encode_audio(user_audio_numpy)  # [T, 768]
        perf_pooled = self._pool(perf_emb).unsqueeze(0)  # [1, D]
        ref_pooled = ref_embedding_pooled.unsqueeze(0).to(self.device)  # [1, D]

        with torch.no_grad():
            scores = self.head(ref_pooled, perf_pooled)

        r, p, c, f = scores[0].cpu().tolist()
        overall = 0.4 * r + 0.3 * p + 0.2 * c + 0.1 * f

        return {
            "rhythm": round(r, 3),
            "pitch": round(p, 3),
            "completeness": round(c, 3),
            "flow": round(f, 3),
            "overall": round(overall, 3),
        }
