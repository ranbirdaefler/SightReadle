"""Full SCMPA model: MERT backbone + scoring head."""

import torch
import torch.nn as nn

from src.model.mert_backbone import build_mert_lora, build_frozen_mert, MERTFeatureExtractor
from src.model.scoring_head import CrossAttentionScoringHead, MLPScoringHead, LinearScoringHead


class SCMPA(nn.Module):
    """Score-Conditioned Music Performance Assessment model.

    Shared MERT backbone extracts embeddings for both reference and performance audio.
    A scoring head compares them to produce quality scores.
    """

    def __init__(
        self,
        feature_extractor: MERTFeatureExtractor,
        scoring_head: nn.Module,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.scoring_head = scoring_head

    def forward(
        self,
        audio_ref: torch.Tensor,
        audio_perf: torch.Tensor,
        ref_mask: torch.Tensor = None,
        perf_mask: torch.Tensor = None,
    ) -> tuple:
        """
        Args:
            audio_ref:  [B, T_ref]  reference waveform at 24kHz
            audio_perf: [B, T_perf] performance waveform at 24kHz
            ref_mask:   [B, T_ref_frames] optional padding mask
            perf_mask:  [B, T_perf_frames] optional padding mask

        Returns:
            (scores [B, 4], z_ref [B, 768], z_perf [B, 768])
        """
        e_ref = self.feature_extractor(audio_ref)    # [B, T_ref_frames, 768]
        e_perf = self.feature_extractor(audio_perf)  # [B, T_perf_frames, 768]

        # Global embeddings for contrastive loss
        z_ref = e_ref.mean(dim=1)    # [B, 768]
        z_perf = e_perf.mean(dim=1)  # [B, 768]

        # Detailed scores from scoring head
        scores = self.scoring_head(e_perf, e_ref, perf_mask=perf_mask, ref_mask=ref_mask)

        return scores, z_ref, z_perf


def build_model(config) -> SCMPA:
    """Factory function to build SCMPA model from config.

    Supports all ablation variants based on config.experiment:
      - "baseline0": frozen MERT, no head (cosine similarity only)
      - "baseline1": frozen MERT, linear probe, supervised
      - "model_a":   frozen MERT, cross-attention head, supervised
      - "model_b":   MERT + LoRA, cross-attention head, supervised
      - "model_c":   MERT + LoRA, cross-attention head, contrastive only
      - "model_d":   MERT + LoRA, cross-attention head, contrastive + supervised (full)
      - "model_e":   MERT + LoRA, MLP head, contrastive + supervised
    """
    experiment = getattr(config, 'experiment', 'model_d')
    model_cfg = config.model

    # Build backbone
    use_lora = model_cfg.lora_rank > 0
    if use_lora:
        mert = build_mert_lora(
            model_name=model_cfg.backbone,
            lora_rank=model_cfg.lora_rank,
            lora_alpha=model_cfg.lora_alpha,
            lora_dropout=model_cfg.lora_dropout,
            target_modules=model_cfg.lora_target_modules,
        )
    else:
        mert = build_frozen_mert(model_name=model_cfg.backbone)

    feature_extractor = MERTFeatureExtractor(mert)

    # Build scoring head
    head_type = getattr(model_cfg, 'head_type', 'cross_attention')
    if head_type == 'cross_attention':
        scoring_head = CrossAttentionScoringHead(
            n_layers=model_cfg.head_layers,
            d_ff=model_cfg.head_dim_ff,
            n_heads=model_cfg.head_n_heads,
        )
    elif head_type == 'mlp':
        scoring_head = MLPScoringHead()
    elif head_type == 'linear':
        scoring_head = LinearScoringHead()
    else:
        raise ValueError(f"Unknown head_type: {head_type}")

    return SCMPA(feature_extractor, scoring_head)
