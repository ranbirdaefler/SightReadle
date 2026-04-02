"""Scoring heads: cross-attention, MLP, and linear variants for ablation."""

import torch
import torch.nn as nn


class CrossAttentionScoringHead(nn.Module):
    """Performance embeddings attend to reference embeddings via cross-attention.

    The attention pattern implicitly encodes alignment and deviation.
    """

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 8,
        n_layers: int = 2,
        d_ff: int = 1024,
        dropout: float = 0.1,
        n_scores: int = 4,
    ):
        super().__init__()

        self.cross_attn_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(n_layers)
        ])

        self.pool_proj = nn.Linear(d_model, d_model)
        self.pool_act = nn.GELU()

        self.score_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )
            for _ in range(n_scores)
        ])

    def forward(
        self,
        e_perf: torch.Tensor,
        e_ref: torch.Tensor,
        perf_mask: torch.Tensor = None,
        ref_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            e_perf: [B, T_perf, 768] performance embeddings (queries)
            e_ref:  [B, T_ref, 768]  reference embeddings (keys/values)
            perf_mask: [B, T_perf] bool mask (True = valid)
            ref_mask:  [B, T_ref] bool mask

        Returns:
            scores: [B, n_scores] predicted quality scores in [0, 1]
        """
        x = e_perf
        for layer in self.cross_attn_layers:
            x = layer(
                x, e_ref,
                tgt_key_padding_mask=~perf_mask if perf_mask is not None else None,
                memory_key_padding_mask=~ref_mask if ref_mask is not None else None,
            )

        # Pool over time: project then adaptive average pool
        x = self.pool_act(self.pool_proj(x))  # [B, T, d_model]
        x = x.permute(0, 2, 1)                 # [B, d_model, T]
        x = torch.nn.functional.adaptive_avg_pool1d(x, 1).squeeze(-1)  # [B, d_model]

        scores = torch.cat([head(x) for head in self.score_heads], dim=-1)  # [B, n_scores]
        return scores

    def forward_with_attention(
        self,
        e_perf: torch.Tensor,
        e_ref: torch.Tensor,
    ) -> tuple:
        """Forward pass that also returns attention weights for visualization."""
        x = e_perf
        attention_weights = []

        for layer in self.cross_attn_layers:
            # Manually call multihead attention to get weights
            attn_out, attn_w = layer.multihead_attn(
                x, e_ref, e_ref,
                need_weights=True,
                average_attn_weights=True,
            )
            x = layer.norm2(x + layer.dropout2(attn_out))
            x = layer.norm3(x + layer.dropout3(layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))))
            attention_weights.append(attn_w)

        x = self.pool_act(self.pool_proj(x))
        x = x.permute(0, 2, 1)
        x = torch.nn.functional.adaptive_avg_pool1d(x, 1).squeeze(-1)
        scores = torch.cat([head(x) for head in self.score_heads], dim=-1)

        return scores, attention_weights


class MLPScoringHead(nn.Module):
    """Simple MLP baseline: concatenate mean-pooled ref + perf embeddings."""

    def __init__(self, d_model: int = 768, hidden: int = 512, dropout: float = 0.1, n_scores: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_scores),
            nn.Sigmoid(),
        )

    def forward(self, e_perf: torch.Tensor, e_ref: torch.Tensor, **kwargs) -> torch.Tensor:
        perf_pooled = e_perf.mean(dim=1)  # [B, 768]
        ref_pooled = e_ref.mean(dim=1)    # [B, 768]
        x = torch.cat([ref_pooled, perf_pooled], dim=-1)  # [B, 1536]
        return self.mlp(x)


class LinearScoringHead(nn.Module):
    """Simplest baseline: single linear layer from concatenated embeddings."""

    def __init__(self, d_model: int = 768, n_scores: int = 4):
        super().__init__()
        self.linear = nn.Linear(d_model * 2, n_scores)
        self.sigmoid = nn.Sigmoid()

    def forward(self, e_perf: torch.Tensor, e_ref: torch.Tensor, **kwargs) -> torch.Tensor:
        perf_pooled = e_perf.mean(dim=1)
        ref_pooled = e_ref.mean(dim=1)
        x = torch.cat([ref_pooled, perf_pooled], dim=-1)
        return self.sigmoid(self.linear(x))
