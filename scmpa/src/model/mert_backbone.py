"""MERT backbone loading with optional LoRA injection."""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mert_lora(
    model_name: str = "m-a-p/MERT-v1-95M",
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
    freeze_base: bool = True,
) -> nn.Module:
    """Load MERT and inject LoRA adapters.

    If lora_rank == 0, returns frozen MERT with no LoRA (for baselines).
    """
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    if lora_rank > 0:
        from peft import LoraConfig, get_peft_model

        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj"]

        # Verify target modules exist
        module_names = [name for name, _ in model.named_modules()]
        found = any(any(t in name for t in target_modules) for name in module_names)
        if not found:
            # Fall back to discovering attention projection layers
            attn_modules = [n for n in module_names if 'attention' in n.lower() and 'proj' in n.lower()]
            if attn_modules:
                print(f"Warning: target_modules {target_modules} not found directly. "
                      f"Found attention modules: {attn_modules[:5]}")

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model


def build_frozen_mert(model_name: str = "m-a-p/MERT-v1-95M") -> nn.Module:
    """Load MERT with all parameters frozen, no LoRA. For Baseline 0/1."""
    return build_mert_lora(model_name=model_name, lora_rank=0, freeze_base=True)


class MERTFeatureExtractor(nn.Module):
    """Wraps MERT to extract a weighted combination of all layer hidden states.

    Learns per-layer weights to produce a single frame-level embedding sequence.
    """

    def __init__(self, mert_model: nn.Module, n_layers: int = 13):
        super().__init__()
        self.mert = mert_model
        self.layer_weights = nn.Parameter(torch.ones(n_layers) / n_layers)
        self.n_layers = n_layers

    def forward(self, audio_values: torch.Tensor) -> torch.Tensor:
        """Extract weighted-sum frame-level embeddings.

        Args:
            audio_values: [B, T_audio] raw waveform at 24kHz

        Returns:
            [B, T_frames, 768] weighted combination of all layer outputs
        """
        outputs = self.mert(audio_values, output_hidden_states=True)
        hidden_states = torch.stack(outputs.hidden_states)  # [n_layers, B, T, 768]
        weights = F.softmax(self.layer_weights, dim=0)       # [n_layers]
        weighted = (hidden_states * weights.view(-1, 1, 1, 1)).sum(dim=0)  # [B, T, 768]
        return weighted

    def extract_per_layer(self, audio_values: torch.Tensor) -> List[torch.Tensor]:
        """Extract per-layer embeddings for probing analysis.

        Returns list of 13 tensors, each [B, T, 768].
        """
        outputs = self.mert(audio_values, output_hidden_states=True)
        return list(outputs.hidden_states)
