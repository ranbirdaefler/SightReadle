"""Loss functions: contrastive regression, supervised score regression, and joint."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SCMPALoss(nn.Module):
    """Joint contrastive regression + supervised score regression loss."""

    def __init__(
        self,
        lambda_contrastive: float = 1.0,
        lambda_score: float = 1.0,
        margin: float = 0.05,
        use_efficient_contrastive: bool = False,
    ):
        super().__init__()
        self.lambda_c = lambda_contrastive
        self.lambda_s = lambda_score
        self.margin = margin
        self.use_efficient = use_efficient_contrastive

    def contrastive_regression_loss(
        self,
        z_ref: torch.Tensor,
        z_perf: torch.Tensor,
        quality_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Pairwise ranking: higher quality should have higher cosine similarity to reference.

        Args:
            z_ref: [B, D] global reference embeddings
            z_perf: [B, D] global performance embeddings
            quality_scores: [B, 5] quality labels (uses overall = index 4)
        """
        sim = F.cosine_similarity(z_ref, z_perf, dim=-1)  # [B]
        overall = quality_scores[:, 4]  # overall score

        if self.use_efficient:
            return F.mse_loss(sim, overall)

        B = sim.size(0)
        loss = torch.tensor(0.0, device=sim.device, requires_grad=True)
        n_pairs = 0

        for i in range(B):
            for j in range(i + 1, B):
                if overall[i] > overall[j]:
                    pair_loss = F.relu(sim[j] - sim[i] + self.margin)
                    loss = loss + pair_loss
                    n_pairs += 1
                elif overall[j] > overall[i]:
                    pair_loss = F.relu(sim[i] - sim[j] + self.margin)
                    loss = loss + pair_loss
                    n_pairs += 1

        if n_pairs > 0:
            loss = loss / n_pairs

        return loss

    def score_regression_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Huber loss on per-dimension quality scores.

        Args:
            predicted: [B, 4] predicted scores (rhythm, pitch, completeness, flow)
            target: [B, 5] ground truth (first 4 dimensions used)
        """
        return F.huber_loss(predicted, target[:, :4], delta=0.1)

    def forward(
        self,
        z_ref: torch.Tensor,
        z_perf: torch.Tensor,
        predicted_scores: torch.Tensor,
        target_scores: torch.Tensor,
    ) -> tuple:
        """Compute joint loss.

        Returns:
            (total_loss, {"contrastive": float, "score": float})
        """
        L_c = self.contrastive_regression_loss(z_ref, z_perf, target_scores)
        L_s = self.score_regression_loss(predicted_scores, target_scores)

        total = self.lambda_c * L_c + self.lambda_s * L_s

        return total, {
            "contrastive": L_c.item(),
            "score": L_s.item(),
        }


class ContrastiveOnlyLoss(nn.Module):
    """Contrastive regression loss only (for Model C ablation)."""

    def __init__(self, margin: float = 0.05, use_efficient: bool = False):
        super().__init__()
        self.inner = SCMPALoss(
            lambda_contrastive=1.0, lambda_score=0.0,
            margin=margin, use_efficient_contrastive=use_efficient,
        )

    def forward(self, z_ref, z_perf, predicted_scores, target_scores):
        return self.inner(z_ref, z_perf, predicted_scores, target_scores)


class SupervisedOnlyLoss(nn.Module):
    """Supervised score regression loss only (for Model B ablation)."""

    def __init__(self):
        super().__init__()
        self.inner = SCMPALoss(lambda_contrastive=0.0, lambda_score=1.0)

    def forward(self, z_ref, z_perf, predicted_scores, target_scores):
        return self.inner(z_ref, z_perf, predicted_scores, target_scores)
