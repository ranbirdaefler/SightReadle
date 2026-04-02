"""Visualization utilities: t-SNE, cosine similarity heatmaps, attention maps."""

from typing import Dict, List, Optional

import numpy as np


def plot_tsne_quality(
    embeddings: np.ndarray,
    quality_scores: np.ndarray,
    save_path: str,
    real_embeddings: Optional[np.ndarray] = None,
    real_scores: Optional[np.ndarray] = None,
    perplexity: int = 30,
    title: str = "t-SNE of Performance Embeddings by Quality",
) -> None:
    """Plot t-SNE of embeddings colored by quality score.

    Args:
        embeddings: [N, D] array of embeddings
        quality_scores: [N] array of overall quality scores
        save_path: path to save the figure
        real_embeddings: optional [M, D] real recording embeddings (plotted as stars)
        real_scores: optional [M] real recording quality scores
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    all_embs = embeddings
    if real_embeddings is not None:
        all_embs = np.vstack([embeddings, real_embeddings])

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    coords = tsne.fit_transform(all_embs)

    fig, ax = plt.subplots(figsize=(10, 8))

    n_synth = len(embeddings)
    scatter = ax.scatter(
        coords[:n_synth, 0], coords[:n_synth, 1],
        c=quality_scores, cmap='viridis', s=30, alpha=0.7,
        vmin=0, vmax=1,
    )

    if real_embeddings is not None and real_scores is not None:
        ax.scatter(
            coords[n_synth:, 0], coords[n_synth:, 1],
            c=real_scores, cmap='viridis', s=150, alpha=0.9,
            marker='*', edgecolors='black', linewidths=0.5,
            vmin=0, vmax=1,
        )

    plt.colorbar(scatter, ax=ax, label='Overall Quality Score')
    ax.set_title(title)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_cosine_vs_quality(
    similarities: np.ndarray,
    quality_scores: np.ndarray,
    save_path: str,
    title: str = "Cosine Similarity vs. Quality Score",
) -> None:
    """Scatter plot of cosine similarity vs ground truth quality."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(quality_scores, similarities, alpha=0.5, s=20)
    ax.set_xlabel('Ground Truth Quality Score')
    ax.set_ylabel('Cosine Similarity to Reference')
    ax.set_title(title)
    ax.set_xlim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    save_path: str,
    title: str = "Cross-Attention Weights",
) -> None:
    """Plot cross-attention heatmap (performance frames × reference frames)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attention_weights, aspect='auto', cmap='hot', interpolation='nearest')
    ax.set_xlabel('Reference Time Frames')
    ax.set_ylabel('Performance Time Frames')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Attention Weight')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
