"""Per-layer embedding analysis: train linear probes on each MERT layer."""

from typing import Dict, List

import numpy as np


def probe_layer(
    embeddings: np.ndarray,
    labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
) -> Dict[str, float]:
    """Train a Ridge regression probe on one layer's embeddings.

    Args:
        embeddings: [N_train, D] mean-pooled layer embeddings
        labels: [N_train, 4] quality labels (rhythm, pitch, completeness, flow)
        test_embeddings: [N_test, D]
        test_labels: [N_test, 4]

    Returns dict with Spearman rho per quality dimension.
    """
    from sklearn.linear_model import Ridge
    from scipy import stats

    dim_names = ["rhythm", "pitch", "completeness", "flow"]
    results = {}

    for i, name in enumerate(dim_names):
        model = Ridge(alpha=1.0)
        model.fit(embeddings, labels[:, i])
        predictions = model.predict(test_embeddings)

        rho, p = stats.spearmanr(predictions, test_labels[:, i])
        results[f"{name}_rho"] = float(rho)
        results[f"{name}_p"] = float(p)

    return results


def run_layer_probing(
    per_layer_train: List[np.ndarray],
    train_labels: np.ndarray,
    per_layer_test: List[np.ndarray],
    test_labels: np.ndarray,
    save_path: str = None,
) -> np.ndarray:
    """Run probing analysis across all MERT layers.

    Args:
        per_layer_train: list of 13 arrays, each [N_train, D]
        train_labels: [N_train, 4]
        per_layer_test: list of 13 arrays, each [N_test, D]
        test_labels: [N_test, 4]
        save_path: optional path to save heatmap figure

    Returns:
        [13, 4] array of Spearman rho values (layers × quality dimensions)
    """
    n_layers = len(per_layer_train)
    n_dims = 4
    rho_matrix = np.zeros((n_layers, n_dims))

    dim_names = ["rhythm", "pitch", "completeness", "flow"]

    for layer_idx in range(n_layers):
        results = probe_layer(
            per_layer_train[layer_idx], train_labels,
            per_layer_test[layer_idx], test_labels,
        )
        for dim_idx, name in enumerate(dim_names):
            rho_matrix[layer_idx, dim_idx] = results[f"{name}_rho"]

    if save_path:
        _plot_probing_heatmap(rho_matrix, dim_names, save_path)

    return rho_matrix


def _plot_probing_heatmap(
    rho_matrix: np.ndarray,
    dim_names: List[str],
    save_path: str,
) -> None:
    """Plot a heatmap of layer × quality dimension Spearman correlations."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(
        rho_matrix,
        annot=True, fmt='.3f',
        xticklabels=dim_names,
        yticklabels=[f'Layer {i}' for i in range(rho_matrix.shape[0])],
        cmap='RdYlGn', vmin=-0.2, vmax=1.0,
        ax=ax,
    )
    ax.set_title('Layer Probing: Spearman ρ per Layer × Quality Dimension')
    ax.set_xlabel('Quality Dimension')
    ax.set_ylabel('MERT Layer')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
