"""
Visualization Utilities

Visualization tools for:
- 3D seismic volumes (slices and volume rendering)
- Well log curves (multi-track display)
- Training curves (loss, metrics)
- Model predictions (fault probability, reservoir maps)
- Cross-modal alignment visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from typing import Optional, List, Tuple, Dict
import torch


# =====================================================================
# Seismic Visualization
# =====================================================================

def plot_seismic_slice(
    seismic: np.ndarray,
    axis: str = "inline",
    slice_idx: Optional[int] = None,
    cmap: str = "seismic",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
):
    """
    Plot a 2D slice from a 3D seismic volume.

    Args:
        seismic: (D, H, W) 3D seismic volume or (1, D, H, W)
        axis: Slice axis ('inline', 'xline', 'depth')
        slice_idx: Index of slice to plot (default: middle)
        cmap: Matplotlib colormap
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    if isinstance(seismic, torch.Tensor):
        seismic = seismic.detach().cpu().numpy()

    # Remove batch/channel dims
    while seismic.ndim > 3:
        seismic = seismic[0]
    if seismic.ndim == 3 and seismic.shape[0] == 1:
        seismic = seismic[0]

    assert seismic.ndim == 3, f"Expected 3D array, got shape {seismic.shape}"

    # Select axis and slice
    if axis == "inline":
        idx = slice_idx if slice_idx is not None else seismic.shape[1] // 2
        data = seismic[:, idx, :]
        xlabel, ylabel = "Xline", "Depth/Time"
    elif axis == "xline":
        idx = slice_idx if slice_idx is not None else seismic.shape[2] // 2
        data = seismic[:, :, idx]
        xlabel, ylabel = "Inline", "Depth/Time"
    elif axis == "depth":
        idx = slice_idx if slice_idx is not None else seismic.shape[0] // 2
        data = seismic[idx, :, :]
        xlabel, ylabel = "Xline", "Inline"
    else:
        raise ValueError(f"Unknown axis: {axis}")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(data.T, cmap=cmap, aspect="auto", origin="lower",
                   norm=Normalize(vmin=-np.abs(data).max(), vmax=np.abs(data).max()))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"Seismic Slice ({axis} = {idx})")
    plt.colorbar(im, ax=ax, label="Amplitude")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_well_logs(
    curves: np.ndarray,
    curve_names: List[str],
    depth: Optional[np.ndarray] = None,
    lithology: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
):
    """
    Plot well log curves in a multi-track display.

    Args:
        curves: (L, C) log values
        curve_names: List of curve names
        depth: (L,) depth values (optional)
        lithology: (L,) lithology labels (optional)
        title: Plot title
        figsize: Figure size
        save_path: Optional save path
    """
    if isinstance(curves, torch.Tensor):
        curves = curves.detach().cpu().numpy()

    n_curves = curves.shape[1]
    if depth is None:
        depth = np.arange(curves.shape[0])

    n_cols = n_curves + (1 if lithology is not None else 0)
    fig, axes = plt.subplots(1, n_cols, figsize=figsize, sharey=True)

    if n_cols == 1:
        axes = [axes]

    # Plot each curve
    for i, name in enumerate(curve_names[:n_curves]):
        ax = axes[i]
        ax.plot(curves[:, i], depth, linewidth=0.8)
        ax.set_xlabel(name)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()

    # Lithology track
    if lithology is not None:
        litho_colors = ["#8B4513", "#F4A460", "#87CEEB", "#2F4F4F"]
        litho_names = ["Shale", "Sand", "Carbonate", "Coal"]
        ax = axes[-1]
        for i in range(4):
            mask = lithology == i
            if mask.any():
                ax.fill_betweenx(depth, 0, 1, where=mask,
                                color=litho_colors[i], alpha=0.7, label=litho_names[i])
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_title("Lithology")
        ax.legend(loc="upper right", fontsize=6)

    axes[0].set_ylabel("Depth (m)")
    fig.suptitle(title or "Well Log Curves")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_curves(
    history: Dict[str, List[float]],
    metrics: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
):
    """
    Plot training curves (loss and metrics).

    Args:
        history: Dict of metric_name -> list of values per epoch
        metrics: Optional list of metric names to plot
        title: Plot title
        figsize: Figure size
        save_path: Optional save path
    """
    if metrics is None:
        metrics = [k for k in history.keys() if "loss" in k.lower()]

    n_metrics = len(metrics)
    if n_metrics == 0:
        print("No metrics to plot.")
        return

    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_metrics == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        if metric in history:
            ax = axes[i]
            epochs = range(1, len(history[metric]) + 1)
            ax.plot(epochs, history[metric], "b-", linewidth=1.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric)
            ax.set_title(metric)
            ax.grid(True, alpha=0.3)

    # Hide unused axes
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(title or "Training Curves")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
