# -*- coding: utf-8 -*-
"""
Visualization helpers for the DMD pendulum notebook.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def load_rgb_frame(frame_path):
    with Image.open(frame_path) as image:
        return np.array(image.convert("RGB"))

def load_grayscale_frame(frame_path):
    with Image.open(frame_path) as image:
        return np.array(image.convert("L"))

def save_and_show_figure(fig, save_path=None, show=True, dpi=160):
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

def show_sample_frames(frame_paths, frame_indices, figsize=(12, 4), save_path=None, show=True, dpi=160):
    fig, axes = plt.subplots(1, len(frame_indices), figsize=figsize)
    axes = np.atleast_1d(axes)

    for ax, idx in zip(axes, frame_indices):
        frame = load_rgb_frame(frame_paths[idx])
        ax.imshow(frame)
        ax.set_title(f"Frame {idx}")
        ax.axis("off")

    save_and_show_figure(fig, save_path=save_path, show=show, dpi=dpi)

    return fig, axes

def plot_ground_truth_signals(ground_truth_df, fit_end_index=None, figsize=(10, 8), save_path=None, show=True, dpi=160):
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    t = ground_truth_df["time_seconds"].to_numpy()

    axes[0].plot(t, ground_truth_df["theta_degrees"])
    axes[0].set_ylabel("Angle (deg)")
    axes[0].set_title("Pendulum angle vs time")
    axes[0].grid(True)

    axes[1].plot(t, ground_truth_df["bob_x"], label="bob_x")
    axes[1].plot(t, ground_truth_df["bob_y"], label="bob_y")
    axes[1].set_ylabel("Pixels")
    axes[1].set_title("Bob coordinates vs time")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(t, ground_truth_df["theta_dot_radians_per_second"])
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Angular velocity")
    axes[2].set_title("Angular velocity vs time")
    axes[2].grid(True)

    if fit_end_index is not None:
        split_time = ground_truth_df.loc[fit_end_index, "time_seconds"]
        for ax in axes:
            ax.axvline(split_time, linestyle="--", label="fit/forecast split")
        axes[0].legend()

    save_and_show_figure(fig, save_path=save_path, show=show, dpi=dpi)

    return fig, axes

def plot_bob_path(
    ground_truth_df,
    figsize=(7, 4),
    show_frame_order=False,
    annotation_count=10,
    annotation_mode="frame",
    save_path=None,
    show=True,
    dpi=160,
):
    x = ground_truth_df["bob_x"].to_numpy()
    y = ground_truth_df["bob_y"].to_numpy()

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x, y, marker="o", markersize=2)
    ax.scatter([x[0]], [y[0]], label="start")
    ax.scatter([x[-1]], [y[-1]], label="end")
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_title("Bob path in image coordinates")
    ax.grid(True)
    ax.legend()

    if show_frame_order:
        annotate_bob_path(
            ax=ax,
            ground_truth_df=ground_truth_df,
            annotation_count=annotation_count,
            annotation_mode=annotation_mode,
        )

    save_and_show_figure(fig, save_path=save_path, show=show, dpi=dpi)

    return fig, ax

def annotate_bob_path(ax, ground_truth_df, annotation_count=10, annotation_mode="frame"):
    if annotation_mode not in {"frame", "time"}:
        raise ValueError("annotation_mode must be either 'frame' or 'time'.")

    x = ground_truth_df["bob_x"].to_numpy()
    y = ground_truth_df["bob_y"].to_numpy()

    step = max(1, len(ground_truth_df) // annotation_count)
    annotation_indices = list(range(0, len(ground_truth_df), step))

    if annotation_indices[-1] != len(ground_truth_df) - 1:
        annotation_indices.append(len(ground_truth_df) - 1)

    for idx in annotation_indices:
        if annotation_mode == "frame":
            label = f"frame {int(ground_truth_df.loc[idx, 'frame_index'])}"
        else:
            label = f"{ground_truth_df.loc[idx, 'time_seconds']:.2f}s"

        ax.annotate(
            label,
            xy=(x[idx], y[idx]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75},
            arrowprops={"arrowstyle": "-", "linewidth": 0.6},
        )