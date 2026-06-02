from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def ensure_parent_dir(path):
    if path is None:
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def reconstruct_dmd_states(Phi, eigenvalues, modal_amplitudes, time_indices):
    """Reconstruct mean-centered states from a fixed DMD model."""
    Phi = np.asarray(Phi, dtype=np.complex128)
    eigenvalues = np.asarray(eigenvalues, dtype=np.complex128)
    modal_amplitudes = np.asarray(modal_amplitudes, dtype=np.complex128)
    time_indices = np.asarray(time_indices, dtype=np.int64)

    if Phi.ndim != 2:
        raise ValueError("Phi must be a 2D array.")

    if eigenvalues.ndim != 1:
        raise ValueError("eigenvalues must be a 1D array.")

    if modal_amplitudes.ndim != 1:
        raise ValueError("modal_amplitudes must be a 1D array.")

    if Phi.shape[1] != eigenvalues.shape[0]:
        raise ValueError("Phi column count must match number of eigenvalues.")

    if eigenvalues.shape[0] != modal_amplitudes.shape[0]:
        raise ValueError("eigenvalues and modal_amplitudes must have the same length.")

    time_dynamics = modal_amplitudes[:, None] * (
        eigenvalues[:, None] ** time_indices[None, :]
    )

    reconstructed_complex = Phi @ time_dynamics
    max_imaginary_residual = float(np.max(np.abs(reconstructed_complex.imag)))
    reconstructed_real = reconstructed_complex.real

    return reconstructed_real, max_imaginary_residual


def add_mean_frame(centered_states, mean_frame_vector):
    """Add the fitting-window mean frame back to centered reconstructions."""
    centered_states = np.asarray(centered_states, dtype=np.float64)
    mean_frame_vector = np.asarray(mean_frame_vector, dtype=np.float64)

    if mean_frame_vector.ndim == 1:
        mean_frame_vector = mean_frame_vector[:, None]

    if mean_frame_vector.ndim != 2 or mean_frame_vector.shape[1] != 1:
        raise ValueError("mean_frame_vector must have shape (n,) or (n, 1).")

    if centered_states.shape[0] != mean_frame_vector.shape[0]:
        raise ValueError("centered_states and mean_frame_vector must have the same row count.")

    return centered_states + mean_frame_vector


def compute_frame_error_summary(true_states, predicted_states, frame_offset=0):
    """Compute per-frame and overall reconstruction error summaries."""
    true_states = np.asarray(true_states, dtype=np.float64)
    predicted_states = np.asarray(predicted_states, dtype=np.float64)

    if true_states.shape != predicted_states.shape:
        raise ValueError("true_states and predicted_states must have the same shape.")

    error = predicted_states - true_states
    abs_error = np.abs(error)

    mse_per_frame = np.mean(error**2, axis=0)
    mae_per_frame = np.mean(abs_error, axis=0)
    max_abs_error_per_frame = np.max(abs_error, axis=0)

    frame_count = true_states.shape[1]
    frame_numbers = np.arange(frame_offset, frame_offset + frame_count)
    relative_frame_numbers = np.arange(frame_count)

    error_df = pd.DataFrame(
        {
            "relative_frame": relative_frame_numbers,
            "frame_number": frame_numbers,
            "mse": mse_per_frame,
            "mae": mae_per_frame,
            "max_abs_error": max_abs_error_per_frame,
        }
    )

    summary = {
        "overall_mse": float(np.mean(error**2)),
        "overall_mae": float(np.mean(abs_error)),
        "overall_rmse": float(np.sqrt(np.mean(error**2))),
        "max_abs_error": float(np.max(abs_error)),
        "mean_mse_per_frame": float(np.mean(mse_per_frame)),
        "mean_mae_per_frame": float(np.mean(mae_per_frame)),
    }

    return error_df, summary


def plot_frame_error_over_time(
    error_df,
    title="Reconstruction Error Over Time",
    save_path=None,
    show=True,
):
    """Plot per-frame MSE and MAE."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(error_df["relative_frame"], error_df["mse"], marker="o", markersize=3)
    axes[0].set_ylabel("MSE")
    axes[0].set_title(title)

    axes[1].plot(error_df["relative_frame"], error_df["mae"], marker="o", markersize=3)
    axes[1].set_xlabel("Frame index within window")
    axes[1].set_ylabel("MAE")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    save_path = ensure_parent_dir(save_path)
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes


def plot_reconstruction_examples(
    true_states,
    predicted_states,
    frame_shape,
    frame_numbers,
    title="Reconstruction Examples",
    save_path=None,
    show=True,
):
    """Plot true frames, reconstructed frames, and absolute error for selected frames."""
    true_states = np.asarray(true_states, dtype=np.float64)
    predicted_states = np.asarray(predicted_states, dtype=np.float64)

    if true_states.shape != predicted_states.shape:
        raise ValueError("true_states and predicted_states must have the same shape.")

    frame_numbers = list(frame_numbers)

    if len(frame_numbers) == 0:
        raise ValueError("frame_numbers must contain at least one frame index.")

    abs_error = np.abs(predicted_states - true_states)
    error_vmax = max(float(np.percentile(abs_error, 99.0)), 1e-8)

    fig, axes = plt.subplots(3, len(frame_numbers), figsize=(3 * len(frame_numbers), 8))

    if len(frame_numbers) == 1:
        axes = np.asarray(axes).reshape(3, 1)

    for col_index, frame_idx in enumerate(frame_numbers):
        true_frame = true_states[:, frame_idx].reshape(frame_shape)
        predicted_frame = predicted_states[:, frame_idx].reshape(frame_shape)
        error_frame = np.abs(predicted_frame - true_frame)

        axes[0, col_index].imshow(true_frame, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, col_index].set_title(f"Frame {frame_idx}")
        axes[0, col_index].axis("off")

        axes[1, col_index].imshow(predicted_frame, cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, col_index].axis("off")

        axes[2, col_index].imshow(error_frame, cmap="magma", vmin=0.0, vmax=error_vmax)
        axes[2, col_index].axis("off")

    axes[0, 0].set_ylabel("True", fontsize=11)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=11)
    axes[2, 0].set_ylabel("Absolute error", fontsize=11)

    fig.suptitle(title)
    fig.tight_layout()

    save_path = ensure_parent_dir(save_path)
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes


def _frame_to_uint8(frame_2d):
    frame_2d = np.asarray(frame_2d, dtype=np.float64)
    frame_2d = np.clip(frame_2d, 0.0, 1.0)
    return (255.0 * frame_2d).round().astype(np.uint8)


def _build_side_by_side_gif_frame(
    true_frame,
    predicted_frame,
    frame_label,
    label_left="True",
    label_right="Prediction",
    gap=8,
    header_height=28,
):
    true_uint8 = _frame_to_uint8(true_frame)
    predicted_uint8 = _frame_to_uint8(predicted_frame)

    h, w = true_uint8.shape
    canvas_width = w * 2 + gap
    canvas_height = h + header_height

    canvas = Image.new("L", (canvas_width, canvas_height), color=255)
    canvas.paste(Image.fromarray(true_uint8, mode="L"), (0, header_height))
    canvas.paste(Image.fromarray(predicted_uint8, mode="L"), (w + gap, header_height))

    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), label_left, fill=0)
    draw.text((w + gap + 4, 4), label_right, fill=0)
    draw.text((canvas_width - 90, 4), frame_label, fill=0)

    return canvas


def save_side_by_side_gif(
    true_states,
    predicted_states,
    frame_shape,
    save_path,
    frame_numbers=None,
    fps=4,
    label_left="True",
    label_right="Prediction",
):
    """Save a grayscale side-by-side GIF comparing true and predicted frames."""
    true_states = np.asarray(true_states, dtype=np.float64)
    predicted_states = np.asarray(predicted_states, dtype=np.float64)

    if true_states.shape != predicted_states.shape:
        raise ValueError("true_states and predicted_states must have the same shape.")

    frame_count = true_states.shape[1]

    if frame_numbers is None:
        frame_numbers = np.arange(frame_count)
    else:
        frame_numbers = np.asarray(frame_numbers, dtype=np.int64)

    gif_frames = []

    for frame_idx in frame_numbers:
        true_frame = true_states[:, frame_idx].reshape(frame_shape)
        predicted_frame = predicted_states[:, frame_idx].reshape(frame_shape)

        gif_frame = _build_side_by_side_gif_frame(
            true_frame=true_frame,
            predicted_frame=predicted_frame,
            frame_label=f"frame {frame_idx}",
            label_left=label_left,
            label_right=label_right,
        )
        gif_frames.append(gif_frame)

    if len(gif_frames) == 0:
        raise ValueError("No frames were selected for the GIF.")

    save_path = ensure_parent_dir(save_path)
    duration_ms = int(round(1000 / fps))

    gif_frames[0].save(
        save_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=duration_ms,
        loop=0,
    )

    return save_path