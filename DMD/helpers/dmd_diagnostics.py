from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_parent_dir(path):
    if path is None:
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def compute_eigenvalue_diagnostics(
    eigenvalues,
    modal_amplitudes=None,
    dt=1.0,
    true_frequency_hz=None,
    fps=None,
    zero_frequency_tol=1e-12,
    nyquist_tol=1e-6,
):
    """Build a diagnostic table for DMD eigenvalues."""
    lambda_complex = np.asarray(eigenvalues, dtype=np.complex128)
    lambda_abs = np.abs(lambda_complex)
    lambda_angle_radians = np.angle(lambda_complex)

    omega = np.log(lambda_complex) / dt
    frequency_hz = omega.imag / (2.0 * np.pi)
    abs_frequency_hz = np.abs(frequency_hz)

    period_seconds = np.full_like(frequency_hz, fill_value=np.nan, dtype=np.float64)
    nonzero_frequency_mask = abs_frequency_hz > zero_frequency_tol
    period_seconds[nonzero_frequency_mask] = 1.0 / abs_frequency_hz[nonzero_frequency_mask]

    data = {
        "mode": np.arange(1, len(lambda_complex) + 1),
        "lambda_real": lambda_complex.real,
        "lambda_imag": lambda_complex.imag,
        "lambda_abs": lambda_abs,
        "lambda_angle_radians": lambda_angle_radians,
        "omega_real": omega.real,
        "omega_imag": omega.imag,
        "frequency_hz": frequency_hz,
        "abs_frequency_hz": abs_frequency_hz,
        "period_seconds": period_seconds,
        "distance_from_unit_circle": np.abs(lambda_abs - 1.0),
    }

    if modal_amplitudes is not None:
        data["amplitude_abs"] = np.abs(modal_amplitudes)

    if true_frequency_hz is not None:
        data["frequency_error_hz"] = np.abs(abs_frequency_hz - true_frequency_hz)

    if fps is not None:
        nyquist_frequency_hz = fps / 2.0
        data["nyquist_frequency_hz"] = np.full_like(frequency_hz, nyquist_frequency_hz)
        data["is_nyquist_like"] = np.isclose(
            abs_frequency_hz,
            nyquist_frequency_hz,
            atol=nyquist_tol,
        )

    diagnostics_df = pd.DataFrame(data)

    if "amplitude_abs" in diagnostics_df.columns:
        diagnostics_df = diagnostics_df.sort_values(
            ["amplitude_abs", "lambda_abs"],
            ascending=[False, False],
        ).reset_index(drop=True)

    return diagnostics_df


def find_closest_frequency_modes(diagnostics_df, true_frequency_hz, frequency_tol=1e-12):
    """Return rows whose absolute frequency is closest to the target frequency."""
    oscillatory_df = diagnostics_df.loc[
        diagnostics_df["abs_frequency_hz"] > frequency_tol
    ].copy()

    if oscillatory_df.empty:
        return oscillatory_df

    oscillatory_df["frequency_error_hz"] = np.abs(
        oscillatory_df["abs_frequency_hz"] - true_frequency_hz
    )
    best_error = oscillatory_df["frequency_error_hz"].min()

    return oscillatory_df.loc[
        np.isclose(oscillatory_df["frequency_error_hz"], best_error)
    ].sort_values("frequency_hz").reset_index(drop=True)


def plot_dmd_eigenvalues_complex_plane(
    eigenvalues,
    dt,
    true_frequency_hz=None,
    title="DMD Eigenvalues in the Complex Plane",
    annotate_modes=True,
    legend_loc="upper left",
    legend_bbox_to_anchor=(-0.35, 1.05),
    save_path=None,
    show=True,
):
    """Plot DMD eigenvalues in the complex plane with the unit circle."""
    lambda_complex = np.asarray(eigenvalues, dtype=np.complex128)

    fig, ax = plt.subplots(figsize=(6, 6))

    angle_grid = np.linspace(0.0, 2.0 * np.pi, 500)
    ax.plot(np.cos(angle_grid), np.sin(angle_grid), linestyle="--", label="unit circle")

    ax.axhline(0.0, linewidth=1)
    ax.axvline(0.0, linewidth=1)

    ax.scatter(lambda_complex.real, lambda_complex.imag, s=45, label="DMD eigenvalues")

    if annotate_modes:
        for mode_index, eigenvalue in enumerate(lambda_complex, start=1):
            ax.annotate(
                str(mode_index),
                (eigenvalue.real, eigenvalue.imag),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )

    if true_frequency_hz is not None:
        theta_true = 2.0 * np.pi * true_frequency_hz * dt
        ideal_positive = np.exp(1j * theta_true)
        ideal_negative = np.exp(-1j * theta_true)

        ax.scatter(
            [ideal_positive.real, ideal_negative.real],
            [ideal_positive.imag, ideal_negative.imag],
            marker="x",
            s=90,
            label="ideal frequency angles",
        )

        ax.plot(
            [0.0, ideal_positive.real],
            [0.0, ideal_positive.imag],
            linestyle=":",
            label=f"+ true angle ≈ {theta_true:.4f} rad/frame",
        )
        ax.plot(
            [0.0, ideal_negative.real],
            [0.0, ideal_negative.imag],
            linestyle=":",
            label=f"- true angle ≈ {-theta_true:.4f} rad/frame",
        )

    max_abs = max(1.05, float(np.max(np.abs(lambda_complex))) * 1.15)
    ax.set_xlim(-max_abs, max_abs)
    ax.set_ylim(-max_abs, max_abs)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("Real part")
    ax.set_ylabel("Imaginary part")
    ax.legend(
        loc=legend_loc,
        bbox_to_anchor=legend_bbox_to_anchor,
        borderaxespad=0.0,
    )
    fig.tight_layout()

    save_path = ensure_parent_dir(save_path)
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def plot_frequency_magnitude_summary(
    diagnostics_df,
    true_frequency_hz=None,
    fps=None,
    title="DMD Frequency and Eigenvalue Magnitude Summary",
    save_path=None,
    show=True,
):
    """Plot absolute frequency against eigenvalue magnitude with grouped labels."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.scatter(
        diagnostics_df["abs_frequency_hz"],
        diagnostics_df["lambda_abs"],
        s=45,
    )

    grouped_points = {}

    for _, row in diagnostics_df.iterrows():
        x_value = float(row["abs_frequency_hz"])
        y_value = float(row["lambda_abs"])

        # Conjugate pairs have the same absolute frequency and magnitude.
        # Rounding groups visually identical points such as modes 7/8.
        key = (round(x_value, 6), round(y_value, 6))

        if key not in grouped_points:
            grouped_points[key] = {
                "x": x_value,
                "y": y_value,
                "modes": [],
            }

        grouped_points[key]["modes"].append(int(row["mode"]))

    for point in grouped_points.values():
        label = "/".join(str(mode) for mode in sorted(point["modes"]))

        ax.annotate(
            label,
            (point["x"], point["y"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    if true_frequency_hz is not None:
        ax.axvline(
            true_frequency_hz,
            linestyle="--",
            label=f"true frequency = {true_frequency_hz:.3f} Hz",
        )

    if fps is not None:
        nyquist_frequency_hz = fps / 2.0
        ax.axvline(
            nyquist_frequency_hz,
            linestyle=":",
            label=f"Nyquist = {nyquist_frequency_hz:.3f} Hz",
        )

    ax.axhline(1.0, linestyle="--", label="unit magnitude")
    ax.set_title(title)
    ax.set_xlabel("Absolute frequency (Hz)")
    ax.set_ylabel("Eigenvalue magnitude")
    ax.legend(loc="best")
    fig.tight_layout()

    save_path = ensure_parent_dir(save_path)
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def plot_eigenvalue_angle_summary(
    diagnostics_df,
    true_frequency_hz=None,
    dt=None,
    title="DMD Eigenvalue Angles by Mode",
    save_path=None,
    show=True,
):
    """Plot discrete eigenvalue angle by mode."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.scatter(
        diagnostics_df["mode"],
        diagnostics_df["lambda_angle_radians"],
        s=45,
    )

    if true_frequency_hz is not None and dt is not None:
        theta_true = 2.0 * np.pi * true_frequency_hz * dt
        ax.axhline(theta_true, linestyle="--", label=f"+ expected angle = {theta_true:.4f}")
        ax.axhline(-theta_true, linestyle="--", label=f"- expected angle = {-theta_true:.4f}")

    ax.axhline(0.0, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Mode")
    ax.set_ylabel("Eigenvalue angle (radians/frame)")
    ax.legend(loc="best")
    fig.tight_layout()

    save_path = ensure_parent_dir(save_path)
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax

def _get_symmetric_vlim(values, percentile=99.0):
    finite_values = np.asarray(values)[np.isfinite(values)]

    if finite_values.size == 0:
        return 1.0

    limit = np.percentile(np.abs(finite_values), percentile)

    if limit <= 0:
        limit = np.max(np.abs(finite_values))

    if limit <= 0:
        limit = 1.0

    return limit

def _get_positive_vmax(values, percentile=99.0):
    finite_values = np.asarray(values)[np.isfinite(values)]

    if finite_values.size == 0:
        return 1.0

    vmax = np.percentile(finite_values, percentile)

    if vmax <= 0:
        vmax = np.max(finite_values)

    if vmax <= 0:
        vmax = 1.0

    return vmax

def _format_mode_title(mode_number, diagnostics_df=None):
    title_parts = [f"Mode {mode_number}"]

    if diagnostics_df is None:
        return "\n".join(title_parts)

    if "mode" not in diagnostics_df.columns:
        return "\n".join(title_parts)

    row = diagnostics_df.loc[diagnostics_df["mode"] == mode_number]

    if row.empty:
        return "\n".join(title_parts)

    row = row.iloc[0]

    if "frequency_hz" in row:
        title_parts.append(f"f = {row['frequency_hz']:.4f} Hz")

    if "period_seconds" in row and np.isfinite(row["period_seconds"]):
        title_parts.append(f"T = {row['period_seconds']:.3f} s")

    if "lambda_abs" in row:
        title_parts.append(f"|λ| = {row['lambda_abs']:.3f}")

    if "omega_real" in row:
        title_parts.append(f"Re(ω) = {row['omega_real']:.3f}")

    return "\n".join(title_parts)

def plot_dmd_mode_images(
    Phi,
    mode_numbers,
    frame_shape,
    diagnostics_df=None,
    save_path=None,
    percentile=99.0,
    figure_width=12,
    row_height=3.2,
    title="Selected DMD Mode Images",
    show=True,
):
    """Plot real, imaginary, and magnitude images for selected one-based DMD modes."""
    Phi = np.asarray(Phi)
    mode_numbers = list(mode_numbers)

    if len(mode_numbers) == 0:
        raise ValueError("mode_numbers must contain at least one mode number.")

    if np.prod(frame_shape) != Phi.shape[0]:
        raise ValueError(
            f"frame_shape {frame_shape} has {np.prod(frame_shape)} pixels, "
            f"but Phi has {Phi.shape[0]} rows."
        )

    for mode_number in mode_numbers:
        if mode_number < 1 or mode_number > Phi.shape[1]:
            raise ValueError(
                f"Mode number {mode_number} is out of range. "
                f"Expected values from 1 to {Phi.shape[1]}."
            )

    n_modes = len(mode_numbers)

    fig, axes = plt.subplots(
        n_modes,
        3,
        figsize=(figure_width, row_height * n_modes),
        squeeze=False,
    )

    for row_index, mode_number in enumerate(mode_numbers):
        mode_index = mode_number - 1
        mode_image = Phi[:, mode_index].reshape(frame_shape)

        real_image = np.real(mode_image)
        imaginary_image = np.imag(mode_image)
        magnitude_image = np.abs(mode_image)

        real_vlim = _get_symmetric_vlim(real_image, percentile=percentile)
        imaginary_vlim = _get_symmetric_vlim(imaginary_image, percentile=percentile)
        magnitude_vmax = _get_positive_vmax(magnitude_image, percentile=percentile)

        axes[row_index, 0].imshow(
            real_image,
            vmin=-real_vlim,
            vmax=real_vlim,
        )
        axes[row_index, 0].set_title("Real part")
        axes[row_index, 0].axis("off")

        axes[row_index, 1].imshow(
            imaginary_image,
            vmin=-imaginary_vlim,
            vmax=imaginary_vlim,
        )
        axes[row_index, 1].set_title("Imaginary part")
        axes[row_index, 1].axis("off")

        axes[row_index, 2].imshow(
            magnitude_image,
            vmin=0,
            vmax=magnitude_vmax,
        )
        axes[row_index, 2].set_title("Magnitude")
        axes[row_index, 2].axis("off")

        row_title = _format_mode_title(
            mode_number=mode_number,
            diagnostics_df=diagnostics_df,
        )

        axes[row_index, 0].set_ylabel(
            row_title,
            rotation=0,
            labelpad=72,
            va="center",
        )

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    save_path = ensure_parent_dir(save_path)
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes