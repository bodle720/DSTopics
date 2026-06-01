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