from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


@dataclass
class CoordinateDMDResult:
    delay_count: int
    rank: int
    dt: float
    coordinate_center: np.ndarray
    coordinate_scale: np.ndarray
    fit_coordinates: np.ndarray
    fit_coordinates_normalized: np.ndarray
    delay_states_fit: np.ndarray
    delay_state_frame_indices_fit: np.ndarray
    Z: np.ndarray
    Z_prime: np.ndarray
    U_r: np.ndarray
    singular_values: np.ndarray
    V_r: np.ndarray
    A_tilde: np.ndarray
    eigenvalues: np.ndarray
    Phi: np.ndarray
    W: np.ndarray


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------

def ensure_parent_dir(path):
    if path is None:
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------
# Coordinate extraction, normalization, and delay-state construction
# ---------------------------------------------------------------------

def extract_bob_coordinates(ground_truth_df, x_col=None, y_col=None):
    """Extract bob x/y coordinates from a ground-truth dataframe."""
    if x_col is not None and y_col is not None:
        if x_col not in ground_truth_df.columns:
            raise ValueError(f"x_col '{x_col}' was not found in the dataframe.")
        if y_col not in ground_truth_df.columns:
            raise ValueError(f"y_col '{y_col}' was not found in the dataframe.")

        coordinates = ground_truth_df[[x_col, y_col]].to_numpy(dtype=np.float64)
        return coordinates, (x_col, y_col)

    candidate_pairs = [
        ("bob_x", "bob_y"),
        ("bob_center_x", "bob_center_y"),
        ("bob_center_x_px", "bob_center_y_px"),
        ("bob_x_px", "bob_y_px"),
        ("x_bob", "y_bob"),
        ("bob_center_col", "bob_center_row"),
    ]

    for candidate_x_col, candidate_y_col in candidate_pairs:
        if (
            candidate_x_col in ground_truth_df.columns
            and candidate_y_col in ground_truth_df.columns
        ):
            coordinates = ground_truth_df[
                [candidate_x_col, candidate_y_col]
            ].to_numpy(dtype=np.float64)
            return coordinates, (candidate_x_col, candidate_y_col)

    raise ValueError(
        "Could not infer bob coordinate columns. "
        f"Available columns: {list(ground_truth_df.columns)}"
    )


def normalize_coordinates(coordinates, coordinate_center=None, coordinate_scale=None):
    """Center and scale coordinates before fitting DMD."""
    coordinates = np.asarray(coordinates, dtype=np.float64)

    if coordinates.ndim != 2:
        raise ValueError("coordinates must have shape (n_frames, n_coordinates).")

    if coordinate_center is None:
        coordinate_center = coordinates.mean(axis=0)
    else:
        coordinate_center = np.asarray(coordinate_center, dtype=np.float64)

    if coordinate_scale is None:
        coordinate_scale = np.ones(coordinates.shape[1], dtype=np.float64)
    else:
        coordinate_scale = np.asarray(coordinate_scale, dtype=np.float64)

    coordinate_scale = coordinate_scale.copy()
    coordinate_scale[np.abs(coordinate_scale) < 1e-12] = 1.0

    normalized_coordinates = (coordinates - coordinate_center) / coordinate_scale

    return normalized_coordinates, coordinate_center, coordinate_scale


def denormalize_coordinates(normalized_coordinates, coordinate_center, coordinate_scale):
    """Convert normalized coordinates back to original coordinate units."""
    normalized_coordinates = np.asarray(normalized_coordinates, dtype=np.float64)
    coordinate_center = np.asarray(coordinate_center, dtype=np.float64)
    coordinate_scale = np.asarray(coordinate_scale, dtype=np.float64)

    return normalized_coordinates * coordinate_scale + coordinate_center


def build_delay_coordinate_states(coordinates, delay_count):
    """Build delay-coordinate states from a coordinate time series."""
    coordinates = np.asarray(coordinates, dtype=np.float64)

    if coordinates.ndim != 2:
        raise ValueError("coordinates must have shape (n_frames, n_coordinates).")

    if delay_count < 1:
        raise ValueError("delay_count must be at least 1.")

    frame_count, coordinate_dim = coordinates.shape

    if frame_count < delay_count + 1:
        raise ValueError(
            "Not enough frames for the requested delay_count. "
            "Need at least delay_count + 1 frames to fit DMD."
        )

    state_count = frame_count - delay_count + 1
    state_dim = coordinate_dim * delay_count

    delay_states = np.empty((state_dim, state_count), dtype=np.float64)
    state_frame_indices = np.arange(delay_count - 1, frame_count)

    for state_col, frame_index in enumerate(state_frame_indices):
        delayed_blocks = [
            coordinates[frame_index - delay_index]
            for delay_index in range(delay_count)
        ]
        delay_states[:, state_col] = np.concatenate(delayed_blocks)

    return delay_states, state_frame_indices


# ---------------------------------------------------------------------
# Core delay-coordinate DMD fitting and prediction
# ---------------------------------------------------------------------

def fit_delay_coordinate_dmd(
    fit_coordinates,
    delay_count,
    rank=None,
    dt=1.0,
    coordinate_center=None,
    coordinate_scale=None,
    numerical_rank_tol=1e-12,
):
    """Fit exact DMD to delay-embedded bob coordinates."""
    fit_coordinates = np.asarray(fit_coordinates, dtype=np.float64)

    fit_coordinates_normalized, coordinate_center, coordinate_scale = normalize_coordinates(
        coordinates=fit_coordinates,
        coordinate_center=coordinate_center,
        coordinate_scale=coordinate_scale,
    )

    delay_states_fit, delay_state_frame_indices_fit = build_delay_coordinate_states(
        coordinates=fit_coordinates_normalized,
        delay_count=delay_count,
    )

    Z = delay_states_fit[:, :-1]
    Z_prime = delay_states_fit[:, 1:]

    U_full, singular_values, Vh_full = np.linalg.svd(Z, full_matrices=False)

    if len(singular_values) == 0:
        raise ValueError("DMD fitting matrix has no singular values.")

    numerical_rank = int(np.sum(singular_values > singular_values[0] * numerical_rank_tol))

    if numerical_rank < 1:
        raise ValueError("Estimated numerical rank is zero.")

    if rank is None:
        rank = numerical_rank

    rank = int(rank)

    if rank < 1:
        raise ValueError("rank must be at least 1.")

    if rank > numerical_rank:
        raise ValueError(
            f"Requested rank {rank}, but estimated numerical rank is {numerical_rank}."
        )

    U_r = U_full[:, :rank]
    singular_values_r = singular_values[:rank]
    Sigma_r_inv = np.diag(1.0 / singular_values_r)
    V_r = Vh_full[:rank, :].conj().T

    A_tilde = U_r.conj().T @ Z_prime @ V_r @ Sigma_r_inv
    eigenvalues, W = np.linalg.eig(A_tilde)

    Phi = Z_prime @ V_r @ Sigma_r_inv @ W

    return CoordinateDMDResult(
        delay_count=delay_count,
        rank=rank,
        dt=dt,
        coordinate_center=coordinate_center,
        coordinate_scale=coordinate_scale,
        fit_coordinates=fit_coordinates,
        fit_coordinates_normalized=fit_coordinates_normalized,
        delay_states_fit=delay_states_fit,
        delay_state_frame_indices_fit=delay_state_frame_indices_fit,
        Z=Z,
        Z_prime=Z_prime,
        U_r=U_r,
        singular_values=singular_values,
        V_r=V_r,
        A_tilde=A_tilde,
        eigenvalues=eigenvalues,
        Phi=Phi,
        W=W,
    )


def predict_delay_states(Phi, eigenvalues, start_state, time_indices):
    """Predict delay states from a fixed DMD model and starting state."""
    Phi = np.asarray(Phi, dtype=np.complex128)
    eigenvalues = np.asarray(eigenvalues, dtype=np.complex128)
    start_state = np.asarray(start_state, dtype=np.complex128)
    time_indices = np.asarray(time_indices, dtype=np.int64)

    if Phi.ndim != 2:
        raise ValueError("Phi must be a 2D array.")

    if eigenvalues.ndim != 1:
        raise ValueError("eigenvalues must be a 1D array.")

    if start_state.ndim != 1:
        raise ValueError("start_state must be a 1D array.")

    if Phi.shape[0] != start_state.shape[0]:
        raise ValueError("Phi row count must match start_state length.")

    if Phi.shape[1] != eigenvalues.shape[0]:
        raise ValueError("Phi column count must match number of eigenvalues.")

    modal_amplitudes = np.linalg.lstsq(Phi, start_state, rcond=None)[0]

    time_dynamics = modal_amplitudes[:, None] * (
        eigenvalues[:, None] ** time_indices[None, :]
    )

    predicted_complex = Phi @ time_dynamics
    max_imaginary_residual = float(np.max(np.abs(predicted_complex.imag)))
    predicted_real = predicted_complex.real

    return predicted_real, max_imaginary_residual, modal_amplitudes


def extract_current_coordinates_from_delay_states(delay_states, coordinate_dim=2):
    """Extract the current coordinate block from delay-coordinate states."""
    delay_states = np.asarray(delay_states, dtype=np.float64)

    if delay_states.ndim != 2:
        raise ValueError("delay_states must have shape (state_dim, state_count).")

    if coordinate_dim < 1:
        raise ValueError("coordinate_dim must be at least 1.")

    return delay_states[:coordinate_dim, :].T


# ---------------------------------------------------------------------
# Reconstruction, forecasting, and experiment wrappers
# ---------------------------------------------------------------------

def reconstruct_fit_coordinates(dmd_result):
    """Reconstruct fitting-window coordinates from the first delay state."""
    state_count = dmd_result.delay_states_fit.shape[1]
    time_indices = np.arange(state_count)

    predicted_delay_states, max_imaginary_residual, modal_amplitudes = predict_delay_states(
        Phi=dmd_result.Phi,
        eigenvalues=dmd_result.eigenvalues,
        start_state=dmd_result.delay_states_fit[:, 0],
        time_indices=time_indices,
    )

    predicted_coordinates_normalized = extract_current_coordinates_from_delay_states(
        predicted_delay_states,
        coordinate_dim=dmd_result.fit_coordinates.shape[1],
    )

    predicted_coordinates = denormalize_coordinates(
        normalized_coordinates=predicted_coordinates_normalized,
        coordinate_center=dmd_result.coordinate_center,
        coordinate_scale=dmd_result.coordinate_scale,
    )

    true_coordinates = dmd_result.fit_coordinates[dmd_result.delay_state_frame_indices_fit]

    return {
        "true_coordinates": true_coordinates,
        "predicted_coordinates": predicted_coordinates,
        "frame_indices": dmd_result.delay_state_frame_indices_fit,
        "max_imaginary_residual": max_imaginary_residual,
        "modal_amplitudes": modal_amplitudes,
        "predicted_delay_states": predicted_delay_states,
    }


def forecast_coordinates_from_last_state(dmd_result, forecast_horizon):
    """Forecast future coordinates from the final observed fitting-window delay state."""
    forecast_horizon = int(forecast_horizon)

    if forecast_horizon < 1:
        raise ValueError("forecast_horizon must be at least 1.")

    time_indices = np.arange(1, forecast_horizon + 1)

    predicted_delay_states, max_imaginary_residual, modal_amplitudes = predict_delay_states(
        Phi=dmd_result.Phi,
        eigenvalues=dmd_result.eigenvalues,
        start_state=dmd_result.delay_states_fit[:, -1],
        time_indices=time_indices,
    )

    predicted_coordinates_normalized = extract_current_coordinates_from_delay_states(
        predicted_delay_states,
        coordinate_dim=dmd_result.fit_coordinates.shape[1],
    )

    predicted_coordinates = denormalize_coordinates(
        normalized_coordinates=predicted_coordinates_normalized,
        coordinate_center=dmd_result.coordinate_center,
        coordinate_scale=dmd_result.coordinate_scale,
    )

    frame_indices = np.arange(
        len(dmd_result.fit_coordinates),
        len(dmd_result.fit_coordinates) + forecast_horizon,
    )

    return {
        "predicted_coordinates": predicted_coordinates,
        "frame_indices": frame_indices,
        "max_imaginary_residual": max_imaginary_residual,
        "modal_amplitudes": modal_amplitudes,
        "predicted_delay_states": predicted_delay_states,
    }


def build_coordinate_prediction_df(true_coordinates, predicted_coordinates, frame_indices, window_name):
    """Build a dataframe with true, predicted, and error coordinates."""
    true_coordinates = np.asarray(true_coordinates, dtype=np.float64)
    predicted_coordinates = np.asarray(predicted_coordinates, dtype=np.float64)
    frame_indices = np.asarray(frame_indices, dtype=np.int64)

    if true_coordinates.shape != predicted_coordinates.shape:
        raise ValueError("true_coordinates and predicted_coordinates must have the same shape.")

    if true_coordinates.shape[0] != frame_indices.shape[0]:
        raise ValueError("frame_indices length must match coordinate row count.")

    error = predicted_coordinates - true_coordinates
    euclidean_error = np.sqrt(np.sum(error**2, axis=1))

    return pd.DataFrame(
        {
            "window": window_name,
            "frame": frame_indices,
            "x_true": true_coordinates[:, 0],
            "y_true": true_coordinates[:, 1],
            "x_predicted": predicted_coordinates[:, 0],
            "y_predicted": predicted_coordinates[:, 1],
            "x_error": error[:, 0],
            "y_error": error[:, 1],
            "euclidean_error": euclidean_error,
            "squared_error": euclidean_error**2,
        }
    )


def summarize_coordinate_errors(prediction_df, group_col="window"):
    """Summarize coordinate prediction errors."""
    grouped = prediction_df.groupby(group_col, dropna=False)

    rows = []

    for group_name, group_df in grouped:
        rows.append(
            {
                group_col: group_name,
                "frame_count": len(group_df),
                "mean_euclidean_error": float(group_df["euclidean_error"].mean()),
                "median_euclidean_error": float(group_df["euclidean_error"].median()),
                "rmse_euclidean": float(np.sqrt(group_df["squared_error"].mean())),
                "max_euclidean_error": float(group_df["euclidean_error"].max()),
                "mean_abs_x_error": float(group_df["x_error"].abs().mean()),
                "mean_abs_y_error": float(group_df["y_error"].abs().mean()),
            }
        )

    return pd.DataFrame(rows)


def run_coordinate_dmd_experiment(
    fit_coordinates,
    forecast_coordinates,
    delay_count,
    rank=None,
    dt=1.0,
    fit_start_frame=0,
    forecast_start_frame=None,
):
    """Fit coordinate DMD, reconstruct fit coordinates, forecast held-out coordinates, and return results."""
    fit_coordinates = np.asarray(fit_coordinates, dtype=np.float64)
    forecast_coordinates = np.asarray(forecast_coordinates, dtype=np.float64)

    if forecast_start_frame is None:
        forecast_start_frame = fit_start_frame + len(fit_coordinates)

    dmd_result = fit_delay_coordinate_dmd(
        fit_coordinates=fit_coordinates,
        delay_count=delay_count,
        rank=rank,
        dt=dt,
    )

    reconstruction = reconstruct_fit_coordinates(dmd_result)

    forecast = forecast_coordinates_from_last_state(
        dmd_result=dmd_result,
        forecast_horizon=len(forecast_coordinates),
    )

    fit_frame_indices = fit_start_frame + reconstruction["frame_indices"]
    forecast_frame_indices = forecast_start_frame + np.arange(len(forecast_coordinates))

    fit_prediction_df = build_coordinate_prediction_df(
        true_coordinates=reconstruction["true_coordinates"],
        predicted_coordinates=reconstruction["predicted_coordinates"],
        frame_indices=fit_frame_indices,
        window_name="fit",
    )

    forecast_prediction_df = build_coordinate_prediction_df(
        true_coordinates=forecast_coordinates,
        predicted_coordinates=forecast["predicted_coordinates"],
        frame_indices=forecast_frame_indices,
        window_name="forecast",
    )

    prediction_df = pd.concat(
        [fit_prediction_df, forecast_prediction_df],
        ignore_index=True,
    )

    error_summary_df = summarize_coordinate_errors(prediction_df)

    return {
        "dmd_result": dmd_result,
        "reconstruction": reconstruction,
        "forecast": forecast,
        "fit_prediction_df": fit_prediction_df,
        "forecast_prediction_df": forecast_prediction_df,
        "prediction_df": prediction_df,
        "error_summary_df": error_summary_df,
    }


# ---------------------------------------------------------------------
# Eigenvalue summaries, singular values, and rank diagnostics
# ---------------------------------------------------------------------

def compute_coordinate_eigenvalue_summary(dmd_result, true_frequency_hz=None):
    """Build a diagnostic table for coordinate-DMD eigenvalues."""
    eigenvalues = np.asarray(dmd_result.eigenvalues, dtype=np.complex128)
    lambda_abs = np.abs(eigenvalues)
    lambda_angle_radians = np.angle(eigenvalues)

    omega = np.log(eigenvalues) / dmd_result.dt
    frequency_hz = omega.imag / (2.0 * np.pi)
    abs_frequency_hz = np.abs(frequency_hz)

    period_seconds = np.full_like(frequency_hz, fill_value=np.nan, dtype=np.float64)
    nonzero_frequency_mask = abs_frequency_hz > 1e-12
    period_seconds[nonzero_frequency_mask] = 1.0 / abs_frequency_hz[nonzero_frequency_mask]

    data = {
        "mode": np.arange(1, len(eigenvalues) + 1),
        "lambda_real": eigenvalues.real,
        "lambda_imag": eigenvalues.imag,
        "lambda_abs": lambda_abs,
        "lambda_angle_radians": lambda_angle_radians,
        "omega_real": omega.real,
        "omega_imag": omega.imag,
        "frequency_hz": frequency_hz,
        "abs_frequency_hz": abs_frequency_hz,
        "period_seconds": period_seconds,
        "distance_from_unit_circle": np.abs(lambda_abs - 1.0),
    }

    if true_frequency_hz is not None:
        data["frequency_error_hz"] = np.abs(abs_frequency_hz - true_frequency_hz)

    return pd.DataFrame(data)


def compute_delay_state_svd_summary(delay_states_fit):
    """Compute singular-value diagnostics for delay-coordinate states."""
    Z = delay_states_fit[:, :-1]
    _, singular_values, _ = np.linalg.svd(Z, full_matrices=False)

    singular_energy = singular_values**2
    total_energy = singular_energy.sum()

    if total_energy <= 0:
        energy_fraction = np.zeros_like(singular_values)
    else:
        energy_fraction = singular_energy / total_energy

    cumulative_energy_fraction = np.cumsum(energy_fraction)
    rank_indices = np.arange(1, len(singular_values) + 1)

    return pd.DataFrame(
        {
            "rank": rank_indices,
            "singular_value": singular_values,
            "energy_fraction": energy_fraction,
            "cumulative_energy_fraction": cumulative_energy_fraction,
        }
    )


def sweep_coordinate_dmd_ranks(
    fit_coordinates,
    forecast_coordinates,
    delay_count,
    rank_values,
    dt=1.0,
    true_frequency_hz=None,
    fit_start_frame=0,
    forecast_start_frame=None,
):
    """Evaluate coordinate-DMD reconstruction and forecast quality for candidate ranks."""
    rows = []

    for rank in rank_values:
        outputs = run_coordinate_dmd_experiment(
            fit_coordinates=fit_coordinates,
            forecast_coordinates=forecast_coordinates,
            delay_count=delay_count,
            rank=rank,
            dt=dt,
            fit_start_frame=fit_start_frame,
            forecast_start_frame=forecast_start_frame,
        )

        dmd_result = outputs["dmd_result"]
        error_summary_df = outputs["error_summary_df"]
        eigenvalue_summary_df = compute_coordinate_eigenvalue_summary(
            dmd_result,
            true_frequency_hz=true_frequency_hz,
        )

        fit_summary = error_summary_df.loc[
            error_summary_df["window"] == "fit"
        ].iloc[0]

        forecast_summary = error_summary_df.loc[
            error_summary_df["window"] == "forecast"
        ].iloc[0]

        oscillatory_df = eigenvalue_summary_df.loc[
            eigenvalue_summary_df["abs_frequency_hz"] > 1e-12
        ].copy()

        if true_frequency_hz is not None and not oscillatory_df.empty:
            closest_frequency_row = oscillatory_df.loc[
                oscillatory_df["frequency_error_hz"].idxmin()
            ]
            closest_abs_frequency_hz = float(closest_frequency_row["abs_frequency_hz"])
            closest_frequency_error_hz = float(closest_frequency_row["frequency_error_hz"])
            closest_lambda_abs = float(closest_frequency_row["lambda_abs"])
        else:
            closest_abs_frequency_hz = np.nan
            closest_frequency_error_hz = np.nan
            closest_lambda_abs = np.nan

        rows.append(
            {
                "rank": rank,
                "fit_mean_euclidean_error": float(fit_summary["mean_euclidean_error"]),
                "fit_rmse_euclidean": float(fit_summary["rmse_euclidean"]),
                "forecast_mean_euclidean_error": float(
                    forecast_summary["mean_euclidean_error"]
                ),
                "forecast_rmse_euclidean": float(forecast_summary["rmse_euclidean"]),
                "forecast_max_euclidean_error": float(
                    forecast_summary["max_euclidean_error"]
                ),
                "closest_abs_frequency_hz": closest_abs_frequency_hz,
                "closest_frequency_error_hz": closest_frequency_error_hz,
                "closest_lambda_abs": closest_lambda_abs,
                "max_lambda_abs": float(eigenvalue_summary_df["lambda_abs"].max()),
                "mean_lambda_abs": float(eigenvalue_summary_df["lambda_abs"].mean()),
            }
        )

    return pd.DataFrame(rows)


def build_coordinate_dmd_report_lines(dmd_result):
    """Build concise printable report lines for a fitted coordinate-DMD result."""
    return [
        f"Selected coordinate DMD rank: {dmd_result.rank}",
        f"Delay count q: {dmd_result.delay_count}",
        f"Coordinate state dimension: {dmd_result.delay_states_fit.shape[0]}",
        f"Fit delay states shape: {dmd_result.delay_states_fit.shape}",
        f"Z shape: {dmd_result.Z.shape}",
        f"Z_prime shape: {dmd_result.Z_prime.shape}",
    ]


def run_selected_coordinate_dmd_rank(
    fit_coordinates,
    forecast_coordinates,
    delay_count,
    rank,
    dt=1.0,
    fit_start_frame=0,
    forecast_start_frame=None,
    true_frequency_hz=None,
    coordinate_rank_sweep_df=None,
):
    """Run a selected coordinate-DMD rank and collect summary tables for notebook display."""
    outputs = run_coordinate_dmd_experiment(
        fit_coordinates=fit_coordinates,
        forecast_coordinates=forecast_coordinates,
        delay_count=delay_count,
        rank=rank,
        dt=dt,
        fit_start_frame=fit_start_frame,
        forecast_start_frame=forecast_start_frame,
    )

    dmd_result = outputs["dmd_result"]

    eigenvalue_summary_df = compute_coordinate_eigenvalue_summary(
        dmd_result,
        true_frequency_hz=true_frequency_hz,
    )

    sort_columns = ["distance_from_unit_circle"]
    sort_ascending = [True]

    if "frequency_error_hz" in eigenvalue_summary_df.columns:
        sort_columns = ["frequency_error_hz", "distance_from_unit_circle"]
        sort_ascending = [True, True]

    eigenvalue_summary_sorted_df = eigenvalue_summary_df.sort_values(
        sort_columns,
        ascending=sort_ascending,
    ).reset_index(drop=True)

    if coordinate_rank_sweep_df is None:
        selected_rank_summary_df = None
    else:
        selected_rank_summary_df = coordinate_rank_sweep_df.loc[
            coordinate_rank_sweep_df["rank"] == rank
        ].reset_index(drop=True)

    return {
        "selected_rank_summary_df": selected_rank_summary_df,
        "outputs": outputs,
        "dmd_result": dmd_result,
        "prediction_df": outputs["prediction_df"],
        "fit_prediction_df": outputs["fit_prediction_df"],
        "forecast_prediction_df": outputs["forecast_prediction_df"],
        "error_summary_df": outputs["error_summary_df"],
        "eigenvalue_summary_df": eigenvalue_summary_df,
        "eigenvalue_summary_sorted_df": eigenvalue_summary_sorted_df,
        "report_lines": build_coordinate_dmd_report_lines(dmd_result),
    }


def run_coordinate_rank_comparison(
    fit_coordinates,
    forecast_coordinates,
    delay_count,
    rank_values,
    dt=1.0,
    fit_start_frame=0,
    forecast_start_frame=None,
):
    """Run coordinate-DMD reconstruction and forecast for multiple ranks."""
    prediction_dfs = []
    error_summary_dfs = []

    for rank in rank_values:
        outputs = run_coordinate_dmd_experiment(
            fit_coordinates=fit_coordinates,
            forecast_coordinates=forecast_coordinates,
            delay_count=delay_count,
            rank=rank,
            dt=dt,
            fit_start_frame=fit_start_frame,
            forecast_start_frame=forecast_start_frame,
        )

        prediction_df = outputs["prediction_df"].copy()
        prediction_df["rank"] = rank
        prediction_dfs.append(prediction_df)

        error_summary_df = outputs["error_summary_df"].copy()
        error_summary_df["rank"] = rank
        error_summary_dfs.append(error_summary_df)

    combined_prediction_df = pd.concat(prediction_dfs, ignore_index=True)
    combined_error_summary_df = pd.concat(error_summary_dfs, ignore_index=True)

    return combined_prediction_df, combined_error_summary_df


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------

def plot_coordinate_singular_values(
    svd_summary_df,
    title="Delay-Coordinate Singular Values",
    save_path=None,
    show=True,
):
    """Plot singular values and cumulative energy for delay-coordinate DMD."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].semilogy(
        svd_summary_df["rank"],
        svd_summary_df["singular_value"],
        marker="o",
    )
    axes[0].set_title("Singular Values")
    axes[0].set_xlabel("Rank")
    axes[0].set_ylabel("Singular value")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        svd_summary_df["rank"],
        svd_summary_df["cumulative_energy_fraction"],
        marker="o",
    )
    axes[1].set_title("Cumulative Energy")
    axes[1].set_xlabel("Rank")
    axes[1].set_ylabel("Cumulative energy fraction")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(True, alpha=0.3)

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


def plot_coordinate_rank_sweep(
    rank_sweep_df,
    true_frequency_hz=None,
    title="Coordinate DMD Rank Sweep",
    save_path=None,
    show=True,
):
    """Plot coordinate-DMD rank sweep metrics."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(
        rank_sweep_df["rank"],
        rank_sweep_df["fit_mean_euclidean_error"],
        marker="o",
        label="fit",
    )
    axes[0].plot(
        rank_sweep_df["rank"],
        rank_sweep_df["forecast_mean_euclidean_error"],
        marker="o",
        label="forecast",
    )
    axes[0].set_ylabel("Mean Euclidean error")
    axes[0].legend(loc="best")

    axes[1].plot(
        rank_sweep_df["rank"],
        rank_sweep_df["closest_abs_frequency_hz"],
        marker="o",
        label="closest DMD frequency",
    )

    if true_frequency_hz is not None:
        axes[1].axhline(
            true_frequency_hz,
            linestyle="--",
            label=f"true frequency = {true_frequency_hz:.3f} Hz",
        )

    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].legend(loc="best")

    axes[2].plot(
        rank_sweep_df["rank"],
        rank_sweep_df["closest_lambda_abs"],
        marker="o",
        label="|lambda| of closest-frequency mode",
    )
    axes[2].axhline(1.0, linestyle="--", label="unit magnitude")
    axes[2].set_xlabel("Rank")
    axes[2].set_ylabel("Eigenvalue magnitude")
    axes[2].ticklabel_format(axis="y", style="plain", useOffset=False)

    lambda_values = rank_sweep_df["closest_lambda_abs"].dropna().to_numpy()
    if len(lambda_values) > 0:
        y_min = min(lambda_values.min(), 1.0)
        y_max = max(lambda_values.max(), 1.0)
        y_padding = max((y_max - y_min) * 0.15, 1e-6)
        axes[2].set_ylim(y_min - y_padding, y_max + y_padding)

    axes[2].legend(loc="best")

    for ax in axes:
        ax.grid(True, alpha=0.3)

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


def plot_coordinate_time_series(
    prediction_df,
    forecast_start_frame=None,
    title="Bob Coordinate Reconstruction and Forecast",
    save_path=None,
    show=True,
):
    """Plot true and predicted x/y coordinates over frame index."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for window_name, group_df in prediction_df.groupby("window", sort=False):
        axes[0].plot(
            group_df["frame"],
            group_df["x_true"],
            label=f"{window_name} true x",
        )
        axes[0].plot(
            group_df["frame"],
            group_df["x_predicted"],
            linestyle="--",
            label=f"{window_name} predicted x",
        )

        axes[1].plot(
            group_df["frame"],
            group_df["y_true"],
            label=f"{window_name} true y",
        )
        axes[1].plot(
            group_df["frame"],
            group_df["y_predicted"],
            linestyle="--",
            label=f"{window_name} predicted y",
        )

    if forecast_start_frame is not None:
        for ax in axes:
            ax.axvline(forecast_start_frame, linestyle=":", label="forecast start")

    axes[0].set_title(title)
    axes[0].set_ylabel("x coordinate")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("y coordinate")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    fig.tight_layout()

    save_path = ensure_parent_dir(save_path)
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes


def plot_coordinate_trajectory(
    prediction_df,
    title="Bob Trajectory Reconstruction and Forecast",
    invert_y_axis=True,
    save_path=None,
    show=True,
):
    """Plot true and predicted bob paths in image-coordinate space."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for window_name, group_df in prediction_df.groupby("window", sort=False):
        ax.plot(
            group_df["x_true"],
            group_df["y_true"],
            label=f"{window_name} true",
        )
        ax.plot(
            group_df["x_predicted"],
            group_df["y_predicted"],
            linestyle="--",
            label=f"{window_name} predicted",
        )

    if invert_y_axis:
        ax.invert_yaxis()

    ax.set_title(title)
    ax.set_xlabel("x coordinate")
    ax.set_ylabel("y coordinate")
    ax.grid(True, alpha=0.3)
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


def plot_coordinate_error(
    prediction_df,
    forecast_start_frame=None,
    title="Bob Coordinate Error Over Time",
    save_path=None,
    show=True,
):
    """Plot Euclidean coordinate error over frame index."""
    fig, ax = plt.subplots(figsize=(10, 4))

    for window_name, group_df in prediction_df.groupby("window", sort=False):
        ax.plot(
            group_df["frame"],
            group_df["euclidean_error"],
            label=f"{window_name} error",
        )

    if forecast_start_frame is not None:
        ax.axvline(forecast_start_frame, linestyle=":", label="forecast start")

    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Euclidean error")
    ax.grid(True, alpha=0.3)
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


def plot_coordinate_trajectory_rank_comparison(
    rank_prediction_df,
    rank_values,
    title="Bob Coordinate DMD Trajectory by Rank",
    invert_y_axis=True,
    save_path=None,
    show=True,
):
    """Plot true bob path and predicted paths for several DMD ranks."""
    fig, ax = plt.subplots(figsize=(7, 6))

    first_rank = rank_values[0]
    true_df = rank_prediction_df.loc[
        rank_prediction_df["rank"] == first_rank
    ].sort_values("frame")

    ax.plot(
        true_df["x_true"],
        true_df["y_true"],
        linewidth=2.5,
        label="true bob path",
    )

    for rank in rank_values:
        rank_df = rank_prediction_df.loc[
            rank_prediction_df["rank"] == rank
        ].sort_values("frame")

        ax.plot(
            rank_df["x_predicted"],
            rank_df["y_predicted"],
            linestyle="--",
            label=f"rank {rank} predicted",
        )

    if invert_y_axis:
        ax.invert_yaxis()

    ax.set_title(title)
    ax.set_xlabel("x coordinate")
    ax.set_ylabel("y coordinate")
    ax.grid(True, alpha=0.3)
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


def plot_coordinate_time_series_rank_comparison(
    rank_prediction_df,
    rank_values,
    forecast_start_frame=None,
    title="Bob Coordinate DMD Time Series by Rank",
    save_path=None,
    show=True,
):
    """Plot true and predicted x/y coordinate time series for several ranks."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    first_rank = rank_values[0]
    true_df = rank_prediction_df.loc[
        rank_prediction_df["rank"] == first_rank
    ].sort_values("frame")

    axes[0].plot(
        true_df["frame"],
        true_df["x_true"],
        linewidth=2.5,
        label="true x",
    )
    axes[1].plot(
        true_df["frame"],
        true_df["y_true"],
        linewidth=2.5,
        label="true y",
    )

    for rank in rank_values:
        rank_df = rank_prediction_df.loc[
            rank_prediction_df["rank"] == rank
        ].sort_values("frame")

        axes[0].plot(
            rank_df["frame"],
            rank_df["x_predicted"],
            linestyle="--",
            label=f"rank {rank} predicted x",
        )
        axes[1].plot(
            rank_df["frame"],
            rank_df["y_predicted"],
            linestyle="--",
            label=f"rank {rank} predicted y",
        )

    if forecast_start_frame is not None:
        for ax in axes:
            ax.axvline(forecast_start_frame, linestyle=":", label="forecast start")

    axes[0].set_title(title)
    axes[0].set_ylabel("x coordinate")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("y coordinate")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    fig.tight_layout()

    save_path = ensure_parent_dir(save_path)
    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes


def plot_coordinate_error_rank_comparison(
    rank_prediction_df,
    rank_values,
    forecast_start_frame=None,
    title="Bob Coordinate DMD Error by Rank",
    use_log_scale=True,
    save_path=None,
    show=True,
):
    """Plot Euclidean coordinate error over time for several ranks."""
    fig, ax = plt.subplots(figsize=(10, 4.5))

    for rank in rank_values:
        rank_df = rank_prediction_df.loc[
            rank_prediction_df["rank"] == rank
        ].sort_values("frame")

        error_values = rank_df["euclidean_error"].to_numpy(dtype=np.float64)

        if use_log_scale:
            error_values = np.where(error_values <= 0.0, np.nan, error_values)

        ax.plot(
            rank_df["frame"],
            error_values,
            label=f"rank {rank}",
        )

    if forecast_start_frame is not None:
        ax.axvline(forecast_start_frame, linestyle=":", label="forecast start")

    if use_log_scale:
        ax.set_yscale("log")

    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Euclidean coordinate error")
    ax.grid(True, alpha=0.3)
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


# ---------------------------------------------------------------------
# GIF helpers
# ---------------------------------------------------------------------

def _load_frame_as_rgb(frame_path):
    with Image.open(frame_path) as image:
        return image.convert("RGB")

def _draw_plus_marker(
    draw,
    x,
    y,
    color,
    size=8,
    width=2,
    outline_color=(255, 255, 255),
    outline_width=4,
):
    x = float(x)
    y = float(y)

    if outline_color is not None and outline_width > 0:
        draw.line(
            [(x - size, y), (x + size, y)],
            fill=outline_color,
            width=outline_width,
        )
        draw.line(
            [(x, y - size), (x, y + size)],
            fill=outline_color,
            width=outline_width,
        )

    draw.line(
        [(x - size, y), (x + size, y)],
        fill=color,
        width=width,
    )
    draw.line(
        [(x, y - size), (x, y + size)],
        fill=color,
        width=width,
    )

def _draw_text_with_background(
    draw,
    xy,
    text,
    text_fill=(255, 255, 255),
    bg_fill=(0, 0, 0),
):
    x, y = xy
    bbox = draw.textbbox((x, y), text)
    padded_bbox = (
        bbox[0] - 2,
        bbox[1] - 1,
        bbox[2] + 2,
        bbox[3] + 1,
    )
    draw.rectangle(padded_bbox, fill=bg_fill)
    draw.text((x, y), text, fill=text_fill)

def _crop_image_around_point(
    image,
    center_x,
    center_y,
    crop_size=(160, 160),
    fill_color=(255, 255, 255),
):
    """
    Crop an image around a target point. If the crop extends past the image
    boundary, pad with fill_color.
    """
    if crop_size is None:
        return image.copy(), 0, 0

    crop_width, crop_height = crop_size
    crop_width = int(crop_width)
    crop_height = int(crop_height)

    center_x = float(center_x)
    center_y = float(center_y)

    left = int(round(center_x - crop_width / 2))
    top = int(round(center_y - crop_height / 2))
    right = left + crop_width
    bottom = top + crop_height

    cropped_image = Image.new(image.mode, (crop_width, crop_height), color=fill_color)

    src_left = max(0, left)
    src_top = max(0, top)
    src_right = min(image.width, right)
    src_bottom = min(image.height, bottom)

    if src_right > src_left and src_bottom > src_top:
        src_crop = image.crop((src_left, src_top, src_right, src_bottom))
        dst_left = src_left - left
        dst_top = src_top - top
        cropped_image.paste(src_crop, (dst_left, dst_top))

    return cropped_image, left, top

def _resize_image_if_needed(image, output_scale=1):
    output_scale = int(output_scale)

    if output_scale <= 1:
        return image

    if hasattr(Image, "Resampling"):
        resample_method = Image.Resampling.NEAREST
    else:
        resample_method = Image.NEAREST

    return image.resize(
        (image.width * output_scale, image.height * output_scale),
        resample=resample_method,
    )


def save_coordinate_overlay_gif(
    frame_paths,
    prediction_df,
    save_path,
    rank=None,
    window=None,
    fps=6,
    true_color=(0, 255, 0),
    predicted_color=(255, 64, 64),
    marker_size=8,
    marker_width=2,
    marker_outline_width=4,
    crop_size=(160, 160),
    output_scale=3,
    show_frame_label=True,
    show_rank_label=True,
    show_window_label=True,
    label_true="True",
    label_predicted="Predicted",
):
    """
    Save a GIF that overlays true and predicted bob coordinates on the original frames.

    The frame is cropped around the true bob position and then enlarged to make
    the forecast easier to inspect visually.
    """
    df = prediction_df.copy()

    if rank is not None:
        if "rank" not in df.columns:
            raise ValueError(
                "prediction_df has no 'rank' column, but rank filtering was requested."
            )
        df = df.loc[df["rank"] == rank]

    if window is not None:
        if "window" not in df.columns:
            raise ValueError(
                "prediction_df has no 'window' column, but window filtering was requested."
            )
        df = df.loc[df["window"] == window]

    if df.empty:
        raise ValueError("No rows remain after filtering prediction_df.")

    required_columns = ["frame", "x_true", "y_true", "x_predicted", "y_predicted"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"prediction_df is missing required columns: {missing_columns}")

    df = df.sort_values("frame").reset_index(drop=True)

    gif_frames = []

    for _, row in df.iterrows():
        frame_number = int(row["frame"])

        if frame_number < 0 or frame_number >= len(frame_paths):
            raise ValueError(
                f"Frame number {frame_number} is out of range for frame_paths "
                f"(length {len(frame_paths)})."
            )

        full_image = _load_frame_as_rgb(frame_paths[frame_number])

        cropped_image, crop_left, crop_top = _crop_image_around_point(
            full_image,
            center_x=row["x_true"],
            center_y=row["y_true"],
            crop_size=crop_size,
        )

        draw = ImageDraw.Draw(cropped_image)

        x_true_local = row["x_true"] - crop_left
        y_true_local = row["y_true"] - crop_top
        x_pred_local = row["x_predicted"] - crop_left
        y_pred_local = row["y_predicted"] - crop_top

        _draw_plus_marker(
            draw,
            x=x_pred_local,
            y=y_pred_local,
            color=predicted_color,
            size=marker_size,
            width=marker_width,
            outline_color=(255, 255, 255),
            outline_width=marker_outline_width,
        )

        _draw_plus_marker(
            draw,
            x=x_true_local,
            y=y_true_local,
            color=true_color,
            size=marker_size,
            width=marker_width,
            outline_color=(255, 255, 255),
            outline_width=marker_outline_width,
        )

        _draw_text_with_background(
            draw,
            (8, 8),
            label_true,
            text_fill=true_color,
        )
        _draw_text_with_background(
            draw,
            (8, 24),
            label_predicted,
            text_fill=predicted_color,
        )

        y_text = 40

        if show_frame_label:
            _draw_text_with_background(draw, (8, y_text), f"frame {frame_number}")
            y_text += 16

        # if show_window_label and "window" in row.index:
        #     _draw_text_with_background(draw, (8, y_text), f"window: {row['window']}")
        #     y_text += 16

        if show_rank_label and rank is not None:
            _draw_text_with_background(draw, (8, y_text), f"rank: {rank}")
            y_text += 16

        cropped_image = _resize_image_if_needed(cropped_image, output_scale=output_scale)
        gif_frames.append(cropped_image)

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

def save_coordinate_overlay_rank_comparison_gif(
    frame_paths,
    rank_prediction_df,
    rank_values,
    save_path,
    window="forecast",
    fps=6,
    true_color=(0, 255, 0),
    predicted_color=(255, 64, 64),
    marker_size=8,
    marker_width=2,
    marker_outline_width=4,
    crop_size=(160, 160),
    output_scale=3,
    panel_gap=12,
    show_frame_label=True,
    show_window_label=True,
    label_true="True",
    label_predicted="Predicted",
):
    """
    Save a side-by-side GIF comparing coordinate predictions for multiple ranks.

    Each panel shows a crop around the bob with:
    - the true bob position
    - the predicted bob position for one rank
    """
    df = rank_prediction_df.copy()

    if "rank" not in df.columns:
        raise ValueError("rank_prediction_df must contain a 'rank' column.")

    if "window" not in df.columns:
        raise ValueError("rank_prediction_df must contain a 'window' column.")

    df = df.loc[df["window"] == window].copy()

    if df.empty:
        raise ValueError(f"No rows found for window='{window}'.")

    required_columns = ["frame", "x_true", "y_true", "x_predicted", "y_predicted"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"rank_prediction_df is missing required columns: {missing_columns}")

    rank_values = list(rank_values)
    rank_groups = {}

    for rank in rank_values:
        rank_df = df.loc[df["rank"] == rank].sort_values("frame").reset_index(drop=True)

        if rank_df.empty:
            raise ValueError(f"No rows found for rank={rank} and window='{window}'.")

        rank_groups[rank] = rank_df

    reference_frames = rank_groups[rank_values[0]]["frame"].to_list()

    for rank in rank_values[1:]:
        frames_this_rank = rank_groups[rank]["frame"].to_list()
        if frames_this_rank != reference_frames:
            raise ValueError(
                f"Frame mismatch across ranks. rank {rank_values[0]} and rank {rank} "
                "do not share the same ordered forecast frames."
            )

    gif_frames = []

    for row_index, frame_number in enumerate(reference_frames):
        panel_images = []

        true_x = rank_groups[rank_values[0]].iloc[row_index]["x_true"]
        true_y = rank_groups[rank_values[0]].iloc[row_index]["y_true"]

        for rank in rank_values:
            row = rank_groups[rank].iloc[row_index]

            full_image = _load_frame_as_rgb(frame_paths[frame_number])

            cropped_image, crop_left, crop_top = _crop_image_around_point(
                full_image,
                center_x=true_x,
                center_y=true_y,
                crop_size=crop_size,
            )

            draw = ImageDraw.Draw(cropped_image)

            x_true_local = row["x_true"] - crop_left
            y_true_local = row["y_true"] - crop_top
            x_pred_local = row["x_predicted"] - crop_left
            y_pred_local = row["y_predicted"] - crop_top

            _draw_plus_marker(
                draw,
                x=x_pred_local,
                y=y_pred_local,
                color=predicted_color,
                size=marker_size,
                width=marker_width,
                outline_color=(255, 255, 255),
                outline_width=marker_outline_width,
            )

            _draw_plus_marker(
                draw,
                x=x_true_local,
                y=y_true_local,
                color=true_color,
                size=marker_size,
                width=marker_width,
                outline_color=(255, 255, 255),
                outline_width=marker_outline_width,
            )

            _draw_text_with_background(draw, (8, 8), f"Rank {rank}")
            _draw_text_with_background(draw, (8, 24), label_true, text_fill=true_color)
            _draw_text_with_background(draw, (8, 40), label_predicted, text_fill=predicted_color)

            y_text = 56

            if show_frame_label:
                _draw_text_with_background(draw, (8, y_text), f"frame {frame_number}")
                y_text += 16

            # if show_window_label:
            #     _draw_text_with_background(draw, (8, y_text), f"window: {window}")
            #     y_text += 16

            cropped_image = _resize_image_if_needed(cropped_image, output_scale=output_scale)
            panel_images.append(cropped_image)

        panel_width, panel_height = panel_images[0].size
        combined_width = len(panel_images) * panel_width + (len(panel_images) - 1) * panel_gap
        combined_height = panel_height

        combined_image = Image.new("RGB", (combined_width, combined_height), color=(255, 255, 255))

        x_offset = 0
        for image in panel_images:
            combined_image.paste(image, (x_offset, 0))
            x_offset += panel_width + panel_gap

        gif_frames.append(combined_image)

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