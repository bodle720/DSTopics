# DMD Pendulum Dynamics

![Synthetic pendulum coordinate forecast](docs/images/dmd_pendulum_coordinate_rank_comparison_forecast_overlay.gif)

This project applies **Dynamic Mode Decomposition (DMD)** to a synthetic pendulum video sequence. The goal is to show how DMD connects high-dimensional video data with low-rank temporal structure, eigenvalues, spatial modes, reconstruction behavior, and forecasting.

The synthetic pendulum is a controlled visual dynamical system: each video frame is a high-dimensional pixel observation, but the underlying motion is governed by a low-dimensional periodic swing. This makes it a cleaner and more interpretable DMD example than noisy real-world forecasting tasks.

The main lesson is that **DMD performance depends strongly on the chosen state representation**. Full-frame pixel DMD recovers meaningful spatial modes and near-correct oscillation frequencies, but its important motion modes are strongly damped, causing reconstruction to decay toward the mean frame. A lower-dimensional delay-coordinate model built from the pendulum bob trajectory recovers the true oscillation and forecasts the motion much more accurately.

## Prerequisite Topics

For background on modes, eigenvalues, matrix exponentials, and DMD, see:

[DMD Background](docs/DMD_Background/README.md)

## Notebook

[Open the notebook](DMD_pendulum_video.ipynb)

The notebook walks through:

* generating a synthetic pendulum video
* converting video frames into DMD snapshot matrices
* selecting fitting and forecast windows
* choosing a DMD rank using singular values and frequency diagnostics
* fitting full-frame DMD on centered grayscale video frames
* interpreting eigenvalues, modes, damping, and frequencies
* visualizing full-frame DMD modes
* reconstructing observed motion and diagnosing why the full-frame model decays
* pivoting to delay-coordinate DMD on the pendulum bob trajectory
* comparing coordinate-DMD reconstruction and forecasting across ranks
* summarizing limitations and possible extensions

## Project Structure

```text
DMD/
├── DMD_pendulum_video.ipynb
├── README.md
├── helpers/
│   └── make_synthetic_pendulum.py
├── docs/
│   ├── DMD_Background/
│   └── images/
│       ├── synthetic_pendulum_preview.gif
│       └── dmd_pendulum_coordinate_rank_comparison_forecast_overlay.gif
└── outputs/
    └── synthetic_pendulum/
```

`outputs/` contains generated frames, masks, metadata, ground-truth coordinates, and preview artifacts. It is treated as generated output rather than source material and is created after running `helpers/make_synthetic_pendulum.py`.

## Generate the Synthetic Pendulum Data

From the `DMD/` folder:

```bash
python helpers/make_synthetic_pendulum.py --overwrite --background-mode grid --noise-std 2
```

This generates:

* RGB video frames
* foreground masks for the arm and bob
* bob-only masks
* ground-truth pendulum angle and bob coordinates
* metadata describing the generated sequence
* a README-friendly preview GIF under `docs/images/`

## Why This Example Works Well for DMD

DMD approximates the time evolution of a system from sequential observations. In this project, each image frame is flattened into a state vector, and DMD learns an approximate linear time-advance model:

$$
x_{k+1} \approx A x_k
$$

For the pendulum sequence, the learned full-frame DMD model captures interpretable structure:

* persistent modes reflect static or slowly changing image content
* oscillatory modes reflect pendulum motion
* eigenvalue phases encode frequency information
* eigenvalue magnitudes describe growth or decay behavior

Because the true pendulum trajectory is known, the notebook can compare learned dynamics against ground-truth bob motion. This makes the project useful not only as a DMD implementation, but also as a diagnostic study of when a representation works well and when it does not.

## Summary

This project is a compact theory-to-implementation demonstration of Dynamic Mode Decomposition on video data. It emphasizes mathematical interpretation, visual diagnostics, eigenvalue and frequency analysis, and the importance of choosing an appropriate state representation for dynamical modeling.
