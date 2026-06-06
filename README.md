# Data Science Topics

This repository contains notebook-centered explorations of mathematical, data science, and Python programming topics.

Many folders are built around a Jupyter notebook that combines conceptual explanation with hands-on implementation. The emphasis is on understanding the method, writing the code, visualizing the results, and connecting the math to practical examples.

This repository is less deployment-focused than my main machine learning portfolio work in [`MLProjects`](https://github.com/bodle720/MLProjects). It is intended to show mathematical intuition, implementation, and practical Python/data-science work.

## Featured Project: Dynamic Mode Decomposition

![Synthetic pendulum coordinate forecast](DMD/docs/images/dmd_pendulum_coordinate_rank_comparison_forecast_overlay.gif)

The strongest project in this repository is the **Dynamic Mode Decomposition** pendulum notebook.

DMD is applied to a synthetic pendulum video to study how state representation affects reconstruction and forecasting. Full-frame pixel DMD recovers meaningful spatial modes and near-correct oscillation frequencies, but its important motion modes are strongly damped, causing reconstruction to decay toward the mean frame. A lower-dimensional delay-coordinate model built from the pendulum bob trajectory recovers the true oscillation and forecasts the motion much more accurately.

Start here:

* [DMD project README](DMD/README.md)
* [DMD pendulum notebook](DMD/DMD_pendulum_video.ipynb)
* [DMD background notes](DMD/docs/DMD_Background/README.md)

## Other Featured Topics

| Topic                   | Main Artifact                                                                                     | Focus                                                                                                                                                                                                                 |
| ----------------------- | ------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Linear PCA              | [pca_linear_oscillation_system.ipynb](PCA_linear/pca_linear_oscillation_system.ipynb)             | Linear PCA on a synthetic oscillating dynamical system, including covariance/eigenvector intuition, explained variance, loadings, and recovery of low-dimensional structure from noisy high-dimensional observations. |
| Kernel PCA              | [pca_kernel.ipynb](PCA_kernel/pca_kernel.ipynb)                                                   | Nonlinear dimensionality reduction using kernel PCA, including RBF-kernel intuition, kernel centering, eigen-decomposition, and comparison against ordinary PCA.                                                      |
| Robust PCA              | [PCA_robust_handwritten_digits_noise.ipynb](PCA_robust/PCA_robust_handwritten_digits_noise.ipynb) | Robust PCA on corrupted handwritten digit images, separating low-rank structure from sparse noise and using cleaned representations for downstream classification.                                                    |
| Sparse PCA              | [PCA_sparse_handwritten_digits.ipynb](PCA_sparse/PCA_sparse_handwritten_digits.ipynb)             | Sparse PCA on handwritten digit images, emphasizing sparse loadings, interpretability, feature visualization, and image reconstruction.                                                                               |
| ICA                     | [ica_on_imagery.ipynb](ICA/ica_on_imagery.ipynb)                                                  | Independent Component Analysis / blind-source separation on imagery, using artificial image mixing and matrix methods to recover approximate source images.                                                           |
| Interactive Brokers API | [IBKR](IBKR/)                                                                                     | Real-world API integration example using the Interactive Brokers API, focused on connection handling, asynchronous callbacks, market-data retrieval, and `pandas` data organization.                                  |
| Python Multiprocessing  | [Python_Multiprocessing](Python_Multiprocessing/)                                                 | Practical Python multiprocessing example using `multiprocessing.Pool`, `imap`, `tqdm`, chunk sizing, and ordered result collection.                                                                                   |

## Topic Groups

### Dynamical Systems and Matrix Methods

The `DMD/` folder is the main project in this repository. It demonstrates Dynamic Mode Decomposition on synthetic pendulum video data, with emphasis on eigenvalues, modes, frequency interpretation, reconstruction diagnostics, and coordinate-level forecasting.

The PCA notebooks form a smaller series on dimensionality reduction and matrix decomposition:

* **Linear PCA**: covariance/eigenvector intuition, explained variance, and recovery of low-dimensional structure.
* **Kernel PCA**: nonlinear dimensionality reduction with RBF kernels and centered kernel matrices.
* **Robust PCA**: low-rank plus sparse decomposition for denoising corrupted handwritten digit images.
* **Sparse PCA**: sparse loadings for more interpretable principal components on digit images.

Together, these notebooks demonstrate dimensionality reduction, eigenvalue methods, visual intuition, and the relationship between mathematical structure and practical data analysis.

### Signal Separation

The `ICA/` folder explores Independent Component Analysis / blind-source separation on imagery. It uses artificial image mixing to demonstrate how matrix methods can recover approximate source images from mixed observations.

### Practical Python and API Work

These folders are smaller practical examples rather than modeling notebooks:

* **IBKR** demonstrates working with a real external API, asynchronous responses, market-data retrieval, and helper utilities for time-series data.
* **Python Multiprocessing** demonstrates a reusable pattern for parallelizing command-line work with progress tracking.

They are included because practical data work often requires more than modeling: it also requires API integration, scripting, data handling, and performance-aware Python code.

## Notes

Some notebooks intentionally take the “long route” instead of only calling a high-level library function. This is deliberate. The goal is to make the underlying method easier to inspect, explain, and reason about.

Not every folder is meant to be a polished application. This repository is a collection of focused explorations that demonstrate mathematical understanding, practical implementation, and curiosity across data science and Python programming topics.

Social Preview photo by <a href="https://unsplash.com/@steve_j?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Steve A Johnson</a> on <a href="https://unsplash.com/photos/a-group-of-white-boxes-with-numbers-on-them-QaM0dr1xN4M?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
      