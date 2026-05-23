# Data Science Topics

This repository contains notebook-centered explorations of mathematical, data science, and Python programming topics.

Many folders are built around a Jupyter notebook that combines conceptual explanation with hands-on implementation. The emphasis is on understanding the method, writing the code, visualizing the results, and connecting the math to practical examples.

This repository is less deployment-focused than my main machine learning portfolio work in [`MLProjects`](https://github.com/bodle720/MLProjects). It is intended to show mathematical intuition, implementation, and practical Python/data-science.

## Featured Topics

| Topic | Main Artifact                                                                                       | Focus |
|---|-----------------------------------------------------------------------------------------------------|---|
| Linear PCA | [`pca_linear_oscillation_system.ipynb`](PCA_linear/pca_linear_oscillation_system.ipynb)             | Linear PCA on a synthetic oscillating dynamical system, including covariance/eigenvector intuition, explained variance, loadings, and recovery of low-dimensional structure from noisy high-dimensional observations. |
| Kernel PCA | [`pca_kernel.ipynb`](PCA_kernel/pca_kernel.ipynb)                                                   | Nonlinear dimensionality reduction using kernel PCA, including RBF-kernel intuition, kernel centering, eigen-decomposition, and comparison against ordinary PCA. |
| Robust PCA | [`PCA_robust_handwritten_digits_noise.ipynb`](PCA_robust/PCA_robust_handwritten_digits_noise.ipynb) | Robust PCA on corrupted handwritten digit images, separating low-rank structure from sparse noise and using cleaned representations for downstream classification. |
| Sparse PCA | [`PCA_sparse_handwritten_digits.ipynb`](PCA_sparse/PCA_sparse_handwritten_digits.ipynb)             | Sparse PCA on handwritten digit images, emphasizing sparse loadings, interpretability, feature visualization, and image reconstruction. |
| ICA | [`ica_on_imagery.ipynb`](ICA/ica_on_imagery.ipynb)                                                  | Independent Component Analysis / blind-source separation on imagery, using artificial image mixing and matrix methods to recover approximate source images. |
| DMD | [`DMD Pendulum Video`](DMD/DMD_pendulum.ipynb)                                                | Dynamic Mode Decomposition on synthetic pendulum video for mode analysis, reconstruction, forecasting, and foreground/background separation. |
| Interactive Brokers API | [`IBKR/`](IBKR/)                                                                                    | Real-world API integration example using the Interactive Brokers API, focused on connection handling, asynchronous callbacks, market-data retrieval, and `pandas` data organization. |
| Python Multiprocessing | [`Python_Multiprocessing/`](Python_Multiprocessing/)                                                | Practical Python multiprocessing example using `multiprocessing.Pool`, `imap`, `tqdm`, chunk sizing, and ordered result collection. |

## Topic Groups

### PCA and Matrix Methods

The PCA notebooks form a small series on dimensionality reduction and matrix decomposition:

- **Linear PCA**: covariance/eigenvector intuition, explained variance, and recovery of low-dimensional structure.
- **Kernel PCA**: nonlinear dimensionality reduction with RBF kernels and centered kernel matrices.
- **Robust PCA**: low-rank plus sparse decomposition for denoising corrupted handwritten digit images.
- **Sparse PCA**: sparse loadings for more interpretable principal components on digit images.

Together, these notebooks demonstrate dimensionality reduction, matrix decomposition, eigenvalue methods, visual intuition, and the relationship between mathematical structure and practical data analysis.

### Signal Separation and Dynamical Systems

The `ICA/` and `DMD/` folders explore data-driven methods for extracting structure from observed data:

* **ICA** focuses on blind-source separation and reconstruction of independent signal components from mixed image observations.
* **DMD** applied to synthetic pendulum video motion, approximates dynamical evolution from time-series snapshots.

These notebooks are exploratory and instructional. They are intended to show mathematical implementation, visualization, and analysis rather than production modeling systems.

### Practical Python and API Work

These folders are smaller practical examples rather than modeling notebooks:

* **IBKR** demonstrates working with a real external API, asynchronous responses, market-data retrieval, and helper utilities for time-series data.
* **Python Multiprocessing** demonstrates a reusable pattern for parallelizing command-line work with progress tracking.

They are included because practical data work often requires more than modeling: it also requires API integration, scripting, data handling, and performance-aware Python code.

## Notes

Some notebooks intentionally take the “long route” instead of only calling a high-level library function. This is deliberate. The goal is to make the underlying method easier to inspect, explain, and reason about.

Not every folder is meant to be a polished application. The repository is a collection of focused explorations that demonstrate mathematical understanding, practical implementation, and curiosity across data science and Python programming topics.