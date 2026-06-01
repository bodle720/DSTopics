# DMD Background Notes

These notes provide the mathematical background for the DMD pendulum video notebook.

They are meant to be read as a short sequence before opening the notebook. The notes cover the core ideas behind modes, eigenvalues, matrix exponentials, continuous versus discrete time, and how DMD applies those ideas to video data.

## Contents

### 1. Modes and Eigenvalues

[01_modes_and_eigenvalues.md](01_modes_and_eigenvalues.md)

Introduces linear dynamical systems, matrix exponentials, eigenvalues, eigenvectors, modes, continuous-time dynamics, discrete-time dynamics, and the relationship between continuous eigenvalues $\omega_i$ and discrete eigenvalues $\lambda_i$.

### 2. A Couple Examples

[02_a_couple_examlpes.md](02_a_couple_examples.md)

Works through two small examples: circular motion as a clean oscillatory linear system, and parabolic motion as an example that requires affine dynamics or a lifted state space. These examples help clarify what linear modes can and cannot represent directly.

### 3. DMD and Videos

[03_dmd_and_videos.md](03_dmd_and_videos.md)

Connects the theory to Dynamic Mode Decomposition. This note explains snapshot matrices, the reduced DMD operator, DMD eigenvalues, DMD modes, video-frame interpretation, scaling and rotation, frequency reporting, and pendulum-specific interpretation.

### 4. Applying DMD

[04_applying_dmd.md](04_applying_dmd.md)

Bridges the background theory into the actual pendulum experiment. This note explains the DMD workflow used in the notebook, including snapshot construction, low-rank approximation, mode reconstruction, forecasting, and frequency interpretation.

### 5. Nyquist Frequency and DMD Frequency Interpretation

[05_nyquist_frequency_and_dmd.md](05_nyquist_frequency_and_dmd.md)

Explains how to interpret DMD frequencies in the pendulum video, including the difference between discrete eigenvalues $\lambda_i$ and continuous-time eigenvalues $\omega_i$, why frequencies are computed from $\mathrm{Im}(\omega_i)/(2\pi)$, what the Nyquist frequency means for a sampled video, why a 30 fps video has a Nyquist frequency of 15 Hz, and why high-rank DMD models can produce sign-flipping Nyquist-like artifacts instead of physically meaningful pendulum motion.

### 6. Complex DMD Modes and Conjugate Pairs

[06_complex_dmd_modes_and_conjugate_pairs.md](06_complex_dmd_modes_and_conjugate_pairs.md)

Explains how to interpret complex DMD modes in real-valued video data, including why complex eigenvalues and eigenvectors appear in conjugate pairs, how conjugate pairs combine to produce real-valued motion, what the real and imaginary parts of a complex mode mean visually, why mode magnitude is usually plotted elementwise across pixels, and how these ideas apply to oscillatory pendulum modes.

## Notebook

After reading the four background notes, continue with the executable notebook:

[DMD_pendulum_video.ipynb](../../DMD_pendulum_video.ipynb)

The notebook applies DMD to synthetic pendulum video data for reconstruction, forecasting, frequency analysis, mode visualization, and foreground/background separation.
