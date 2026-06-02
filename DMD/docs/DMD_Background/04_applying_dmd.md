# Part 4: Applying DMD: Dynamic Mode Decomposition for Pendulum Video Motion

This note discusses the application of **Dynamic Mode Decomposition (DMD)** to a synthetic pendulum video sequence.
See [part 5](05_nyquist_frequency_and_dmd.md) next. After that, see the [notebook](/DMD/DMD_pendulum_video.ipynb) for the code implementation.

The background notes before this one introduced the main mathematical ideas: modes, eigenvalues, matrix exponentials, continuous-time and discrete-time dynamics, frequency interpretation, and the relationship between DMD and video data. This note is more practical. It focuses on how those ideas are used in the pendulum video notebook.

The central idea is that a video is high-dimensional when represented as raw pixels, but the motion generating the video may be much lower-dimensional. A pendulum is a useful example because its motion is smooth, coherent, and periodic. Each frame contains many pixel values, but the true underlying motion is mostly governed by a small number of physical quantities: angle, angular velocity, bob position, and time.

The goal of the notebook is to use a controlled visual dynamical system to study:

* DMD snapshot matrices,
* low-rank approximation through SVD,
* DMD eigenvalues, modes, frequencies, and growth rates,
* reconstruction of observed video frames,
* forecasting of held-out future frames,
* background/foreground separation,
* bob-centroid motion analysis,
* and the effect of rank selection on reconstruction and forecasting quality.

The notebook is meant to be instructional. The goal is not merely to run DMD as a black-box algorithm, but to connect the implementation back to the mathematics: state vectors, linear time-advance operators, eigenvalues, modes, modal amplitudes, SVD truncation, and frequency interpretation.

A previous version of this project applied DMD to financial time-series data. That was useful as an exploratory sequential-modeling exercise, but financial data is noisy, non-stationary, difficult to validate visually, and strongly affected by variables that are not observed. The synthetic pendulum is a better instructional example because the true motion is known, the future frames are known, and the model output can be evaluated both visually and numerically.

## DMD in One-Step Prediction Form

DMD learns an approximate one-step linear model from snapshot data:

$$
\vec{x}_{k+1} \approx A\vec{x}_k.
$$

Here:

* $\vec{x}_k$ is the observed state at time step $k$,
* $\vec{x}_{k+1}$ is the next observed state,
* and $A$ is an unknown linear operator that advances the state forward by one sampled step.

This mirrors the modal-solution idea from Part 1: a state can be represented as a sum of spatial modes, each with its own time behavior and amplitude. DMD keeps that modal viewpoint, but estimates the modes and eigenvalues from snapshot data rather than from a known system matrix.

For this notebook, each state vector is a flattened grayscale video frame. If a frame has height $h$ and width $w$, then

$$
\vec{x}_k \in \mathbb{R}^{h \cdot w}.
$$

The full transition matrix $A$ would therefore have shape

$$
(h \cdot w) \times (h \cdot w).
$$

Even for modest image sizes, this matrix is too large to form and analyze directly. DMD avoids this by estimating the dominant dynamics in a lower-dimensional subspace.

The standard DMD formulation used here is based on the formulation described in *Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control* by Brunton and Kutz.

## Pendulum Frames as State Vectors

The input to DMD is a sequence of video frames:

$$
\text{frame}_1, \text{frame}_2, \ldots, \text{frame}_m.
$$

Each frame is converted to grayscale and flattened into a column vector:

$$
\vec{x}_1, \vec{x}_2, \ldots, \vec{x}_m.
$$

This turns the video into a sequence of high-dimensional points. DMD then tries to find a linear rule that approximately advances one point to the next.

For the pendulum video, this is useful because the frames are high-dimensional, but the visible motion is organized. The bob and arm move coherently through time, while much of the background is static. DMD attempts to represent this video as a sum of spatial patterns with simple temporal behavior.

## Snapshot Matrices

Suppose we observe $m$ snapshots:

$$
\vec{x}_1, \vec{x}_2, \ldots, \vec{x}_m.
$$

DMD forms two snapshot matrices:

$$
X =[\vec{x}_1 ; \vec{x}_2 ; \cdots ; \vec{x}_{m-1}]
$$

and

$$
X' =[\vec{x}_2 ; \vec{x}_3 ; \cdots ; \vec{x}_m].
$$

Equivalently, using a more visual column notation,

$$
X = \begin{bmatrix}
| & | &  & | \\
\vec{x}_1 & \vec{x}_2 & \cdots & \vec{x}_{m-1} \\
| & | &  & |
\end{bmatrix}
$$

and

$$
X' = \begin{bmatrix}
| & | &  & | \\
\vec{x}_2 & \vec{x}_3 & \cdots & \vec{x}_{m} \\
| & | &  & |
\end{bmatrix}.
$$


The columns of $X$ contain the earlier snapshots. The columns of $X'$ contain the same sequence shifted forward by one time step.

The DMD assumption is

$$
X' \approx AX.
$$

where

$$
AX = A
\begin{bmatrix}
\vec{x}_1 & \vec{x}_2 & \cdots & \vec{x}_{m-1}
\end{bmatrix}
= \begin{bmatrix}
A\vec{x}_1 & A\vec{x}_2 & \cdots & A\vec{x}_{m-1}
\end{bmatrix}.
$$

Each column of $X'$ is approximated by each column of the above, this means

$$
A\vec{x}_j \approx \vec{x}_{j+1},
\qquad j = 1,2,\ldots,m-1.
$$

So DMD is trying to find one linear operator that approximately maps each frame to the next frame. In this setup, each $\vec{x}_{j+1}$ is $\Delta t$ units of time after $\vec{x}_j$.


## Choosing the Fitting Window and Timestep

Two practical choices matter before fitting DMD:

1. how many frames to use,
2. and what timestep to associate with consecutive snapshots.


The number of frames controls how much temporal behavior is available for fitting. If the fitting window is too short, DMD may not see enough of the pendulum swing to identify the dominant oscillation. If the fitting window is too long, the data may include behavior that is no longer well approximated by one fixed linear time-advance operator. For the synthetic pendulum, the situation is cleaner because the system is controlled, periodic, and generated by a stable rule. Even here, the fitting window should include enough of the swing to learn the dominant oscillation. Fitting on only a tiny fraction of a period may not reveal the full periodic motion, while fitting on one or more full periods gives DMD a better chance to identify the oscillatory structure.

Standard DMD assumes the snapshots are sampled at a constant, uniform timestep, so consecutive columns of the snapshot matrices should represent evenly spaced times. Variants and preprocessing strategies can handle more complicated sampling situations, but in this notebook we use evenly spaced video frames so that each DMD step has a clear physical meaning.


The timestep controls the physical meaning of one DMD step. If every video frame is used and the video has frame rate $\mathrm{fps}$, then

$$
\Delta t = \frac{1}{\mathrm{fps}}.
$$

For example, at $30$ frames per second, consecutive frames are separated by $\Delta t = 1/30$ seconds.

If the video is downsampled by keeping every second frame, then the effective timestep becomes

$$
\Delta t = \frac{2}{\mathrm{fps}}.
$$

More generally, if every $q$-th frame is kept, then

$$
\Delta t = \frac{q}{\mathrm{fps}}.
$$

The DMD operator estimates one-step evolution between the snapshots it sees. Therefore, $\Delta t$ determines what “one step forward” means physically.

## Least-Squares View of the Transition Matrix

The most direct way to estimate $A$ is to solve

$$
X' \approx AX.
$$

Using the Moore-Penrose pseudoinverse of $X$, denoted $X^\dagger$, the least-squares solution is

$$
A = X'X^\dagger.
$$

Mathematically, this $A$ is the matrix that best fits the observed transitions from one snapshot to the next in a least-squares sense:

$$
A\vec{x}_j \approx \vec{x}_{j+1}.
$$

This formulation is often referred to as **exact DMD**.

The algebraic least-squares equation does not explicitly depend on the physical size of the timestep. However, $\Delta t$ is still important because it determines the physical meaning of one step forward and is needed when interpreting eigenvalues as growth rates, frequencies, and periods.

The direct solution is conceptually simple, but it is usually not practical for image data. If each flattened frame has dimension $n$, then $A$ has shape

$$
n \times n.
$$

For a $256 \times 256$ grayscale frame,

$$
n = 256 \cdot 256 = 65{,}536.
$$

The full matrix $A$ would have shape

$$
65{,}536 \times 65{,}536.
$$

That is far too large to compute and analyze directly in a simple notebook. Instead, DMD estimates the important dynamics in a reduced low-dimensional subspace.

## Low-Rank Structure and the SVD

The key practical assumption is that the high-dimensional video has useful low-rank structure.

Even though each frame contains many pixels, most of the important variation may be described by a smaller number of dominant spatiotemporal patterns. In other words, we hope that the data can be compressed into a lower-dimensional subspace while still preserving the main structure needed for reconstruction, forecasting, and interpretation. For the pendulum video, this is plausible because the background is mostly static and the moving pendulum follows a coherent periodic path.

DMD uses the Singular Value Decomposition (SVD) to identify a low-dimensional subspace for the data. This is closely related to Principal Component Analysis (PCA). PCA uses the SVD to identify dominant directions of variation in data. DMD also uses the SVD, but then goes further by estimating how the reduced coordinates evolve through time. See my [notebook on PCA](/PCA_linear/pca_linear_oscillation_system.ipynb) for a related discussion of dimensionality reduction and low-rank approximation.

The first computational step is to compute a rank $r$ truncated SVD of the snapshot matrix $X$:

$$
X \approx U_r \Sigma_r V_r^*.
$$

Here:

* $U_r$ contains the first $r$ left singular vectors,
* $\Sigma_r$ contains the first $r$ singular values,
* $V_r$ contains the first $r$ right singular vectors,
* $V_r^*$ denotes the conjugate transpose of $V_r$,
* and $r$ is the chosen truncation rank.

The columns of $U_r$ define the reduced subspace. Instead of trying to learn dynamics in the full pixel space, DMD learns dynamics inside this smaller rank $r$ subspace.

The choice of $r$ is important. A small value of $r$ may underfit by discarding meaningful dynamics. A large value of $r$ may retain noise, weak modes, or unstable behavior that hurts forecasting.

A simple first approach is to inspect the singular values, often on a log plot, and choose a cutoff where the values decay sharply or become small. More careful approaches may use cumulative-energy thresholds, noise-aware rank selection, cross-validation, or application-specific criteria.

In the notebook, the singular values will be plotted before selecting a main rank. Later, the effect of rank selection will be studied directly.

## Practical DMD Algorithm

The practical DMD procedure is:

1. Build snapshot matrices $X$ and $X'$.
2. Compute a rank $r$ truncated SVD:

$$
X \approx U_r \Sigma_r V_r^*.
$$

3. Build the reduced operator, denoted $\tilde{A}$, by using the SVD factors to build an $r \times r$ reduced substitute for the full operator $A$.:

$$
\tilde{A} = U_r^* X' V_r \Sigma_r^{-1}.
$$

4. Compute the eigenvalues and eigenvectors of $\tilde{A}$:

$$
\tilde{A}W = W\Lambda.
$$

5. Map the reduced eigenvectors back to the original pixel space to obtain DMD modes:

$$
\Phi = X' V_r \Sigma_r^{-1} W.
$$

After these steps, the main DMD ingredients are:

* $\Phi$, the DMD modes,
* $\Lambda$, the diagonal matrix of DMD eigenvalues,
* and modal amplitudes, which describe how strongly each mode contributes to a chosen starting frame.

## Reduced Linear Operator

Conceptually, the reduced operator represents the full transition matrix projected into the rank $r$ subspace:

$$
\tilde{A} \approx U_r^* A U_r.
$$

Here, $U_r$ maps reduced coordinates back toward the original state space, and $U_r^*$ projects original state-space vectors down into the reduced coordinates.

The direct least-squares estimate of the full transition matrix is

$$
A = X'X^\dagger.
$$

This full matrix $A$ represents the best-fit one-step transition operator for the observed snapshot pairs. In practical DMD, however, we do not know the true continuous-time generator $A_c$ or the true full discrete-time operator $A_d$. We only have data snapshots, and forming the full least-squares matrix $A$ is usually too expensive for high-dimensional data such as images.

Using the truncated SVD approximation

$$
X \approx U_r\Sigma_rV_r^*,
$$

DMD computes the reduced operator directly from the snapshot matrices:

$$
\tilde{A} = U_r^* X' V_r \Sigma_r^{-1}.
$$

This $\tilde{A}$ is the matrix actually computed and analyzed in practical DMD.

When the data approximately follow a one-step linear model,

$$
X' \approx AX,
$$

the computed reduced operator approximates the projection of the full transition matrix into the rank $r$ subspace:

$$
\tilde{A} \approx U_r^* A U_r.
$$

The matrix $\tilde{A}$ has shape

$$
r \times r.
$$

It represents the approximate frame-to-frame dynamics inside the subspace spanned by the columns of $U_r$.

This is the key computational shortcut. Instead of forming the huge matrix $A$ in the original pixel space, DMD forms a much smaller matrix $\tilde{A}$ in the reduced SVD coordinate system.

The eigenvalues of $\tilde{A}$ are the DMD eigenvalues. They approximate the dominant eigenvalues of the unknown full time-advance operator, and they describe the learned frame-to-frame behavior of the DMD modes.


## Eigenvalues of the Reduced Operator

The next step is to solve

$$
\tilde{A}W = W\Lambda.
$$

Here:

* the columns of $W$ are eigenvectors of $\tilde{A}$,
* $\Lambda$ is a diagonal matrix containing DMD eigenvalues,
* and each DMD eigenvalue describes the frame-to-frame behavior of one mode.

If $\lambda_i$ is a DMD eigenvalue, then:

* $|\lambda_i| < 1$ suggests a decaying mode,
* $|\lambda_i| > 1$ suggests a growing mode,
* $|\lambda_i| \approx 1$ suggests a persistent mode,
* and a complex $\lambda_i$ suggests oscillatory behavior.

For the pendulum video, the most important non-background behavior should be oscillatory. Therefore, a strong DMD result should reveal one or more complex-conjugate eigenvalue pairs associated with the swinging motion.

## DMD Modes in the Original Pixel Space

The eigenvectors in $W$ are eigenvectors of the reduced operator $\tilde{A}$. They live in the reduced rank $r$ coordinate system, not in the original $n$-dimensional state space.

For video data, the original state space is the flattened pixel space. If each frame has height $h$ and width $w$, then each state vector has dimension

$$
n = h \cdot w.
$$

Therefore, to interpret the reduced eigenvectors as image-like spatial patterns, we need to map them back to the original pixel space.

Conceptually, the DMD modes can be written as

$$
\Phi = AU_rW.
$$

Here, $U_rW$ maps the reduced eigenvectors back toward the original state space, and multiplication by $A$ advances those mapped vectors through the learned time dynamics.

In practice, we do not explicitly form the full matrix $A$. Instead, DMD uses a data-driven expression. Recall that

$$
X \approx U_r\Sigma_rV_r^*.
$$

Using this truncated SVD approximation, the least-squares transition operator can be written approximately as

$$
A \approx X'V_r\Sigma_r^{-1}U_r^*.
$$

Therefore,

$$
AU_rW
\approx
X'V_r\Sigma_r^{-1}U_r^*U_rW.
$$

Since the columns of $U_r$ are orthonormal,

$$
U_r^*U_r = I.
$$

So,

$$
AU_rW
\approx
X'V_r\Sigma_r^{-1}W.
$$

This gives the exact DMD mode formula used in practice:

$$
\Phi = X'V_r\Sigma_r^{-1}W.
$$

The columns of $\Phi$ are the DMD modes:

$$
\Phi =
[\phi_1 ; \phi_2 ; \cdots ; \phi_r].
$$

Each mode $\phi_i$ has the same dimension as a flattened video frame:

$$
\phi_i \in \mathbb{C}^{h \cdot w}.
$$

Therefore, each mode can be reshaped into the shape of a video frame and visualized as an image.

A DMD mode is not usually a literal frame from the video. It is a spatial pattern paired with a particular temporal behavior. In the pendulum video:

* a nearly stationary mode may represent static background structure,
* oscillatory modes may emphasize the pendulum arm, bob, and swept path,
* complex-valued mode pairs may represent phase-shifted parts of the swing,
* and weak or unstable modes may represent noise, fitting artifacts, or small corrections.

The visual interpretation of DMD modes is one of the most useful parts of applying DMD to video. The method is not only producing a reconstruction or forecast; it is also producing spatial patterns tied to specific temporal behavior.

## Frequency Reporting in the Notebook

DMD eigenvalues are computed in discrete time. They describe what happens from one snapshot to the next.

To connect a DMD eigenvalue to a physical frequency, use

$$
\omega_i = \frac{\log(\lambda_i)}{\Delta t}.
$$

This conversion is not arbitrary. It comes from matching a continuous-time exponential mode to the same mode observed only at discrete sampling times.

In practical DMD, $\omega_i$ is not measured directly from a known continuous-time system. It is inferred from the learned DMD eigenvalue $\lambda_i$ and the timestep $\Delta t$.

Here:

* $\lambda_i$ is the discrete-time DMD eigenvalue,
* $\omega_i$ is the inferred continuous-time rate,
* and $\Delta t$ is the physical time between snapshots.

If

$$
\lambda_i = r_i e^{i\theta_i},
$$

then

$$
\omega_i
= \frac{\log(r_i)}{\Delta t}
+i\frac{\theta_i}{\Delta t}.
$$

The real part of $\omega_i$ gives the growth or decay rate:

$$
\mathrm{Re}(\omega_i)
= \frac{\log(r_i)}{\Delta t}.
$$

The imaginary part gives angular frequency:

$$
\mathrm{Im}(\omega_i)
= \frac{\theta_i}{\Delta t}.
$$

The physical frequency in cycles per second is

$$
f_i = \frac{\mathrm{Im}(\omega_i)}{2\pi}.
$$

The corresponding cycles per frame can be computed directly from the eigenvalue angle:

$$
f_{i,\mathrm{frame}} = \frac{\theta_i}{2\pi}.
$$

If the frequency is nonzero, the periods are

$$
T_{i,\mathrm{seconds}} = \frac{1}{|f_i|}
$$

and

$$
T_{i,\mathrm{frames}} = \frac{1}{|f_{i,\mathrm{frame}}|}.
$$

With $\Delta t$ measured in seconds:

- $\omega_i$ has units of inverse seconds,
- $\mathrm{Re}(\omega_i)$ is a growth or decay rate per second,
- $\mathrm{Im}(\omega_i)$ is an angular frequency in radians per second,
- $f_i$ is a frequency in cycles per second,
- $f_{i,\mathrm{frame}}$ is a frequency in cycles per frame,
- $T_{i,\mathrm{seconds}}$ is a period in seconds,
- and $T_{i,\mathrm{frames}}$ is a period in frames.

For the synthetic pendulum notebook, it is useful to report all four quantities:

* frequency in cycles per second,
* frequency in cycles per frame,
* period in seconds,
* and period in frames.

This makes the connection between physical time and frame-index time explicit.

There is one practical numerical warning. The complex logarithm is multi-valued because angles can differ by multiples of $2\pi$:

$$
\log(\lambda_i)
= \log(r_i) + i(\theta_i + 2\pi q),
\qquad q \in \mathbb{Z}.
$$

Most numerical code uses the principal branch of the logarithm, where the angle is restricted to a standard interval such as $(-\pi, \pi]$. This is usually reasonable when the sampling rate is high enough relative to the true oscillation frequency. If the system oscillates too quickly relative to the frame rate, frequency aliasing can occur.

For this synthetic pendulum example, the motion is slow relative to the video frame rate, so the principal-frequency interpretation should be reasonable.

## Reconstruction from DMD Modes

Once the DMD modes $\Phi$ and eigenvalues $\Lambda$ are known, the video can be approximated as a sum of modal contributions. The key idea is that repeated applications of the learned time-advance dynamics can be represented through the DMD modes and eigenvalues.

Conceptually, if the full transition operator were diagonalizable in terms of the DMD modes, we would write

$$
A \approx \Phi \Lambda \Phi^{-1}.
$$

Repeated application of $A$ means applying the one-step rule

$$
\vec{x}_k = A\vec{x}_{k-1}
$$

again and again. Starting from $\vec{x}_1$, this gives

$$
\vec{x}_k
= A^{k-1}\vec{x}_1.
$$

If the learned transition operator is approximately diagonalized by the DMD modes,

$$
A \approx \Phi\Lambda\Phi^{-1},
$$

then

$$
\vec{x}_k
= A^{k-1}\vec{x}_1
\approx
\Phi \Lambda^{k-1}\Phi^{-1}\vec{x}_1.
$$

This expression says that the state at time step $k$ can be approximated by decomposing the initial state into DMD modes, evolving each mode forward in time using the corresponding eigenvalue, and then mapping the result back into the original state space.

If $\Phi$ were square and invertible, the modal amplitude vector would be

$$
\vec{b} = \Phi^{-1}\vec{x}_1.
$$

In practice, $\Phi$ is usually not square, so we compute this using the Moore-Penrose pseudoinverse:

$$
\vec{b} = \Phi^\dagger \vec{x}_1.
$$

Here, $\Phi^\dagger$ is the Moore-Penrose pseudoinverse of $\Phi$. The vector $\vec{b}$ contains the modal amplitudes for the chosen starting frame, usually the first frame in the fitting window.

Then the DMD reconstruction formula is

$$
\hat{\vec{x}}_k \approx \Phi \Lambda^{k-1}\vec{b}.
$$

Equivalently, in modal-sum form,

$$
\hat{\vec{x}}_k
\approx
\sum_{i=1}^{r}
b_i \phi_i \lambda_i^{k-1}.
$$

This is the discrete-time analog of the continuous-time modal solution

$$
\vec{x}(t)
=\sum_{i=1}^{n}
b_i \vec{v}_i e^{\omega_i t}.
$$

The interpretation is parallel:

* $\phi_i$ is a DMD mode, or spatial pattern,
* $\lambda_i^{k-1}$ evolves that mode forward by $k-1$ discrete time steps,
* and $b_i$ determines how strongly the mode contributes to the starting frame.

Reconstruction asks:

> Can the learned DMD model reproduce the frames used during fitting?

If DMD is fit on frames

$$
\vec{x}_1, \vec{x}_2, \ldots, \vec{x}_m,
$$

then reconstruction uses the learned modes and eigenvalues to approximate those same frames.

Low reconstruction error indicates that the selected rank and modes capture the dominant structure in the fitting window. However, low reconstruction error alone does not guarantee good forecasting. A model can fit the observed frames well while still extrapolating poorly.

## Forecasting Held-Out Frames

Forecasting asks:

> Can the learned DMD model predict frames beyond the fitting window?

Forecasting is harder than reconstruction because the model must extrapolate beyond the data used to fit it. For periodic motion, short-horizon forecasts should remain coherent if the dominant DMD frequency is close to the true pendulum frequency. However, small errors in eigenvalue magnitude or phase can accumulate over time.

One way to forecast is to use the first fitting frame as the starting state and evaluate the DMD model beyond the fitting window.

Another way is to start from a later observed state, such as $\vec{x}_j$, compute a new amplitude vector,

$$
\vec{b}_j = \Phi^\dagger \vec{x}_j,
$$

and forecast $s$ steps forward:

$$
\hat{\vec{x}}_{j+s} \approx \Phi \Lambda^s \vec{b}_j.
$$

Here:

* $\vec{x}_j$ is the starting state,
* $s$ is the forecast horizon,
* $\Phi$ contains the DMD modes,
* $\Lambda$ contains the DMD eigenvalues,
* and $\vec{b}_j$ contains the modal amplitudes for the chosen starting state.

The choice of starting state matters for forecasting. There are several related ways to use the learned DMD model:

1. Fix the first snapshot as the initial condition and evaluate the model at later values of $k$.
2. Recompute the modal amplitudes from a more recent observed state and forecast forward from there.
3. Predict one step ahead, treat that prediction as the new starting point, and repeat the process iteratively.

The iterative approach is conceptually close to repeatedly applying the learned time-advance dynamics, but it can also accumulate error because each predicted frame becomes the input for the next prediction.

In the notebook, forecasting means using the learned DMD model to extrapolate beyond the fitting window and compare against held-out future frames.

Forecast quality can be evaluated visually and numerically. Useful checks include:

* side-by-side predicted and true frames,
* pixel-level error images,
* reconstruction and forecast error curves,
* dominant learned frequencies compared with the known pendulum frequency,
* and bob-centroid trajectories extracted from predicted frames.

## Background and Foreground Separation

The pendulum video contains both static and moving content. This makes it a natural setting for a simple DMD-based background/foreground experiment.

The basic idea is:

* slow or nearly stationary modes may represent background,
* oscillatory modes may represent pendulum motion,
* and the difference between the original frame and the background estimate can highlight moving foreground content.

A common practical approach is to identify modes with very low frequency or eigenvalues close to $1$. These modes change slowly from frame to frame and are candidates for background structure.

A simple DMD-based decomposition is:

$$
\text{background} \approx \text{low-frequency DMD reconstruction}
$$

and

$$
\text{foreground} \approx \text{original frame} - \text{background reconstruction}.
$$

For the synthetic pendulum, the stationary background should be captured by persistent modes, while the arm and bob should appear in the foreground residual.

For example, a background reconstruction can be formed using a subset of modes:

$$
\hat{\vec{x}}_{k,\mathrm{background}}
=\Phi_{\mathrm{background}}
\Lambda_{\mathrm{background}}^{k-1}
\vec{b}_{\mathrm{background}}.
$$

Then a foreground estimate can be formed from the residual:

$$
\hat{\vec{x}}_{k,\mathrm{foreground}}
=\vec{x}_k - \hat{\vec{x}}_{k,\mathrm{background}}.
$$

In practice, the foreground image may use the absolute value of the residual:

$$
|\vec{x}_k - \hat{\vec{x}}_{k,\mathrm{background}}|.
$$

This is not a perfect segmentation method. DMD is not being trained with foreground labels. The goal is to show how modal decomposition can separate slowly changing structure from coherent moving structure.

For the pendulum video, a reasonable result would show the static background in the low-frequency component and the moving arm and bob in the residual or oscillatory components.

## Bob-Centroid Motion Analysis

The synthetic pendulum gives an additional way to evaluate the DMD results: the bob position can be tracked through time.

DMD predicts pixels, not physical coordinates. However, after reconstructing or forecasting frames, we can estimate the bob centroid from those frames and compare it with the known or measured bob trajectory.

This turns the video forecast into a lower-dimensional trajectory comparison.

The notebook can compare:

* the true bob centroid from the generated data,
* the centroid estimated from original frames,
* the centroid estimated from DMD reconstructed frames,
* and the centroid estimated from DMD forecast frames.

If DMD captures the dominant pendulum motion well, then the predicted centroid should follow the same periodic path as the true bob, at least over a short forecast horizon.

This is useful because pixel-level errors can be hard to interpret. A centroid trajectory gives a more physical summary of the result.

For example, even if the predicted frame is slightly blurry, the predicted bob center may still follow the correct path. Conversely, a visually plausible forecast may still drift in phase, which would appear clearly in the centroid trajectory.

## Rank Sensitivity

The truncation rank $r$ controls how many DMD modes are retained.

This choice strongly affects reconstruction, forecasting, and interpretability.

A very small rank may capture only the strongest structures. This can produce a simple model, but it may miss details of the pendulum motion.

A moderate rank may capture the static background and the dominant oscillatory motion while ignoring weak noise or artifacts.

A very large rank may reconstruct the fitting window more accurately, but it may also retain unstable modes, noise, or small numerical artifacts. This can hurt forecasting.

The notebook will study rank sensitivity by fitting DMD with several values of $r$ and comparing:

* singular-value energy retained,
* reconstruction error,
* forecast error,
* dominant learned frequencies,
* eigenvalue stability,
* visual mode quality,
* and bob-centroid trajectory quality.

The goal is not simply to find the rank with the lowest reconstruction error. The goal is to understand the tradeoff between compression, interpretability, reconstruction, and forecasting.

## What the Notebook Will Evaluate

The notebook will use the synthetic pendulum video to evaluate DMD from several angles.

First, it will inspect the data and construct snapshot matrices.

Second, it will fit DMD using a chosen rank and examine the singular values, eigenvalues, modes, and inferred frequencies.

Third, it will reconstruct the fitting frames and measure how well the model represents the observed data.

Fourth, it will forecast held-out future frames and compare them against the known future frames.

Fifth, it will use selected DMD modes to explore background/foreground separation.

Sixth, it will estimate bob-centroid trajectories from reconstructed and forecast frames and compare them to the true motion.

Finally, it will repeat the workflow across several ranks to show how rank selection changes the learned modes, frequencies, reconstruction quality, and forecast quality.

The overall goal is to connect DMD theory to a visual, interpretable experiment:

> Learn spatial modes from a video, use eigenvalues to describe their time behavior, reconstruct and forecast frames, and compare the learned dynamics against known pendulum motion.
