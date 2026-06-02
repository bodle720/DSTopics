# Part 4: Applying DMD to Pendulum Video Motion

This note explains how **Dynamic Mode Decomposition (DMD)** is applied to the synthetic pendulum video notebook.

The earlier notes introduced modes, eigenvalues, matrix exponentials, continuous-time and discrete-time dynamics, frequency interpretation, and the relationship between DMD and video data. This note connects those ideas to the actual pendulum experiment.

The central idea is that a video is high-dimensional when represented as pixels, but the motion that generates the video may be much lower-dimensional. A pendulum is a useful example because its motion is smooth, coherent, and periodic. Each frame contains many pixel values, but the underlying motion is mostly governed by a small number of quantities: angle, angular velocity, bob position, and time.

The notebook uses this controlled visual dynamical system to study:

* DMD snapshot matrices,
* low-rank approximation through SVD,
* DMD eigenvalues, modes, frequencies, and growth rates,
* full-frame reconstruction behavior,
* the limitations of pixel-space DMD,
* delay-coordinate DMD on the bob trajectory,
* coordinate-level reconstruction and forecasting,
* and the effect of rank selection on learned dynamics.

The goal is not merely to run DMD as a black-box algorithm. The goal is to connect the implementation back to state vectors, linear time-advance operators, eigenvalues, modes, modal amplitudes, SVD truncation, and frequency interpretation.

A previous version of this project applied DMD to financial time-series data. That was useful as an exploratory sequential-modeling exercise, but financial data is noisy, non-stationary, difficult to validate visually, and affected by many unobserved variables. The synthetic pendulum is a cleaner instructional example because the true motion is known, the future trajectory is known, and the model output can be evaluated visually and numerically.

## 1. DMD in One-Step Prediction Form

DMD learns an approximate one-step linear model from snapshot data:

$$
\vec{x}_{k+1} \approx A\vec{x}_k.
$$

Here:

* $\vec{x}_k$ is the observed state at time step $k$,
* $\vec{x}_{k+1}$ is the next observed state,
* and $A$ is an unknown linear operator that advances the state by one sampled step.

This mirrors the modal-solution idea from Part 1: a state can be represented as a sum of spatial modes, each with its own time behavior and amplitude. DMD keeps that modal viewpoint, but estimates the modes and eigenvalues from snapshot data rather than from a known system matrix.

For the full-frame part of the notebook, each state vector is a flattened grayscale video frame. If a frame has height $h$ and width $w$, then

$$
\vec{x}_k \in \mathbb{R}^{h \cdot w}.
$$

The full transition matrix $A$ would therefore have shape

$$
(h \cdot w) \times (h \cdot w).
$$

Even for modest image sizes, this matrix is too large to form and analyze directly. DMD avoids this by estimating the dominant dynamics in a lower-dimensional subspace.

The standard DMD formulation used here follows the formulation described in *Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control* by Brunton and Kutz.

## 2. Pendulum Frames as State Vectors

The input to DMD is a sequence of video frames:

$$
\mathrm{frame}_1, \mathrm{frame}_2, \ldots, \mathrm{frame}_m.
$$

Each frame is converted to grayscale and flattened into a column vector:

$$
\vec{x}_1, \vec{x}_2, \ldots, \vec{x}_m.
$$

This turns the video into a sequence of high-dimensional points. DMD then tries to find a linear rule that approximately advances one point to the next.

For the pendulum video, this is useful because the frames are high-dimensional but organized. The bob and arm move coherently through time, while much of the background is static. DMD attempts to represent the video as a sum of spatial patterns with simple temporal behavior.

In the notebook, the full-frame DMD model is fit to mean-centered frames:

$$
\vec{y}_k = \vec{x}_k - \bar{\vec{x}}.
$$

Mean-centering removes the average frame so that the DMD model focuses more directly on motion and deviations from the static background.

## 3. Snapshot Matrices

Suppose we observe $m$ snapshots:

$$
\vec{x}_1, \vec{x}_2, \ldots, \vec{x}_m.
$$

DMD forms two snapshot matrices:

$$
X = [\vec{x}_1 \quad \vec{x}_2 \quad \cdots \quad \vec{x}_{m-1}]
$$

and

$$
X' = [\vec{x}_2 \quad \vec{x}_3 \quad \cdots \quad \vec{x}_m].
$$

The columns of $X$ contain the earlier snapshots. The columns of $X'$ contain the same sequence shifted forward by one time step.

The DMD assumption is

$$
X' \approx AX.
$$

Equivalently,

$$
A[\vec{x}_1 \quad \vec{x}_2 \quad \cdots \quad \vec{x}_{m-1}] = [A\vec{x}_1 \quad A\vec{x}_2 \quad \cdots \quad A\vec{x}_{m-1}].
$$

So DMD is trying to find one linear operator such that

$$
A\vec{x}_j \approx \vec{x}_{j+1}, \qquad j = 1, 2, \ldots, m-1.
$$

In the mean-centered full-frame setup, the same construction is applied to centered snapshots:

$$
Y = [\vec{y}_1 \quad \vec{y}*2 \quad \cdots \quad \vec{y}_{m-1}]
$$

and

$$
Y' = [\vec{y}_2 \quad \vec{y}_3 \quad \cdots \quad \vec{y}_m].
$$

Then the learned relationship is

$$
Y' \approx AY.
$$

Each column of $Y'$ is one sampled timestep after the corresponding column of $Y$.

## 4. Choosing the Fitting Window and Timestep

Two practical choices matter before fitting DMD:

1. how many frames to use,
2. what timestep to associate with consecutive snapshots.

The number of frames controls how much temporal behavior is available for fitting. If the fitting window is too short, DMD may not see enough of the pendulum swing to identify the dominant oscillation. If the fitting window is too long, the data may include behavior that is not well approximated by one fixed linear time-advance operator.

For the synthetic pendulum, the situation is cleaner because the system is controlled, periodic, and generated by a stable rule. Even here, the fitting window should include enough of the swing to learn the dominant oscillation. Fitting on only a tiny fraction of a period may not reveal the full periodic motion, while fitting on one or more full periods gives DMD a better chance to identify the oscillatory structure.

Standard DMD assumes the snapshots are sampled at a constant timestep. In this notebook, the snapshots are evenly spaced video frames, so each DMD step has a clear physical meaning.

If every video frame is used and the video has frame rate $\mathrm{fps}$, then

$$
\Delta t = \frac{1}{\mathrm{fps}}.
$$

For example, at 30 frames per second, consecutive frames are separated by $\Delta t = 1/30$ seconds.

If the video is downsampled by keeping every second frame, then

$$
\Delta t = \frac{2}{\mathrm{fps}}.
$$

More generally, if every $q$-th frame is kept, then

$$
\Delta t = \frac{q}{\mathrm{fps}}.
$$

The DMD operator estimates one-step evolution between the snapshots it sees. Therefore, $\Delta t$ determines what "one step forward" means physically.

## 5. Least-Squares View of the Transition Matrix

The most direct way to estimate $A$ is to solve

$$
X' \approx AX.
$$

Using the Moore-Penrose pseudoinverse of $X$, denoted $X^\dagger$, the least-squares solution is

$$
A = X'X^\dagger.
$$

This $A$ is the matrix that best fits the observed transitions from one snapshot to the next in a least-squares sense:

$$
A\vec{x}_j \approx \vec{x}_{j+1}.
$$

This formulation is often referred to as **exact DMD**.

The least-squares equation does not explicitly depend on the physical timestep. However, $\Delta t$ is still important because it determines the physical meaning of one step forward and is needed when interpreting eigenvalues as growth rates, frequencies, and periods.

The direct solution is conceptually simple but usually not practical for image data. If each flattened frame has dimension $n$, then $A$ has shape

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

## 6. Low-Rank Structure and the SVD

The key practical assumption is that the high-dimensional video has useful low-rank structure.

Even though each frame contains many pixels, most of the important variation may be described by a smaller number of dominant spatiotemporal patterns. For the pendulum video, this is plausible because the background is mostly static and the moving pendulum follows a coherent periodic path.

DMD uses the Singular Value Decomposition (SVD) to identify a low-dimensional subspace for the data. This is closely related to Principal Component Analysis (PCA). PCA uses the SVD to identify dominant directions of variation in data. DMD also uses the SVD, but then goes further by estimating how the reduced coordinates evolve through time.

See my [notebook on PCA](/PCA_linear/pca_linear_oscillation_system.ipynb) for a related discussion of dimensionality reduction and low-rank approximation.

The first computational step is to compute a rank $r$ truncated SVD of the snapshot matrix:

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

The choice of $r$ is important. A small value of $r$ may underfit by discarding meaningful dynamics. A large value of $r$ may retain noise, weak modes, or unstable behavior that hurts interpretation or forecasting.

A simple first approach is to inspect the singular values, often on a log plot, and choose a cutoff where the values decay sharply or become small. More careful approaches may use cumulative-energy thresholds, noise-aware rank selection, cross-validation, or application-specific criteria.

In the notebook, singular values and rank-frequency diagnostics are used to choose a practical full-frame DMD rank.

## 7. Practical DMD Algorithm

The practical DMD procedure is:

1. Build snapshot matrices $X$ and $X'$.
2. Compute a rank $r$ truncated SVD:

$$
X \approx U_r \Sigma_r V_r^*.
$$

3. Build the reduced operator $\tilde{A}$:

$$
\tilde{A} = U_r^*X'V_r\Sigma_r^{-1}.
$$

4. Compute the eigenvalues and eigenvectors of $\tilde{A}$:

$$
\tilde{A}W = W\Lambda.
$$

5. Map the reduced eigenvectors back to the original state space to obtain DMD modes:

$$
\Phi = X'V_r\Sigma_r^{-1}W.
$$

After these steps, the main DMD ingredients are:

* $\Phi$, the DMD modes,
* $\Lambda$, the diagonal matrix of DMD eigenvalues,
* and modal amplitudes, which describe how strongly each mode contributes to a chosen starting frame.

## 8. Reduced Linear Operator

The reduced operator is the smaller matrix actually computed and analyzed in practical DMD. Conceptually, it approximates the full transition matrix projected into the rank $r$ subspace:

$$
\tilde{A} \approx U_r^*AU_r.
$$

Here, $U_r^*$ projects original state-space vectors down into reduced coordinates, and $U_r$ maps reduced coordinates back toward the original state space.

The full least-squares transition matrix is

$$
A = X'X^\dagger.
$$

In high-dimensional image data, forming this full matrix is usually too expensive. Using the truncated SVD approximation,

$$
X \approx U_r\Sigma_rV_r^*,
$$

DMD computes the reduced operator directly from the snapshot matrices:

$$
\tilde{A} = U_r^*X'V_r\Sigma_r^{-1}.
$$

The matrix $\tilde{A}$ has shape

$$
r \times r.
$$

This is the key computational shortcut. Instead of forming the huge matrix $A$ in the original pixel space, DMD forms a much smaller matrix in the reduced SVD coordinate system.

The eigenvalues of $\tilde{A}$ are the DMD eigenvalues. They approximate the dominant eigenvalues of the unknown full time-advance operator and describe the learned frame-to-frame behavior of the DMD modes.

## 9. Eigenvalues of the Reduced Operator

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

For the pendulum video, the most important non-background behavior should be oscillatory. Therefore, an interpretable DMD result should reveal one or more complex-conjugate eigenvalue pairs associated with the swinging motion.

## 10. DMD Modes in the Original Pixel Space

The eigenvectors in $W$ are eigenvectors of the reduced operator $\tilde{A}$. They live in the reduced rank $r$ coordinate system, not in the original $n$-dimensional state space.

For video data, the original state space is flattened pixel space. If each frame has height $h$ and width $w$, then each state vector has dimension

$$
n = h \cdot w.
$$

To interpret reduced eigenvectors as image-like spatial patterns, they must be mapped back to the original pixel space.

Conceptually, the DMD modes can be written as

$$
\Phi = AU_rW.
$$

In practice, we do not explicitly form the full matrix $A$. Using the truncated SVD approximation,

$$
X \approx U_r\Sigma_rV_r^*,
$$

the least-squares transition operator can be approximated as

$$
A \approx X'V_r\Sigma_r^{-1}U_r^*.
$$

Therefore,

$$
AU_rW \approx X'V_r\Sigma_r^{-1}U_r^*U_rW.
$$

Since the columns of $U_r$ are orthonormal,

$$
U_r^*U_r = I.
$$

So the exact DMD mode formula used in practice is

$$
\Phi = X'V_r\Sigma_r^{-1}W.
$$

The columns of $\Phi$ are the DMD modes:

$$
\Phi = [\phi_1 \quad \phi_2 \quad \cdots \quad \phi_r].
$$

Each mode has the same dimension as a flattened video frame:

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

## 11. Frequency Reporting in the Notebook

DMD eigenvalues are computed in discrete time. They describe what happens from one snapshot to the next.

To connect a DMD eigenvalue to a physical frequency, use

$$
\omega_i = \frac{\log(\lambda_i)}{\Delta t}.
$$

This conversion comes from matching a continuous-time exponential mode to the same mode observed only at discrete sampling times.

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
\omega_i = \frac{\log(r_i)}{\Delta t} + i\frac{\theta_i}{\Delta t}.
$$

The real part gives the growth or decay rate:

$$
\mathrm{Re}(\omega_i) = \frac{\log(r_i)}{\Delta t}.
$$

The imaginary part gives angular frequency:

$$
\mathrm{Im}(\omega_i) = \frac{\theta_i}{\Delta t}.
$$

The physical frequency in cycles per second is

$$
f_i = \frac{\mathrm{Im}(\omega_i)}{2\pi}.
$$

The corresponding frequency in cycles per frame can be computed directly from the eigenvalue angle:

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

* $\omega_i$ has units of inverse seconds,
* $\mathrm{Re}(\omega_i)$ is a growth or decay rate per second,
* $\mathrm{Im}(\omega_i)$ is an angular frequency in radians per second,
* $f_i$ is a frequency in cycles per second,
* $f_{i,\mathrm{frame}}$ is a frequency in cycles per frame,
* $T_{i,\mathrm{seconds}}$ is a period in seconds,
* and $T_{i,\mathrm{frames}}$ is a period in frames.

For the synthetic pendulum notebook, it is useful to report frequency in cycles per second, frequency in cycles per frame, period in seconds, and period in frames. This makes the connection between physical time and frame-index time explicit.

There is one practical numerical warning. The complex logarithm is multi-valued because angles can differ by multiples of $2\pi$:

$$
\log(\lambda_i) = \log(r_i) + i(\theta_i + 2\pi q), \qquad q \in \mathbb{Z}.
$$

Most numerical code uses the principal branch of the logarithm, where the angle is restricted to a standard interval such as $(-\pi, \pi]$. This is usually reasonable when the sampling rate is high enough relative to the true oscillation frequency. If the system oscillates too quickly relative to the frame rate, frequency aliasing can occur.

For this synthetic pendulum example, the motion is slow relative to the video frame rate, so the principal-frequency interpretation is reasonable.

## 12. Reconstruction from DMD Modes

Once the DMD modes $\Phi$ and eigenvalues $\Lambda$ are known, the video can be approximated as a sum of modal contributions. Repeated applications of the learned time-advance dynamics are represented through the DMD modes and eigenvalues.

Conceptually, if the full transition operator were diagonalizable in terms of the DMD modes, we would write

$$
A \approx \Phi\Lambda\Phi^{-1}.
$$

Starting from $\vec{x}_1$, repeated application of $A$ gives

$$
\vec{x}_k = A^{k-1}\vec{x}_1.
$$

If the learned transition operator is approximately diagonalized by the DMD modes, then

$$
\vec{x}_k \approx \Phi\Lambda^{k-1}\Phi^{-1}\vec{x}_1.
$$

This expression says that the state at time step $k$ can be approximated by decomposing the initial state into DMD modes, evolving each mode forward using its eigenvalue, and mapping the result back into the original state space.

If $\Phi$ were square and invertible, the modal amplitude vector would be

$$
\vec{b} = \Phi^{-1}\vec{x}_1.
$$

In practice, $\Phi$ is usually not square, so the amplitudes are computed using the Moore-Penrose pseudoinverse:

$$
\vec{b} = \Phi^\dagger\vec{x}_1.
$$

The DMD reconstruction formula is then

$$
\hat{\vec{x}}_k \approx \Phi\Lambda^{k-1}\vec{b}.
$$

Equivalently, in modal-sum form,

$$
\hat{\vec{x}}_k \approx \sum_{i=1}^{r} b_i\phi_i\lambda_i^{k-1}.
$$

This is the discrete-time analog of the continuous-time modal solution

$$
\vec{x}(t) = \sum_{i=1}^{n} b_i\vec{v}_i e^{\omega_i t}.
$$

The interpretation is parallel:

* $\phi_i$ is a DMD mode, or spatial pattern,
* $\lambda_i^{k-1}$ evolves that mode forward by $k-1$ discrete time steps,
* and $b_i$ determines how strongly the mode contributes to the starting frame.

Reconstruction asks:

> Can the learned DMD model reproduce the frames used during fitting?

Low reconstruction error indicates that the selected rank and modes capture dominant structure in the fitting window. However, low reconstruction error alone does not guarantee good forecasting. A model can fit observed frames well while still extrapolating poorly.

In the finalized pendulum notebook, this distinction is important: the full-frame DMD model finds meaningful modes and near-correct frequencies, but its important motion modes are strongly damped, so the reconstruction decays toward the mean frame.

## 13. Forecasting and Starting-State Choices

Forecasting asks:

> Can the learned DMD model predict states beyond the fitting window?

Forecasting is harder than reconstruction because the model must extrapolate beyond the data used to fit it. For periodic motion, short-horizon forecasts should remain coherent if the dominant DMD frequency is close to the true pendulum frequency and the relevant eigenvalues have appropriate magnitudes. However, small errors in eigenvalue magnitude or phase can accumulate over time.

One way to forecast is to use the first fitting snapshot as the starting state and evaluate the DMD model beyond the fitting window.

Another way is to start from a later observed state, such as $\vec{x}_j$, compute a new amplitude vector,

$$
\vec{b}_j = \Phi^\dagger\vec{x}_j,
$$

and forecast $s$ steps forward:

$$
\hat{\vec{x}}_{j+s} \approx \Phi\Lambda^s\vec{b}_j.
$$

Here:

* $\vec{x}_j$ is the starting state,
* $s$ is the forecast horizon,
* $\Phi$ contains the DMD modes,
* $\Lambda$ contains the DMD eigenvalues,
* and $\vec{b}_j$ contains the modal amplitudes for the chosen starting state.

There are several related ways to use a learned DMD model:

1. Fix the first snapshot as the initial condition and evaluate the model at later values of $k$.
2. Recompute modal amplitudes from a more recent observed state and forecast forward from there.
3. Predict one step ahead, treat that prediction as the new starting point, and repeat the process iteratively.

The iterative approach is close to repeatedly applying the learned time-advance dynamics, but it can accumulate error because each predicted state becomes the input for the next prediction.

In the finalized notebook, the successful forecasting result comes from delay-coordinate DMD on the bob trajectory rather than from full-frame pixel DMD. This is the main modeling pivot: changing the state representation makes the learned linear dynamics much more accurate.

## 14. Background and Foreground Separation as an Extension

The pendulum video contains both static and moving content, so it is a natural setting for a simple DMD-based background/foreground experiment.

The basic idea is:

* slow or nearly stationary modes may represent background,
* oscillatory modes may represent pendulum motion,
* and the difference between the original frame and the background estimate can highlight moving foreground content.

A common practical approach is to identify modes with very low frequency or eigenvalues close to $1$. These modes change slowly from frame to frame and are candidates for background structure.

A simple DMD-based decomposition is

$$
\mathrm{background} \approx \mathrm{low\text{-}frequency\ DMD\ reconstruction}
$$

and

$$
\mathrm{foreground} \approx \mathrm{original\ frame} - \mathrm{background\ reconstruction}.
$$

For example, a background reconstruction can be formed using a subset of modes:

$$
\hat{\vec{x}}_{k,\mathrm{background}} = \Phi_{\mathrm{background}}\Lambda_{\mathrm{background}}^{k-1}\vec{b}_{\mathrm{background}}.
$$

Then a foreground estimate can be formed from the residual:

$$
\hat{\vec{x}}_{k,\mathrm{foreground}} = \vec{x}_k - \hat{\vec{x}}_{k,\mathrm{background}}.
$$

In practice, the foreground image may use the absolute value of the residual:

$$
|\vec{x}_k - \hat{\vec{x}}_{k,\mathrm{background}}|.
$$

This is not a perfect segmentation method. DMD is not trained with foreground labels. The idea is only to show how modal decomposition can separate slowly changing structure from coherent moving structure.

For the finalized notebook, this is best treated as a possible extension rather than a main result.

## 15. Bob-Centroid and Coordinate-Level Motion Analysis

The synthetic pendulum gives an additional way to evaluate DMD results: the bob position is known through time.

DMD on full video frames predicts pixels, not physical coordinates. However, the generated data includes the true bob trajectory, so the notebook can compare learned dynamics to a lower-dimensional physical summary of the motion.

This turns the video problem into a trajectory problem.

Useful comparisons include:

* the true bob centroid from the generated data,
* the trajectory implied by reconstructed or forecast states,
* and the trajectory predicted by a coordinate-level DMD model.

This is useful because pixel-level errors can be hard to interpret. A centroid trajectory gives a more physical summary of the result. A visually blurry frame might still preserve the bob's approximate path, while a visually plausible frame might still drift in phase.

In the finalized notebook, the coordinate-level analysis becomes the successful modeling pivot. A delay-coordinate state built from the bob trajectory is much closer to the true low-dimensional pendulum dynamics than the full pixel frame. This representation recovers the true oscillation and forecasts the bob trajectory much more accurately.

## 16. Rank Sensitivity

The truncation rank $r$ controls how many DMD modes are retained.

This choice strongly affects reconstruction, forecasting, and interpretability.

A very small rank may capture only the strongest structures. This can produce a simple model, but it may miss details of the pendulum motion.

A moderate rank may capture the static background and dominant oscillatory motion while ignoring weak noise or artifacts.

A very large rank may reconstruct the fitting window more accurately, but it may also retain unstable modes, noise, or small numerical artifacts. This can hurt forecasting and physical interpretation.

Rank selection can be studied by comparing:

* singular-value energy retained,
* reconstruction error,
* forecast error where forecasting is evaluated,
* dominant learned frequencies,
* eigenvalue stability,
* visual mode quality,
* and bob-trajectory quality.

The goal is not simply to find the rank with the lowest reconstruction error. The goal is to understand the tradeoff between compression, interpretability, reconstruction, and dynamical behavior.

## 17. What the Notebook Evaluates

The notebook uses the synthetic pendulum video to evaluate DMD from several angles.

First, it generates and inspects the data.

Second, it constructs full-frame snapshot matrices from mean-centered grayscale video frames.

Third, it chooses a full-frame DMD rank using singular values and rank-frequency diagnostics.

Fourth, it fits full-frame DMD and examines eigenvalues, modes, damping, and inferred frequencies.

Fifth, it reconstructs observed motion and diagnoses why the full-frame model decays toward the mean frame.

Sixth, it pivots to delay-coordinate DMD on the pendulum bob trajectory.

Seventh, it compares coordinate-DMD reconstruction and forecasting across ranks.

The overall goal is to connect DMD theory to a visual, interpretable experiment:

> Learn spatial modes from video, use eigenvalues to describe their time behavior, diagnose where pixel-space DMD succeeds and fails, and show how a better state representation can recover the underlying pendulum motion.

See part 5 next, [Nyquist Frequency and DMD](05_nyquist_frequency_and_dmd.md).