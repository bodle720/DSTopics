### Step 1: Compute the SVD and Choose $r$

The first step is to compute a rank-r truncated SVD of the snapshot matrix $X$:

$$
X \approx U_r \Sigma_r V_r^*.
$$

Here:

- $U_r$ contains the first $r$ left singular vectors,
- $\Sigma_r$ contains the first $r$ singular values,
- $V_r$ contains the first $r$ right singular vectors,
- $V_r^*$ denotes the conjugate transpose of $V_r$,
- and $r$ is the chosen truncation rank.

The columns of $U_r$ store the $r$ dominant left singular vectors. These vectors define the reduced subspace and act as the key for transforming between the original $n$-dimensional state space and the lower-dimensional rank-r approximation.

The choice of $r$ is important and has been the subject of much research. A simple first approach is to inspect the singular values, often on a log plot, and choose a cutoff where the values decay sharply or become small. A more careful approach may use energy thresholds, noise-aware rank selection, cross-validation, or application-specific criteria.

A small $r$ may underfit by discarding meaningful dynamics. A large $r$ may retain noise, weak modes, or unstable behavior that hurts forecasting.

In this notebook, the singular values will be plotted before selecting a main rank, and later the effect of rank will be studied directly.

## Reduced Linear Operator

### Step 2: Find $\tilde{A}$

DMD avoids directly forming the large matrix $A$ by constructing a reduced operator

$$
\tilde{A} = U_r^* A U_r.
$$

This matrix represents the dynamics projected into the rank-r subspace spanned by the columns of $U_r$. It is a similarity-transform-style reduced representation of the full transition dynamics.

Using

$$
A = X'X^\dagger
$$

and the truncated SVD

$$
X \approx U_r \Sigma_r V_r^*,
$$

the reduced operator can be computed without explicitly forming $A$:

$$
\tilde{A} = U_r^* X' V_r \Sigma_r^{-1}.
$$

The matrix $\tilde{A}$ has shape

$$
r \times r.
$$

This is much smaller than the original $n \times n$ operator and is therefore easy to analyze.

## Eigen-Decomposition of the Reduced Operator

### Step 3: Compute the Eigenvectors and Eigenvalues of $\tilde{A}$

Next, compute the eigenvalues and eigenvectors of $\tilde{A}$:

$$
\tilde{A}W = W\Lambda.
$$

Here:

- the columns of $W$ are eigenvectors of $\tilde{A}$,
- $\Lambda$ is a diagonal matrix containing the DMD eigenvalues,
- and the entries of $\Lambda$ approximate the important eigenvalues of the full time-advance operator.

In discrete-time DMD, the eigenvalues describe how modes evolve from one snapshot to the next.

If $\lambda_i$ is a DMD eigenvalue, then:

- $|\lambda_i| < 1$ suggests a decaying mode,
- $|\lambda_i| > 1$ suggests a growing mode,
- $|\lambda_i| \approx 1$ suggests a persistent mode,
- complex conjugate eigenvalues often correspond to oscillatory behavior.

For a pendulum, oscillatory behavior is exactly what we expect. Therefore, a strong DMD result should reveal eigenvalues associated with the swinging motion.

## DMD Modes: Mapping Back to the Original State Space

### Step 4: Map the Results Back to the Original Space

The eigenvectors stored in $W$ are eigenvectors of the reduced operator $\tilde{A}$. They live in the reduced $r$-dimensional coordinate system, not in the original $n$-dimensional state space.

However, our original data lives in $n$-dimensional space. For video data, $n$ is the number of pixels in a flattened frame. Therefore, to interpret the learned eigenvectors as spatial patterns in the original data, we need to map the reduced eigenvectors back to the original state space.

To obtain the DMD modes, we compute

$$
\Phi = A U_r W.
$$

Here, $U_r W$ maps the reduced eigenvectors back toward the original state space, and the multiplication by $A$ advances those mapped vectors through the learned time dynamics.

In practice, we do not explicitly form the full matrix $A$. Instead, we use the data-driven expression

$$
\Phi = X' V_r \Sigma_r^{-1} W.
$$

These two expressions are connected through the truncated SVD approximation used in DMD.

Recall that

$$
X \approx U_r \Sigma_r V_r^*.
$$

Using this approximation, the least-squares transition operator can be written approximately as

$$
A \approx X' V_r \Sigma_r^{-1} U_r^*.
$$

Therefore,

$$
A U_r W \approx X' V_r \Sigma_r^{-1} U_r^* U_r W.
$$

Since the columns of $U_r$ are orthonormal,

$$
U_r^* U_r = I.
$$

So,

$$
A U_r W \approx X' V_r \Sigma_r^{-1} W.
$$

This gives the exact DMD mode formula:

$$
\Phi = X' V_r \Sigma_r^{-1} W.
$$

The columns of $\Phi$ are called the **DMD modes**:

$$
\Phi =[\phi_1 \; \phi_2 \; \cdots \; \phi_r].
$$

Each mode $\phi_i$ is an $n$-dimensional spatial pattern in the original state space. For video data, this means each DMD mode can be reshaped back into the shape of a video frame and viewed as an image.

This mapping step is important. The reduced eigenvectors $W$ describe dynamics inside the low-dimensional SVD coordinate system. The DMD modes $\Phi$ describe corresponding spatial patterns in the original high-dimensional data space.

The multiplication by $A$, or equivalently the data-driven replacement $X' V_r \Sigma_r^{-1}$, is what produces the exact DMD modes. These modes can be shown to approximate eigenvectors of the original full transition operator $A$, while the entries of $\Lambda$ are the associated DMD eigenvalues.

So, after this step, we have the essential modal ingredients:

- DMD modes $\Phi$,
- DMD eigenvalues $\Lambda$,
- and a way to represent repeated multiplication by the learned transition dynamics.

In the pendulum video, this means:

- some DMD modes may resemble static background structure,
- some modes may emphasize the pendulum arm and bob,
- complex-valued oscillatory modes may capture phase-shifted parts of the swing,
- and the corresponding eigenvalues describe how those spatial patterns evolve through time.

For video data, this visual interpretation is one of the most useful parts of DMD. The method is not only producing a reconstruction or forecast; it is also producing spatial modes tied to specific temporal behavior.

## Reconstruction and Forecasting from DMD Modes

Once the DMD modes $\Phi$ and eigenvalues $\Lambda$ are known, we can use them to represent repeated applications of the learned time-advance dynamics.

Conceptually, if the full transition operator were diagonalizable in terms of the DMD modes, we would write

$$
A \approx \Phi \Lambda \Phi^{-1}.
$$

Repeated application of $A$ then gives

$$
\vec{x}_k = A\vec{x}_{k-1} \approx \Phi \Lambda^{k-1} \Phi^{-1}\vec{x}_1.
$$

This expression says that the state at time step $k$ can be approximated by decomposing the initial state into DMD modes, evolving each mode forward in time using the corresponding eigenvalue, and then mapping the result back into the original state space.

We define the modal amplitude vector

$$
\vec{b} = \Phi^{-1}\vec{x}_1.
$$

In practice, $\Phi$ is usually not square, so we compute this using the Moore-Penrose pseudoinverse:

$$
\vec{b} = \Phi^\dagger \vec{x}_1.
$$

Then the DMD reconstruction formula becomes

$$
\vec{x}_k \approx \Phi \Lambda^{k-1} \vec{b}.
$$

Equivalently, in modal-sum form,

$$
\vec{x}_k \approx \sum_{i=1}^{r} \phi_i \lambda_i^{k-1} b_i.
$$

This is the discrete-time analog of the continuous-time modal solution

$$
\vec{x}(t)
= \sum_{i=1}^{n}
b_i \vec{v}_i e^{\lambda_i t}.
$$

The interpretation is parallel:

- $\phi_i$ is a DMD mode, or spatial pattern,
- $\lambda_i^{k-1}$ describes how that mode evolves after $k-1$ discrete time steps,
- $b_i$ is the amplitude of that mode for the chosen initial condition.

This formula can be used for two related tasks.

### Reconstruction

Reconstruction asks:

> Can the learned DMD model reproduce the frames used during fitting?

If DMD is fit on frames

$$
\vec{x}_1, \vec{x}_2, \dots, \vec{x}_m,
$$

then reconstruction uses the learned modes and eigenvalues to approximate those same frames.

Low reconstruction error indicates that the selected rank and modes capture the dominant structure in the fitting window.

### Forecasting

Forecasting asks:

> Can the learned DMD model predict frames beyond the fitting window?

Forecasting is harder because the model must extrapolate beyond the data used to fit it. For periodic motion, short-horizon forecasts should remain coherent if the dominant DMD frequency is close to the true pendulum frequency. However, small errors in eigenvalue magnitude or phase can accumulate over time.

It is important to note that the initial condition does not have to be the first snapshot $\vec{x}_1$.

We can also start from a later observed state, such as $\vec{x}_j$, and compute a new amplitude vector

$$
\vec{b}_j = \Phi^\dagger \vec{x}_j.
$$

Then an $s$-step forecast from that state is

$$
\vec{x}_{j+s} \approx \Phi \Lambda^s \vec{b}_j.
$$

Here:

- $\vec{x}_j$ is the starting state,
- $s$ is the forecast horizon,
- $\Phi$ contains the DMD modes,
- $\Lambda$ contains the DMD eigenvalues,
- and $\vec{b}_j$ contains the modal amplitudes for the chosen starting state.

This distinction matters for forecasting. There are several related ways to use the learned DMD model:

1. We can fix the first snapshot as the initial condition and vary $k$.
2. We can recompute the modal amplitudes from a more recent observed state and forecast forward from there.
3. We can predict one step ahead, treat that prediction as the new starting point, and repeat the process iteratively.

The iterative approach is conceptually close to repeatedly applying the learned time-advance dynamics, but it can also accumulate error because each predicted frame becomes the input for the next prediction.

In this notebook, reconstruction means using the learned DMD model to reproduce frames from the fitting window. Forecasting means using the learned model to extrapolate beyond the fitting window and compare against held-out future frames.

---