# Part 4: Applying DMD: Dynamic Mode Decomposition for Pendulum Video Motion

This note discusses the application of **Dynamic Mode Decomposition (DMD)** to a synthetic pendulum video sequence.
See the [notebook](/DMD/DMD_pendulum_video.ipynb) for the code implementation.

The central idea is that a video is high-dimensional when represented as raw pixels, but the motion generating the video may be much lower-dimensional. A pendulum is a useful example because its motion is smooth, coherent, and periodic. Each frame contains many pixel values, but the true underlying motion is mostly governed by a small number of physical quantities: angle, angular velocity, bob position, and time.

The goal of this notebook is to use a controlled visual dynamical system to study:

- DMD theory and snapshot matrices,
- low-rank approximation through SVD,
- DMD eigenvalues, modes, frequencies, and growth rates,
- reconstruction of observed video frames,
- forecasting of held-out future frames,
- background/foreground separation,
- bob-centroid motion analysis,
- and the effect of rank selection on reconstruction and forecasting quality.

The notebook is meant to be instructional. The goal is not merely to run DMD as a black-box algorithm, but to connect the implementation back to the mathematics: state vectors, linear time-advance operators, eigenvalues, eigenvectors, modal amplitudes, SVD truncation, frequency interpretation, and the relationship between continuous-time and discrete-time dynamical systems.

A previous version of this notebook applied DMD to financial time-series data. That was useful as an exploratory sequential-modeling exercise, but financial data is noisy, non-stationary, difficult to validate visually, and strongly affected by variables that are not observed. The synthetic pendulum is a better instructional example because the true motion is known, the future frames are known, and the model output can be evaluated both visually and numerically.

## Dynamic Mode Decomposition (DMD)

DMD seeks to approximate the dynamics of a system using a linear transformation learned from data. DMD's results can be used for predicting the next step in a dynamic system. In its simplest discrete-time form, DMD tries to learn a linear time-advance model

$$
\vec{x}_{k+1} \approx A\vec{x}_k,
$$

where $\vec{x}_k$ is the observed state of the system at time step $k$, and $A$ is an unknown linear operator that advances the system forward by one time step.

In this sense, DMD is directly tied to next-step prediction. If the learned operator captures the dominant dynamics well, then applying it to the current state gives an approximation of the next state. Repeated application can then be used for reconstruction or forecasting.

The method is useful when a system is evolving through time and appears to contain underlying structures that influence its progression. For video processing, each frame represents one sample in time. For this notebook, those samples are pendulum frames. In other contexts, the samples could be fluid snapshots, sensor measurements, or financial observations.

This is a deliberately simplified model. Many real systems are nonlinear, noisy, and only partially observed. However, DMD is useful because it can often identify dominant spatiotemporal patterns even when the full governing equations are unknown.

A few common applications of DMD include:

- fluid dynamics,
- video processing,
- oscillatory systems,
- dynamical systems analysis,
- control,
- and exploratory time-series modeling.

In each case, a system evolves through time and is assumed to contain some underlying structure that influences its progression. For video processing, each frame of a video represents a single sample in time. In this notebook, each video frame will be flattened into a vector, so the pendulum video becomes a sequence of high-dimensional state vectors.

The introduction and overview that follows are based heavily on the standard DMD formulation described in *Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control* by Brunton and Kutz. For a deeper overview, that text is a very useful reference.

## Continuous-Time Motivation

Let $\vec{x}$ represent the state of a system. The state vector contains the quantities we wish to measure or model. In this notebook, $\vec{x}$ will eventually represent the pixels in a single video frame. More generally, $\vec{x}$ could represent fluid velocities, temperatures, sensor readings, stock-market features, or any collection of measured quantities.

A general continuous-time dynamical system can be written as

$$
\frac{d\vec{x}}{dt} = f(\vec{x}, t, \vec{\mu}),
$$

where:

- $\vec{x}$ is the state vector,
- $t$ is time,
- $\vec{\mu}$ represents possible system parameters,
- and $f$ describes how the state changes.

The left-hand side,

$$
\frac{d\vec{x}}{dt},
$$

is the derivative of the state vector. It describes precisely how the state changes at a given time $t$.

In many real systems, the function $f$ may be nonlinear, unknown, difficult to model from first principles, or dependent on unobserved variables. Discovering and studying $f$ is highly context-dependent and can represent an entire field of study.

One crucial simplification behind DMD is that we approximate the observed dynamics with a linear model. Instead of trying to discover the full nonlinear function $f$, DMD asks whether the observed evolution can be approximated by a linear operator.

A simplified continuous-time linear system has the form

$$
\frac{d\vec{x}}{dt} = A\vec{x}.
$$

Here, $A$ is a matrix that describes how the components of the state interact linearly.

If $\vec{x} \in \mathbb{R}^n$, then $A$ has shape

$$
n \times n.
$$

For image or video data, $n$ can be very large. For example, if a grayscale frame has height $h$ and width $w$, then the flattened frame has dimension

$$
n = h \cdot w.
$$

So even a modest image can produce a very large state vector.

This is a major simplification. The original system may be nonlinear, time-dependent, or affected by parameters we do not observe. However, once we assume a linear, time-invariant approximation,

$$
\frac{d\vec{x}}{dt}=A\vec{x},
$$

we enter the setting of constant-coefficient linear systems of ordinary differential equations.

This is the setting where the matrix-exponential solution is available:

$$
\vec{x}(t)=e^{At}\vec{x}(0).
$$

Here, $\vec{x}(0)$ is the initial condition, or starting state, at time $t = 0$.

This equation says that the state at time $t$ can be obtained by applying the matrix exponential $e^{At}$ to the initial state.

The eigenvalue/eigenvector representation below depends on this linear assumption. In the fully nonlinear case, there is generally no single matrix $A$ whose eigenvectors and eigenvalues describe the whole system globally.

## Eigenvalues, Eigenvectors, and Modal Solutions

The solution becomes more interpretable when $A$ can be diagonalized.

Suppose $A$ is diagonalizable, so that

$$
A = V\Lambda V^{-1},
$$

where

$$
V =
[\vec{v}_1 \; \vec{v}_2 \; \cdots \; \vec{v}_n]
$$

is the matrix whose columns are eigenvectors of $A$, and

$$
\Lambda = \mathrm{diag}(\lambda_1, \lambda_2, \dots, \lambda_n)
$$

is the diagonal matrix of eigenvalues.

Substituting the eigendecomposition of $A$ into the solution gives

$$
\vec{x}(t)
= V e^{\Lambda t} V^{-1}\vec{x}(0).
$$

Because $\Lambda$ is diagonal, its matrix exponential is easy to interpret:

$$
e^{\Lambda t}
= \begin{pmatrix}
e^{\lambda_1 t} & 0 & \cdots & 0 \\
0 & e^{\lambda_2 t} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & e^{\lambda_n t}
\end{pmatrix}.
$$

Therefore,

$$
\vec{x}(t)
= V e^{\Lambda t} V^{-1}\vec{x}(0)
= V
\begin{pmatrix}
e^{\lambda_1 t} & 0 & \cdots & 0 \\
0 & e^{\lambda_2 t} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & e^{\lambda_n t}
\end{pmatrix}
V^{-1} \vec{x}(0).
$$

This form shows that the time evolution of the system can be decomposed into separate modal components.

Equivalently, the solution can be written as

$$
\vec{x}(t)
= \sum_{i=1}^{n} b_i \vec{v}_i e^{\lambda_i t}.
$$

This expression is important.

The eigenvectors $\vec{v}_i$ describe spatial directions, patterns, or modes. The eigenvalues $\lambda_i$ describe how those modes evolve in time. The coefficients $b_i$ describe how strongly each mode contributes to the initial condition.

The vector of coefficients

$$
\vec{b}
= \begin{bmatrix}
b_1 \\
b_2 \\
\vdots \\
b_n
\end{bmatrix}
$$

is computed from

$$
\vec{b} = V^{-1}\vec{x}(0).
$$

Equivalently, $\vec{b}$ solves

$$
\vec{x}(0) = V\vec{b}.
$$

This is the same idea that appears in ordinary differential equations: the general solution is a linear combination of independent solutions, and the initial condition determines the coefficients in that linear combination.

So the formula

$$
\vec{x}(t)
= \sum_{i=1}^{n} b_i \vec{v}_i e^{\lambda_i t}
$$

can be read as:

> The state at time $t$ is a sum of modes. Each mode has a spatial direction $\vec{v}_i$, a time behavior $e^{\lambda_i t}$, and an amplitude $b_i$ determined by the initial condition.

This is the conceptual foundation of DMD.

DMD uses this same modal idea, but estimates the modes and eigenvalues directly from data rather than from a known matrix $A$.

This entire eigenvalue/eigenvector solution depends on the linearity assumption. The formula above is available because we have assumed a constant linear system with a matrix $A$. In this idealized scenario, $A$ is known, so the dynamics can be described directly in terms of its eigenvectors and eigenvalues.

In real data-driven problems, $A$ is not handed to us. For a video, we do not begin with a known matrix that advances one frame to the next. For financial data, if such a reliable transition matrix were known, it would likely be exploited quickly and stop being useful. More generally, the true governing process may be nonlinear, noisy, partially observed, or unknown. DMD begins from the linear picture above and then tries to estimate the relevant modal structure from observed data.

## From Known Dynamics to Data-Driven Dynamics

The above derivation assumes that $A$ is known.

In many real problems, $A$ is not known. We do not usually have direct access to the true transition matrix governing the system. For example, in video analysis, we do not begin with an explicit matrix that maps one frame to the next. In finance, if a reliable transition matrix for future prices were known, market behavior would likely change as people exploited it.

The point of DMD is to estimate an approximate transition operator from observed data.

We now shift from the continuous-time equation

$$
\frac{d\vec{x}}{dt} = A\vec{x}
$$

to a related discrete-time state transition model:

$$
\vec{x}_{t+1} = A\vec{x}_t.
$$

Here, $A$ is responsible for progressing the state one step forward in time.

The simplest DMD setup assumes that the snapshots are sampled at a constant, uniform time interval. In this notebook, that means consecutive frames are evenly spaced in time. This assumption can be relaxed in more advanced variants, but the standard formulation assumes a fixed $\Delta t$.

For this notebook, one time step corresponds to one video frame. If the video has frames per second value `fps`, then the physical time between consecutive frames is

$$
\Delta t = \frac{1}{\text{fps}}.
$$

For example, if the synthetic video is generated at 30 frames per second, then

$$
\Delta t = \frac{1}{30} \text{ seconds}.
$$

This $\Delta t$ matters when we interpret DMD eigenvalues as physical frequencies and growth rates.

## Discrete-Time Snapshot Matrices

Suppose we observe $m$ snapshots of a system:

$$
\vec{x}_1, \vec{x}_2, \dots, \vec{x}_m.
$$

Each $\vec{x}_i$ is one observed state vector.

For video data, each $\vec{x}_i$ is one flattened video frame. If the frame has height $h$ and width $w$, then

$$
\vec{x}_i \in \mathbb{R}^{h \cdot w}.
$$

DMD forms two snapshot matrices:

$$
X =[\vec{x}_1 \; \vec{x}_2 \; \cdots \; \vec{x}_{m-1}]
$$

and

$$
X' =[\vec{x}_2 \; \vec{x}_3 \; \cdots \; \vec{x}_{m}].
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

The columns of $X$ contain the earlier snapshots. The columns of $X'$ contain the same snapshots shifted forward by one time step.

DMD assumes there is an approximate linear operator $A$ such that

$$
X' \approx AX
$$

This means that one matrix $A$ approximately advances each snapshot column of $X$ to the next snapshot column in $X'$:

The DMD assumption is

$$
X' \approx AX.
$$

Expanding the snapshot matrices gives

$$
X' =
\begin{bmatrix}
\vec{x}_2 & \vec{x}_3 & \cdots & \vec{x}_m
\end{bmatrix}
$$

and

$$
AX = A
\begin{bmatrix}
\vec{x}_1 & \vec{x}_2 & \cdots & \vec{x}_{m-1}
\end{bmatrix}
= \begin{bmatrix}
A\vec{x}_1 & A\vec{x}_2 & \cdots & A\vec{x}_{m-1}
\end{bmatrix}.
$$

So column by column,

$$
A\vec{x}_j \approx \vec{x}_{j+1},
\qquad j = 1,2,\ldots,m-1.
$$

So column by column, DMD is trying to make

$$
A\vec{x}_i \approx \vec{x}_{i+1},
\qquad i = 1,2,\ldots,m-1.
$$

In this setup, each $\vec{x}_{i+1}$ is $\Delta t$ units of time after $\vec{x}_i$.

## What Should $m$ and $\Delta t$ Be?

This brings up two important questions:

1. What should $m$ be?
2. What should $\Delta t$ be?

The answer to each question depends on context and requires domain knowledge.

The value $m$ controls how many snapshots are used to estimate the dynamics. If $m$ is too small, there may not be enough temporal information to estimate meaningful modes. If $m$ is too large, the data may include behavior that is no longer well approximated by one linear time-advance operator.

In financial time series, this choice is especially difficult because market behavior is noisy, non-stationary, and affected by many unobserved variables. A long window may include contradictory regimes, while a short window may miss useful trends. Past performance is often not a reliable indicator of future performance, so large historical windows can become misleading if the system changes behavior. On the other hand, if the window is too short, we may fail to detect trends or repeated patterns that are genuinely present.

For the synthetic pendulum, the situation is cleaner. The system is controlled, periodic, and generated by a stable rule. Even here, $m$ still matters. A fitting window should include enough of the swing to learn the dominant oscillation. For example, fitting on only a tiny fraction of a period may not reveal the full periodic motion. Fitting on one or more full periods gives DMD a better chance to identify the oscillatory structure.

The value $\Delta t$ controls the time spacing between snapshots.

For a video, if every frame is used, then

$$
\Delta t = \frac{1}{\text{fps}}.
$$

If we downsample the video by taking every second frame, then the effective time step doubles:

$$
\Delta t = \frac{2}{\text{fps}}.
$$

More generally, if we keep every $q$-th frame, then

$$
\Delta t = \frac{q}{\text{fps}}.
$$

The discrete DMD operator estimates one-step evolution between the snapshots it sees. Therefore, $\Delta t$ determines what “one step forward” means physically.

## Converting Discrete-Time DMD Eigenvalues to Continuous-Time Frequencies

DMD eigenvalues are usually first computed in discrete time. That means they describe what happens from one snapshot to the next.

For this notebook, one snapshot is one video frame. If the video is sampled at a fixed frame rate, then consecutive frames are separated by a time step $\Delta t$.

So DMD naturally learns a discrete-time relationship of the form

$$
\vec{x}_{k+1} \approx A_d \vec{x}_k,
$$

where $A_d$ is a discrete-time time-advance operator. The subscript $d$ stands for “discrete.”

If $\lambda_i$ is an eigenvalue of $A_d$, then the corresponding DMD mode evolves by multiplying by $\lambda_i$ each time step. After $k$ discrete steps, that mode has been multiplied by

$$
\lambda_i^k.
$$

So $\lambda_i$ answers the question:

> What factor multiplies this mode every time we move forward by one sampled time step?

If $|\lambda_i| < 1$, that mode decays from frame to frame.
If $|\lambda_i| > 1$, that mode grows from frame to frame.
If $|\lambda_i| \approx 1$, that mode persists.
If $\lambda_i$ is complex, its angle also causes oscillation.

However, the underlying dynamical-systems theory is often written in continuous time. In continuous time, a linear system is written as

$$
\frac{d\vec{x}}{dt} = A_c \vec{x},
$$

where $A_c$ is the continuous-time generator of the dynamics. The subscript $c$ stands for “continuous.”

If $\omega_i$ is an eigenvalue of $A_c$, then the corresponding continuous-time mode evolves like

$$
e^{\omega_i t}.
$$

So $\omega_i$ answers a different but related question:

> What exponential growth, decay, and oscillation rate does this mode have per unit of continuous time?

These two descriptions are connected by sampling. Suppose a continuous-time mode evolves as

$$
e^{\omega_i t}.
$$

The video does not observe this mode at every possible time $t$. It observes the system only at discrete frame times

$$
t_k = k\Delta t.
$$

At those sampled times, the continuous-time mode becomes

$$
e^{\omega_i t_k}
= e^{\omega_i k\Delta t}.
$$

Using exponent rules, this can be rewritten as

$$
e^{\omega_i k\Delta t}
= \left(e^{\omega_i \Delta t}\right)^k.
$$

But the DMD discrete-time evolution of the same mode has the form

$$
\lambda_i^k.
$$

Therefore, to make the continuous-time and discrete-time descriptions agree at the sampled frame times, we identify

$$
\lambda_i = e^{\omega_i \Delta t}.
$$

Taking the logarithm of both sides gives

$$
\log(\lambda_i) = \omega_i \Delta t.
$$

Solving for $\omega_i$ gives

$$
\omega_i = \frac{\log(\lambda_i)}{\Delta t}.
$$

This is the origin of the formula. It is not an arbitrary conversion. It comes from matching a continuous-time exponential mode to the same mode observed at discrete sampling times.

In the ideal theoretical case, if we knew the true continuous-time matrix $A_c$, then $\omega_i$ would be an eigenvalue of $A_c$. The corresponding discrete-time one-step operator would be

$$
A_d = e^{A_c \Delta t}.
$$

In that ideal case, the eigenvalues of $A_d$ would be related to the eigenvalues of $A_c$ by

$$
\lambda_i = e^{\omega_i \Delta t}.
$$

So in exact theory:

- $\omega_i$ is a continuous-time eigenvalue of the generator $A_c$,
- $\lambda_i$ is a discrete-time eigenvalue of the sampled time-advance operator $A_d$,
- and the two are related by $\lambda_i = e^{\omega_i \Delta t}$.

In DMD, we usually do not know either $A_c$ or $A_d$. Instead, we have data snapshots. We build

$$
X = [\vec{x}_1 \; \vec{x}_2 \; \cdots \; \vec{x}_{m-1}]
$$

and

$$
X' = [\vec{x}_2 \; \vec{x}_3 \; \cdots \; \vec{x}_{m}].
$$

DMD estimates an approximate time-advance relationship

$$
X' \approx A_d X.
$$

Because the full matrix $A_d$ would be enormous for image data, DMD works through a low-rank projected operator

$$
\tilde{A} = U_r^* X' V_r \Sigma_r^{-1}.
$$

The eigenvalues computed in DMD are the eigenvalues of this reduced operator $\tilde{A}$:

$$
\tilde{A}W = W\Lambda.
$$

The diagonal entries of $\Lambda$ are the DMD eigenvalues $\lambda_i$.

So, in the practical data-driven setting:

- $\lambda_i$ is an eigenvalue of the reduced DMD operator $\tilde{A}$,
- $\lambda_i$ approximates a dominant eigenvalue of the unknown discrete-time dynamics that maps one frame to the next,
- and $\omega_i = \log(\lambda_i)/\Delta t$ is an inferred continuous-time growth, decay, and oscillation rate.

So $\omega_i$ is not directly measured from the system. It is inferred from the DMD eigenvalue $\lambda_i$ and the sampling time step $\Delta t$.

To interpret this more concretely, write a complex DMD eigenvalue in polar form:

$$
\lambda_i = r_i e^{i\theta_i}.
$$

Then

$$
\log(\lambda_i)
= \log(r_i) + i\theta_i.
$$

Therefore

$$
\omega_i
= \frac{\log(r_i)}{\Delta t}
+i\frac{\theta_i}{\Delta t}.
$$

The real part is

$$
\mathrm{Re}(\omega_i)
= \frac{\log(r_i)}{\Delta t}.
$$

This is the continuous-time growth or decay rate.

The imaginary part is

$$
\mathrm{Im}(\omega_i)
= \frac{\theta_i}{\Delta t}.
$$

This is an angular frequency measured in radians per unit time.

To convert angular frequency to ordinary frequency in cycles per unit time, divide by $2\pi$:

$$
f_i = \frac{\mathrm{Im}(\omega_i)}{2\pi}.
$$

If $f_i \neq 0$, the corresponding period is

$$
T_i = \frac{1}{|f_i|}.
$$

In this notebook, $\Delta t$ is measured in seconds because the synthetic video has a frame rate measured in frames per second:

$$
\Delta t = \frac{1}{\mathrm{fps}}.
$$

For example, if the video is generated at $30$ frames per second, then

$$
\Delta t = \frac{1}{30}
$$

seconds per frame.

With this choice:

- $\omega_i$ has units of inverse seconds,
- $\mathrm{Re}(\omega_i)$ is a growth or decay rate per second,
- $\mathrm{Im}(\omega_i)$ is an angular frequency in radians per second,
- $f_i$ is a frequency in cycles per second,
- and $T_i$ is a period in seconds.


If we choose frame index as the time variable, then one DMD step is one unit of time, so $\Delta t = 1$. In that case, the frequency is measured in cycles per frame. If we choose physical time in seconds as the time variable, then for a 30 fps video $\Delta t = 1/30$, and the frequency is measured in cycles per second and we get physical frequencies in Hz.

For the synthetic pendulum, this conversion gives a useful sanity check. If DMD learns the dominant oscillatory behavior correctly, then one of the dominant DMD frequencies should be related to the known pendulum period used to generate the data.

There is one subtle numerical issue. The complex logarithm is multi-valued because angles can differ by multiples of $2\pi$:

$$
\log(\lambda_i)
= \log(r_i) + i(\theta_i + 2\pi q),
\qquad q \in \mathbb{Z}.
$$

Most numerical code uses the principal branch of the logarithm, where the angle is restricted to a standard interval such as $(-\pi, \pi]$. This is usually reasonable when the sampling rate is high enough relative to the true oscillation frequency. If the system oscillates too quickly relative to the frame rate, frequency aliasing can occur.

For this synthetic pendulum example, the motion is slow relative to the video frame rate, so the principal-frequency interpretation should be reasonable.

## Least-Squares Estimate of the Transition Matrix

The most direct way to estimate $A$ is to solve

$$
X' \approx AX.
$$

Using the Moore-Penrose pseudoinverse of $X$, denoted $X^\dagger$, the least-squares solution is

$$
A = X'X^\dagger.
$$

Mathematically, this $A$ is the matrix that best fits the observed transitions from $\vec{x}_i$ to $\vec{x}_{i+1}$ in a least-squares sense. This formulation is often referred to as **exact DMD**.

The algebraic least-squares equation does not explicitly depend on the physical size of the timestep. However, $\Delta t$ is still important because it determines the physical meaning of one step forward and is needed when interpreting eigenvalues as growth rates, frequencies, and periods.

This is conceptually simple, but it is usually not practical to form $A$ directly.

If each state vector has dimension $n$, then $A$ has shape

$$
n \times n.
$$

For image data, $n$ is the number of pixels in a flattened frame. Even with modest grayscale frames, this can be large. For example, a $256 \times 256$ grayscale frame has

$$
n = 256 \cdot 256 = 65{,}536
$$

entries. The full matrix $A$ would then have shape

$$
65{,}536 \times 65{,}536.
$$

That is far too large to compute and analyze directly in a simple notebook.

This is typical in applications of DMD, especially when the number of features is large relative to the number of snapshots. Therefore, DMD takes a roundabout but powerful approach: it estimates the important dynamics in a reduced low-dimensional subspace.

## Practical DMD Procedure

The practical DMD procedure can be summarized as follows:

1. Approximate the data matrix $X$ using a rank-$r$ truncation from the Singular Value Decomposition.
2. Use the SVD factors to build an $r \times r$ reduced substitute for the full operator $A$, denoted $\tilde{A}$.
3. Compute the eigenvectors and eigenvalues of $\tilde{A}$.
4. Map those reduced eigenvectors and eigenvalues back to the original state space to obtain DMD modes and the desired modal approximation.

The rest of the derivation expands these steps in more detail.

## Low-Rank Structure and the SVD

The key assumption is that the high-dimensional data has useful low-rank structure.

That is, even though each video frame has many pixels, most of the important variation may be described by a much smaller number of dominant spatial-temporal patterns. In other words, we hope that we can reduce the dimension and still capture most of the important variation in the data.

This is closely related to Principal Component Analysis (PCA). PCA uses the singular value decomposition to identify dominant directions of variation in data. DMD also uses the SVD, but then goes further by estimating how the reduced coordinates evolve through time. See my [notebook on PCA](/PCA_linear/pca_linear_oscillation_system.ipynb) for a related discussion of dimensionality reduction and low-rank approximation.

### Step 1: Compute the SVD and Choose $r$

The first step is to compute a rank-$r$ truncated SVD of the snapshot matrix $X$:

$$
X \approx U_r \Sigma_r V_r^*.
$$

Here:

- $U_r$ contains the first $r$ left singular vectors,
- $\Sigma_r$ contains the first $r$ singular values,
- $V_r$ contains the first $r$ right singular vectors,
- $V_r^*$ denotes the conjugate transpose of $V_r$,
- and $r$ is the chosen truncation rank.

The columns of $U_r$ store the $r$ dominant left singular vectors. These vectors define the reduced subspace and act as the key for transforming between the original $n$-dimensional state space and the lower-dimensional rank-$r$ approximation.

The choice of $r$ is important and has been the subject of much research. A simple first approach is to inspect the singular values, often on a log plot, and choose a cutoff where the values decay sharply or become small. A more careful approach may use energy thresholds, noise-aware rank selection, cross-validation, or application-specific criteria.

A small $r$ may underfit by discarding meaningful dynamics. A large $r$ may retain noise, weak modes, or unstable behavior that hurts forecasting.

In this notebook, the singular values will be plotted before selecting a main rank, and later the effect of rank will be studied directly.

## Reduced Linear Operator

### Step 2: Find $\tilde{A}$

DMD avoids directly forming the large matrix $A$ by constructing a reduced operator

$$
\tilde{A} = U_r^* A U_r.
$$

This matrix represents the dynamics projected into the rank-$r$ subspace spanned by the columns of $U_r$. It is a similarity-transform-style reduced representation of the full transition dynamics.

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