# Crash Course: Modes, Eigenvalues, Matrix Exponentials, and DMD

This note gives background for understanding Dynamic Mode Decomposition (DMD), especially the relationship between continuous-time dynamics, discrete-time dynamics, eigenvalues, modes, and video data.

DMD is easiest to understand as a data-driven method for discovering approximate linear dynamical structure from a sequence of snapshots.

For a video, each snapshot is a frame. After flattening each frame into a vector, DMD tries to represent the video as a sum of spatial patterns that each evolve in time according to a simple exponential rule.

---

## 1. Linear Dynamics in Continuous Time

A continuous-time linear dynamical system has the form

$$
\frac{d\vec{x}}{dt} = A_c \vec{x}.
$$

Here:

- $\vec{x}(t)$ is the state of the system at time $t$,
- $A_c$ is the continuous-time system matrix,
- and the subscript $c$ stands for continuous.

The matrix $A_c$ is sometimes called the continuous-time generator because it generates the time evolution of the system.

The solution is

$$
\vec{x}(t) = e^{A_c t}\vec{x}(0).
$$

The notation $e^{A_c t}$ is called the matrix exponential.

---

## 2. What Does $e^{A_c t}$ Mean?

The expression $e^{A_c t}$ is not an ordinary scalar exponential. It is a matrix.

If $A_c$ is an $n \times n$ matrix, then $e^{A_c t}$ is also an $n \times n$ matrix.

It is defined by the same power series used for the scalar exponential:

$$
e^{A_c t}
=
I
+A_c t
+\frac{(A_c t)^2}{2!}
+\frac{(A_c t)^3}{3!}
+\cdots
$$

Equivalently,

$$
e^{A_c t}
=
\sum_{j=0}^{\infty}
\frac{(A_c t)^j}{j!}.
$$

So if

$$
A_c =
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix},
$$

then

$$
e^{A_c t}
$$

is another $2 \times 2$ matrix:

$$
e^{A_c t}
=
\begin{bmatrix}
m_{11}(t) & m_{12}(t) \\
m_{21}(t) & m_{22}(t)
\end{bmatrix}.
$$

The entries $m_{ij}(t)$ are functions of time determined by the entries of $A_c$.

The matrix exponential is important because it maps an initial condition forward in time:

$$
\vec{x}(0)
\mapsto
\vec{x}(t).
$$

That is,

$$
\vec{x}(t) = e^{A_c t}\vec{x}(0).
$$

So $e^{A_c t}$ is the time-$t$ flow map of the linear system.

---

## 3. Matrix Exponential Through Eigenvalues

If $A_c$ is diagonalizable, then

$$
A_c = V\Omega V^{-1},
$$

where

$$
\Omega =
\begin{bmatrix}
\omega_1 & 0 & \cdots & 0 \\
0 & \omega_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \omega_n
\end{bmatrix}.
$$

The columns of $V$ are eigenvectors of $A_c$, and the diagonal entries $\omega_i$ are eigenvalues of $A_c$.

Then

$$
e^{A_c t}
=
V e^{\Omega t} V^{-1}.
$$

Because $\Omega$ is diagonal, its exponential is easy to compute:

$$
e^{\Omega t}
=
\begin{bmatrix}
e^{\omega_1 t} & 0 & \cdots & 0 \\
0 & e^{\omega_2 t} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & e^{\omega_n t}
\end{bmatrix}.
$$

So the matrix exponential evolves each eigenvector direction independently.

This is the main reason eigenvalues and eigenvectors are useful for linear dynamical systems.

---

## 4. What Is a Mode?

A mode is a spatial pattern associated with simple time evolution.

In exact linear algebra, a mode is closely related to an eigenvector.

For a continuous-time system,

$$
\frac{d\vec{x}}{dt} = A_c\vec{x},
$$

an eigenvector satisfies

$$
A_c\vec{v}_i = \omega_i \vec{v}_i.
$$

Here:

- $\vec{v}_i$ is an eigenvector, or mode,
- $\omega_i$ is the corresponding continuous-time eigenvalue.

If the system starts exactly in the direction of this eigenvector,

$$
\vec{x}(0) = \vec{v}_i,
$$

then the solution is

$$
\vec{x}(t) = e^{\omega_i t}\vec{v}_i.
$$

The state stays in the same eigenvector direction. Only its scalar multiplier changes.

This is why eigenvectors are dynamically important: they are directions that do not get mixed with other directions by the linear dynamics.

---

## 5. General ODE Solution as a Sum of Modes

Most initial conditions are not exactly one eigenvector. But if the eigenvectors form a basis, then the initial state can be written as

$$
\vec{x}(0)
=
b_1\vec{v}_1
+b_2\vec{v}_2
+\cdots
+b_n\vec{v}_n.
$$

Each eigenvector component then evolves independently:

$$
\vec{x}(t)
=
b_1 e^{\omega_1 t}\vec{v}_1
+b_2 e^{\omega_2 t}\vec{v}_2
+\cdots
+b_n e^{\omega_n t}\vec{v}_n.
$$

The solution has three ingredients:

1. The mode $\vec{v}_i$, which gives the spatial direction or pattern.
2. The eigenvalue $\omega_i$, which gives the time behavior.
3. The coefficient $b_i$, which says how much of that mode appears in the initial condition.

So a mode is not just “important” because it is large. A mode is important because it is a dynamically meaningful building block of the system.

---

## 6. Discrete-Time Dynamics

DMD is usually computed from discrete snapshots.

A discrete-time linear system has the form

$$
\vec{x}_{k+1} = A_d \vec{x}_k.
$$

Here:

- $\vec{x}_k$ is the state at discrete time step $k$,
- $A_d$ is the one-step time-advance matrix,
- and the subscript $d$ stands for discrete.

An eigenvector of $A_d$ satisfies

$$
A_d\vec{v}_i = \lambda_i \vec{v}_i.
$$

Here:

- $\vec{v}_i$ is a discrete-time mode,
- $\lambda_i$ is the corresponding discrete-time eigenvalue.

If

$$
\vec{x}_0 = \vec{v}_i,
$$

then

$$
\vec{x}_1 = A_d\vec{v}_i = \lambda_i\vec{v}_i,
$$

and

$$
\vec{x}_2 = A_d\vec{x}_1 = \lambda_i^2\vec{v}_i.
$$

After $k$ steps,

$$
\vec{x}_k = \lambda_i^k\vec{v}_i.
$$

For a general initial condition,

$$
\vec{x}_k
=
b_1\lambda_1^k\vec{v}_1
+b_2\lambda_2^k\vec{v}_2
+\cdots
+b_n\lambda_n^k\vec{v}_n.
$$

So in discrete time, the eigenvalues $\lambda_i$ determine how each mode changes from one snapshot to the next.

---

## 7. Why Eigenvalues Describe Growth, Decay, and Oscillation

In discrete time, each mode is multiplied by $\lambda_i$ at every time step.

If $\lambda_i$ is real and positive:

- $0 < \lambda_i < 1$ means the mode decays,
- $\lambda_i = 1$ means the mode persists,
- $\lambda_i > 1$ means the mode grows.

For example,

$$
0.9^k
$$

decays as $k$ increases, while

$$
1.1^k
$$

grows as $k$ increases.

If $\lambda_i$ is real and negative, the sign flips every step. For example,

$$
(-0.9)^k
$$

decays in magnitude but alternates sign.

If $\lambda_i$ is complex, write it in polar form:

$$
\lambda_i = r_i e^{i\theta_i}.
$$

Then

$$
\lambda_i^k
=
r_i^k e^{ik\theta_i}.
$$

The magnitude $r_i$ controls growth or decay:

- $r_i < 1$ means decay,
- $r_i = 1$ means persistence,
- $r_i > 1$ means growth.

The angle $\theta_i$ controls oscillation.

Using Euler's formula,

$$
e^{ik\theta_i}
=
\cos(k\theta_i)
+i\sin(k\theta_i).
$$

Therefore complex eigenvalues produce oscillatory behavior.

This is why DMD eigenvalues are often plotted in the complex plane:

- distance from the origin gives growth or decay,
- angle around the origin gives oscillation frequency.

---

## 8. Continuous-Time Complex Eigenvalues

In continuous time, suppose an eigenvalue is complex:

$$
\omega_i = \alpha_i + i\beta_i.
$$

Then the time evolution is

$$
e^{\omega_i t}
=
e^{(\alpha_i+i\beta_i)t}
=
e^{\alpha_i t}e^{i\beta_i t}.
$$

Using Euler's formula,

$$
e^{i\beta_i t}
=
\cos(\beta_i t)
+i\sin(\beta_i t).
$$

So

$$
e^{\omega_i t}
=
e^{\alpha_i t}
\left(
\cos(\beta_i t)
+i\sin(\beta_i t)
\right).
$$

The real part $\alpha_i$ controls growth or decay.

The imaginary part $\beta_i$ controls angular oscillation frequency.

The ordinary frequency in cycles per unit time is

$$
f_i = \frac{\beta_i}{2\pi}.
$$

If $f_i \neq 0$, the period is

$$
T_i = \frac{1}{|f_i|}.
$$

---

## 9. Connecting Continuous and Discrete Time

The continuous-time system is

$$
\frac{d\vec{x}}{dt} = A_c\vec{x}.
$$

Its solution is

$$
\vec{x}(t) = e^{A_c t}\vec{x}(0).
$$

Suppose the system is sampled every $\Delta t$ seconds.

The frame times are

$$
t_k = k\Delta t.
$$

The state at frame $k$ is

$$
\vec{x}_k = \vec{x}(t_k) = \vec{x}(k\Delta t).
$$

From the initial condition,

$$
\vec{x}_k = e^{A_c k\Delta t}\vec{x}(0).
$$

The next state is

$$
\vec{x}_{k+1} = e^{A_c (k+1)\Delta t}\vec{x}(0).
$$

This can be rewritten as

$$
\vec{x}_{k+1}
=
e^{A_c \Delta t}
e^{A_c k\Delta t}
\vec{x}(0).
$$

Since

$$
\vec{x}_k = e^{A_c k\Delta t}\vec{x}(0),
$$

we get

$$
\vec{x}_{k+1}
=
e^{A_c \Delta t}\vec{x}_k.
$$

This is why the one-step discrete-time map is

$$
A_d = e^{A_c \Delta t}.
$$

The factor $k\Delta t$ is used to evolve all the way from the initial time to time step $k$.

The factor $\Delta t$ is used to evolve from one sampled state to the next sampled state.

So:

$$
\vec{x}_k = e^{A_c k\Delta t}\vec{x}_0
$$

but

$$
\vec{x}_{k+1} = e^{A_c \Delta t}\vec{x}_k.
$$

The first formula is the time-$k$ solution from the initial condition.

The second formula is the one-step transition from snapshot $k$ to snapshot $k+1$.

---

## 10. Why $e^{A_c\Delta t}\vec{v}_i = e^{\omega_i\Delta t}\vec{v}_i$

Suppose $\vec{v}_i$ is an eigenvector of $A_c$:

$$
A_c\vec{v}_i = \omega_i\vec{v}_i.
$$

Then applying $A_c$ twice gives

$$
A_c^2\vec{v}_i
=
A_c(A_c\vec{v}_i)
=
A_c(\omega_i\vec{v}_i)
=
\omega_i A_c\vec{v}_i
=
\omega_i^2 \vec{v}_i.
$$

Similarly,

$$
A_c^3\vec{v}_i = \omega_i^3\vec{v}_i.
$$

In general,

$$
A_c^j\vec{v}_i = \omega_i^j\vec{v}_i.
$$

Now use the power series definition of the matrix exponential:

$$
e^{A_c\Delta t}
=
I
+A_c\Delta t
+\frac{(A_c\Delta t)^2}{2!}
+\frac{(A_c\Delta t)^3}{3!}
+\cdots.
$$

Apply this matrix to $\vec{v}_i$:

$$
e^{A_c\Delta t}\vec{v}_i
=
\left(
I
+A_c\Delta t
+\frac{A_c^2\Delta t^2}{2!}
+\frac{A_c^3\Delta t^3}{3!}
+\cdots
\right)
\vec{v}_i.
$$

Using

$$
A_c^j\vec{v}_i = \omega_i^j\vec{v}_i,
$$

we get

$$
e^{A_c\Delta t}\vec{v}_i
=
\left(
1
+\omega_i\Delta t
+\frac{\omega_i^2\Delta t^2}{2!}
+\frac{\omega_i^3\Delta t^3}{3!}
+\cdots
\right)
\vec{v}_i.
$$

The scalar series in parentheses is exactly

$$
e^{\omega_i\Delta t}.
$$

Therefore

$$
e^{A_c\Delta t}\vec{v}_i
=
e^{\omega_i\Delta t}\vec{v}_i.
$$

This shows that if $\omega_i$ is an eigenvalue of the continuous-time generator $A_c$, then

$$
e^{\omega_i\Delta t}
$$

is the corresponding eigenvalue of the discrete-time map

$$
A_d = e^{A_c\Delta t}.
$$

So the discrete and continuous eigenvalues satisfy

$$
\lambda_i = e^{\omega_i\Delta t}.
$$

Solving for $\omega_i$ gives

$$
\omega_i = \frac{\log(\lambda_i)}{\Delta t}.
$$

---

## 11. Interpreting $\lambda_i$ and $\omega_i$

The discrete-time eigenvalue $\lambda_i$ describes what happens to a mode from one snapshot to the next.

The continuous-time eigenvalue $\omega_i$ describes what happens to a mode per unit of continuous time.

They are related by

$$
\lambda_i = e^{\omega_i\Delta t}.
$$

So:

- $\lambda_i$ is a per-step multiplier,
- $\omega_i$ is a per-time exponential rate.

For a video sampled at frame rate $\mathrm{fps}$,

$$
\Delta t = \frac{1}{\mathrm{fps}}.
$$

Using this physical $\Delta t$ gives frequencies in cycles per second.

If instead frame index is used as the time variable, then one DMD step is one unit of time:

$$
\Delta t = 1.
$$

In that case, frequencies are measured in cycles per frame.

Both are valid time coordinates, but they answer different questions.

For the synthetic pendulum video, the physical interpretation is usually more useful because the generator defines a period in seconds.

---

## 12. DMD as a Data-Driven Approximation

In theory, if the true system matrix $A_d$ were known, its eigenvectors and eigenvalues could be computed directly.

In DMD, the true $A_d$ is not known.

Instead, DMD receives snapshot data:

$$
X =
[
\vec{x}_1
\;
\vec{x}_2
\;
\cdots
\;
\vec{x}_{m-1}
]
$$

and

$$
X' =
[
\vec{x}_2
\;
\vec{x}_3
\;
\cdots
\;
\vec{x}_{m}
].
$$

DMD assumes that there is an approximate linear relationship

$$
X' \approx A_d X.
$$

For image data, the full matrix $A_d$ would be enormous. If each grayscale frame is $256 \times 256$, then each flattened frame has dimension

$$
n = 256 \cdot 256 = 65{,}536.
$$

The full matrix $A_d$ would have size

$$
65{,}536 \times 65{,}536.
$$

DMD avoids constructing this full matrix directly.

Instead, it computes a low-rank SVD of the snapshot matrix:

$$
X \approx U_r\Sigma_r V_r^*.
$$

Then it forms the reduced operator

$$
\tilde{A}
=
U_r^* X' V_r \Sigma_r^{-1}.
$$

This smaller matrix approximates the time-advance dynamics inside the rank-$r$ subspace.

DMD then solves the eigenvalue problem

$$
\tilde{A}W = W\Lambda.
$$

The diagonal entries of $\Lambda$ are the DMD eigenvalues $\lambda_i$.

The DMD modes are mapped back to the original state space using

$$
\Phi = X'V_r\Sigma_r^{-1}W.
$$

The columns of $\Phi$ are the DMD modes:

$$
\Phi =
[
\phi_1
\;
\phi_2
\;
\cdots
\;
\phi_r
].
$$

So in practical DMD:

- $\lambda_i$ is an eigenvalue of the reduced learned operator $\tilde{A}$,
- $\phi_i$ is a DMD mode mapped back into the original state space,
- $\lambda_i$ approximates a dominant eigenvalue of the unknown one-step dynamics,
- and $\phi_i$ approximates a dynamically meaningful spatial pattern.

---

## 13. Modes in a Pendulum Video

For the pendulum video, each grayscale frame is flattened into a vector:

$$
\vec{x}_k \in \mathbb{R}^{h\cdot w}.
$$

A DMD mode has the same dimension:

$$
\phi_i \in \mathbb{C}^{h\cdot w}.
$$

Therefore a DMD mode can be reshaped into an image.

However, a DMD mode is not usually a literal frame from the video.

A frame is the full state at one time:

$$
\vec{x}_k.
$$

A DMD mode is a reusable spatial pattern that contributes to many frames over time.

The reconstructed frame is a sum of mode contributions:

$$
\vec{x}_k
\approx
\sum_{i=1}^{r}
b_i\phi_i\lambda_i^k.
$$

Each term contains:

- $\phi_i$: the image-like spatial pattern,
- $\lambda_i^k$: the time behavior,
- $b_i$: the amplitude determined by the initial condition.

So a DMD mode is best interpreted as an image-like building block paired with a specific temporal behavior.

---

## 14. What DMD Modes May Represent in the Pendulum Example

Because the pendulum video contains both static and moving content, DMD may learn different types of modes.

A near-zero-frequency mode may represent the static background.

A complex-conjugate pair of modes may represent the oscillatory pendulum motion.

Modes with eigenvalues near the unit circle,

$$
|\lambda_i| \approx 1,
$$

represent persistent behavior.

Modes inside the unit circle,

$$
|\lambda_i| < 1,
$$

represent decaying behavior.

Modes outside the unit circle,

$$
|\lambda_i| > 1,
$$

represent growing behavior, which can be dangerous for forecasting if the growth is not physically meaningful.

For real-valued video data, complex eigenvalues and modes usually appear in conjugate pairs. Individual complex modes are not directly real images, but their combined contribution produces real oscillatory motion.

Mode visualizations can include:

- real part of a mode,
- imaginary part of a mode,
- magnitude of a mode,
- and phase of a mode.

For a pendulum, oscillatory modes may highlight the arm, bob, and swept path of the motion. A background mode may highlight the static image structure.

---

## 15. Why DMD Is Useful for Video

A video frame is high-dimensional. A $256 \times 256$ grayscale frame is a point in

$$
\mathbb{R}^{65{,}536}.
$$

But the synthetic pendulum motion is low-dimensional. Much of the motion is governed by one angle:

$$
\theta(t).
$$

DMD tries to represent the high-dimensional video as a small number of spatial patterns with simple time evolution.

Conceptually, DMD tries to write the video as

$$
\text{video}
\approx
\text{background mode}
+\text{oscillatory pendulum modes}
+\text{correction modes}.
$$

Mathematically,

$$
\vec{x}_k
\approx
\sum_{i=1}^{r}
b_i\phi_i\lambda_i^k.
$$

This is the bridge between linear algebra and video interpretation:

- the modes $\phi_i$ describe spatial patterns,
- the eigenvalues $\lambda_i$ describe temporal behavior,
- the amplitudes $b_i$ describe how strongly each mode contributes,
- and the sum reconstructs or forecasts video frames.

---

## 16. Summary

In exact continuous-time theory:

- $A_c$ is the continuous-time generator,
- $\omega_i$ is an eigenvalue of $A_c$,
- $\vec{v}_i$ is an eigenvector or mode,
- and the mode evolves as $e^{\omega_i t}\vec{v}_i$.

In exact discrete-time theory:

- $A_d$ is the one-step time-advance matrix,
- $\lambda_i$ is an eigenvalue of $A_d$,
- $\vec{v}_i$ is an eigenvector or mode,
- and the mode evolves as $\lambda_i^k\vec{v}_i$.

The continuous and discrete eigenvalues are connected by

$$
\lambda_i = e^{\omega_i\Delta t}
$$

and

$$
\omega_i = \frac{\log(\lambda_i)}{\Delta t}.
$$

In DMD:

- the true time-advance matrix is unknown,
- the dynamics are estimated from snapshots,
- the eigenvalues are computed from a reduced learned operator,
- and the modes are data-driven spatial patterns mapped back into the original state space.

For video DMD, each mode can be reshaped into an image-like pattern. The mode says what spatial structure is changing, and the eigenvalue says how that structure changes over time.