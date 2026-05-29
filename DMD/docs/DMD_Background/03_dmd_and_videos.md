# Part 3: DMD and Videos: Connecting DMD to the Video Domain

This is a continuation of [part 1](01_modes_and_eigenvalues.md) and [part 2](02_a_couple_examples.md).

---

DMD is easiest to understand as a data-driven method for discovering approximate linear dynamical structure from a sequence of snapshots.

For a video, each snapshot is a frame. After flattening each frame into a vector, DMD tries to represent the video as a sum of spatial patterns that each evolve in time according to a simple exponential rule.

The central idea is:

> A mode describes a spatial pattern, and an eigenvalue describes how that spatial pattern changes over time.

For video DMD, each mode can often be reshaped into an image-like object. The mode says what spatial pattern is changing, and the eigenvalue says whether that pattern persists, grows, decays, or oscillates.

---

## 19. DMD as a Data-Driven Approximation

In theory, if the true system matrix $A_d$ were known, its eigenvectors and eigenvalues could be computed directly.

In DMD, the true $A_d$ is not known.

Instead, DMD receives snapshot data:

$$
X =
[
\vec{x}_1
;
\vec{x}_2
;
\cdots
;
\vec{x}_{m-1}
]
$$

and

$$
X' =
[
\vec{x}_2
;
\vec{x}_3
;
\cdots
;
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
\tilde{A} = U_r^* X' V_r \Sigma_r^{-1}.
$$

This smaller matrix approximates the time-advance dynamics inside the rank-r subspace: $\tilde{A}$ is the low-dimensional representation of the unknown full time-advance operator $A_d$ after projecting into the rank-r SVD/POD subspace (it describes how the dynamics act inside the low-rank subspace spanned by the columns of $U_r$).

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
;
\phi_2
;
\cdots
;
\phi_r
].
$$

So in practical DMD:

* $\lambda_i$ is an eigenvalue of the reduced learned operator $\tilde{A}$, and they approximate dominant eigenvalues of the unknown full $A_d$.
* $\phi_i$ is a DMD mode mapped back into the original state space, and approximates a dynamically meaningful spatial pattern.

DMD usually estimates discrete-time eigenvalues first because the data are snapshots. Continuous-time rates are then inferred using

$$
\omega_i = \frac{\log(\lambda_i)}{\Delta t}.
$$

This asks:

> What continuous-time exponential rate would produce the observed per-snapshot multiplier?

---

## 20. Modes in a Pendulum Video

For the pendulum video, each grayscale frame is flattened into a vector:

$$
\vec{x}_k \in \mathbb{R}^{h\cdot w}.
$$

A DMD mode has the same dimension:

$$
\phi_i \in \mathbb{C}^{h\cdot w}.
$$

Therefore a DMD mode can be reshaped into an image. However, a DMD mode is not usually a literal frame from the video. A frame is the full state at one time:

$$
\vec{x}_k.
$$

A DMD mode is a reusable spatial pattern that contributes to many frames over time.

The reconstructed frame is a sum of mode contributions:

$$
\vec{x}_k \approx \sum_{i=1}^{r} b_i\phi_i\lambda_i^k.
$$

Each term contains:

* $\phi_i$: the image-like spatial pattern,
* $\lambda_i^k$: the time behavior,
* $b_i$: the amplitude determined by the initial condition.

So a DMD mode is best interpreted as an image-like building block paired with a specific temporal behavior.

---

## 21. What DMD Modes May Represent in the Pendulum Example

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

* real part of a mode,
* imaginary part of a mode,
* magnitude of a mode,
* and phase of a mode.

For a pendulum, oscillatory modes may highlight the arm, bob, and swept path of the motion. A background mode may highlight the static image structure.

---

## 22. Why DMD Is Useful for Video

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
\text{video} \approx \text{background mode} + \text{oscillatory pendulum modes} + \text{correction modes}.
$$

Mathematically,

$$
\vec{x}_k \approx \sum_{i=1}^{r} b_i\phi_i\lambda_i^k.
$$

This is the bridge between linear algebra and video interpretation:

* the modes $\phi_i$ describe spatial patterns,
* the eigenvalues $\lambda_i$ describe temporal behavior,
* the amplitudes $b_i$ describe how strongly each mode contributes,
* and the sum reconstructs or forecasts video frames.

For the pendulum video, the most important modal interpretation is:

* a persistent mode may correspond to the static background,
* oscillatory modes may correspond to the swinging arm and bob,
* the angle of the eigenvalues may reveal the swing frequency,
* and the learned frequency can be compared to the known synthetic ground-truth period.

---

## 23. Reporting Frequencies: Seconds and Frames

For the synthetic pendulum video, both physical-time and frame-index interpretations are useful.

If the generator defines a frame rate $\mathrm{fps}$, then

$$
\Delta t = \frac{1}{\mathrm{fps}}.
$$

Using this $\Delta t$ gives frequencies in cycles per second.

For a DMD eigenvalue $\lambda_i$, compute

$$
\omega_i = \frac{\log(\lambda_i)}{\Delta t}.
$$

If

$$
\omega_i = \alpha_i + i\beta_i,
$$

then the physical frequency is

$$
f_i = \frac{\beta_i}{2\pi}.
$$

This has units of cycles per second.

The corresponding cycles per frame are

$$
f_{i,\text{frame}} = \frac{f_i}{\mathrm{fps}}.
$$

Equivalently, if

$$
\lambda_i = r_i e^{i\theta_i},
$$

then

$$
f_{i,\text{frame}} = \frac{\theta_i}{2\pi}.
$$

The corresponding periods are:

$$
T_{i,\text{seconds}} = \frac{1}{|f_i|},
$$

and

$$
T_{i,\text{frames}} = \frac{1}{|f_{i,\text{frame}}|}.
$$

For the synthetic pendulum notebook, it is useful to report both:

* frequency in cycles per second,
* frequency in cycles per frame,
* period in seconds,
* period in frames.

This makes the connection between physical time and frame-index time explicit.

---

## 24. Scaling, Rotation, and Frequency Interpretation

DMD eigenvalues are usually complex, and complex eigenvalues are easiest to interpret in polar form.

Suppose a discrete-time DMD eigenvalue is

$$
\lambda_i = r_i e^{i\theta_i}.
$$

Using Euler's formula,

$$
e^{i\theta_i} = \cos(\theta_i) + i\sin(\theta_i).
$$

So multiplying by $\lambda_i$ has two effects:

1. $r_i$ scales the mode amplitude.
2. $e^{i\theta_i}$ rotates the mode phase by $\theta_i$ radians.

The complex factor $e^{i\theta_i}$ can also be represented as a real rotation matrix:

$$
R(\theta_i) =
\begin{bmatrix}
\cos(\theta_i) & -\sin(\theta_i) \\
\sin(\theta_i) & \cos(\theta_i)
\end{bmatrix}.
$$

So multiplication by

$$
\lambda_i = r_i e^{i\theta_i}
$$

corresponds to scaling by $r_i$ and rotating by $\theta_i$.

In matrix form, the corresponding scale-rotation action is

$$
r_iR(\theta_i)
= r_i
\begin{bmatrix}
\cos(\theta_i) & -\sin(\theta_i) \\
\sin(\theta_i) & \cos(\theta_i)
\end{bmatrix}.
$$

For a discrete-time eigenpair,

$$
A_d\vec{v}_i = \lambda_i\vec{v}_i,
$$

repeated time stepping gives

$$
A_d^k\vec{v}_i = \lambda_i^k\vec{v}_i.
$$

Since

$$
\lambda_i^k = (r_i e^{i\theta_i})^k = r_i^k e^{ik\theta_i},
$$

the interpretation is:

* $r_i^k$ controls amplitude after $k$ steps,
* $e^{ik\theta_i}$ controls phase after $k$ steps,
* $r_i$ is the amplitude multiplier per step,
* $\theta_i$ is the phase advance per step.

If one DMD step is one video frame, then $\theta_i$ is measured in radians per frame.

The frequency in cycles per frame is

$$
f_{i,\mathrm{frame}} = \frac{\theta_i}{2\pi}.
$$

The corresponding period in frames is

$$
T_{i,\mathrm{frames}} = \frac{1}{|f_{i,\mathrm{frame}}|}.
$$

For real-valued data, such as grayscale video frames, complex eigenvalues usually appear in conjugate pairs:

$$
\lambda_i = r_i e^{i\theta_i},
\qquad
\overline{\lambda_i} = r_i e^{-i\theta_i}.
$$

The associated modes also appear as conjugate pairs. A single complex mode is not usually a directly observable real image by itself. Instead, the conjugate pair combines to produce a real oscillatory pattern.

So a complex DMD eigenvalue does not usually mean that the entire image literally rotates. It means that a modal coefficient rotates in phase inside a two-dimensional oscillatory subspace.

For the pendulum video, this kind of mode pair may represent the swinging motion of the arm and bob.

---

### Continuous-Time Interpretation

In continuous time, write a continuous-time eigenvalue as

$$
\omega_i = \alpha_i + i\beta_i.
$$

The corresponding modal time factor is

$$
e^{\omega_i t}.
$$

Substituting $\omega_i = \alpha_i + i\beta_i$ gives

$$
e^{\omega_i t}
= e^{(\alpha_i+i\beta_i)t}
= e^{\alpha_i t}e^{i\beta_i t}.
$$

So the continuous-time interpretation parallels the discrete-time interpretation:

* $e^{\alpha_i t}$ controls continuous growth or decay,
* $e^{i\beta_i t}$ controls continuous oscillation,
* $\alpha_i$ is the growth or decay rate per unit time,
* $\beta_i$ is the angular frequency in radians per unit time.

The physical frequency in cycles per unit time is

$$
f_i = \frac{\beta_i}{2\pi}.
$$

The corresponding period is

$$
T_i = \frac{1}{|f_i|}.
$$

So $\alpha_i$ and $\beta_i$ are continuous-time rate quantities, while $r_i$ and $\theta_i$ are discrete-time per-step quantities.

---

### Connecting the Discrete and Continuous Interpretations

The discrete and continuous eigenvalues are related by

$$
\lambda_i = e^{\omega_i\Delta t}.
$$

Now write the continuous eigenvalue as

$$
\omega_i = \alpha_i + i\beta_i.
$$

Then

$$
\lambda_i
= e^{(\alpha_i+i\beta_i)\Delta t}
= e^{\alpha_i\Delta t}e^{i\beta_i\Delta t}.
$$

But the discrete eigenvalue can also be written as

$$
\lambda_i = r_i e^{i\theta_i}.
$$

Matching the two forms gives

$$
r_i = e^{\alpha_i\Delta t}
$$

and

$$
\theta_i = \beta_i\Delta t.
$$

Therefore,

$$
\alpha_i = \frac{\log(r_i)}{\Delta t}
$$

and

$$
\beta_i = \frac{\theta_i}{\Delta t}.
$$

This explains the real and imaginary parts of

$$
\omega_i = \frac{\log(\lambda_i)}{\Delta t}.
$$

The real part is

$$
\mathrm{Re}(\omega_i) = \alpha_i = \frac{\log(r_i)}{\Delta t}.
$$

The imaginary part is

$$
\mathrm{Im}(\omega_i) = \beta_i = \frac{\theta_i}{\Delta t}.
$$

So:

* $r_i$ is the amplitude multiplier per frame,
* $\alpha_i$ is the amplitude growth or decay rate per second,
* $\theta_i$ is the phase advance per frame,
* $\beta_i$ is the phase advance rate in radians per second.

This is the main unit conversion:

> The discrete eigenvalue says what happens per sample.
> The continuous eigenvalue says what rate would produce that same effect per unit time.

---

### Example with Pendulum-Style Numbers

Suppose a mode has one full oscillation every $2$ seconds. Then the physical frequency is

$$
f = \frac{1}{2} = 0.5
$$

cycles per second.

The angular frequency is

$$
\beta = 2\pi f = \pi
$$

radians per second.

If the video is sampled at $30$ frames per second, then

$$
\Delta t = \frac{1}{30}.
$$

The phase advance per frame is

$$
\theta = \beta\Delta t = \pi\frac{1}{30} = \frac{\pi}{30}.
$$

So the corresponding discrete eigenvalue, assuming no growth or decay, is

$$
\lambda = e^{i\pi/30}.
$$

This means the modal phase advances by

$$
\frac{\pi}{30}
$$

radians per frame, or $6^\circ$ per frame.

The cycles per frame are

$$
f_{\mathrm{frame}}
=\frac{\theta}{2\pi}
=\frac{\pi/30}{2\pi}
=\frac{1}{60}.
$$

So the mode completes one full cycle every $60$ frames.

This agrees with the physical timing:

$$
30 \text{ frames/second} \times 2 \text{ seconds/cycle} = 60 \text{ frames/cycle}.
$$

For the synthetic pendulum notebook, this is why it is useful to report both:

* frequency in cycles per second,
* frequency in cycles per frame,
* period in seconds,
* period in frames.

The DMD eigenvalue $\lambda_i$ is learned from frame-to-frame evolution. The continuous-time value $\omega_i$ is inferred from $\lambda_i$ using the known frame spacing $\Delta t = 1/\mathrm{fps}$.

For a strong pendulum DMD result, one of the dominant non-background oscillatory modes should have:

* $|\lambda_i| \approx 1$, meaning persistent oscillation,
* an angle $\theta_i$ close to the true phase advance per frame,
* and a continuous-time frequency close to the known pendulum frequency.

---

## 25. Note Summary

In exact continuous-time theory:

* $A_c$ is the continuous-time generator,
* $\omega_i$ is an eigenvalue of $A_c$,
* $\vec{v}_i$ is an eigenvector or mode,
* and the mode evolves as $e^{\omega_i t}\vec{v}_i$.

In exact discrete-time theory:

* $A_d$ is the one-step time-advance matrix,
* $\lambda_i$ is an eigenvalue of $A_d$,
* $\vec{v}_i$ is an eigenvector or mode,
* and the mode evolves as $\lambda_i^k\vec{v}_i$.

The continuous and discrete eigenvalues are connected by

$$
\lambda_i = e^{\omega_i\Delta t}
$$

and

$$
\omega_i = \frac{\log(\lambda_i)}{\Delta t}.
$$

This relationship is exact for sampled linear continuous-time dynamics.

The continuous eigenvalue $\omega_i$ is a rate per unit time. It describes continuous growth, decay, and oscillation.

The discrete eigenvalue $\lambda_i$ is a multiplier per sample step. It describes how much a mode changes from one frame to the next.

In DMD:

* the true time-advance matrix is unknown,
* the dynamics are estimated from snapshots,
* the eigenvalues are computed from a reduced learned operator,
* and the modes are data-driven spatial patterns mapped back into the original state space.

For video DMD, each mode can be reshaped into an image-like pattern. The mode says what spatial structure is changing, and the eigenvalue says how that structure changes over time.

The pendulum example is especially useful because the video frames are high-dimensional, but the underlying motion is low-dimensional, coherent, and periodic. This makes it possible to connect the DMD eigenvalues and modes back to an interpretable physical motion.

---

For the specifics on our DMD implementation, see [Applying DMD](04_applying_dmd.md) (part 4) next.