# Modes, Eigenvalues, Matrix Exponentials, and DMD

This note gives background for understanding Dynamic Mode Decomposition (DMD), especially the relationship between continuous-time dynamics, discrete-time dynamics, eigenvalues, modes, matrix exponentials, and video data.

DMD is easiest to understand as a data-driven method for discovering approximate linear dynamical structure from a sequence of snapshots.

For a video, each snapshot is a frame. After flattening each frame into a vector, DMD tries to represent the video as a sum of spatial patterns that each evolve in time according to a simple exponential rule.

The central idea is:

> A mode describes a spatial pattern, and an eigenvalue describes how that spatial pattern changes over time.

For video DMD, each mode can often be reshaped into an image-like object. The mode says what spatial pattern is changing, and the eigenvalue says whether that pattern persists, grows, decays, or oscillates.

---

## 1. Linear Dynamics in Continuous Time

A continuous-time linear dynamical system has the form

$$
\frac{d\vec{x}}{dt} = A_c \vec{x}.
$$

Here:

* $\vec{x}(t)$ is the state of the system at time $t$,
* $A_c$ is the continuous-time system matrix,
* and the subscript $c$ stands for continuous.

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
e^{A_c t} = I + A_c t + \frac{(A_c t)^2}{2!} + \frac{(A_c t)^3}{3!} + \cdots
$$

Equivalently,

$$
e^{A_c t} = \sum_{j=0}^{\infty} \frac{(A_c t)^j}{j!}.
$$

So if

$$
A_c =
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix},
$$

then $e^{A_c t}$ is another $2 \times 2$ matrix:

$$
e^{A_c t} =
\begin{bmatrix}
m_{11}(t) & m_{12}(t) \\
m_{21}(t) & m_{22}(t)
\end{bmatrix}.
$$

The entries $m_{ij}(t)$ are functions of time determined by the entries of $A_c$.

The matrix exponential is important because it maps an initial condition forward in time:

$$
\vec{x}(0) \mapsto \vec{x}(t).
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
A_c = V D V^{-1},
$$

where

$$
D =
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
e^{A_c t} = V e^{D t} V^{-1}.
$$

Because $D$ is diagonal, its exponential is easy to compute:

$$
e^{D t} =
\begin{bmatrix}
e^{\omega_1 t} & 0 & \cdots & 0 \\
0 & e^{\omega_2 t} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & e^{\omega_n t}
\end{bmatrix}.
$$

So the matrix exponential evolves each eigenvector direction independently.

This is the main reason eigenvalues and eigenvectors are useful for linear dynamical systems. In the eigenvector basis, the system separates into independent modal pieces.

---

## 4. Modes

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

* $\vec{v}_i$ is an eigenvector, or mode,
* $\omega_i$ is the corresponding continuous-time eigenvalue.

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

In applied DMD language, the word mode is often used for the spatial pattern $\phi_i$ that is paired with a specific eigenvalue $\lambda_i$. Strictly speaking, the mode is the spatial vector, while the eigenvalue controls how that vector evolves in time.

---

## 5. General ODE Solution as a Sum of Modes

Most initial conditions are not exactly one eigenvector. But if the eigenvectors form a basis, then the initial state can be written as

$$
\vec{x}(0) = b_1\vec{v}_1 + b_2\vec{v}_2 + \cdots + b_n\vec{v}_n.
$$

Each eigenvector component then evolves independently:

$$
\vec{x}(t) = b_1 e^{\omega_1 t}\vec{v}_1 + b_2 e^{\omega_2 t}\vec{v}_2 + \cdots + b_n e^{\omega_n t}\vec{v}_n.
$$

The solution has three ingredients:

1. The mode $\vec{v}_i$, which gives the spatial/eigenvector direction or pattern.

2. The eigenvalue $\omega_i$ of $A_c$, which gives the continuous-time behavior of that mode. The scalar factor $e^{\omega_i t}$ is the time-evolution multiplier for the mode at time $t$, and $e^{\omega_i t}$ is the corresponding eigenvalue of the matrix exponential $e^{A_c t}$.

3. The coefficient $b_i$, which says how much of that mode appears in the initial condition. It is the initial amplitude of that mode.

So a mode is not just “important” because it is large. A mode is important because it is a dynamically meaningful building block of the system.

The mode says:

> What direction or spatial pattern is present?

The eigenvalue says:

> How does that pattern evolve over time?

The coefficient says:

> How strongly does that pattern contribute to the initial condition?

The modal-sum solution comes from combining the matrix exponential solution with the eigendecomposition of the system matrix:

Start with the continuous-time linear system

$$
\frac{d\vec{x}}{dt} = A_c\vec{x}.
$$

The general solution is

$$
\vec{x}(t) = e^{A_c t}\vec{x}(0).
$$

Now assume $A_c$ is diagonalizable. Then

$$
A_c = VDV^{-1},
$$

where

$$
V = [\vec{v}_1 \; \vec{v}_2 \; \cdots \; \vec{v}_n]
$$

is the matrix whose columns are eigenvectors, and

$$
D =
\begin{bmatrix}
\omega_1 & 0 & \cdots & 0 \\
0 & \omega_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \omega_n
\end{bmatrix}
$$

is the diagonal matrix of continuous-time eigenvalues.

Because $A_c = VDV^{-1}$, the matrix exponential can be written as

$$
e^{A_c t} = V e^{Dt} V^{-1}.
$$

This is useful because $D$ is diagonal, so $e^{Dt}$ is easy to compute:

$$
e^{Dt} =
\begin{bmatrix}
e^{\omega_1t} & 0 & \cdots & 0 \\
0 & e^{\omega_2t} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & e^{\omega_nt}
\end{bmatrix}.
$$

Now express the initial condition in the eigenvector basis:

$$
\vec{x}(0) = V\vec{b},
$$

where

$$
\vec{b} =
\begin{bmatrix}
b_1 \\
b_2 \\
\vdots \\
b_n
\end{bmatrix}.
$$

Equivalently,

$$
\vec{b} = V^{-1}\vec{x}(0).
$$

The entries $b_i$ tell how much of each eigenvector direction is present in the initial condition.

Substitute this into the solution:

$$
\vec{x}(t)
= e^{A_c t}\vec{x}(0)
= V e^{Dt} V^{-1} V\vec{b}.
$$

Since $V^{-1}V = I$, this becomes

$$
\vec{x}(t) = V e^{Dt}\vec{b}.
$$

Now compute $e^{Dt}\vec{b}$:

$$
e^{Dt}\vec{b}
=
\begin{bmatrix}
b_1e^{\omega_1t} \\
b_2e^{\omega_2t} \\
\vdots \\
b_ne^{\omega_nt}
\end{bmatrix}.
$$

Multiplying by $V$ forms a linear combination of the columns of $V$:

$$
\vec{x}(t)
=
b_1e^{\omega_1t}\vec{v}_1
+b_2e^{\omega_2t}\vec{v}_2
+\cdots
+b_ne^{\omega_nt}\vec{v}_n.
$$

So the modal-sum formula is

$$
\vec{x}(t)
=
\sum_{i=1}^{n}
b_i e^{\omega_i t}\vec{v}_i.
$$

This is the same solution as

$$
\vec{x}(t) = e^{A_c t}\vec{x}(0),
$$

but written in a way that separates the dynamics into independent modal pieces.

So the eigenvectors provide the directions or spatial patterns, the eigenvalues provide the time behavior, and the coefficients determine how strongly each mode contributes to the initial condition.

This is the main modal interpretation of a linear dynamical system:

> Decompose the initial state into eigenvector directions, evolve each direction independently according to its eigenvalue, and add the pieces back together.

---

## 6. Discrete-Time Dynamics

DMD is usually computed from discrete snapshots.

A discrete-time linear system has the form

$$
\vec{x}_{k+1} = A_d \vec{x}_k.
$$

Here:

* $\vec{x}_k$ is the state at discrete time step $k$,
* $A_d$ is the one-step time-advance matrix,
* and the subscript $d$ stands for discrete.

An eigenvector of $A_d$ satisfies

$$
A_d\vec{v}_i = \lambda_i \vec{v}_i.
$$

Here:

* $\vec{v}_i$ is a discrete-time mode,
* $\lambda_i$ is the corresponding discrete-time eigenvalue.

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
\vec{x}_k = b_1\lambda_1^k\vec{v}_1 + b_2\lambda_2^k\vec{v}_2 + \cdots + b_n\lambda_n^k\vec{v}_n.
$$

So in discrete time, the eigenvalues $\lambda_i$ determine how each mode changes from one snapshot to the next.

The most important distinction is:

* $\omega_i$ is a continuous-time eigenvalue, or rate per unit time.
* $\lambda_i$ is a discrete-time eigenvalue, or multiplier per sample step.

---

## 7. Why Eigenvalues Describe Growth, Decay, and Oscillation

In discrete time, each mode is multiplied by $\lambda_i$ at every time step.

If $\lambda_i$ is real and positive:

* $0 < \lambda_i < 1$ means the mode decays,
* $\lambda_i = 1$ means the mode persists,
* $\lambda_i > 1$ means the mode grows.

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
\lambda_i^k = r_i^k e^{ik\theta_i}.
$$

The magnitude $r_i$ controls growth or decay:

* $r_i < 1$ means decay,
* $r_i = 1$ means persistence,
* $r_i > 1$ means growth.

The angle $\theta_i$ controls oscillation.

Using Euler's formula,

$$
e^{ik\theta_i} = \cos(k\theta_i) + i\sin(k\theta_i).
$$

Therefore complex eigenvalues produce oscillatory behavior.

This is why DMD eigenvalues are often plotted in the complex plane:

* distance from the origin gives growth or decay,
* angle around the origin gives oscillation frequency.

A discrete eigenvalue $\lambda_i = r_i e^{i\theta_i}$ says:

> Each time step multiplies the mode magnitude by $r_i$ and advances its phase by $\theta_i$ radians.

So $\lambda_i$ is a per-step update rule for a mode.

---

## 8. Continuous-Time Complex Eigenvalues

In continuous time, suppose an eigenvalue is complex:

$$
\omega_i = \alpha_i + i\beta_i.
$$

Then the time evolution is

$$
e^{\omega_i t} = e^{(\alpha_i+i\beta_i)t} = e^{\alpha_i t}e^{i\beta_i t}.
$$

Using Euler's formula,

$$
e^{i\beta_i t} = \cos(\beta_i t) + i\sin(\beta_i t).
$$

So

$$
e^{\omega_i t} = e^{\alpha_i t}\left(\cos(\beta_i t) + i\sin(\beta_i t)\right).
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

A continuous eigenvalue $\omega_i = \alpha_i + i\beta_i$ says:

> The mode grows or decays at rate $\alpha_i$ per unit time and oscillates at angular frequency $\beta_i$ radians per unit time.

So $\omega_i$ is a continuous-time rate, not a per-step multiplier.

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

The sample times are

$$
t_k = k\Delta t.
$$

The state at sample $k$ is

$$
\vec{x}_k = \vec{x}(t=t_k) = \vec{x}(k\Delta t).
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
\vec{x}_{k+1} = e^{A_c \Delta t} e^{A_c k\Delta t}\vec{x}(0).
$$

Since

$$
\vec{x}_k = e^{A_c k\Delta t}\vec{x}(0),
$$

we get

$$
\vec{x}_{k+1} = e^{A_c \Delta t}\vec{x}_k.
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

The second formula is the one-step transition from sample $k$ to sample $k+1$.

---

## 10. Why $e^{A_c\Delta t}\vec{v}_i = e^{\omega_i\Delta t}\vec{v}_i$

Suppose $\vec{v}_i$ is an eigenvector of $A_c$:

$$
A_c\vec{v}_i = \omega_i\vec{v}_i.
$$

Then applying $A_c$ twice gives

$$
A_c^2\vec{v}_i = A_c(A_c\vec{v}_i) = A_c(\omega_i\vec{v}_i) = \omega_i A_c\vec{v}_i = \omega_i^2 \vec{v}_i.
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
e^{A_c\Delta t} = I + A_c\Delta t + \frac{(A_c\Delta t)^2}{2!} + \frac{(A_c\Delta t)^3}{3!} + \cdots.
$$

Apply this matrix to $\vec{v}_i$:

$$
e^{A_c\Delta t}\vec{v}_i = \left(I + A_c\Delta t + \frac{A_c^2\Delta t^2}{2!} + \frac{A_c^3\Delta t^3}{3!} + \cdots\right)\vec{v}_i.
$$

Using

$$
A_c^j\vec{v}_i = \omega_i^j\vec{v}_i,
$$

we get

$$
e^{A_c\Delta t}\vec{v}_i = \left(1 + \omega_i\Delta t + \frac{\omega_i^2\Delta t^2}{2!} + \frac{\omega_i^3\Delta t^3}{3!} + \cdots\right)\vec{v}_i.
$$

The scalar series in parentheses is exactly

$$
e^{\omega_i\Delta t}.
$$

Therefore

$$
e^{A_c\Delta t}\vec{v}_i = e^{\omega_i\Delta t}\vec{v}_i.
$$

This shows that if $\omega_i$ is an eigenvalue of the continuous-time generator $A_c$, then $e^{\omega_i\Delta t}$ is the corresponding eigenvalue of the discrete-time map

$$
A_d = e^{A_c\Delta t}.
$$

You can also reason through this by recognizing that since $A_c\vec{v}_i = \omega_i\vec{v}_i$, the eigenvector $\vec{v}_i$ is also an eigenvector of the matrix exponential $e^{A_c t}$, with eigenvalue $e^{\omega_i t}$:

$$
e^{A_c t}\vec{v}_i = e^{\omega_i t}\vec{v}_i.
$$

Evaluating this relationship at one sampled time step, $t=\Delta t$, gives

$$
e^{A_c \Delta t}\vec{v}_i = e^{\omega_i \Delta t}\vec{v}_i.
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

* $\lambda_i$ is a per-step multiplier,
* $\omega_i$ is a per-time exponential rate.

In continuous time, a mode evolves as

$$
e^{\omega_i t}.
$$

In discrete time, a mode evolves as

$$
\lambda_i^k.
$$

These match at sampled times $t_k = k\Delta t$ because

$$
\lambda_i^k = \left(e^{\omega_i\Delta t}\right)^k = e^{\omega_i k\Delta t}.
$$

This is the essential bridge:

> $\omega_i$ says what happens per unit of physical time.
> $\lambda_i$ says what happens after one sampled time step.

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

For the synthetic pendulum video, the physical interpretation is usually more useful because the generator defines a period in seconds. It is still useful to also report cycles per frame, because DMD is fit to frame-to-frame data.

---

## 12. The Limit View: How the Exponential Arises

The relationship

$$
\lambda = e^{\omega\Delta t}
$$

is an exact finite-time relationship once the continuous-time ODE solution is known.

However, the exponential can also be motivated as a limit of many tiny discrete steps.

Start with the scalar continuous-time equation

$$
\frac{dz}{dt} = \omega z.
$$

A simple forward Euler step over a small interval $\Delta t$ gives

$$
z(t+\Delta t) \approx z(t) + \Delta t \omega z(t).
$$

So

$$
z(t+\Delta t) \approx (1+\omega\Delta t)z(t).
$$

This gives the approximate one-step multiplier

$$
\lambda_{\text{Euler}} = 1+\omega\Delta t.
$$

But this is only an approximation.

To get the exact exponential, split a finite interval $\Delta t$ into $N$ smaller intervals of length $\Delta t/N$. Each tiny step has approximate multiplier

$$
1+\omega\frac{\Delta t}{N}.
$$

After $N$ tiny steps, the total multiplier is approximately

$$
\left(1+\omega\frac{\Delta t}{N}\right)^N.
$$

Taking the limit as $N \to \infty$ gives

$$
\lim_{N\to\infty}\left(1+\omega\frac{\Delta t}{N}\right)^N = e^{\omega\Delta t}.
$$

So the exponential can be understood as the result of infinitely many infinitesimal linear updates.

This helps connect the derivative definition of continuous time to the finite multiplier used in discrete time.

---

## 13. Worked Example: Circular Motion

A clean example for understanding modes, eigenvalues, oscillation, and the continuous/discrete relationship is circular motion.

Let $s$ be a physical angular speed in radians per second. Use $s$ instead of $\omega$ here to avoid confusing the angular speed parameter with the continuous-time eigenvalues $\omega_i$.

Consider

$$
\frac{d}{dt}
\begin{bmatrix}
x \\
y
\end{bmatrix}
= A_c
\begin{bmatrix}
x \\
y
\end{bmatrix},
$$

where

$$
A_c =
\begin{bmatrix}
0 & -s \\
s & 0
\end{bmatrix}.
$$

This means

$$
\frac{dx}{dt} = -sy,
$$

and

$$
\frac{dy}{dt} = sx.
$$

This system rotates points around the origin.

To see this, compute the derivative of the squared radius:

$$
\frac{d}{dt}(x^2+y^2) = 2x\frac{dx}{dt} + 2y\frac{dy}{dt}.
$$

Substitute the ODE:

$$
\frac{d}{dt}(x^2+y^2) = 2x(-sy) + 2y(sx).
$$

Therefore,

$$
\frac{d}{dt}(x^2+y^2) = -2sxy + 2sxy = 0.
$$

So $x^2+y^2$ is constant. The trajectory stays on a circle centered at the origin.

If the initial condition is

$$
\vec{x}(0) =
\begin{bmatrix}
1 \\
0
\end{bmatrix},
$$

then the solution is

$$
\vec{x}(t) =
\begin{bmatrix}
\cos(st) \\
\sin(st)
\end{bmatrix}.
$$

So the continuous trajectory is a smooth circle.

The matrix exponential for this system is a rotation matrix:

$$
e^{A_c t} =
\begin{bmatrix}
\cos(st) & -\sin(st) \\
\sin(st) & \cos(st)
\end{bmatrix}.
$$

Thus

$$
\vec{x}(t) = e^{A_c t}\vec{x}(0)
$$

means:

> Rotate the initial state by angle $st$ radians.

The continuous-time eigenvalues of $A_c$ are

$$
\omega_1 = is,
$$

and

$$
\omega_2 = -is.
$$

These have zero real part and nonzero imaginary part, so they produce oscillation without growth or decay.

Now suppose the system is sampled every $\Delta t$ seconds. The sampled states are

$$
\vec{x}_k = \vec{x}(k\Delta t).
$$

The one-step discrete map is

$$
A_d = e^{A_c\Delta t}.
$$

For the circular-motion system,

$$
A_d =
\begin{bmatrix}
\cos(s\Delta t) & -\sin(s\Delta t) \\
\sin(s\Delta t) & \cos(s\Delta t)
\end{bmatrix}.
$$

So the discrete system is

$$
\vec{x}_{k+1} = A_d\vec{x}_k.
$$

This means:

> Each discrete step rotates the point by $s\Delta t$ radians.

The discrete-time eigenvalues are

$$
\lambda_1 = e^{is\Delta t},
$$

and

$$
\lambda_2 = e^{-is\Delta t}.
$$

These lie on the unit circle because their magnitudes are $1$.

So the continuous-time and discrete-time eigenvalues are connected by

$$
\lambda_i = e^{\omega_i\Delta t}.
$$

For example, from $\omega_1 = is$,

$$
\lambda_1 = e^{is\Delta t}.
$$

The continuous eigenvalue $\omega_1=is$ says:

> The mode oscillates at angular frequency $s$ radians per second.

The discrete eigenvalue $\lambda_1=e^{is\Delta t}$ says:

> The mode advances its phase by $s\Delta t$ radians per sampled step.

If $s=\pi$ radians per second, then the physical frequency is

$$
f = \frac{s}{2\pi} = \frac{1}{2}.
$$

So the system completes $0.5$ cycles per second, or one full cycle every $2$ seconds.

If the system is sampled at $30$ frames per second, then

$$
\Delta t = \frac{1}{30}.
$$

The discrete eigenvalue is

$$
\lambda = e^{i\pi/30}.
$$

That means each frame advances the phase by

$$
\frac{\pi}{30}
$$

radians, or $6^\circ$ per frame.

After $60$ frames,

$$
\lambda^{60} = \left(e^{i\pi/30}\right)^{60} = e^{i2\pi} = 1.
$$

So the system completes one full cycle every $60$ frames.


$$
\text{frames/cycle} = \frac{30 \text{ frames/second}}{0.5 \text{ cycles/second}} = 60 \text{ frames/cycle}.
$$

So this system can be interpreted in both units:

* $0.5$ cycles per second,
* $1/60$ cycles per frame,
* $2$ seconds per cycle,
* $60$ frames per cycle.

---

## 14. Continuous Time Versus Discrete Time in the Circular Example

The continuous trajectory is

$$
\vec{x}(t) =
\begin{bmatrix}
\cos(st) \\
\sin(st)
\end{bmatrix}.
$$

This gives the state at every possible time $t$.

The discrete sampled trajectory is

$$
\vec{x}_k =
\begin{bmatrix}
\cos(sk\Delta t) \\
\sin(sk\Delta t)
\end{bmatrix}.
$$

This gives the state only at sampled times.

So the continuous system moves smoothly around the circle, while the discrete system jumps from one sampled point to the next.

If $\Delta t$ is small, the sampled points are close together and look like a smooth circle.

If $\Delta t$ is large, the sampled points are spread farther apart.

The continuous and discrete systems are not different physical systems here. They are two descriptions of the same motion:

* continuous time describes every instant,
* discrete time describes sampled instants.

For video data, the pendulum moves continuously, but the camera records frames at discrete times. DMD sees the discrete samples.

---

## 15. Worked Example: A Parabola Is Not a Homogeneous 2D Linear ODE

Consider the curve

$$
x(t)=t,
$$

and

$$
y(t)=t^2.
$$

Then the state is

$$
\vec{z}(t) =
\begin{bmatrix}
x(t) \\
y(t)
\end{bmatrix}
= \begin{bmatrix}
t \\
t^2
\end{bmatrix}.
$$

The derivative is

$$
\frac{d\vec{z}}{dt} =
\begin{bmatrix}
1 \\
2t
\end{bmatrix}.
$$

Suppose there were a constant $2 \times 2$ matrix

$$
A =
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
$$

such that

$$
\frac{d\vec{z}}{dt} = A\vec{z}.
$$

Then we would need

$$
\begin{bmatrix}
1 \\
2t
\end{bmatrix}
= \begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
\begin{bmatrix}
t \\
t^2
\end{bmatrix}
= \begin{bmatrix}
at + bt^2 \\
ct + dt^2
\end{bmatrix}.
$$

The first component would require

$$
1 = at + bt^2
$$

for all $t$. This is impossible, because the right side has no constant term. At $t=0$, it gives $1=0$.

So there is no constant $2 \times 2$ matrix $A$ such that

$$
\frac{d}{dt}
\begin{bmatrix}
t \\
t^2
\end{bmatrix}
= A
\begin{bmatrix}
t \\
t^2
\end{bmatrix}.
$$

This is an important lesson:

> Moving along a simple curve does not automatically mean the motion comes from a homogeneous linear ODE in the original variables.

---

## 16. The Parabola as an Affine Linear System

The natural equations for the parabola are

$$
\frac{dx}{dt}=1,
$$

and

$$
\frac{dy}{dt}=2t.
$$

Since $x(t)=t$, the second equation can be written as

$$
\frac{dy}{dt}=2x.
$$

So the system is

$$
\frac{dx}{dt}=1,
$$

$$
\frac{dy}{dt}=2x.
$$

In vector form,

$$
\frac{d}{dt}
\begin{bmatrix}
x \\
y
\end{bmatrix}
=
\begin{bmatrix}
0 & 0 \\
2 & 0
\end{bmatrix}
\begin{bmatrix}
x \\
y
\end{bmatrix}
+\begin{bmatrix}
1 \\
0
\end{bmatrix}
$$

This has the form

$$
\vec{z}' = A\vec{z} + \vec{c}.
$$

So the parabola is an affine linear system, not a homogeneous linear system.

With initial condition

$$
x(0)=0,
$$

and

$$
y(0)=0,
$$

the solution is

$$
x(t)=t,
$$

and

$$
y(t)=t^2.
$$

The constant term $\vec{c}$ is what allows $x$ to start moving even when $x=0$ and $y=0$.

---

## 17. Lifting the Parabola to a Homogeneous Linear System

The affine system can be converted into a homogeneous linear system by adding a constant feature.

Define an augmented state

$$
\vec{s}(t) =
\begin{bmatrix}
1 \\
x(t) \\
y(t)
\end{bmatrix}.
$$

Then

$$
\frac{d}{dt}
\begin{bmatrix}
1 \\
x \\
y
\end{bmatrix}
=\begin{bmatrix}
0 \\
1 \\
2x
\end{bmatrix}.
$$

This can be written as

$$
\frac{d\vec{s}}{dt} = B\vec{s},
$$

where

$$
B =
\begin{bmatrix}
0 & 0 & 0 \\
1 & 0 & 0 \\
0 & 2 & 0
\end{bmatrix}.
$$

Check:

$$
B
\begin{bmatrix}
1 \\
x \\
y
\end{bmatrix}
= \begin{bmatrix}
0 \\
1 \\
2x
\end{bmatrix}.
$$

So the parabola becomes a homogeneous linear ODE in the augmented state space.

With

$$
\vec{s}(0) =
\begin{bmatrix}
1 \\
0 \\
0
\end{bmatrix},
$$

the solution is

$$
\vec{s}(t) =
\begin{bmatrix}
1 \\
t \\
t^2
\end{bmatrix}.
$$

This is related to a broader idea: nonlinear or affine dynamics in one coordinate system can sometimes become linear after lifting the state into a richer feature space.

This is conceptually related to Koopman-style thinking and to why DMD can sometimes discover useful linear structure in high-dimensional measurements.

---

## 18. The Nilpotent Matrix in the Parabola Example

The lifted parabola matrix is

$$
B =
\begin{bmatrix}
0 & 0 & 0 \\
1 & 0 & 0 \\
0 & 2 & 0
\end{bmatrix}.
$$

This matrix has only one eigenvalue:

$$
\omega = 0.
$$

Because this is a continuous-time generator, the eigenvalue is a continuous-time eigenvalue.

The eigenvectors satisfy

$$
B\vec{v}=0.
$$

Let

$$
\vec{v} =
\begin{bmatrix}
a \\
b \\
c
\end{bmatrix}.
$$

Then

$$
B\vec{v} =
\begin{bmatrix}
0 \\
a \\
2b
\end{bmatrix}.
$$

For this to equal zero, we need

$$
a=0,
$$

and

$$
b=0.
$$

So the eigenspace is

$$
\operatorname{span}\left\{
\begin{bmatrix}
0 \\
0 \\
1
\end{bmatrix}
\right\}
$$

The only true eigenvector direction is the pure $y$ direction.

Along this true eigenvector, the system is stationary:

$$
e^{0t}
\begin{bmatrix}
0 \\
0 \\
1
\end{bmatrix}
=\begin{bmatrix}
0 \\
0 \\
1
\end{bmatrix}.
$$

So the true eigenvector mode does not grow, decay, or oscillate.

However, the actual initial condition for the parabola is

$$
\vec{s}(0) =
\begin{bmatrix}
1 \\
0 \\
0
\end{bmatrix}.
$$

That vector is not an eigenvector.

The matrix $B$ is nilpotent:

$$
B^3 = 0.
$$

Therefore the matrix exponential terminates after finitely many terms:

$$
e^{Bt} = I + Bt + \frac{B^2t^2}{2}.
$$

Now apply this to the initial condition:

$$
\vec{s}(t) = e^{Bt}\vec{s}(0).
$$

Using

$$
\vec{s}(0) =
\begin{bmatrix}
1 \\
0 \\
0
\end{bmatrix},
$$

we get

$$
B
\begin{bmatrix}
1 \\
0 \\
0
\end{bmatrix}
=\begin{bmatrix}
0 \\
1 \\
0
\end{bmatrix},
$$

and

$$
B^2
\begin{bmatrix}
1 \\
0 \\
0
\end{bmatrix}
=\begin{bmatrix}
0 \\
0 \\
2
\end{bmatrix}.
$$

So

$$
\vec{s}(t)
=\begin{bmatrix}
1 \\
0 \\
0
\end{bmatrix}
+t
\begin{bmatrix}
0 \\
1 \\
0
\end{bmatrix}
+\frac{t^2}{2}
\begin{bmatrix}
0 \\
0 \\
2
\end{bmatrix}.
$$

Therefore,

$$
\vec{s}(t) =
\begin{bmatrix}
1 \\
t \\
t^2
\end{bmatrix}.
$$

The parabola comes from polynomial terms in the matrix exponential, not from separate oscillatory eigenmodes.

This happens because $B$ is not diagonalizable. The motion is governed by a generalized eigenvector chain:

$$
\begin{bmatrix}
1 \\
0 \\
0
\end{bmatrix}
\mapsto
\begin{bmatrix}
0 \\
1 \\
0
\end{bmatrix}
\mapsto
\begin{bmatrix}
0 \\
0 \\
2
\end{bmatrix}
\mapsto
\begin{bmatrix}
0 \\
0 \\
0
\end{bmatrix}.
$$

Ignoring the factor of $2$, the chain is:

$$
\text{constant feature}
\rightarrow
x\text{-direction}
\rightarrow
y\text{-direction}
\rightarrow 0.
$$

This example teaches a different lesson from the circular-motion example:

* the circular-motion example shows ordinary oscillatory eigenmodes,
* the parabola example shows affine lifting, nilpotent matrices, generalized eigenvectors, and polynomial dynamics.

Both are useful, but the circular-motion example is closer to the pendulum/DMD frequency story.

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

This smaller matrix approximates the time-advance dynamics inside the rank-$r$ subspace: $\tilde{A}$ is the low-dimensional representation of the unknown full time-advance operator $A_d$ after projecting into the rank-$r$ SVD/POD subspace (it describes how the dynamics act inside the low-rank subspace spanned by the columns of $U_r$).

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

Therefore a DMD mode can be reshaped into an image.

However, a DMD mode is not usually a literal frame from the video.

A frame is the full state at one time:

$$
\vec{x}_k.
$$

A DMD mode is a reusable spatial pattern that contributes to many frames over time.

The reconstructed frame is a sum of mode contributions:

$$
\vec{x}*k \approx \sum*{i=1}^{r} b_i\phi_i\lambda_i^k.
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
\vec{x}*k \approx \sum*{i=1}^{r} b_i\phi_i\lambda_i^k.
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

## 24. Summary

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
