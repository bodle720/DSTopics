# Part 1: Modes, Eigenvalues, Matrix Exponentials, and DMD

This series of notes gives background for understanding Dynamic Mode Decomposition (DMD), especially the relationship between continuous-time dynamics, discrete-time dynamics, eigenvalues, modes, matrix exponentials, and video data.

See [part 2](02_a_couple_examples.md) after this note.

## 0. Continuous-Time Motivation

Let $\vec{x}$ represent the state of a system. The state vector contains the quantities we wish to measure or model. In these notes, $\vec{x}$ may eventually represent the pixels in a single video frame. More generally, $\vec{x}$ could represent fluid velocities, temperatures, sensor readings, financial features, or any collection of measured quantities.

A general continuous-time dynamical system can be written as

$$
\frac{d\vec{x}}{dt} = f(\vec{x}, t, \vec{\mu}),
$$

where:

* $\vec{x}$ is the state vector,
* $t$ is time,
* $\vec{\mu}$ represents possible system parameters,
* and $f$ describes how the state changes.

The left-hand side,

$$
\frac{d\vec{x}}{dt},
$$

is the derivative of the state vector. It describes how the state changes at a given time $t$.

In many real systems, the function $f$ may be nonlinear, unknown, difficult to model from first principles, or dependent on unobserved variables. Discovering and studying $f$ is highly context-dependent and can represent an entire field of study.

One crucial simplification behind DMD is that we approximate the observed dynamics with a linear model. Instead of trying to discover the full nonlinear function $f$, DMD asks whether the observed evolution can be approximated by a linear operator.

A simplified continuous-time linear system has the form

$$
\frac{d\vec{x}}{dt} = A_c\vec{x}.
$$

Here, $A_c$ is a matrix that describes how the components of the state interact linearly. The subscript $c$ stands for continuous.

If $\vec{x} \in \mathbb{R}^n$, then $A_c$ has shape

$$
n \times n.
$$

For image or video data, $n$ can be very large. For example, if a grayscale frame has height $h$ and width $w$, then the flattened frame has dimension

$$
n = h \cdot w.
$$

So even a modest image can produce a very large state vector.

This linear model is a major simplification. The original system may be nonlinear, time-dependent, or affected by parameters we do not observe. However, once we assume a linear, time-invariant approximation,

$$
\frac{d\vec{x}}{dt}=A_c\vec{x},
$$

we enter the setting of constant-coefficient linear systems of ordinary differential equations.

This is the setting where the matrix-exponential solution is available:

$$
\vec{x}(t)=e^{A_ct}\vec{x}(0).
$$

Here, $\vec{x}(0)$ is the initial condition, or starting state, at time $t = 0$.

This equation says that the state at time $t$ can be obtained by applying the matrix exponential $e^{A_ct}$ to the initial state.

The clean eigenvalue/eigenvector representation that follows depends on this linear approximation. In the fully nonlinear case, there is generally no single finite-dimensional matrix $A_c$ whose eigenvectors and eigenvalues describe the whole system globally. DMD begins from this linear picture and then estimates an approximate linear time-advance model directly from data.


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

So $e^{A_c t}$ is the time-t flow map of the linear system.

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

This is the same idea that appears in ordinary differential equations: the general solution is a linear combination of independent solutions, and the initial condition determines the coefficients in that linear combination.
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
= \begin{bmatrix}
b_1e^{\omega_1t} \\
b_2e^{\omega_2t} \\
\vdots \\
b_ne^{\omega_nt}
\end{bmatrix}.
$$

Multiplying by $V$ forms a linear combination of the columns of $V$:

$$
\vec{x}(t)
= b_1e^{\omega_1t}\vec{v}_1
+b_2e^{\omega_2t}\vec{v}_2
+\cdots
+b_ne^{\omega_nt}\vec{v}_n.
$$

So the modal-sum formula is

$$
\vec{x}(t)
= \sum_{i=1}^{n}
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

The first formula is the time-k solution from the initial condition.

The second formula is the one-step transition from sample $k$ to sample $k+1$.

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

See [part 2](02_a_couple_examples.md) next for a couple worked examples.