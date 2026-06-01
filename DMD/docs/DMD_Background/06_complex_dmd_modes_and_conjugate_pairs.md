# Part 6: Complex DMD Modes, Conjugate Pairs, and Visual Interpretation

This note explains how to interpret complex eigenvalues and complex DMD modes, especially for real-valued video data.

After this, see the [notebook](/DMD/DMD_pendulum_video.ipynb) for the code implementation.

The main questions are:

* If eigenvalues appear as complex conjugates, what does that imply about the eigenvectors or DMD modes?
* If the original matrix and input snapshots are real, why do complex eigenvalues not make the reconstructed data complex?
* What do the real part, imaginary part, and magnitude of a complex DMD mode mean when visualized as images?

## 1. Complex Conjugate Eigenvalues and Eigenvectors

Suppose a real-valued matrix or operator $A$ has an eigenvalue-eigenvector pair

$$
A v = \lambda v.
$$

If $A$ is real-valued and $\lambda$ is complex, then the complex conjugate eigenvalue is also an eigenvalue:

$$
\bar{\lambda}.
$$

The corresponding eigenvector can be chosen as the complex conjugate of $v$:

$$
\bar{v}.
$$

So if

$$
\lambda_1 = \alpha + i\beta
$$

and

$$
v_1 = a + ib,
$$

then the conjugate pair is

$$
\lambda_2 = \alpha - i\beta
$$

and

$$
v_2 = a - ib.
$$

This does not mean the two eigenvectors are necessarily perpendicular. It also does not mean they are ordinary geometric reflections in the real plane. The relationship is complex conjugation: same real part, opposite imaginary part.

For DMD, this means complex-conjugate mode pairs should usually be interpreted together. One member of the pair has positive frequency, and the other has negative frequency. Together, they produce a real-valued oscillatory pattern.

## 2. Why a Real Matrix Still Maps Real Vectors to Real Vectors

If $A$ is a real-valued matrix and $x$ is a real-valued vector, then

$$
Ax
$$

is guaranteed to be real-valued. This is true even if $A$ has complex eigenvalues.

The reason is that the complex eigenvectors are not ordinary real state directions. They are a convenient way to represent rotation and scaling inside a real two-dimensional invariant subspace.

Suppose

$$
\lambda = \alpha + i\beta
$$

and

$$
v = a + ib,
$$

where $a$ and $b$ are real vectors.

Starting from the eigenvalue equation,

$$
A v = \lambda v,
$$

substitute $v = a + ib$:

$$
A(a + ib) = (\alpha + i\beta)(a + ib).
$$

The left side is

$$
Aa + iAb.
$$

The right side expands to

$$
(\alpha a - \beta b) + i(\beta a + \alpha b).
$$

Matching real and imaginary parts gives

$$
Aa = \alpha a - \beta b
$$

and

$$
Ab = \beta a + \alpha b.
$$

This shows that $A$ maps the real span of $a$ and $b$ back into itself. In the real two-dimensional subspace spanned by $a$ and $b$, the action of $A$ behaves like a rotation plus scaling.

That is the real interpretation of complex eigenvalues: they describe rotating and scaling behavior inside a real plane.

## 3. How the Complex Parts Cancel in Real Reconstructions

A real-valued state can be represented using a complex-conjugate eigenvector pair.

For example, a real vector can be written as

$$
x = c v + \bar{c}\bar{v},
$$

where $c$ is a complex coefficient, $v$ is a complex eigenvector, and the second term is the conjugate counterpart.

Since the two terms are conjugates of each other, their sum is real:

$$
c v + \bar{c}\bar{v}
=2\mathrm{Re}(c v).
$$

After applying $A$, we get

$$
Ax = c\lambda v + \bar{c}\bar{\lambda}\bar{v}.
$$

Again, the two terms are conjugates of each other, so their sum is real.

This is why complex eigenvalues and eigenvectors do not imply that a real dynamical system suddenly produces complex-valued physical states. The complex representation is a mathematical bookkeeping tool. The conjugate pair together produces real-valued motion.

For DMD, this is why complex-conjugate modes should be interpreted as a pair rather than as two unrelated image patterns.

## 4. DMD Modes in Video Data

In the pendulum video notebook, DMD is fit to mean-centered grayscale frames:

$$
\vec{y}_k = \vec{x}_k - \bar{\vec{x}}.
$$

Each DMD mode has the same dimension as a flattened frame:

$$
\phi_i \in \mathbb{C}^n.
$$

Since $n = h \cdot w$, each mode can be reshaped into an $h \times w$ image.

However, a DMD mode is not itself an ordinary video frame. It is a learned spatial pattern tied to a temporal behavior. The time behavior is controlled by the corresponding eigenvalue.

A DMD reconstruction has the form

$$
\hat{\vec{y}}_k =\sum_i b_i \phi_i \lambda_i^k,
$$

where:

* $\phi_i$ is a DMD mode,
* $\lambda_i$ is the corresponding discrete DMD eigenvalue,
* $b_i$ is the modal amplitude,
* and $k$ is the snapshot step.

Because the model is fit to mean-centered frames, $\hat{\vec{y}}_k$ is a predicted deviation from the average frame. To convert back to ordinary grayscale image space, the mean frame is added back:

$$
\hat{\vec{x}}_k = \bar{\vec{x}} + \hat{\vec{y}}_k.
$$

## 5. Real-Valued Modes

If a DMD mode is real-valued, it can be visualized directly as an image.

Positive and negative regions show where that mode adds or subtracts brightness relative to the mean frame.

For a real-valued mode, a mode image can often be interpreted as:

> This is the spatial region or pattern controlled by this mode.

In the pendulum video, a real mode might highlight broad deviations from the average pendulum image, moving-edge structure, or other spatial patterns involved in the video dynamics.

## 6. Complex-Valued Modes

A complex DMD mode can be written as

$$
\phi = a + ib,
$$

where $a$ and $b$ are real-valued spatial patterns.

Here:

* $a = \mathrm{Re}(\phi)$ is the real-part image,
* $b = \mathrm{Im}(\phi)$ is the imaginary-part image.

The real and imaginary parts are not two separate physical frames. Instead, they are phase-shifted spatial components of one oscillatory mode pair.

If the eigenvalue has angle $\theta$, then the oscillatory contribution changes phase over time. A simplified real-valued contribution from the mode pair has the form

$$
a\cos(k\theta) - b\sin(k\theta).
$$

This means the real-part image and imaginary-part image combine with changing cosine and sine weights as time advances.

For video interpretation:

* the real part shows one spatial phase of the oscillatory pattern,
* the imaginary part shows a phase-shifted companion pattern,
* the two together describe how the pattern oscillates over time.

In the pendulum example, a complex mode pair near the pendulum frequency may show spatial activity around the arm, bob, or moving edges. The real and imaginary parts can highlight different phases of that swinging motion.

## 7. What “Mode Magnitude” Means

The word “magnitude” can refer to two different things.

The vector norm of a DMD mode is one scalar:

$$
|\phi_i|.
$$

But when we visualize the “magnitude of a mode” as an image, we usually mean the elementwise complex magnitude at each pixel.

For a mode entry

$$
\phi_i[j] = a_j + ib_j,
$$

the elementwise magnitude is

$$
|\phi_i[j]| =\sqrt{a_j^2 + b_j^2}.
$$

Doing this for every pixel gives an image:

$$
|\phi_i|.
$$

This image answers:

> Where is this mode spatially active, regardless of phase?

The mode magnitude image loses sign and phase information, but it is useful for seeing where the mode has support. In a video, this can reveal whether a mode is concentrated around the bob, the arm, moving edges, or broad background regions.

So:

* the real part shows one phase-sensitive spatial pattern,
* the imaginary part shows the phase-shifted companion pattern,
* the elementwise magnitude shows where the mode is active regardless of phase.

## 8. Why the Real and Imaginary Split Is Not Unique

One subtle issue is that a complex DMD mode can be multiplied by a complex phase factor without changing the underlying oscillatory subspace.

For example,

$$
\phi
$$

and

$$
e^{i\gamma}\phi
$$

represent the same mode up to a phase shift, as long as the amplitude is adjusted consistently.

This means the exact visual appearance of the real and imaginary parts can change depending on phase convention. The real part and imaginary part are still useful, but they should not be overinterpreted as uniquely defined physical frames.

The elementwise magnitude image is often more stable for answering:

> Where is this mode active?

The real and imaginary images are useful for answering:

> What are the phase-shifted spatial components of this oscillatory pattern?

## 9. Interpreting Complex DMD Modes in the Pendulum Notebook

In the pendulum video notebook, modes 7 and 8 form a complex-conjugate pair whose frequency is close to the true pendulum frequency.

That means the pair is associated with an oscillatory image pattern near the visible swing rate.

The pair should be interpreted together:

* one mode has positive frequency,
* the conjugate mode has negative frequency,
* their spatial modes are complex conjugates,
* together they produce a real-valued oscillatory contribution.

If the mode magnitude is concentrated around the arm, bob, or moving edges, that supports the interpretation that the pair is connected to the pendulum motion.

However, the pair may still be damped. If $|\lambda_i| < 1$ or $\mathrm{Re}(\omega_i) < 0$, the mode decays over time in the DMD model. This means DMD may detect the pendulum-like frequency while still representing the full-frame video dynamics through a mixture of decaying spatial modes.

This is an important distinction:

> Finding a frequency near the pendulum swing does not mean DMD has found one perfect physical pendulum oscillator.

It means DMD has found an image-space pattern whose time evolution has a frequency close to the visible pendulum motion.

## 10. Practical Visualization Guide

For each important DMD mode or conjugate pair, useful plots include:

1. **Real part**
   Shows one phase-sensitive spatial pattern.

2. **Imaginary part**
   Shows the phase-shifted companion spatial pattern.

3. **Elementwise magnitude**
   Shows where the mode is active regardless of phase.

4. **Eigenvalue location in the complex plane**
   Shows growth or decay through distance from the unit circle and oscillation through angle.

5. **Frequency and period table**
   Connects eigenvalue angle to physical-time frequency using the timestep.

For the pendulum notebook, the most important interpretation is not simply which mode has the largest amplitude. Instead, the key question is:

> Which mode pair has a frequency close to the known pendulum frequency, and where is that mode pair spatially active?

This is why the notebook examines both eigenvalue diagnostics and mode images.

## 11. How Complex Mode Parts Combine Into Real Pixel Values

A complex DMD mode should be thought of as an oscillating image pattern, not as one literal image.

For one DMD mode, the time evolution has the form

$$
\hat{\vec{y}}_k = \phi_i b_i \lambda_i^k.
$$

Here, $\phi_i$ is the spatial mode, $b_i$ is the modal amplitude, $\lambda_i^k$ evolves the mode through time, and $\hat{\vec{y}}_k$ is still in mean-centered image space.

If $\phi_i$, $b_i$, and $\lambda_i$ are complex, this single term is generally complex-valued by itself. A physical video frame cannot have complex pixel intensities. The real-valued pixel interpretation comes from combining complex-conjugate pairs.

For a conjugate pair,

$$
\lambda_2 = \bar{\lambda}_1,
\qquad
\phi_2 = \bar{\phi}_1,
\qquad
b_2 = \bar{b}_1.
$$

The combined contribution is

$$
\phi b \lambda^k + \bar{\phi}\bar{b}\bar{\lambda}^k
=2\mathrm{Re}\left(\phi b \lambda^k\right).
$$

This final expression is real-valued. That is what gives the conjugate-pair contribution real pixel meaning.

Suppose one complex mode is written as

$$
\phi = a + ic,
$$

where

$$
a = \mathrm{Re}(\phi),
\qquad
c = \mathrm{Im}(\phi).
$$

After reshaping the mode into image form, $a$ is the real-part image and $c$ is the imaginary-part image. These are not two separate physical frames. They are two phase components of one oscillatory spatial pattern.

For an oscillatory eigenvalue written as

$$
\lambda = \rho e^{i\theta},
$$

a simplified real-valued contribution from the conjugate pair has the form

$$
\vec{y}_k
\approx
\rho^k
\left[
a \cos(k\theta)
c \sin(k\theta)
\right],
$$

up to scaling and phase from the modal amplitude $b$.

This equation is the key interpretation. The reconstructed centered frame does not use only the real-part image or only the imaginary-part image. Instead, it blends the two with changing cosine and sine weights as time advances. At one phase of the oscillation, the real-part image may dominate. A quarter-cycle later, the imaginary-part image may dominate. Between those phases, the contribution is a mixture of both.

This means the real part is not automatically "the physical image" while the imaginary part is "nonphysical." Both are real-valued images after extracting their components, but both are phase-dependent pieces of a complex oscillatory mode. The most directly physical object is the combined time-dependent contribution of the conjugate pair:

$$
2\mathrm{Re}\left(\phi b \lambda^k\right).
$$

That gives a real-valued centered-frame contribution at a specific time step $k$. To return to ordinary grayscale image space, the mean frame is added back:

$$
\hat{\vec{x}}_k
=\bar{\vec{x}} + \hat{\vec{y}}_k.
$$

There is also an important phase-convention issue. A complex mode can be multiplied by a complex phase factor,

$$
\phi \rightarrow e^{i\alpha}\phi,
$$

as long as the modal amplitude is adjusted consistently:

$$
b \rightarrow e^{-i\alpha}b.
$$

The reconstruction stays the same, but the plotted real and imaginary parts change. Therefore, the individual real-part and imaginary-part images are useful diagnostics, but they should not be overinterpreted as uniquely defined physical frames.

A practical hierarchy is:

```text
Most directly physical:
conjugate-pair contribution at a specific time

Useful but phase-dependent:
real part and imaginary part of one complex mode

Useful for spatial support:
mode magnitude image
```

The magnitude image

$$
|\phi[j]|
=\sqrt{
\mathrm{Re}(\phi[j])^2
+\mathrm{Im}(\phi[j])^2
}
$$

removes the phase information and answers:

> Where is this mode active in the image?

For the pendulum notebook, this is often the easiest mode image to interpret visually. If the mode pair is related to the pendulum-frequency motion, its magnitude image should show activity near the arm, bob, moving edges, or swing path. However, magnitude alone does not show whether the mode is positive or negative at a particular time, and it does not show the phase of the oscillation.

For modes 7 and 8 in the notebook, the expected interpretation is:

* the two modes form a conjugate pair,
* their real parts should be similar,
* their imaginary parts should be sign-flipped or phase-flipped,
* their magnitude images should be nearly identical,
* and the pair together represents the near-pendulum-frequency centered image pattern.

## 12. Summary

For real-valued DMD problems:

* complex eigenvalues appear in conjugate pairs,
* their eigenvectors or DMD modes can also be chosen as conjugate pairs,
* individual complex modes are not standalone real physical states,
* the conjugate pair together produces real-valued oscillatory motion,
* the real and imaginary parts of a complex mode are phase-shifted spatial patterns,
* the elementwise magnitude of a complex mode shows where the mode is active,
* and the eigenvalue controls how that spatial pattern evolves over time.

In video DMD, this means complex modes should be interpreted as oscillatory image patterns. The real and imaginary parts describe phase-shifted spatial components, the magnitude image shows spatial activity, and the conjugate pair together explains a real-valued oscillation in the reconstructed video.