# Nyquist Frequency, DMD Eigenvalues, and Pendulum Video Interpretation

This note explains how to interpret DMD frequencies in the synthetic pendulum video project, especially the difference between the physically meaningful pendulum frequency and high-frequency artifacts near the Nyquist frequency.

After this, see the [notebook](/DMD/DMD_pendulum_video.ipynb) for the code implementation.

The goal is to connect four ideas:

* discrete DMD eigenvalues, denoted by $\lambda$,
* continuous-time DMD eigenvalues, denoted by $\omega$,
* frequency in cycles per second, measured in Hz,
* and the Nyquist frequency imposed by the video frame rate.

## 1. What Frequency Means in This Pendulum Video

The synthetic pendulum is generated so that its visible motion repeats every 2 seconds. That means the main visible pendulum pattern has frequency

$$
f_{\text{true}} = \frac{1}{2} = 0.5 \text{ Hz}.
$$

This does not mean that DMD knows anything about pendulums as physical objects. DMD only sees snapshot data. In this project, the snapshots are grayscale video frames.

So when DMD identifies a frequency near $0.5$ Hz, the correct interpretation is:

> DMD has found a video pattern whose time evolution repeats about once every 2 seconds.

That pattern is connected to the visible pendulum swing because the image sequence itself repeats with the pendulum motion.

A “cycle” is not just any arbitrary event we name. A meaningful cycle in the data is a repeated observable pattern. For this video, the full visible pendulum motion approximately repeats every 2 seconds, so $0.5$ Hz is meaningful in the snapshot data.

## 2. Discrete DMD Eigenvalues: Per-Frame Dynamics

DMD learns a linear time-advance model from snapshot pairs. In the mean-centered video setup, the model is fit to centered frames:

$$
\vec{y}_k = \vec{x}_k - \bar{\vec{x}}.
$$

The paired DMD matrices are

$$
Y = [\vec{y}_1 ; \vec{y}*2 ; \cdots ; \vec{y}*{m-1}]
$$

and

$$
Y' = [\vec{y}_2 ; \vec{y}_3 ; \cdots ; \vec{y}_m].
$$

DMD estimates a time-advance relationship

$$
Y' \approx AY.
$$

The DMD eigenvalues $\lambda_i$ describe how each DMD mode changes from one frame to the next. A discrete eigenvalue can be written as

$$
\lambda = \rho e^{i\theta}.
$$

Here:

* $\rho = |\lambda|$ is the magnitude,
* $\theta = \arg(\lambda)$ is the phase angle,
* $\rho$ controls growth or decay per frame,
* $\theta$ controls phase rotation per frame.

If $|\lambda| < 1$, the corresponding mode decays over time.

If $|\lambda| > 1$, the corresponding mode grows over time.

If $|\lambda| \approx 1$, the corresponding mode is persistent.

If $\theta = 0$, the mode does not oscillate in phase.

If $\theta \neq 0$, the mode rotates in the complex plane from frame to frame, producing oscillatory behavior.

## 3. From Discrete Eigenvalues to Continuous-Time Eigenvalues

The discrete eigenvalue $\lambda$ describes one step forward in frame index.

To interpret the dynamics in physical time, DMD converts $\lambda$ into a continuous-time eigenvalue $\omega$ using

$$
\omega = \frac{\log(\lambda)}{\Delta t}.
$$

This comes from the relationship

$$
\lambda = e^{\omega \Delta t}.
$$

The continuous-time eigenvalue can be written as

$$
\omega = \alpha + i\beta.
$$

Here:

* $\alpha = \operatorname{Re}(\omega)$ is the growth or decay rate per second,
* $\beta = \operatorname{Im}(\omega)$ is the angular frequency in radians per second.

This is an important separation:

* $\lambda$ is the discrete per-frame eigenvalue.
* $\omega$ is the continuous-time eigenvalue after accounting for the frame timestep $\Delta t$.

The angle of $\lambda$ tells us phase motion per frame.

The imaginary part of $\omega$ tells us angular frequency per second.

## 4. Why Divide by $2\pi$?

The imaginary part of $\omega$ is an angular frequency in radians per second.

But Hz means cycles per second.

One full rotation around the complex plane is

$$
2\pi \text{ radians}.
$$

So to convert radians per second into cycles per second, divide by $2\pi$:

$$
f = \frac{\operatorname{Im}(\omega)}{2\pi}.
$$

This is why the code uses

$$
\text{frequency_hz} = \frac{\operatorname{Im}(\omega)}{2\pi}.
$$

This conversion takes the rotation rate of a DMD mode in the complex plane and expresses it as a frequency in ordinary physical units: cycles per second.

## 5. The Complex Plane Interpretation

A DMD mode with a complex eigenvalue rotates in the complex plane as time advances.

If

$$
\lambda = \rho e^{i\theta},
$$

then after $k$ frame steps, the modal contribution includes

$$
\lambda^k = \rho^k e^{ik\theta}.
$$

The value $k\theta$ is the accumulated phase after $k$ frames.

If the angle $\theta$ is small, the mode rotates slowly around the complex plane.

If the angle $\theta$ is large, the mode rotates quickly.

A full numerical cycle corresponds to a $2\pi$ radian rotation. That is the complex-plane version of “one cycle.”

For the pendulum, the visible motion repeats every 2 seconds. A DMD mode near $0.5$ Hz means the mode’s complex coefficient completes about one full rotation every 2 seconds.

That is how DMD translates repeated video motion into an eigenvalue frequency.

## 6. Expected Phase Step for the Pendulum

The video is sampled at 30 frames per second, so

$$
\Delta t = \frac{1}{30}.
$$

The true pendulum frequency is

$$
f_{\text{true}} = 0.5 \text{ Hz}.
$$

The expected phase step per frame for a 0.5 Hz oscillation is

$$
\theta_{\text{true}} = 2\pi f_{\text{true}} \Delta t.
$$

Substituting the values gives

$$
\theta_{\text{true}}
= 2\pi(0.5)\left(\frac{1}{30}\right)
= \frac{\pi}{30}
\approx 0.105 \text{ radians/frame}.
$$

So a DMD eigenvalue associated with the pendulum’s main oscillation should have an angle near

$$
\pm \frac{\pi}{30}.
$$

Equivalently, after converting to continuous time, the DMD frequency should be near

$$
\pm 0.5 \text{ Hz}.
$$

The positive and negative signs usually appear as a complex-conjugate pair. Together, that pair represents an oscillation.

## 7. What the Nyquist Frequency Means

The video is sampled at a fixed frame rate. A 30 fps video gives 30 samples per second.

But 30 samples per second does not mean we can represent 30 cycles per second.

To observe one oscillation cycle, we need at least two samples: one sample on one side of the cycle and another sample on the opposite side. Therefore, the fastest frequency that can be represented without ambiguity is half the sampling rate.

This is the Nyquist frequency:

$$
f_{\text{Nyquist}} = \frac{f_s}{2},
$$

where $f_s$ is the sampling rate.

For video,

$$
f_s = \text{fps}.
$$

So

$$
f_{\text{Nyquist}} = \frac{\text{fps}}{2}.
$$

At 30 fps,

$$
f_{\text{Nyquist}} = \frac{30}{2} = 15 \text{ Hz}.
$$

This means 15 Hz is the fastest representable oscillation in a 30 fps video.

A 15 Hz oscillation has a period of

$$
\frac{1}{15} \text{ seconds}.
$$

At 30 fps, that is exactly 2 frames:

$$
\frac{2}{30} = \frac{1}{15}.
$$

So a 15 Hz sampled oscillation alternates every frame:

$$
+,; -,; +,; -,; \ldots
$$

## 8. Why a Negative Real Eigenvalue Gives a Nyquist Frequency

A negative real eigenvalue has phase angle

$$
\theta = \pi.
$$

That means each frame step advances the phase by $\pi$ radians.

Since a full cycle is $2\pi$ radians, a step of $\pi$ radians is half a cycle per frame.

This also means the mode flips sign every frame:

$$
\lambda^k = \rho^k e^{ik\pi} = \rho^k(-1)^k.
$$

So the time pattern is

$$
+,; -,; +,; -,; \ldots
$$

This has a 2-frame period. At 30 fps, a 2-frame period corresponds to 15 Hz.

That is why negative real eigenvalues often appear as Nyquist-frequency modes in DMD frequency tables.

## 9. What Aliasing Means

Aliasing happens when a continuous signal is sampled too slowly to distinguish its true frequency.

If a signal oscillates faster than the Nyquist frequency, the sampled data can make it appear like a different, lower frequency. The classic visual example is a wagon wheel in a video: the wheel may be spinning forward quickly, but because the camera samples it only at discrete times, the wheel can appear to spin slowly or backward.

In the DMD setting, aliasing is related to the fact that eigenvalue phase is observed only at discrete frame intervals.

The discrete eigenvalue angle is effectively limited to a principal range such as

$$
-\pi \leq \theta \leq \pi.
$$

A phase step of $\pi$ radians per frame is already the boundary case: the fastest distinguishable sampled oscillation.

That boundary corresponds to the Nyquist frequency.

So when a DMD mode appears at 15 Hz in a 30 fps video, it is sitting at the maximum representable sampled frequency.

In this pendulum project, a 15 Hz mode should not be interpreted as the pendulum physically swinging at 15 Hz. The pendulum is generated with a 2-second period, so its meaningful physical frequency is 0.5 Hz.

A 15 Hz DMD mode is more likely a pixel-space artifact, residual correction, or high-rank sign-flipping mode.

## 10. Why Higher Rank Can Be Harmful

The DMD rank controls how many singular directions are retained.

A higher rank preserves more pixel-space variation, which can improve reconstruction of observed frames. However, preserving more variation is not always better for learning meaningful dynamics.

In full-frame video, higher-rank directions can include:

* small moving-edge residuals,
* pixel-level correction patterns,
* weak directions with little dynamical importance,
* numerical artifacts,
* or modes that alternate sign every frame.

These extra directions can produce eigenvalues near the negative real axis, which appear as Nyquist-frequency modes.

This is why higher-rank DMD can produce worse physical interpretation even while preserving more centered image energy.

The rank should therefore be chosen using both:

1. the singular-value spectrum, which shows how much image variation is retained, and
2. the eigenvalue/frequency diagnostic, which checks whether the learned dynamics are meaningful.

In this notebook, the rank-frequency diagnostic showed that an intermediate rank recovered a DMD frequency closer to the known 0.5 Hz pendulum motion, while some higher ranks produced Nyquist-like artifacts.

The lesson is:

> The best DMD rank is not necessarily the rank that preserves the most pixel-space energy.

## 11. How This Connects to the Pendulum Video

The pendulum video is a controlled synthetic system. The visible pendulum motion repeats every 2 seconds, so the target physical frequency is 0.5 Hz.

DMD does not directly know the pendulum angle, the pendulum equation, or the generator settings. It only sees snapshots.

When DMD finds a mode pair near 0.5 Hz, it means that the selected observable contains a repeated pattern that DMD can represent as a rotating mode in the complex plane.

For full-frame video, that observable is the mean-centered grayscale image. The mode is not literally “the pendulum” in an object-tracking sense. Instead, it is a spatial pattern over pixels whose coefficient oscillates at a rate close to the visible swing.

This matches the practical interpretation:

> DMD is finding a video pattern that oscillates at the pendulum’s visible frequency.

The frequency comes from the mode’s eigenvalue angle, and the conversion to Hz uses the frame timestep.

## 12. Reiterating the Main Intuition

The video is sampled in frames, but DMD represents each mode’s time evolution through eigenvalues.

The discrete eigenvalue $\lambda$ describes the mode’s step from one frame to the next.

The eigenvalue angle tells us how far the mode rotates in the complex plane per frame.

The timestep $\Delta t$ converts that per-frame rotation into a per-second rate.

The continuous-time eigenvalue $\omega$ is computed by

$$
\omega = \frac{\log(\lambda)}{\Delta t}.
$$

The imaginary part of $\omega$ is an angular frequency in radians per second.

Dividing by $2\pi$ converts radians per second into cycles per second:

$$
f = \frac{\operatorname{Im}(\omega)}{2\pi}.
$$

So the image dynamics are mapped into a complex-plane representation where one full $2\pi$ rotation is one cycle.

If the mode completes one rotation every 2 seconds, it has frequency 0.5 Hz.

If the mode flips sign every frame in a 30 fps video, it has a 2-frame period and appears at 15 Hz, the Nyquist frequency.

This is why the 0.5 Hz modes are physically meaningful for the pendulum, while 15 Hz modes are usually interpreted as high-frequency sampled artifacts.

## 13. Sampling Rate Thought Experiment

Suppose the same physical pendulum motion is sampled at 10 fps instead of 30 fps.

The true pendulum frequency is still

$$
0.5 \text{ Hz}.
$$

If the sampling is still adequate, DMD should still be able to find a mode near 0.5 Hz.

However, the Nyquist frequency would change because the sampling rate changed:

$$
f_{\text{Nyquist}} = \frac{10}{2} = 5 \text{ Hz}.
$$

So the meaningful pendulum frequency stays the same, but the maximum representable sampled frequency changes.

At 30 fps, sign-flipping modes appear at 15 Hz.

At 10 fps, sign-flipping modes appear at 5 Hz.

This distinction is important:

* The physical frequency is a property of the underlying motion.
* The Nyquist frequency is a property of the sampling rate.
* DMD frequency interpretation depends on both the learned eigenvalue and the timestep used to sample the data.

## 14. Summary

In this pendulum DMD project:

* $\lambda$ is the discrete DMD eigenvalue.
* $\lambda$ describes how a mode changes from one frame to the next.
* $|\lambda|$ controls growth or decay per frame.
* $\arg(\lambda)$ controls phase rotation per frame.
* $\omega$ is the continuous-time eigenvalue.
* $\omega = \log(\lambda) / \Delta t$ converts frame-step dynamics into physical-time dynamics.
* $\operatorname{Im}(\omega)$ is angular frequency in radians per second.
* Dividing by $2\pi$ converts radians per second to Hz.
* The pendulum’s true visible frequency is 0.5 Hz because the generated motion repeats every 2 seconds.
* The Nyquist frequency is half the frame rate.
* At 30 fps, the Nyquist frequency is 15 Hz.
* A 15 Hz DMD mode in this video usually means a sign-flipping frame-to-frame artifact, not the physical pendulum swing.
* Higher rank can preserve more image detail but produce less meaningful dynamics.
* A good DMD rank balances image reconstruction, frequency interpretation, and forecast behavior.

The central idea is that DMD translates repeated visual patterns into eigenvalue rotations. The timestep connects those rotations to physical time. That is why a mode near 0.5 Hz can correspond to the pendulum swing, while a mode near 15 Hz reflects the sampling limit of a 30 fps video.
