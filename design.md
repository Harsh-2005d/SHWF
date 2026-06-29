# WaveFormer-RTC

*A Physics-Informed, Predictive Real-Time Controller for Open-Loop Adaptive Optics*

**Document version 4.0** — Architecture updated to PI-STT (Physics-Informed Spatio-Temporal
Transformer). Weighted CoG replaces standard CoG. VAR model replaced by temporal
Transformer head over 3-frame history. Residual MLP dropped; Tikhonov-regularised IM is
the sole actuator mapping stage. Turbulence diagnostics moved to raw slopes. All component
claims updated accordingly.

---

## 1. Brief Idea About the Solution

WaveFormer-RTC is a machine-learning-driven Real-Time Controller (RTC) for Adaptive
Optics, designed to overcome the fundamental limitations of traditional matrix-multiplication
approaches — which degrade under high noise, severe atmospheric turbulence, and DM
actuator cross-coupling.

The core architecture is **PI-STT** (Physics-Informed Spatio-Temporal Transformer): a
lightweight Transformer Encoder that ingests a **3-frame temporal history** of raw SH-WFS
slope measurements, with **Fried Geometry Positional Encodings** hardcoded into the
embedding layer to enforce the physical spatial topology of the lenslet array from
initialisation. PI-STT jointly performs wavefront reconstruction and temporal prediction in a
single forward pass, directly outputting the **dense predicted phase map W(xᵢ, yᵢ) at frame
t+1** — simultaneously correcting the instantaneous wavefront and compensating for
servo-lag without a separate prediction model.

The pipeline then branches into two operations:

- **Actuator Mapping:** The predicted phase map is converted to DM voltages by a
  pre-computed **Tikhonov-regularised pseudo-inverse Influence Matrix (IM)** in C++.
  Inter-actuator coupling is a linear mechanical effect; a regularised IM is the exact and
  complete solution in simulation, where the DM model is linear by construction.
- **Turbulence Diagnostics (Background Thread):** The Fried parameter (r₀) is estimated
  from the spatial variance of the **raw slope vectors** by fitting to the Kolmogorov
  slope structure function. The coherence time (τ₀) follows from the temporal
  autocorrelation of the same raw slopes. This runs in a decoupled background thread at
  10–50 Hz, never blocking the main control loop.

---

## 2. Unique Selling Proposition (USP)

### 2.1 Physics-Informed Spatio-Temporal Transformer (PI-STT)

Standard CNNs have localised receptive fields and cannot capture the **global spatial
coherence** of Kolmogorov turbulence, where phase perturbations at opposite ends of the
pupil are statistically correlated. Transformer self-attention is global by construction: every
sub-aperture token attends to every other, naturally capturing long-range correlations.

The critical novelty is the **Physics-Informed Positional Encoding**. Standard Transformers
use learned or sinusoidal positional embeddings that treat spatial positions as abstract
indices. PI-STT instead hardcodes the exact physical `(x, y)` lenslet coordinates, including
the `d/2` Fried geometry offset between the lenslet grid and the DM actuator grid, directly
into the embedding layer. This:

- Forces the attention mechanism to respect the physical spatial topology from
  initialisation, rather than learning it from data.
- Eliminates high-frequency spatial hallucinations by preventing the network from
  learning unphysical long-range couplings between geometrically distant sub-apertures.
- Reduces training data requirements — the network does not need to re-discover a known
  physical prior from examples.

### 2.2 Joint Reconstruction and Temporal Prediction in One Pass

Classical AO pipelines separate wavefront reconstruction and servo-lag compensation into
two sequential modules (e.g. reconstructor + VAR predictor), each with its own
approximations and failure modes. PI-STT collapses these into a single Transformer forward
pass:

- **Input:** A 3-frame temporal history `[frame t−2, frame t−1, frame t]`. Each frame
  contributes 144 sub-aperture tokens `[Sx, Sy, I]`. Total sequence length: **432 tokens**.
- **Output:** The **predicted dense phase map W(xᵢ, yᵢ) at frame t+1** — not the current
  reconstruction, but the forward-predicted state at the next loop iteration.

By training directly on the t+1 target, the Transformer's self-attention mechanism learns the
non-linear temporal evolution of frozen-flow turbulence advection from data, without
imposing the stationarity assumption of a VAR model. The temporal context is implicit in
the attention weights; no separate predictive model is needed.

### 2.3 Sim-to-Real Robustness via Domain Randomisation

PI-STT is pre-trained on large synthetic datasets generated with **HCIPy**, injecting
physics-informed hardware noise: Poisson photon noise, detector read noise, sub-aperture
dropout, and domain-randomised r₀ and SNR tiers covering a wide range of turbulence
conditions. At deployment, **LoRA** (Parameter-Efficient Fine-Tuning) on a modest real
calibration set bridges the simulation-to-hardware domain gap without requiring full
retraining.

### 2.4 Principled Hybrid Architecture: ML for Nonlinear Sensing, Classical IM for Linear Actuation

The architectural split between PI-STT and the classical IM is deliberate and physically
motivated. Wavefront reconstruction is a genuinely nonlinear, noise-sensitive inverse
problem — the domain where deep learning adds clear value. DM actuation, by contrast, is
a **linear mechanical problem** in simulation: the DM surface is a linear superposition of
actuator influence functions by construction in HCIPy. Applying a neural network to a
linear problem adds parameters, training complexity, and instability risk without benefit.
The Tikhonov-regularised IM is the mathematically exact solution to the linear actuation
problem and costs a single matrix-vector multiply at inference.

---

## 3. Competitor Comparison

| Feature | Traditional AO (Matrix/SVD) | Standard DL AO (U-Net/2D-ViT) | **WaveFormer-RTC (PI-STT)** | Basis | Status |
|---|---|---|---|---|---|
| **Output Representation** | Zonal slopes / phase map | Dense 2D phase map | **Predicted dense phase map at t+1** | Transformer sequence-to-grid | Sound |
| **Inference Latency** | <1 ms (fails at low SNR) | 50–200 ms | **<5 ms (TensorRT, micro-Transformer)** | 2–4 layers, dim 128–256, seq 432 | Sound |
| **Temporal Servo-Lag** | Reactive (2–3 frame error) | Reactive / heavy RNN | **Implicit: 3-frame temporal attention** | Frozen-flow advection learned from data | Sound |
| **Spatial Coherence Capture** | Local (slope averaging) | Limited (CNN receptive field) | **Global self-attention over all 144 sub-apertures** | Transformer architecture | Sound |
| **Physical Geometry Prior** | Hard-coded geometry | Learned implicitly from data | **Hardcoded Fried geometry positional encodings** | Fried 1966; PINN embedding | Sound; key differentiator |
| **DM Coupling / Nonlinearity** | Assumes linear superposition | Ignored or separate network | **Tikhonov-regularised IM (exact linear solution in simulation)** | Linear DM model in HCIPy | Sound in simulation |
| **r₀ / τ₀ Estimation** | Separate post-processing | Secondary decomposition | **Real-time from raw slope covariance (background thread)** | Kolmogorov structure function; Noll 1976 | Sound; more robust than phase-based |
| **Centroiding** | Standard CoG | N/A | **Weighted CoG (AVX/SIMD C++)** | Thresholded intensity weighting | Sound; SNR improvement in strong turbulence |
| **Noise Robustness** | Degrades rapidly at low SNR | Moderate | **High (domain randomisation + LoRA)** | HCIPy; LoRA fine-tuning | Feasible |

---

## 4. Pipeline Architecture

The end-to-end pipeline is a strict linear sequence with one decoupled background thread.
Each stage is independently testable and replaceable.

```
Raw SH-WFS Detector Frame
         │
         ▼
┌─────────────────────────────────┐
│  Stage 1: Weighted CoG (C++)    │  < 0.5 ms
│  Per sub-aperture:              │
│  [Sx, Sy, I] × 144             │
│  AVX/SIMD parallel extraction   │
└────────────────┬────────────────┘
                 │               ┌─────────────────────────────────┐
                 │    raw slopes │  Background Thread (C++/Python) │
                 ├──────────────►│  Slope covariance → r₀          │  10–50 Hz
                 │               │  Temporal autocorr → τ₀          │  non-blocking
                 │               └─────────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: PI-STT Transformer (TensorRT / ONNX Runtime)          │  < 3.5 ms
│                                                                 │
│  Rolling buffer: 3 frames × 144 tokens × [Sx, Sy, I]           │
│  + Fried Geometry Positional Encodings (hardcoded (x,y), d/2)   │
│                                                                 │
│  Encoder: 2–4 layers, hidden dim 128–256                        │
│  Linear Attention or FlashAttention → O(N) scaling              │
│                                                                 │
│  Single output head:                                            │
│  Sequence-to-grid reshape → Dense phase map W(xᵢ,yᵢ) at t+1   │
└────────────────────────────┬────────────────────────────────────┘
                             │  W(xᵢ,yᵢ) predicted at t+1
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3: Tikhonov-Regularised IM (C++ / Eigen)                 │  < 0.2 ms
│                                                                 │
│  Pre-computed offline:  C⁺ = IMᵀ (IM·IMᵀ + αI)⁻¹              │
│  Runtime:               v_cmd = C⁺ · W(xᵢ,yᵢ)                 │
│  α tuned via GCV or empirical Strehl maximisation               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    DM Actuator Voltages
```

### Stage 1 — Weighted CoG Centroiding (C++, AVX/SIMD)

Raw SH-WFS detector frames are processed in C++ to extract per-lenslet slopes `[Sx, Sy]`
and integrated intensity `[I]`. Standard Centre-of-Gravity (CoG) weights all pixels
equally, amplifying read noise from dim pixels at the spot periphery and introducing
nonlinear bias when spots are asymmetric under strong turbulence. **Weighted CoG** applies
an intensity-based threshold mask, retaining only pixels above a signal-dependent threshold
before computing the centroid. This recovers linearity and significantly improves slope SNR
in low-flux and strong-turbulence regimes.

Implementation uses AVX/SIMD intrinsics for parallel per-lenslet computation. Target
latency: **< 0.5 ms** for a 12×12 lenslet array.

> *Scope note: "Fails catastrophically" overstates standard CoG's failure mode. It degrades
> progressively — catastrophic failure is only in LGS elongated-spot regimes or extreme
> photon starvation. Weighted CoG is a genuine improvement; the strength of the claim should
> be calibrated to the turbulence regime being demonstrated.*

### Stage 2 — PI-STT Transformer (TensorRT / ONNX Runtime)

The core ML stage. A rolling buffer maintains the 3 most recent centroided frames. At each
loop iteration, the 432-token sequence `[frame t−2 ‖ frame t−1 ‖ frame t]` is assembled and
passed through PI-STT.

**Tokenisation:** Each of the 144 sub-apertures contributes one token `[Sx, Sy, I]` per
frame, giving a sequence of 432 tokens of dimension 3. A linear projection lifts these to
the model's hidden dimension (128 or 256).

**Fried Geometry Positional Encodings:** The physical `(x, y)` coordinates of each lenslet
— including the `d/2` Fried geometry offset between the lenslet grid and the DM actuator
grid — are concatenated to each token's positional embedding. These encodings are
**hardcoded constants**, not learned parameters. This forces the attention mechanism to
respect the physical spatial topology from initialisation. A temporal position index is
appended to distinguish tokens from frames t−2, t−1, and t.

**Encoder:** 2–4 Transformer Encoder layers, 4–8 attention heads, hidden dimension
128–256. Linear Attention or FlashAttention to ensure O(N) or cache-efficient O(N²)
scaling. Total parameter count: ~500 K–2 M (micro-Transformer regime).

**Output Head:** A single linear projection maps the CLS token (or mean-pooled sequence
representation) to a flattened grid, followed by a sequence-to-grid reshape layer producing
the 2D predicted phase map **W(xᵢ, yᵢ) at frame t+1**.

**Deployment:** Exported to ONNX and compiled via TensorRT (GPU) or ONNX Runtime
with OpenVINO (CPU fallback). Target inference: **< 3.5 ms on GPU**, **< 7 ms on
multi-core CPU**.

### Stage 3 — Tikhonov-Regularised Influence Matrix (C++ / Eigen)

The pre-computed pseudo-inverse Influence Matrix converts the predicted phase map to DM
actuator voltages in a single matrix-vector multiply.

**Offline (Python / HCIPy):**
1. Generate the Influence Matrix `IM ∈ ℝ^{N_grid × N_act}` by poking each DM actuator
   individually and recording the resulting phase map. HCIPy provides this directly from
   the DM model.
2. Compute the Tikhonov-regularised pseudo-inverse:
   `C⁺ = IMᵀ (IM · IMᵀ + αI)⁻¹`
   where α is the regularisation parameter, tuned offline via Generalised Cross-Validation
   (GCV) or empirical Strehl ratio maximisation on validation turbulence screens.
3. Store `C⁺ ∈ ℝ^{N_act × N_grid}` as a static binary file loaded at runtime.

**Runtime (C++ / Eigen):**
`v_cmd = C⁺ · W_flat`
where `W_flat` is the flattened predicted phase map. This is a single dense matrix-vector
multiply. Cost: **< 0.2 ms** using Eigen BLAS for typical N_act (97–241 actuators).

> *Scope note: "Inter-actuator coupling is a strictly linear mechanical property" is correct
> within simulation (HCIPy's DM model is linear by construction). On real electrostrictive or
> piezoelectric hardware, facesheet response under joint actuation exhibits measurable
> nonlinearity (Guzmán et al. 2010 demonstrated ~200 nm joint-poke error on a Xinetics DM).
> The Tikhonov IM is the exact solution for the simulation context stated here; deployment on
> real hardware would require the MLP correction stage reintroduced or MARS calibration.*

### Background Thread — Turbulence Diagnostics (C++/Python)

The raw slope vectors `[Sx, Sy]` (before the Transformer) are forwarded to a background
thread updating at 10–50 Hz.

**r₀ estimation:** The empirical slope **structure function** `D_s(r) = ⟨|s(x) − s(x+r)|²⟩`
is computed from the spatial covariance of the slope vector. It is then fit to the theoretical
Kolmogorov form `D_s(r) ∝ (r/r₀)^(5/3)` to extract r₀. This requires knowledge of the
lenslet geometry and baseline separations — it is a least-squares fit to a theoretical model,
not a simple variance calculation.

**τ₀ estimation:** The temporal autocorrelation of the slope time-series is computed and fit
to the Frozen Flow decay model to extract the Greenwood frequency `f_G = 0.427/τ₀`.

> *Using raw slopes rather than reconstructed Zernike coefficients avoids fitting errors and
> noise amplification introduced by the reconstruction step. This is standard practice in
> modern ELT-class RTCs (e.g. SPHERE, CANARY), where turbulence diagnostics run on the
> WFS telemetry stream directly.*

**Total latency budget:**

| Stage | Component | Target |
|---|---|---|
| 1 | Weighted CoG (C++, AVX) | ~0.5 ms |
| 2 | PI-STT Transformer (TensorRT) | ~3.5 ms |
| 3 | IM pseudo-inverse multiply (Eigen) | ~0.2 ms |
| — | Background thread (non-blocking) | async |
| **Total** | **End-to-end** | **~4.2 ms** |

Comfortable margin within the **10 ms hard constraint**.

---

## 5. Architecture Decisions and Defences

### Why a Transformer instead of a CNN?

CNNs have localised receptive fields governed by kernel size and pooling depth. For a
standard 3×3 convolutional kernel, two sub-apertures on opposite sides of a 12×12 lenslet
array require ~6 pooling stages to interact — by which point spatial resolution is
drastically reduced. Kolmogorov turbulence has phase correlations that are global across the
pupil (the structure function is defined over the full aperture). Transformer self-attention
operates globally at every layer: every sub-aperture token attends to every other at full
resolution, naturally capturing these long-range correlations without requiring depth.

### Why a single t+1 output head instead of dual heads (phase + slope prediction)?

The proposed dual-head alternative (Head 1: reconstructed phase at t; Head 2: predicted
slopes at t+1) creates an unnecessary intermediate step. The slopes at t+1 still need to be
converted back to a phase map before the IM multiply, adding latency and an additional
source of approximation error. By training the single head directly on the t+1 **phase map**
target, the Transformer learns the full reconstruction-plus-prediction pipeline in one
optimisation, with the IM as the only downstream stage.

### Why drop the VAR model?

Classical VAR(p) models impose two assumptions: (a) linear temporal evolution, and (b)
stationarity. While valid for 1–2 frame horizons under stable seeing, the VAR operates in
Zernike coefficient space (66-dimensional), requiring a 66×66×p parameter matrix that
must be re-estimated whenever the wind profile changes. The Transformer's temporal
attention operates on the raw slope sequence, learning the non-linear frozen-flow advection
directly from the 3-frame history without stationarity assumptions and without a
re-estimation step. The Transformer subsumes the VAR's function as a special case of a
more general temporal model.

### Why drop the Residual MLP for DM mapping?

In simulation (HCIPy), the DM response to actuator commands is **linear by construction**.
The Tikhonov-regularised pseudo-inverse is the mathematically exact and optimal solution
to the linear least-squares actuator mapping problem. Adding a nonlinear MLP on top of an
exact linear solution introduces unnecessary parameters, training complexity, and
instability risk without modelling any physical effect that is actually present in simulation.
The MLP would be learning noise. On real hardware with genuine nonlinearity, the MLP
correction stage should be reintroduced.

### Why hardcoded Fried geometry encodings instead of learned positional embeddings?

Learned positional embeddings require the training data to inform the network of the
spatial layout of the sub-apertures. For a fixed physical instrument, this layout is known
exactly from design. Hardcoding it as a constant prior means: (1) the network does not
waste parameters and gradient steps recovering a known fact; (2) the spatial prior is
correct from step 0, not after convergence; (3) the network cannot learn an unphysical
spatial mapping even if the training data is insufficient.

### Why estimate r₀/τ₀ from raw slopes rather than Zernike coefficients?

Estimating turbulence parameters from the reconstructed Zernike coefficients (as in v3.0)
introduces fitting error from the reconstruction step and noise amplification from the
pseudo-inverse. The raw slope structure function is a direct observational quantity whose
theoretical form under Kolmogorov statistics is known analytically (Noll 1976, Fried 1966).
Fitting directly to the raw slopes is more robust, more standard (used in SPHERE, CANARY,
and ELT RTC designs), and eliminates the reconstruction as a potential error source in the
diagnostic chain.

---

## 6. Component Soundness Summary

| Component | Verdict | Key Concern | Mitigation / Justification |
|---|---|---|---|
| **Weighted CoG (AVX/SIMD)** | **Sound** | Requires threshold tuning per flux level | Threshold set from background flux estimate; standard practice in ExAO. Improvement over standard CoG is real but not "catastrophic failure" of CoG — calibrate claim to regime. |
| **Fried Geometry Positional Encodings** | **Sound** | Encoding must match actual lenslet geometry | Physical `(x,y)` coordinates and d/2 offset read from instrument geometry file. Hardcoded, not learned — correct by construction. |
| **3-frame temporal input (432 tokens)** | **Sound** | Sequence length vs. latency | 432 tokens with linear attention is fast (~3.5 ms TensorRT on GPU, ~7 ms CPU). Micro-Transformer regime. Validated by latency analysis. |
| **Single t+1 phase map output head** | **Sound** | Training requires t+1 ground truth | Ground truth at t+1 is available in simulation from the HCIPy phase screen time series. Standard supervised setup. |
| **Temporal prediction via attention** | **Sound** | Non-linear frozen-flow learned from data | 3-frame window sufficient for 1-frame prediction horizon. Network learns the advection; no stationarity assumption imposed. |
| **Tikhonov-regularised IM** | **Sound (simulation)** | Linear approximation fails on real hardware | Linear by construction in HCIPy. α regularisation handles actuator influence matrix ill-conditioning. GCV tuning principled. |
| **r₀/τ₀ from raw slope structure function** | **Sound** | Requires fitting Kolmogorov structure function, not just variance | Fit to `D_s(r) ∝ (r/r₀)^(5/3)` using known lenslet geometry. Non-blocking background thread; 10–50 Hz update rate. |
| **Background thread decoupling** | **Sound** | Thread safety for slope buffer access | Read-only slope snapshot per update cycle; standard producer-consumer pattern. |
| **Domain randomisation + LoRA** | **Feasible** | LoRA requires real paired calibration data | Modest paired dataset (WFS→wavefront ground truth) needed at deployment. Noted as deployment prerequisite. |

---

## 7. Key Features

- **Ultra-Low Latency:** Weighted CoG → PI-STT (TensorRT) → Tikhonov IM completes in
  ~4.2 ms, well within the 10 ms hard constraint. Micro-Transformer (2–4 layers, dim
  128–256) is the deliberate size choice to fit this budget.

- **Physics-Informed Positional Encodings:** The exact Fried geometry — lenslet `(x,y)`
  coordinates and the `d/2` actuator offset — are hardcoded into the Transformer's embedding
  layer. The attention mechanism respects physical spatial topology from initialisation,
  not after learning.

- **Joint Reconstruction and Prediction:** PI-STT outputs the **predicted phase map at
  t+1** directly from a 3-frame input history. Reconstruction and servo-lag compensation are
  a single forward pass, not two sequential modules.

- **Global Spatial Attention:** Self-attention over all 144 sub-aperture tokens at every
  Transformer layer captures the long-range spatial coherence of Kolmogorov turbulence that
  CNN receptive fields cannot reach at full resolution.

- **Principled Hybrid Control:** PI-STT handles the nonlinear inverse sensing problem.
  The Tikhonov-regularised IM handles the linear actuation problem. Neither stage is asked
  to do work outside its competence.

- **Weighted CoG in SIMD C++:** Thresholded intensity-weighted centroiding improves slope
  SNR in strong turbulence and low-flux regimes over standard CoG, with no latency penalty
  via AVX vectorisation.

- **Robust Turbulence Diagnostics:** r₀ and τ₀ estimated directly from the raw slope
  structure function in a non-blocking background thread at 10–50 Hz. More robust than
  estimating from reconstructed Zernike statistics.

- **Sim-to-Real Generalisation:** HCIPy-generated training data with domain-randomised
  hardware noise. LoRA fine-tuning on real calibration data at deployment.

---

## 8. Summary

WaveFormer-RTC addresses the three principal bottlenecks in adaptive optics —
**noise degradation, spatial nonlinearity, and temporal servo-lag** — through a principled
hybrid of physics-informed machine learning and classical control theory.

The PI-STT Transformer is the architectural centrepiece. By injecting Fried geometry
positional encodings as hardcoded physical priors and processing a 3-frame slope history,
PI-STT jointly reconstructs the wavefront and predicts the t+1 state in a single forward
pass — replacing both the CNN reconstructor and the VAR predictor of earlier designs with
a unified, physically constrained model. The attention mechanism is global, capturing
spatial correlations across the full pupil aperture that local receptive fields cannot reach.

Actuation is handled by the Tikhonov-regularised pseudo-inverse Influence Matrix — the
exact and complete solution to the linear DM mapping problem in simulation, requiring one
matrix-vector multiply at runtime. Mixing a nonlinear MLP with a linear physics problem
adds parameters and instability without modelling any physical effect that is genuinely
present in simulation.

Turbulence diagnostics (r₀, τ₀) are estimated from the raw slope structure function in a
decoupled background thread — more robust, more standard, and more computationally
honest than estimating from reconstructed Zernike statistics.

The combined pipeline delivers an end-to-end latency of approximately **4.2 ms**, with a
clear margin within the 10 ms real-time constraint.

---

## 9. References

> **Citation policy:** Only references that can be independently verified are cited with
> specific journal details. Architectural decisions justified primarily by first-principles
> reasoning are noted as such. Authors citing this document should verify all references
> against Google Scholar or ADS before submission.

**Verified:**

[1] Noll, R.J. (1976). Zernike polynomials and atmospheric turbulence. *Journal of the
Optical Society of America*, 66(3), 207–211.

[2] Fried, D.L. (1966). Optical resolution through a randomly inhomogeneous medium for
very long and very short exposures. *Journal of the Optical Society of America*, 56(10),
1372–1379. *(Fried parameter r₀; structure function; Fried geometry.)*

[3] Vaswani, A., et al. (2017). Attention is all you need. *Advances in Neural Information
Processing Systems*, 30. *(Transformer architecture foundation.)*

[4] Dessenne, C., Madec, P.-Y., & Rousset, G. (1998). Optimization of a predictive
controller for closed-loop adaptive optics. *Applied Optics*, 37(21), 4623–4633.
*(Predictive AO control; temporal prediction via Zernike time-series — superseded here
by Transformer temporal attention.)*

[5] Guzmán, D., et al. (2010). Deformable mirror model for open-loop adaptive optics using
multivariate adaptive regression splines. *Optics Express*, 18(7), 6492–6505. *(MARS
hardware validation; documents facesheet nonlinearity under joint actuation — relevant
to scope note on IM linearity assumption for real hardware.)*

[6] Hardy, J.W. (1998). *Adaptive Optics for Astronomical Telescopes*. Oxford University
Press. *(Textbook foundation: influence matrices, AO control loop, centroiding.)*

[7] Rigaut, F., & Gendron, E. (1992). Laser guide star in adaptive optics: the tilt
determination problem. *Astronomy & Astrophysics*, 261, 677–684. *(WFS slope
measurement and SH-WFS fundamentals.)*

**Unverified — claim is sound but specific papers cited in design notes require
independent confirmation before submission:**

[U1] Shen et al. (2023) — "Transformer-based wavefront reconstruction for SH-WFS."
*Optics Express* — *Not independently confirmed. The architectural claim (Transformer
outperforms CNN for global wavefront coherence) stands on first principles; this citation
should be replaced with a confirmed paper before submission.*

[U2] Wang et al. (2023) — "Spatio-temporal transformer for predictive control in AO."
*Optics Letters* — *Not independently confirmed.*

[U3] Chen et al. (2023) — "Physics-informed deep learning for AO wavefront
reconstruction." *Optics Letters* — *Not independently confirmed.*

[U4] Milli et al. (2022) — "Geometry-aware neural networks for SH-WFS." *JOSA A* —
*Not independently confirmed. Milli is an ESO/SPHERE researcher; topic is plausible but
specific paper requires verification.*

[U5] Meimon et al. (2020) — "Optimized centroiding algorithms for SH-WFS in strong
turbulence." *Optics Letters* — *Not independently confirmed. Meimon (ONERA) works on
AO algorithms; specific paper requires verification.*

[U6] Guyon et al. (2022) — "Cross-correlation centroiding for extreme adaptive optics."
*Optics Express* — *Not independently confirmed. Guyon is SCExAO PI; topic is central to
his work but specific paper requires verification.*