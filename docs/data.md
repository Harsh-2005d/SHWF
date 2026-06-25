# Architecture & Implementation Reference: High-Throughput SH-WFS Data Generation Pipeline

**Target Architecture:** ZERN-Trans (Tokenized Transformer for Wavefront Sensing)
**Pipeline Output:** 500,000 paired tokens/targets encoded in HDF5 (`LZF` compressed).

---

## 1. Executive Summary

This script implements a heavily optimized, GPU-accelerated physical simulation of a Shack-Hartmann Wavefront Sensor (SH-WFS) operating under von Kármán atmospheric turbulence. Designed specifically for training a sequence-based Transformer model, the pipeline eschews traditional 2D image output. Instead, it computes physical phase gradients, models log-normal scintillation, injects hardware noise models, and outputs **sequence tokens** containing localized slope and intensity data.

To achieve the required throughput for 500,000 frames, the architecture is split into two distinct operational domains:

1. **CPU Phase (Initialization):** Utilizes `hcipy` for rigorous analytical definitions of optical geometries, spatial grids, and Zernike basis polynomials.
2. **GPU Phase (Runtime):** Transfers the mathematical constants to PyTorch tensors, enabling highly parallelized FFT operations, gradient computations, and matrix projections across massive batches.

---

## 2. Dependency Matrix

| Library | Version / Role | Justification |
| --- | --- | --- |
| **PyTorch** (`torch`) | Core computation engine. | Handles batched FFTs, matrix multiplications, pooling, and gradient calculations natively on the GPU. |
| **HCIPy** (`hcipy`) | High-Contrast Imaging Python. | Provides analytically perfect models for circular apertures and localized Zernike polynomials over specific spatial grids. |
| **NumPy** (`numpy`) | Bridge computation. | Handles CPU-side array manipulations and memory-efficient pseudo-inverse calculations before tensor conversion. |
| **h5py** | Data persistence. | Facilitates chunked, compressed writing to HDF5 format, which is critical for efficient disk I/O during PyTorch `DataLoader` traversal. |

---

## 3. Module Breakdown & System Flow

### Module A: Optical System Initialization (`setup_optical_system`)

**Execution:** CPU $\rightarrow$ GPU Transfer (Executes Once)

This function mathematically defines the telescope and the wavefront sensor geometry. It prepares the transformation matrices required to bridge the physical phase screens to the target Zernike coefficients.

* **Telescope Geometry:** Configures a 1.8m diameter ($D$) telescope mapped to a grid where a $20 \times 20$ lenslet array covers the pupil. Each lenslet is resolved by a $16 \times 16$ pixel grid, resulting in a global computational domain of $320 \times 320$ pixels.
* **Zernike Basis ($M_Z$):** Generates 64 Zernike modes (excluding Piston/Mode 1) using HCIPy.
* **Memory Optimization (The Masking Step):** To prevent allocating a massive dense matrix for the pseudo-inverse calculation, the Zernike basis transformation matrix is masked against the circular pupil. Only pixels with an aperture value $> 0.5$ are retained.
* **Pseudo-Inverse Projection:** Computes $(M_Z^T M_Z)^{-1} M_Z^T$. This precomputed matrix allows the GPU to extract target Zernike coefficients from random phase screens via a single, rapid matrix multiplication.
* **Coordinate System:** Generates normalized $[-1, 1]$ $(x, y)$ coordinates for the center of every valid sub-aperture inside the pupil, explicitly ignoring corners outside the circular aperture.

### Module B: Fast GPU Phase Screen Generator (`generate_von_karman_batch`)

**Execution:** GPU Batched (Executes per batch)

Generates random atmospheric phase screens ($\Phi$) directly on the GPU using Fourier-domain filtering.

* **Von Kármán Power Spectral Density (PSD):** Computes the PSD based on the user-defined outer scale ($L_0 = 25\text{m}$) and a batched vector of Fried parameters ($r_0$).

$$PSD(f) = 0.023 r_0^{-5/3} (f^2 + L_0^{-2})^{-11/6}$$


* **FFT Filtering:** Generates random complex Gaussian noise, filters it in the frequency domain using the square root of the PSD, and performs a 2D Inverse Fast Fourier Transform (iFFT) to yield the spatial phase screen.
* **Pupil Masking:** The final phase screen is multiplied by the aperture mask to zero out data outside the telescope pupil.

### Module C: Core Simulation & Tokenization (`generate_transformer_dataset`)

**Execution:** GPU Batched I/O Loop

This is the primary orchestration loop. It handles domain randomization, physics-based centroiding, noise injection, and formatting the data into Transformer tokens.

1. **Domain Randomization ($r_0$):** Samples a uniform distribution of seeing conditions per frame ($r_0 \in [0.05, 0.20]$ meters), representing strong to weak turbulence.
2. **Target Generation ($Y$):** Flattens the valid pixels of the phase screen and multiplies them by the precomputed pseudo-inverse matrix to yield the exact ground truth Zernike coefficients.
3. **Physical Gradients (Slopes):** Uses `torch.gradient` to analytically compute the instantaneous slope of the wavefront in the $x$ and $y$ dimensions.
4. **Lenslet Integration:** Employs `F.avg_pool2d` to simulate the spatial integration of a lenslet. By pooling the $320 \times 320$ gradient field with a kernel size of 16, it yields the exact centroid shift per sub-aperture.
5. **Log-Normal Scintillation Model:** * Computes the wavefront Laplacian ($\nabla^2 \Phi$) to determine local curvature (focal/defocus effects).
* Calculates a base intensity using an exponential decay model: $I_{base} = \exp(-0.5 \nabla^2 \Phi)$.
* Multiplies this by a log-normal statistical noise distribution to perfectly simulate atmospheric speckling and deep fades.


6. **Sim-to-Real Hardware Noise:**
* **Photon Noise:** Adds Poisson-like noise to the intensity $I$.
* **Read Noise:** Injects a base level of zero-mean Gaussian noise directly to the slopes. Photon noise on the slopes is scaled inversely proportional to the local intensity (dimmer lenslets have noisier centroids).


7. **Transformer Normalization:** Applies a global scaling factor (`SLOPE_NORM_FACTOR = 50.0`) to the raw slopes. This is critical to ensure the self-attention mechanism is not overpowered by massive gradient values during strong turbulence.
8. **Token Construction ($X$):** Stacks the features into a sequence tensor of shape `(Batch, N_valid, 5)` where the feature vector is $[S_x, S_y, I, x, y]$.
9. **Sensor Dropout:** Randomly applies a 5% dropout mask to zero out entire tokens, forcing the Transformer to learn spatial interpolation and robustness against dead pixels.

---

## 4. Output HDF5 Schema

The resulting HDF5 file (`ZERN_Trans_Dataset_500k.h5`) uses `LZF` compression and is chunked to ensure maximum sequential read speed for PyTorch `DataLoaders`.

| Dataset Name | Shape | Data Type | Description |
| --- | --- | --- | --- |
| `X_Tokens` | `(500000, N_valid, 5)` | `float32` | The input sequence for the Transformer. Dimension 1 is the sequence length (number of valid sub-apertures). Dimension 2 contains the normalized features: $[S_x, S_y, I, x, y]$. |
| `Y_Zernikes` | `(500000, 64)` | `float32` | The target Zernike coefficients (Modes 2 through 65). |

*(Note: `N_valid` is the precise count of sub-apertures strictly inside the circular pupil, calculated dynamically during initialization).*