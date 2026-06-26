

# Project Title: WaveFormer-RTC 
**A Physics-Informed, Predictive Real-Time Controller for Open-Loop Adaptive Optics**

## 1. Brief Idea About the Solution
**WaveFormer-RTC** is a machine learning-driven Real-Time Controller (RTC) for Adaptive Optics, designed to replace traditional matrix-multiplication methods that fail under high noise and severe atmospheric turbulence. 

The core architecture features a lightweight 1D Linear Transformer utilizing a 3-arm physics-informed design (inspired by ISNet [1]). It ingests localized sub-aperture features $[S_x, S_y, I, x, y]$ extracted from Shack-Hartmann Wavefront Sensor (SH-WFS) images via a highly optimized C++ preprocessing pipeline. Instead of outputting a dense 2D phase map, the Transformer directly regresses the **Zernike modal coefficients up to the 120th order** in a single forward pass.

The pipeline then branches into critical real-time operations:
*   **Turbulence Characterization:** The Fried parameter ($r_0$) and coherence time ($\tau_0$) are analytically inferred directly from the time-series of the predicted Zernike coefficients [2].
*   **Temporal Prediction (Servo-Lag Mitigation):** To eliminate time-delay errors inherent in AO loops, a lightweight **Vector Auto-Regressive (VAR)** model operating in pure C++ predicts the future state of the Zernike coefficients based on the Frozen Flow Hypothesis, anticipating the wavefront before it reaches the mirror.
*   **Non-Linear Actuator Mapping:** The predicted Zernike coefficients are spatially resampled and fed into a pre-trained **Multivariate Adaptive Regression Splines (MARS)** model [3]. This correctly maps the conjugate wavefront to the Deformable Mirror (DM) actuator voltages, natively handling non-linear hysteresis and inter-actuator cross-coupling in a Fried geometry.

## 2. Unique Selling Proposition (USP)
*   **Novel Physics-Informed Architecture:** Traditionally, CNNs map sensor images to Zernike coefficients, while heavy Transformers map images to dense phase maps (requiring costly secondary fitting) [2]. WaveFormer-RTC introduces a 1D Transformer that tokenizes the *sensor sub-apertures* rather than raw pixels, directly outputting Zernike coefficients. This bypasses the dense-to-modal bottleneck, keeping the architecture ultra-lightweight without sacrificing inference accuracy.
*   **Sim-to-Real Robustness via Domain Randomization:** The Transformer is highly performant under extreme turbulence, scintillation, and optical cross-talk. It is pre-trained on massive synthetic datasets generated via the HCIPy library, injecting physics-informed hardware noise (Poisson photon noise, read noise, and sensor dropout). When deployed, it utilizes Parameter-Efficient Fine-Tuning (LoRA) on sparse real-time lab data to seamlessly bridge the Sim-to-Real gap.
*   **Zero-Latency Predictive & Non-Linear Control:** Traditional systems suffer from temporal servo-lag and rely on linear Influence Matrices that fail at large stroke lengths. We decouple these problems: a **VAR model** handles temporal prediction in pure C++ (taking <50 microseconds) to stay ahead of atmospheric advection, while **MARS** [3] handles the spatial non-linearities and Fried-geometry cross-coupling of the DM. This dual approach guarantees sub-millisecond control logic, securing the 10ms latency budget.

## 3. Competitor Comparison

| Feature | Traditional AO (Matrix / SVD) | Standard DL AO (U-Net / 2D-ViT) | Our Solution (WaveFormer-RTC) |
| :--- | :--- | :--- | :--- |
| **Output Representation** | Zonal slopes or dense Phase Map | Dense 2D Phase Map | **Direct Zernike Coefficients** |
| **Latency (Inference)** | < 1 ms *(fails on noise)* | 50 - 200 ms *(blows 10ms budget)* | **< 5 ms *(TensorRT optimized)*** |
| **Temporal Control (Servo-Lag)**| Reactive *(High lag error)* | Reactive or heavy RNN/LSTM | **Predictive via C++ VAR (<50 µs)** |
| **DM Coupling / Hysteresis** | Assumes linear superposition | Usually ignored or requires separate net | **Handled natively via MARS** |
| **Turbulence Params ($r_0, \tau_0$)**| Requires separate post-processing | Requires secondary Zernike decomposition | **Calculated analytically on-the-fly** |
| **Noise Robustness** | Degrades rapidly in low SNR | Moderate | **High (via Domain Randomization)** |

## 4. Summary
WaveFormer-RTC addresses the three major bottlenecks in Adaptive Optics: **noise degradation, spatial non-linearities, and temporal servo-lag**. Our model tackles these issues with a robust 1D Transformer architecture, turbulence-characteristics-aware diagnostics, and a hybrid VAR-MARS control loop. By utilizing a mix of traditional, science-backed control theory and the latest advancements in Machine Learning, we deliver a State-of-the-Art (SOTA) solution for Wavefront Reconstruction. Crucially, the model remains ultra-lightweight, executing the entire end-to-end pipeline well within the strict 10 ms constraint of the Hackathon challenge.

## 5. Key Features
*   **Ultra-Low Latency Pipeline:** The end-to-end execution (C++ Preprocessing $\rightarrow$ TensorRT Inference $\rightarrow$ VAR Prediction $\rightarrow$ MARS Actuation) completes in < 10 ms, strictly satisfying the 10 ms real-time constraint.
*   **3-Arm Feature Fusion:** Simultaneously leverages wavefront gradients ($S_x, S_y$) and local scintillation/intensity ($I$) to resolve phase wrapping ambiguities inherent in strong turbulence regimes.
*   **Fried Geometry Cross-Coupling:** The MARS model is explicitly trained to map the spatially offset WFS lenslet grid to the DM actuator grid, learning the complex mechanical cross-talk without requiring a physical model of the mirror.
*   **Predictive Servo-Lag Mitigation:** Traditional systems are purely reactive, suffering from 2-3 frame time-delay errors. By deploying a lightweight VAR model on the Zernike time-series, our RTC anticipates the advected turbulence and commands the DM for the *future* state of the atmosphere in microseconds.
*   **Real-Time Turbulence Diagnostics:** Continuous telemetry output of coherence time ($\tau_0$) and Fried parameter ($r_0$) allows the system to dynamically assess atmospheric conditions and monitor temporal lag errors on the fly.
*   **Hardware Agnostic Control:** The MARS model is purely mathematical and data-driven, requiring no physical parameters of the DM, making the control scheme portable across different mirror technologies.

