# Matrix-Based Wavefront Reconstruction Base Model

This folder contains the classical adaptive-optics baseline described in the
review paper: reconstruct wavefronts from Shack-Hartmann WFS measurements using
least-squares / SVD pseudoinverse matrix reconstruction.

The baseline follows:

```text
s = A phi
phi_hat = A_plus s
```

where:

- `s` is the measured SHWFS slope / centroid-displacement vector.
- `A` is the interaction matrix learned from generated data.
- `phi` is the Zernike coefficient vector.
- `A_plus` is the SVD pseudoinverse reconstruction matrix.

## Files

- `slopes.py` extracts centroid displacement vectors from each SHWFS image.
- `reconstructor.py` fits and saves the SVD pseudoinverse matrix model.
- `train.py` trains the base model from `Xtrain` and `Ytrain`.
- `predict.py` predicts Zernike coefficients for one frame.
- `visualize_prediction.py` compares input, true phase, predicted phase, and residual.
- `optics.py` rebuilds phase maps from Zernike coefficients.

## Usage

Generate data first from the project root:

```bash
python shcnndata.py
```

Train the upgraded matrix model:

```bash
python -m BaseModal.train --file Sum_NewData_299_5000.h5 --modes 100
```

The upgraded model standardizes the 224 slope measurements, selects a
Tikhonov regularization strength on a calibration split, and directly fits the
modal reconstruction matrix with SVD. Prediction is still a fast matrix
multiplication. A classical reconstructor should start with fewer low-order
modes than slope measurements; `--modes 100` is a stable first baseline.

Predict one frame:

```bash
python -m BaseModal.predict --file Sum_NewData_299_100.h5 --frame 0
```

Visualize reconstruction:

```bash
python -m BaseModal.visualize_prediction --file Sum_NewData_299_100.h5 --frame 0
```

The newer four-panel pupil-masked visualization is:

```bash
python -m BaseModal.plot_reconstruction --file Sum_NewData_299_5000.h5 --frame 0
```

Evaluate metrics over the dataset:

```bash
python -m BaseModal.evaluate --file Sum_NewData_299_100.h5
```

For held-out per-frame CSV metrics and an aggregate JSON report:

```bash
python -m BaseModal.benchmark --file Sum_NewData_299_5000.h5
```

The trained model is saved by default at:

```text
BaseModal/artifacts/matrix_reconstructor_v2.npz
```

The original `matrix_reconstructor.npz` is retained as the legacy
double-pseudoinverse baseline.
