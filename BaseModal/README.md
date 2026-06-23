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

Train the base matrix model:

```bash
python -m BaseModal.train --file Sum_NewData_299_100.h5 --modes 100
```

The generated dataset has 224 slope measurements and 299 Zernike coefficients.
A classical matrix reconstructor should start with fewer low-order modes than
the number of slope measurements. `--modes 100` is a stable first baseline.

Predict one frame:

```bash
python -m BaseModal.predict --file Sum_NewData_299_100.h5 --frame 0
```

Visualize reconstruction:

```bash
python -m BaseModal.visualize_prediction --file Sum_NewData_299_100.h5 --frame 0
```

Evaluate metrics over the dataset:

```bash
python -m BaseModal.evaluate --file Sum_NewData_299_100.h5
```

The trained model is saved by default at:

```text
BaseModal/artifacts/matrix_reconstructor.npz
```
