import json
from pathlib import Path

import numpy as np


class MatrixWavefrontReconstructor:
    """SVD/least-squares baseline for reconstructing Zernike coefficients."""

    def __init__(self, rcond=1e-6):
        self.rcond = rcond
        self.slope_mean = None
        self.coeff_mean = None
        self.interaction_matrix = None
        self.reconstruction_matrix = None
        self.train_indices = None
        self.test_indices = None
        self.metadata = {}

    def fit(self, slopes, coefficients):
        slopes = np.asarray(slopes, dtype=np.float64)
        coefficients = np.asarray(coefficients, dtype=np.float64)
        if slopes.ndim != 2 or coefficients.ndim != 2:
            raise ValueError("Expected slopes and coefficients to be 2D matrices.")
        if slopes.shape[0] != coefficients.shape[0]:
            raise ValueError("Slopes and coefficients must contain the same frames.")

        self.slope_mean = slopes.mean(axis=0)
        self.coeff_mean = coefficients.mean(axis=0)
        centered_slopes = slopes - self.slope_mean
        centered_coeffs = coefficients - self.coeff_mean

        # Interaction matrix A maps modal coefficients to SHWFS slopes: s = A phi.
        self.interaction_matrix = np.linalg.pinv(centered_coeffs, rcond=self.rcond) @ centered_slopes

        # Reconstruction matrix R maps slopes back to coefficients: phi_hat = R s.
        self.reconstruction_matrix = np.linalg.pinv(self.interaction_matrix, rcond=self.rcond)
        return self

    def predict(self, slopes):
        if self.reconstruction_matrix is None:
            raise RuntimeError("Model is not fitted. Call fit() or load() first.")

        slopes = np.asarray(slopes, dtype=np.float64)
        one_sample = slopes.ndim == 1
        if one_sample:
            slopes = slopes[None, :]

        centered = slopes - self.slope_mean
        coeffs = centered @ self.reconstruction_matrix + self.coeff_mean
        return coeffs[0] if one_sample else coeffs

    def save(self, path):
        if self.reconstruction_matrix is None:
            raise RuntimeError("Cannot save an unfitted model.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "rcond": np.asarray(self.rcond),
            "slope_mean": self.slope_mean,
            "coeff_mean": self.coeff_mean,
            "interaction_matrix": self.interaction_matrix,
            "reconstruction_matrix": self.reconstruction_matrix,
        }
        if self.train_indices is not None:
            payload["train_indices"] = self.train_indices
        if self.test_indices is not None:
            payload["test_indices"] = self.test_indices
        np.savez_compressed(path, **payload)

        meta_path = path.with_suffix(".json")
        meta_path.write_text(
            json.dumps(
                {
                    "model": "svd_pseudoinverse_matrix_reconstructor",
                    "formula": "s = A phi, phi_hat = A_plus s",
                    "rcond": self.rcond,
                    "n_slopes": int(self.slope_mean.shape[0]),
                    "n_modes": int(self.coeff_mean.shape[0]),
                    **self.metadata,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path):
        data = np.load(path)
        model = cls(rcond=float(data["rcond"]))
        model.slope_mean = data["slope_mean"]
        model.coeff_mean = data["coeff_mean"]
        model.interaction_matrix = data["interaction_matrix"]
        model.reconstruction_matrix = data["reconstruction_matrix"]
        model.train_indices = data["train_indices"] if "train_indices" in data else None
        model.test_indices = data["test_indices"] if "test_indices" in data else None
        return model

    def split_membership(self, frame_idx):
        if self.train_indices is None or self.test_indices is None:
            return "unknown"
        if frame_idx in set(self.train_indices.tolist()):
            return "train"
        if frame_idx in set(self.test_indices.tolist()):
            return "test"
        return "outside_split"
