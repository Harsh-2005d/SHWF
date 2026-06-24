from pathlib import Path

import h5py
import numpy as np
import torch

from model import ISNet


DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "ISNet_PhaseData_5000.h5"
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "ISNet_weights.pth"


def pupil_mask(size=32):
    coordinates = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    x, y = np.meshgrid(coordinates, coordinates)
    return x * x + y * y <= 1.0


def remove_piston(phase, mask):
    phase = np.asarray(phase, dtype=np.float64).copy()
    phase[mask] -= phase[mask].mean()
    return phase


def phase_metrics(target, prediction, mask):
    """Calculate wavefront statistics inside the pupil after piston removal."""
    target = remove_piston(target, mask)
    prediction = remove_piston(prediction, mask)
    true_values = target[mask]
    pred_values = prediction[mask]
    residual = true_values - pred_values
    denominator = np.sqrt(np.mean(true_values**2))

    correlation = 0.0
    if true_values.std() > 0 and pred_values.std() > 0:
        correlation = float(np.corrcoef(true_values, pred_values)[0, 1])

    return {
        "true_rms": float(np.sqrt(np.mean(true_values**2))),
        "pred_rms": float(np.sqrt(np.mean(pred_values**2))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "residual_pv": float(np.ptp(residual)),
        "relative_rmse": float(np.sqrt(np.mean(residual**2)) / max(denominator, 1e-12)),
        "correlation": correlation,
    }


def load_model(weights, device):
    model = ISNet().to(device)
    state = torch.load(weights, map_location=device, weights_only=True)
    # Also accept a conventional training checkpoint if one is supplied later.
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def read_frame(data_file, frame):
    with h5py.File(data_file, "r") as file:
        length = len(file["Y_Phase"])
        if not -length <= frame < length:
            raise IndexError(f"Frame {frame} is outside the valid range 0..{length - 1}")
        frame %= length
        arrays = tuple(
            np.asarray(file[key][frame], dtype=np.float32)
            for key in ("X_I", "X_Sx", "X_Sy", "Y_Phase")
        )
    return frame, arrays


@torch.inference_mode()
def predict_frame(model, arrays, device):
    intensity, slope_x, slope_y, target = arrays
    inputs = [
        torch.from_numpy(array).unsqueeze(0).to(device)
        for array in (intensity, slope_x, slope_y)
    ]
    prediction = model(*inputs).squeeze().cpu().numpy()
    return target.squeeze(), prediction
