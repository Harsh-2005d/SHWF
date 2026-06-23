import argparse
import json
from pathlib import Path

import numpy as np

from .data import make_slope_dataset
from .metrics import coefficient_metrics
from .optics import reconstruct_phase
from .reconstructor import MatrixWavefrontReconstructor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the matrix wavefront reconstructor over many frames."
    )
    parser.add_argument("--model", default="BaseModal/artifacts/matrix_reconstructor.npz")
    parser.add_argument("--file", default="Sum_NewData_299_100.h5")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--phase-samples", type=int, default=100)
    parser.add_argument("--output", default="BaseModal/artifacts/evaluation_metrics.json")
    return parser.parse_args()


def phase_residual_metrics(y_true, y_pred, phase_samples):
    count = min(phase_samples, y_true.shape[0])
    rms_values = []
    pv_values = []

    for idx in range(count):
        true_phase = reconstruct_phase(y_true[idx], size=240)
        pred_phase = reconstruct_phase(y_pred[idx], size=240)
        residual = pred_phase - true_phase
        mask = true_phase != 0
        values = residual[mask] if np.any(mask) else residual.ravel()
        rms_values.append(float(np.sqrt(np.mean(values**2))))
        pv_values.append(float(np.max(values) - np.min(values)))

    return {
        "phase_samples": count,
        "phase_residual_rms_mean": float(np.mean(rms_values)),
        "phase_residual_rms_std": float(np.std(rms_values)),
        "phase_residual_pv_mean": float(np.mean(pv_values)),
        "phase_residual_pv_std": float(np.std(pv_values)),
    }


def collect_metrics(name, slopes, truth, pred, phase_samples):
    coeff = coefficient_metrics(truth, pred)
    phase = phase_residual_metrics(truth, pred, phase_samples)
    return {
        "split": name,
        "frames": int(slopes.shape[0]),
        "slope_features": int(slopes.shape[1]),
        "modes": int(pred.shape[1]),
        **coeff,
        **phase,
    }


def print_metrics(metrics):
    print(f"\n[{metrics['split']}]")
    for key, value in metrics.items():
        if key == "split":
            continue
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


def main():
    args = parse_args()
    model = MatrixWavefrontReconstructor.load(args.model)

    slopes, coeffs = make_slope_dataset(args.file, limit=args.limit)
    pred = model.predict(slopes)
    truth = coeffs[:, : pred.shape[1]]

    results = {
        "full": collect_metrics("full", slopes, truth, pred, args.phase_samples)
    }

    if model.train_indices is not None and model.test_indices is not None:
        train_idx = model.train_indices[model.train_indices < slopes.shape[0]]
        test_idx = model.test_indices[model.test_indices < slopes.shape[0]]
        results["train"] = collect_metrics(
            "train",
            slopes[train_idx],
            truth[train_idx],
            pred[train_idx],
            args.phase_samples,
        )
        results["test"] = collect_metrics(
            "test",
            slopes[test_idx],
            truth[test_idx],
            pred[test_idx],
            args.phase_samples,
        )

    for metrics in results.values():
        print_metrics(metrics)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved metrics to {out}")


if __name__ == "__main__":
    main()
