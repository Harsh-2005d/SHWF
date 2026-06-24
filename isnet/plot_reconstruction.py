import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from eval_utils import (
    DEFAULT_DATASET,
    DEFAULT_WEIGHTS,
    load_model,
    phase_metrics,
    predict_frame,
    pupil_mask,
    read_frame,
    remove_piston,
)


def masked_phase(phase, mask):
    return np.ma.array(phase, mask=~mask)


def plot_reconstruction(model, data_file, frame, output, device, dpi=180):
    frame, arrays = read_frame(data_file, frame)
    intensity, slope_x, slope_y, _ = arrays
    target, prediction = predict_frame(model, arrays, device)
    mask = pupil_mask(target.shape[-1])
    target = remove_piston(target, mask)
    prediction = remove_piston(prediction, mask)
    residual = target - prediction
    metrics = phase_metrics(target, prediction, mask)

    phase_limit = max(
        float(np.max(np.abs(target[mask]))),
        float(np.max(np.abs(prediction[mask]))),
        1e-12,
    )
    residual_limit = max(float(np.max(np.abs(residual[mask]))), 1e-12)

    phase_cmap = plt.colormaps["jet"].copy()
    residual_cmap = plt.colormaps["coolwarm"].copy()
    phase_cmap.set_bad("#d9d9d9")
    residual_cmap.set_bad("#d9d9d9")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.7), constrained_layout=True)
    slope_x = slope_x.squeeze()
    slope_y = slope_y.squeeze()
    intensity = intensity.squeeze()
    slope_magnitude = np.hypot(slope_x, slope_y)
    input_image = axes[0].imshow(slope_magnitude, origin="lower", cmap="viridis")
    yy, xx = np.mgrid[: slope_x.shape[0], : slope_x.shape[1]]
    active = intensity > 0
    axes[0].quiver(
        xx[active], yy[active], slope_x[active], slope_y[active],
        color="white", angles="xy", scale_units="xy", scale=None,
        width=0.006, headwidth=3,
    )
    axes[0].set_title("SHWFS input (slope field)")
    fig.colorbar(input_image, ax=axes[0], fraction=0.046, pad=0.04, label="slope magnitude")

    true_image = axes[1].imshow(
        masked_phase(target, mask), origin="lower", cmap=phase_cmap,
        vmin=-phase_limit, vmax=phase_limit,
    )
    axes[1].set_title("True wavefront")
    fig.colorbar(true_image, ax=axes[1], fraction=0.046, pad=0.04)

    pred_image = axes[2].imshow(
        masked_phase(prediction, mask), origin="lower", cmap=phase_cmap,
        vmin=-phase_limit, vmax=phase_limit,
    )
    axes[2].set_title("ISNet reconstruction")
    fig.colorbar(pred_image, ax=axes[2], fraction=0.046, pad=0.04)

    residual_image = axes[3].imshow(
        masked_phase(residual, mask), origin="lower", cmap=residual_cmap,
        vmin=-residual_limit, vmax=residual_limit,
    )
    axes[3].set_title("Residual (true − predicted)")
    fig.colorbar(residual_image, ax=axes[3], fraction=0.046, pad=0.04)

    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(
        f"Frame {frame} | true RMS={metrics['true_rms']:.4f}, "
        f"pred RMS={metrics['pred_rms']:.4f}, residual RMSE={metrics['rmse']:.4f}, "
        f"corr={metrics['correlation']:.4f}"
    )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Plot one ISNet wavefront reconstruction")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = args.output or Path(f"isnet_reconstruction_frame_{args.frame}.png")
    model = load_model(args.weights, device)
    metrics = plot_reconstruction(
        model, args.data, args.frame, output, device, dpi=args.dpi
    )
    print(f"Saved: {output.resolve()}")
    print("  ".join(f"{key}={value:.6f}" for key, value in metrics.items()))


if __name__ == "__main__":
    main()
