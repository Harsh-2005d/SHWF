import argparse
from pathlib import Path

import numpy as np

from .optics import phase_metrics, reconstruct_phase
from .reconstructor import MatrixWavefrontReconstructor
from .slopes import image_to_slopes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize true vs matrix-reconstructed wavefront for one frame."
    )
    parser.add_argument("--model", default="BaseModal/artifacts/matrix_reconstructor.npz")
    parser.add_argument("--file", default="Sum_NewData_299_100.h5")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output", default="BaseModal/artifacts/reconstruction_frame_0.png")
    return parser.parse_args()


def main():
    args = parse_args()
    import h5py
    import matplotlib.pyplot as plt

    model = MatrixWavefrontReconstructor.load(args.model)

    with h5py.File(args.file, "r") as h5f:
        image = h5f["Xtrain"][args.frame]
        truth = h5f["Ytrain"][args.frame]

    pred = model.predict(image_to_slopes(image))
    truth = truth[: pred.shape[0]]
    true_phase = reconstruct_phase(truth, size=image.shape[0])
    pred_phase = reconstruct_phase(pred, size=image.shape[0])
    residual = pred_phase - true_phase

    panels = [
        ("SHWFS input", image, "jet"),
        ("True wavefront", true_phase, "jet"),
        ("Matrix reconstruction", pred_phase, "jet"),
        ("Residual", residual, "coolwarm"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (title, data, cmap) in zip(axes, panels):
        vmax = np.max(np.abs(data))
        if title == "SHWFS input":
            im = ax.imshow(data, cmap=cmap, origin="lower")
        else:
            im = ax.imshow(data, cmap=cmap, origin="lower", vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    true_metrics = phase_metrics(true_phase)
    pred_metrics = phase_metrics(pred_phase)
    fig.suptitle(
        "Frame "
        f"{args.frame} | true RMS={true_metrics['rms']:.4f}, "
        f"pred RMS={pred_metrics['rms']:.4f}",
        fontsize=11,
    )
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization to {out}")


if __name__ == "__main__":
    main()
