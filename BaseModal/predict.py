import argparse
from pathlib import Path

import numpy as np

from .reconstructor import MatrixWavefrontReconstructor
from .slopes import image_to_slopes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict Zernike coefficients for one SHWFS frame using the base matrix model."
    )
    parser.add_argument("--model", default="BaseModal/artifacts/matrix_reconstructor.npz")
    parser.add_argument("--file", default="Sum_NewData_299_100.h5")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output", default=None, help="Optional .npy output for predicted coefficients.")
    return parser.parse_args()


def main():
    args = parse_args()
    import h5py

    model = MatrixWavefrontReconstructor.load(args.model)
    membership = model.split_membership(args.frame)

    with h5py.File(args.file, "r") as h5f:
        image = h5f["Xtrain"][args.frame]
        truth = h5f["Ytrain"][args.frame] if "Ytrain" in h5f else None

    slopes = image_to_slopes(image)
    pred = model.predict(slopes)

    if truth is not None:
        truth = truth[: pred.shape[0]]
        rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
        print(f"Frame {args.frame} coefficient RMSE: {rmse:.6f}")
    print(f"Frame {args.frame} split: {membership}")

    print("First 10 predicted coefficients:")
    print(np.array2string(pred[:10], precision=6, separator=", "))
    print(f"Predicted modes: {pred.shape[0]}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, pred)
        print(f"Saved coefficients to {out}")


if __name__ == "__main__":
    main()
