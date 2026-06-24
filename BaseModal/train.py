import argparse
from pathlib import Path

from .data import make_slope_dataset, split_indices
from .metrics import coefficient_metrics
from .reconstructor import MatrixWavefrontReconstructor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a classical SVD/least-squares wavefront reconstruction matrix."
    )
    parser.add_argument("--file", default="Sum_NewData_299_100.h5", help="Input HDF5 dataset.")
    parser.add_argument("--output", default="BaseModal/artifacts/matrix_reconstructor.npz")
    parser.add_argument("--limit", type=int, default=None, help="Optional frame limit for quick tests.")
    parser.add_argument(
        "--modes",
        type=int,
        default=100,
        help="Number of low-order Zernike modes to reconstruct.",
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--rcond", type=float, default=1e-6, help="SVD cutoff for pseudoinverse.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    h5_path = Path(args.file)
    if not h5_path.exists():
        raise FileNotFoundError(
            f"{h5_path} was not found. Generate it first with: python shcnndata.py"
        )

    print(f"Loading dataset from {h5_path}...")
    slopes, coeffs = make_slope_dataset(h5_path, limit=args.limit)
    if args.modes < 1 or args.modes > coeffs.shape[1]:
        raise ValueError(f"--modes must be between 1 and {coeffs.shape[1]}.")
    coeffs = coeffs[:, : args.modes]
    print(f"Slope matrix: {slopes.shape}")
    print(f"Coefficient matrix: {coeffs.shape}")

    train_idx, test_idx = split_indices(
        slopes.shape[0], test_fraction=args.test_fraction, seed=args.seed
    )
    x_train, x_test = slopes[train_idx], slopes[test_idx]
    y_train, y_test = coeffs[train_idx], coeffs[test_idx]

    model = MatrixWavefrontReconstructor(rcond=args.rcond).fit(x_train, y_train)
    model.train_indices = train_idx
    model.test_indices = test_idx
    model.metadata = {
        "source_file": str(h5_path),
        "frames": int(slopes.shape[0]),
        "test_fraction": args.test_fraction,
        "seed": args.seed,
    }
    model.save(args.output)
    print(f"Saved base model to {args.output}")
    print(f"Train frames: {train_idx.size}")
    print(f"Test frames: {test_idx.size}")

    if x_test.shape[0] > 0:
        pred = model.predict(x_test)
        metrics = coefficient_metrics(y_test, pred)
        print("Validation metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")
    else:
        print("Validation skipped because test split is empty.")


if __name__ == "__main__":
    main()
