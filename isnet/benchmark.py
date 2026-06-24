import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from eval_utils import DEFAULT_DATASET, DEFAULT_WEIGHTS, load_model, phase_metrics, pupil_mask


METRIC_NAMES = (
    "true_rms",
    "pred_rms",
    "rmse",
    "mae",
    "residual_pv",
    "relative_rmse",
    "correlation",
)


def validation_indices(length, fraction, seed):
    if not 0.0 < fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")
    val_size = min(max(1, round(length * fraction)), length - 1)
    # Matches random_split(..., generator=manual_seed(seed)) in train.py.
    permutation = torch.randperm(length, generator=torch.Generator().manual_seed(seed))
    return permutation[length - val_size :].numpy()


def summarize(rows):
    summary = {"frames": len(rows)}
    for metric in METRIC_NAMES:
        values = np.asarray([row[metric] for row in rows])
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
        }
    return summary


@torch.inference_mode()
def benchmark(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.weights, device)
    mask = pupil_mask(32)
    rows = []

    with h5py.File(args.data, "r") as file:
        length = len(file["Y_Phase"])
        if args.split == "validation":
            indices = validation_indices(length, args.val_fraction, args.seed)
        else:
            indices = np.arange(length)
        if args.max_frames is not None:
            indices = indices[: args.max_frames]

        for start in tqdm(range(0, len(indices), args.batch_size), desc="benchmark"):
            batch_indices = indices[start : start + args.batch_size]
            # h5py requires monotonically increasing fancy indices.
            order = np.argsort(batch_indices)
            sorted_indices = batch_indices[order]
            inverse = np.argsort(order)
            arrays = [
                np.asarray(file[key][sorted_indices], dtype=np.float32)[inverse]
                for key in ("X_I", "X_Sx", "X_Sy", "Y_Phase")
            ]
            inputs = [torch.from_numpy(array).to(device) for array in arrays[:3]]
            predictions = model(*inputs).cpu().numpy()[:, 0]
            targets = arrays[3][:, 0]

            for frame, target, prediction in zip(batch_indices, targets, predictions):
                row = {"frame": int(frame)}
                row.update(phase_metrics(target, prediction, mask))
                rows.append(row)

    return rows, summarize(rows), device


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark ISNet wavefront reconstruction")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=Path("isnet_benchmark"))
    parser.add_argument("--split", choices=("validation", "all"), default="validation")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1 or (args.max_frames is not None and args.max_frames < 1):
        raise ValueError("batch-size and max-frames must be positive")
    rows, summary, device = benchmark(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "per_frame_metrics.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("frame", *METRIC_NAMES))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "device": str(device),
        "data": str(args.data.resolve()),
        "weights": str(args.weights.resolve()),
        "split": args.split,
        "piston_removed": True,
        "pupil_masked": True,
        "summary": summary,
    }
    json_path = args.output_dir / "summary.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Evaluated {len(rows)} frames on {device}")
    for metric in METRIC_NAMES:
        stats = summary[metric]
        print(f"{metric:14s}: {stats['mean']:.6f} ± {stats['std']:.6f}")
    print(f"Per-frame CSV: {csv_path.resolve()}")
    print(f"Summary JSON:  {json_path.resolve()}")


if __name__ == "__main__":
    main()
