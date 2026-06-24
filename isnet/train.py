import argparse
from pathlib import Path

import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from model import ISNet


DATASET_KEYS = ("X_I", "X_Sx", "X_Sy", "Y_Phase")
EXPECTED_SHAPES = {
    "X_I": (1, 16, 16),
    "X_Sx": (1, 16, 16),
    "X_Sy": (1, 16, 16),
    "Y_Phase": (1, 32, 32),
}
DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "ISNet_PhaseData_5000.h5"


class SHWFSDataset(Dataset):
    """Lazily opens the generated HDF5 file once in each DataLoader worker."""

    def __init__(self, h5_file):
        self.h5_file = str(Path(h5_file).expanduser().resolve())
        self.file = None
        self.arrays = None

        with h5py.File(self.h5_file, "r") as file:
            missing = set(DATASET_KEYS) - set(file.keys())
            if missing:
                raise ValueError(f"Dataset is missing keys: {sorted(missing)}")

            lengths = {key: len(file[key]) for key in DATASET_KEYS}
            if len(set(lengths.values())) != 1:
                raise ValueError(f"Dataset arrays have different lengths: {lengths}")

            for key, expected in EXPECTED_SHAPES.items():
                actual = tuple(file[key].shape[1:])
                if actual != expected:
                    raise ValueError(
                        f"{key} has sample shape {actual}; expected {expected}"
                    )
            self.length = lengths["X_I"]

    def _open(self):
        if self.file is None:
            self.file = h5py.File(self.h5_file, "r")
            self.arrays = tuple(self.file[key] for key in DATASET_KEYS)

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None
            self.arrays = None

    def __del__(self):
        self.close()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self._open()
        # h5py returns independent NumPy arrays here. as_tensor preserves their
        # float32 dtype without an unnecessary second copy.
        return tuple(torch.as_tensor(array[idx]) for array in self.arrays)


def make_loaders(dataset, batch_size, workers, val_fraction, seed, pin_memory):
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")
    if len(dataset) < 2:
        raise ValueError("At least two samples are required for train/validation")

    val_size = max(1, round(len(dataset) * val_fraction))
    val_size = min(val_size, len(dataset) - 1)
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(
        dataset, [len(dataset) - val_size, val_size], generator=generator
    )
    loader_args = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    return train_loader, val_loader


def move_batch(batch, device):
    return tuple(tensor.to(device, non_blocking=True) for tensor in batch)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum = 0.0
    progress = tqdm(loader, desc="train", leave=False)
    for batch in progress:
        intensity, slope_x, slope_y, target = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(intensity, slope_x, slope_y)
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * target.size(0)
        progress.set_postfix(loss=f"{loss.item():.6f}")
    return loss_sum / len(loader.dataset)


@torch.inference_mode()
def validate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    for batch in tqdm(loader, desc="valid", leave=False):
        intensity, slope_x, slope_y, target = move_batch(batch, device)
        prediction = model(intensity, slope_x, slope_y)
        loss_sum += criterion(prediction, target).item() * target.size(0)
    return loss_sum / len(loader.dataset)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ISNet on gpu_data.py output")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=Path("ISNet_weights.pth"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_isnet(args):
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("epochs/batch-size must be positive and workers non-negative")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}; dataset: {args.data}")

    dataset = SHWFSDataset(args.data)
    train_loader, val_loader = make_loaders(
        dataset,
        args.batch_size,
        args.workers,
        args.val_fraction,
        args.seed,
        pin_memory=device.type == "cuda",
    )
    print(f"Samples: {len(train_loader.dataset)} train, {len(val_loader.dataset)} validation")

    model = ISNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), args.output)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{args.epochs}: train={train_loss:.6f} "
            f"val={val_loss:.6f} lr={lr:.2e}"
        )

    dataset.close()
    print(f"Best validation loss: {best_loss:.6f}")
    print(f"Best weights saved to: {args.output.resolve()}")


if __name__ == "__main__":
    train_isnet(parse_args())
