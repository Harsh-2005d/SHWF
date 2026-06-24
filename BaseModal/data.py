import numpy as np

from .slopes import batch_to_slopes


def read_h5_training_data(h5_path, x_key="Xtrain", y_key="Ytrain", limit=None):
    import h5py

    with h5py.File(h5_path, "r") as h5f:
        total = h5f[x_key].shape[0]
        count = total if limit is None else min(limit, total)
        images = h5f[x_key][:count]
        coeffs = h5f[y_key][:count]
    return images, coeffs


def make_slope_dataset(h5_path, x_key="Xtrain", y_key="Ytrain", limit=None):
    images, coeffs = read_h5_training_data(h5_path, x_key=x_key, y_key=y_key, limit=limit)
    slopes = batch_to_slopes(images)
    return slopes, coeffs


def train_test_split(x, y, test_fraction=0.2, seed=42):
    train_idx, test_idx = split_indices(x.shape[0], test_fraction=test_fraction, seed=seed)

    if test_idx.size == 0:
        return x[train_idx], x[:0], y[train_idx], y[:0]

    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def split_indices(n_samples, test_fraction=0.2, seed=42):
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be in the range [0, 1).")

    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    test_size = int(round(n_samples * test_fraction))
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    if train_idx.size == 0:
        raise ValueError("Not enough samples left for training.")

    return train_idx, test_idx
