import numpy as np


def coefficient_metrics(y_true, y_pred):
    err = np.asarray(y_pred) - np.asarray(y_true)
    rmse_per_frame = np.sqrt(np.mean(err**2, axis=1))
    mae_per_frame = np.mean(np.abs(err), axis=1)
    return {
        "rmse_mean": float(np.mean(rmse_per_frame)),
        "rmse_std": float(np.std(rmse_per_frame)),
        "mae_mean": float(np.mean(mae_per_frame)),
        "mae_std": float(np.std(mae_per_frame)),
    }
