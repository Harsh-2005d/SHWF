import torch
import torch.nn as nn
import numpy as np
import h5py

class WeightedMSELoss(nn.Module):
    """
    Inverse variance weight mapping with strict numerical clamping boundaries
    to prevent gradient explosions on high-order variables.
    """
    def __init__(self, variances: torch.Tensor):
        super().__init__()
        # Clamp to 1e-4 floor to eliminate infinite scaling trends
        stabilized_vars = torch.clamp(variances, min=1e-4)
        weights = 1.0 / stabilized_vars
        
        # Normalize weights so the mean is 1.0 to preserve gradient scaling
        weights = weights / weights.mean()
        self.register_buffer('weights', weights)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sq = (pred - target) ** 2
        return (sq * self.weights).mean()

    @classmethod
    def from_h5(cls, h5_path: str, n_val: int = 0) -> 'WeightedMSELoss':
        with h5py.File(h5_path, 'r') as f:
            N = f['Ytrain'].shape[0]
            end = N - n_val if n_val > 0 else N
            y = torch.from_numpy(f['Ytrain'][:end].astype(np.float32))
        variances = y.var(dim=0)
        return cls(variances)