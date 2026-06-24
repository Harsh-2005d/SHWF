import torch
import torch.nn as nn
import torch.nn.functional as F
from gat import MultiHeadGATLayer

class GSHWS_GAT_3D(nn.Module):
    """
    Lightweight formulation targeting 3-channel input node features 
    while retaining structural properties for simple deployment.
    """
    _LATENT = 16
    _K = 4
    _HEAD_MID = 4
    _HEAD_OUT = 16
    _GRAPH_DIM = 64
    _N_ZERNIKE = 299  # Matched to output data target

    def __init__(self, spot_feat_dim: int = 3, dropout: float = 0.1):
        super().__init__()
        in_dim = spot_feat_dim + 2  # Spatial [dx, dy, I] paired with [xr, yr]

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, self._LATENT),
            nn.ELU(),
        )

        gat_mid = dict(in_dim=self._LATENT, head_dim=self._HEAD_MID, num_heads=self._K, concat=True, dropout=dropout)
        self.gat2 = MultiHeadGATLayer(**gat_mid)
        self.gat3 = MultiHeadGATLayer(**gat_mid)
        self.gat4 = MultiHeadGATLayer(**gat_mid)

        self.gat5 = MultiHeadGATLayer(
            in_dim=self._LATENT, head_dim=self._HEAD_OUT, num_heads=self._K, 
            out_dim=self._GRAPH_DIM, concat=False, dropout=dropout
        )

        self.mlp = nn.Sequential(
            nn.Linear(self._GRAPH_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self._N_ZERNIKE),
        )
        self.feat_drop = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, spot_feat: torch.Tensor, subap_feat: torch.Tensor, 
                edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = torch.cat([spot_feat, subap_feat], dim=-1)
        x = self.input_proj(x)

        x = self.feat_drop(self.gat2(x, edge_index))
        x = self.feat_drop(self.gat3(x, edge_index))
        x = self.feat_drop(self.gat4(x, edge_index))
        x = self.gat5(x, edge_index)

        # Global Average Pooling across dynamic graph instances
        # ── Optimized Global average pooling (No Graph Breaks) ────────────────
# Use shape properties rather than pulling values to the CPU via .item()
        B = batch[-1] + 1 if batch.numel() > 0 else 1

        g_rep = torch.zeros(B, self._GRAPH_DIM, device=x.device, dtype=x.dtype)
        cnt   = torch.zeros(B, 1,               device=x.device, dtype=x.dtype)

        g_rep.scatter_add_(0, batch.unsqueeze(-1).expand_as(x), x)
        cnt.scatter_add_(0,   batch.unsqueeze(-1), torch.ones(x.size(0), 1, device=x.device))

        g_rep = g_rep / (cnt + 1e-8)   # [B, 64]
        return self.mlp(g_rep)