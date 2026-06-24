import torch
import torch.nn as nn
import torch.nn.functional as F

def _scatter_softmax(e: torch.Tensor, dst_idx: torch.Tensor, N: int) -> torch.Tensor:
    K = e.shape[1]
    idx_exp = dst_idx.unsqueeze(-1).expand(-1, K)

    e_max = torch.full((N, K), float('-inf'), device=e.device, dtype=e.dtype)
    e_max.scatter_reduce_(0, idx_exp, e, reduce='amax', include_self=True)

    exp_e = (e - e_max[dst_idx]).exp()
    denom = torch.zeros(N, K, device=e.device, dtype=e.dtype)
    denom.scatter_add_(0, idx_exp, exp_e)

    return exp_e / (denom[dst_idx] + 1e-16)

class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim: int, head_dim: int, num_heads: int, out_dim: int = None, 
                 concat: bool = True, dropout: float = 0.0, leaky_slope: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.concat = concat

        self.W = nn.Linear(in_dim, num_heads * head_dim, bias=False)
        self.attn = nn.Parameter(torch.empty(num_heads, 2 * head_dim))

        if (not concat) and (out_dim is not None):
            self.Wout = nn.Linear(head_dim, out_dim, bias=False)
            self.out_dim = out_dim
        else:
            self.Wout = None
            self.out_dim = num_heads * head_dim if concat else head_dim

        self.leaky_relu = nn.LeakyReLU(leaky_slope)
        self.attn_drop = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.attn.unsqueeze(0))
        if self.Wout is not None:
            nn.init.xavier_uniform_(self.Wout.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        N, E = x.size(0), edge_index.size(1)
        src_idx, dst_idx = edge_index[0], edge_index[1]

        Wh = self.W(x).view(N, self.num_heads, self.head_dim)
        cat = torch.cat([Wh[dst_idx], Wh[src_idx]], dim=-1)
        
        e = (self.attn * cat).sum(dim=-1)
        e = self.leaky_relu(e)

        alpha = self.attn_drop(_scatter_softmax(e, dst_idx, N))

        weighted = alpha.unsqueeze(-1) * Wh[src_idx]
        agg = torch.zeros(N, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
        agg.scatter_add_(0, dst_idx.view(-1, 1, 1).expand(E, self.num_heads, self.head_dim), weighted)
        agg = F.elu(agg)

        if self.concat:
            return agg.view(N, self.num_heads * self.head_dim)
        else:
            avg = agg.mean(dim=1)
            if self.Wout is not None:
                avg = F.elu(self.Wout(avg))
            return avg