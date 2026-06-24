import math
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

def make_subap_centers(sub_num: int = 12, pixel_lens: int = 20) -> np.ndarray:
    """Reconstructs subaperture reference centroids for a circular aperture."""
    Na = sub_num * pixel_lens
    half = Na / 2.0
    rows = []
    for row in range(sub_num):
        for col in range(sub_num):
            cy = row * pixel_lens + pixel_lens / 2.0
            cx = col * pixel_lens + pixel_lens / 2.0
            if math.sqrt((cx - half) ** 2 + (cy - half) ** 2) <= half:
                rows.append([cx, cy, float(row), float(col)])
    return np.array(rows, dtype=np.float32)

def build_static_edges(subap_info: np.ndarray) -> np.ndarray:
    """Builds the static 4-connected topology graph + self-attribution loops."""
    rows = subap_info[:, 2].astype(int)
    cols = subap_info[:, 3].astype(int)
    pos = {(r, c): i for i, (r, c) in enumerate(zip(rows, cols))}

    src_list, dst_list = [], []
    for i, (r, c) in enumerate(zip(rows, cols)):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            j = pos.get((r + dr, c + dc))
            if j is not None:
                src_list.append(i)
                dst_list.append(j)
        # Self loop for node tracking
        src_list.append(i)
        dst_list.append(i)

    return np.array([src_list, dst_list], dtype=np.int64)

class SHWSGraphDataset3D(Dataset):
    """
    Optimized 3-dimensional pipeline dataset loader. 
    Processes spot images into [del_x, del_y, normalized_intensity] arrays.
    """
    def __init__(self, h5_path: str, subap_info: np.ndarray, focal_length_px: float, 
                 split: str = 'train', val_fraction: float = 0.2, augment: bool = True):
        self.h5_path = h5_path
        self.subap_info = subap_info
        self.focal_length_px = focal_length_px
        self.augment = augment and (split == 'train')
        self.pixel_lens = int(focal_length_px)
        
        self.edge_index = torch.from_numpy(build_static_edges(subap_info))

        with h5py.File(h5_path, 'r') as f:
            N = f['Xtrain'].shape[0]

        n_val = max(1, int(N * val_fraction))
        if split == 'train':
            self.indices = list(range(N - n_val))
        else:
            self.indices = list(range(N - n_val, N))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        real_idx = self.indices[idx]

        with h5py.File(self.h5_path, 'r') as f:
            img = torch.from_numpy(f['Xtrain'][real_idx].astype(np.float32))
            y = torch.from_numpy(f['Ytrain'][real_idx].astype(np.float32))

        N_sub = len(self.subap_info)
        spot_feat = torch.zeros((N_sub, 3), dtype=torch.float32)
        subap_feat = torch.from_numpy(self.subap_info[:, :2] / self.focal_length_px)

        # Vectorized block grid extraction instead of pixel loops
        for i, (xr, yr, _, _) in enumerate(self.subap_info):
            r_start, c_start = int(yr - self.pixel_lens/2), int(xr - self.pixel_lens/2)
            patch = img[r_start:r_start+self.pixel_lens, c_start:c_start+self.pixel_lens]
            
            # Sub-aperture background thresholding
            thresholded = torch.where(patch > 0.05, patch, torch.zeros_like(patch))
            total_intensity = torch.sum(thresholded)

            if total_intensity > 0:
                y_idx, x_idx = torch.meshgrid(torch.arange(self.pixel_lens), torch.arange(self.pixel_lens), indexing='ij')
                dx = (torch.sum(x_idx.float() * thresholded) / total_intensity - (self.pixel_lens / 2)) / self.focal_length_px
                dy = (torch.sum(y_idx.float() * thresholded) / total_intensity - (self.pixel_lens / 2)) / self.focal_length_px
                spot_feat[i] = torch.tensor([dx, dy, total_intensity])

        # Data augmentation simulation (10% to 30% random dropouts)
        if self.augment and np.random.rand() < 0.5:
            drop_rate = np.random.uniform(0.10, 0.30)
            n_drop = max(1, int(N_sub * drop_rate))
            drop_idx = np.random.choice(N_sub, n_drop, replace=False)
            spot_feat[drop_idx] = 0.0

        return {
            'spot_feat': spot_feat,
            'subap_feat': subap_feat,
            'edge_index': self.edge_index,
            'y': y
        }

def collate_graphs(batch: list[dict]) -> dict:
    spot_feats, subap_feats, edges, ys, batch_ids = [], [], [], [], []
    offset = 0

    for gid, sample in enumerate(batch):
        N = sample['spot_feat'].size(0)
        spot_feats.append(sample['spot_feat'])
        subap_feats.append(sample['subap_feat'])
        edges.append(sample['edge_index'] + offset)
        ys.append(sample['y'])
        batch_ids.append(torch.full((N,), gid, dtype=torch.long))
        offset += N

    return {
        'spot_feat': torch.cat(spot_feats, dim=0),
        'subap_feat': torch.cat(subap_feats, dim=0),
        'edge_index': torch.cat(edges, dim=1),
        'y': torch.stack(ys, dim=0),
        'batch': torch.cat(batch_ids, dim=0),
    }