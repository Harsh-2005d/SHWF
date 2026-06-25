import hcipy as hc
import numpy as np
import h5py
import torch
import torch.nn.functional as F
import time
import math

# ==========================================
# 1. HCIPy OPTICAL SYSTEM SETUP
# ==========================================
def setup_optical_system(device):
    print("Initializing HCIPy Optical Geometry...")
    D = 1.8              # Telescope diameter (m)
    num_subaps = 20      # 20x20 SH-WFS grid
    pixels_per_lens = 16 # Pixels per sub-aperture for accurate physics
    grid_res = num_subaps * pixels_per_lens # 320x320 pixel grid
    
    # Define Grids and Pupil
    pupil_grid = hc.make_pupil_grid(grid_res, D)
    aperture_func = hc.make_circular_aperture(D)
    aperture = aperture_func(pupil_grid)
    
    # Define Zernike Basis (Exclude Piston, Mode 1)
    num_zernike = 65
    zernike_basis = hc.make_zernike_basis(num_zernike, D, pupil_grid, starting_mode=2)
    
    # ---------------------------------------------------------
    # FIX 1: HCIPy Memory Trap & Efficiency
    # Extract only the pixels inside the pupil before doing the pseudo-inverse.
    # This prevents creating a massive 102,400 x 65 dense matrix.
    # ---------------------------------------------------------
    Z_mat = zernike_basis.transformation_matrix 
    Z_mat_masked = Z_mat[aperture > 0.5, :]
    Z_pinv = np.linalg.pinv(Z_mat_masked) 
    
    # Define Sub-aperture mapping coordinates
    x_coords = np.linspace(-1, 1, num_subaps)
    X_grid, Y_grid = np.meshgrid(x_coords, x_coords)
    
    subap_radius = np.sqrt(X_grid**2 + Y_grid**2)
    valid_subaps_mask = subap_radius <= 1.0
    N_valid = np.sum(valid_subaps_mask)
    
    print(f"Valid Sub-apertures inside pupil: {N_valid} out of {num_subaps**2}")
    
    valid_X = X_grid[valid_subaps_mask]
    valid_Y = Y_grid[valid_subaps_mask]

    system = {
        'Z_pinv': torch.tensor(Z_pinv, dtype=torch.float32, device=device), # (Modes, Valid_Pixels)
        'aperture': torch.tensor(aperture, dtype=torch.float32, device=device).view(grid_res, grid_res),
        'valid_subaps_mask': torch.tensor(valid_subaps_mask, device=device),
        'valid_X': torch.tensor(valid_X, dtype=torch.float32, device=device),
        'valid_Y': torch.tensor(valid_Y, dtype=torch.float32, device=device),
        'D': D, 'grid_res': grid_res, 'pixels_per_lens': pixels_per_lens, 
        'num_zernike': num_zernike, 'N_valid': N_valid
    }
    return system

# ==========================================
# 2. FAST GPU PHASE SCREEN GENERATOR
# ==========================================
def generate_von_karman_batch(B, sys, r0_batch, device):
    grid_res = sys['grid_res']
    D = sys['D']
    L0 = 25.0 
    
    dq = 1.0 / D
    q = torch.fft.fftfreq(grid_res, d=D/grid_res, device=device)
    Qx, Qy = torch.meshgrid(q, q, indexing='ij')
    Q2 = Qx**2 + Qy**2
    Q2[0, 0] = 1e-8 
    
    PSD = 0.023 * (r0_batch.view(B, 1, 1)**(-5/3)) * ((Q2 + (1.0/L0)**2)**(-11/6))
    PSD[:, 0, 0] = 0
    
    noise = torch.randn((B, grid_res, grid_res), device=device) + \
            1j * torch.randn((B, grid_res, grid_res), device=device)
    
    phase_f = noise * torch.sqrt(PSD) * (dq * grid_res)
    phase_screens = torch.fft.ifft2(phase_f).real
    
    return phase_screens * sys['aperture']

# ==========================================
# 3. CORE SIMULATION PIPELINE
# ==========================================
def generate_transformer_dataset(h5_filename="ZERN_Trans_Dataset_500k.h5", FrameNum=500000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running generation on: {device}")
    
    sys = setup_optical_system(device)
    N_valid = sys['N_valid']
    
    # Normalization constant for slopes (rad/m). Adjust based on max expected turbulence.
    SLOPE_NORM_FACTOR = 50.0 
    
    with h5py.File(h5_filename, 'w') as h5f:
        X_dataset = h5f.create_dataset(
            "X_Tokens", shape=(FrameNum, N_valid, 5), 
            dtype='float32', chunks=(1000, N_valid, 5), compression="lzf"
        )
        Y_dataset = h5f.create_dataset(
            "Y_Zernikes", shape=(FrameNum, sys['num_zernike']), 
            dtype='float32', chunks=(1000, sys['num_zernike']), compression="lzf"
        )
        
        batch_size = 500
        t0 = time.time()
        
        for start_idx in range(0, FrameNum, batch_size):
            end_idx = min(start_idx + batch_size, FrameNum)
            B = end_idx - start_idx
            
            r0_batch = torch.rand(B, device=device) * 0.15 + 0.05 
            
            phase = generate_von_karman_batch(B, sys, r0_batch, device) 
            
            phase_flat = phase.view(B, -1)
            phase_masked = phase_flat[:, sys['aperture'].view(-1) > 0.5]
            Y_target = torch.matmul(phase_masked, sys['Z_pinv'].T) 
            
            dx = torch.gradient(phase, dim=2)[0]
            dy = torch.gradient(phase, dim=1)[0]
            
            sub_Sx = F.avg_pool2d(dx.unsqueeze(1), kernel_size=sys['pixels_per_lens']).squeeze(1)
            sub_Sy = F.avg_pool2d(dy.unsqueeze(1), kernel_size=sys['pixels_per_lens']).squeeze(1)
            
            # ---------------------------------------------------------
            # FIX 3: Log-Normal Scintillation Model
            # Replaced linear clamping with physics-based exponential decay.
            # ---------------------------------------------------------
            laplacian = torch.gradient(dx, dim=2)[0] + torch.gradient(dy, dim=1)[0]
            sub_lap = F.avg_pool2d(laplacian.unsqueeze(1), kernel_size=sys['pixels_per_lens']).squeeze(1)
            
            # Base intensity driven by wavefront curvature
            sub_I_base = torch.exp(-0.5 * sub_lap)
            
            # Inject multiplicative log-normal statistical noise
            scintillation_variance = 0.1 
            log_normal_noise = torch.exp(torch.randn_like(sub_I_base) * math.sqrt(scintillation_variance))
            sub_I = sub_I_base * log_normal_noise
            
            Sx_valid = sub_Sx[:, sys['valid_subaps_mask']] 
            Sy_valid = sub_Sy[:, sys['valid_subaps_mask']]
            I_valid = sub_I[:, sys['valid_subaps_mask']]
            
            photon_noise_scale = 0.05
            read_noise_scale = 0.02
            
            I_noisy = I_valid + photon_noise_scale * torch.randn_like(I_valid) * torch.sqrt(I_valid)
            I_noisy = torch.clamp(I_noisy, min=0.0)
            
            noise_Sx = read_noise_scale * torch.randn_like(Sx_valid) + \
                       photon_noise_scale * torch.randn_like(Sx_valid) / (I_noisy + 1e-3)
            noise_Sy = read_noise_scale * torch.randn_like(Sy_valid) + \
                       photon_noise_scale * torch.randn_like(Sy_valid) / (I_noisy + 1e-3)
                       
            Sx_noisy = Sx_valid + noise_Sx
            Sy_noisy = Sy_valid + noise_Sy
            
            # ---------------------------------------------------------
            # FIX 2: Transformer Slope Normalization
            # We scale by a global constant rather than zero-mean instance normalization, 
            # because subtracting the mean would destroy Tip and Tilt (Modes 2 & 3).
            # ---------------------------------------------------------
            Sx_norm = Sx_noisy / SLOPE_NORM_FACTOR
            Sy_norm = Sy_noisy / SLOPE_NORM_FACTOR
            
            x_coords = sys['valid_X'].unsqueeze(0).expand(B, -1)
            y_coords = sys['valid_Y'].unsqueeze(0).expand(B, -1)
            
            # Stack into Transformer Tokens [Sx, Sy, I, x, y]
            Tokens = torch.stack([Sx_norm, Sy_norm, I_noisy, x_coords, y_coords], dim=-1)
            
            # Random Sub-aperture Masking (Simulate Dead Pixels/Dropout)
            dropout_mask = (torch.rand((B, N_valid, 1), device=device) > 0.05).float()
            Tokens = Tokens * dropout_mask
            
            X_dataset[start_idx:end_idx] = Tokens.cpu().numpy()
            Y_dataset[start_idx:end_idx] = Y_target.cpu().numpy()
            
            if (start_idx % 5000 == 0) and start_idx > 0:
                print(f"Generated {start_idx} / {FrameNum} frames. Time elapsed: {(time.time() - t0):.2f}s")

if __name__ == '__main__':
    generate_transformer_dataset()