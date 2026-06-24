import torch
import h5py
import numpy as np
import math
from scipy.special import gamma
import time
import torch.nn.functional as F

# ==========================================
# MATHEMATICAL HELPERS (CPU Precomputation)
# ==========================================
def matlab_round(x):
    return np.floor(x + 0.5)

def nmzern(nz):
    csum = np.cumsum(np.arange(1, nz + 2))
    n = np.sum(csum < nz)
    if n == 0: return 0, 0
    if n % 2 == 0:
        m = int(np.fix((nz - csum[n-1]) / 2.0) * 2)
    else:
        m = int(matlab_round((nz - csum[n-1]) / 2.0) * 2 - 1)
    return n, m

def pupil(Na):
    x = np.linspace(-1, 1, Na)
    X, Y = np.meshgrid(x, x)
    return (np.sqrt(X**2 + Y**2) <= 1.0).astype(np.float32)

def zernike(mode, Na):
    n, m = nmzern(mode)
    x = np.linspace(-1, 1, Na)
    X, Y = np.meshgrid(x, x)
    r = np.sqrt(X**2 + Y**2)
    th = np.arctan2(Y, X)
    
    R_poly = np.zeros_like(r)
    s = 0
    while s >= 0 and s <= (n - abs(m)) / 2:
        a = (-1)**s * math.factorial(n - s)
        b = math.factorial(s) * math.factorial(int((n + abs(m))/2 - s)) * math.factorial(int((n - abs(m))/2 - s))
        R_poly += (a / b) * r**(n - 2*s)
        s += 1
        
    if m == 0:
        z = np.sqrt(n + 1) * R_poly
    else:
        if mode % 2 == 0:
            z = np.sqrt(2 * (n + 1)) * R_poly * np.cos(abs(m) * th)
        else:
            z = np.sqrt(2 * (n + 1)) * R_poly * np.sin(abs(m) * th)
            
    return (z * pupil(Na)).astype(np.float32)

def ZernikeCovarianceMat(nZernike):
    Cz = np.zeros((nZernike, nZernike))
    for i in range(1, nZernike + 1):
        ni, mi = nmzern(i + 1)
        for j in range(1, nZernike + 1):
            nj, mj = nmzern(j + 1)
            if mi == mj:
                m = mi
                if m == 0 or (m != 0 and (i - j) % 2 == 0):
                    t1 = gamma(14/3) * (4.8 * gamma(1.2))**(5/6) * (gamma(11/6))**2 / (2**(8/3) * math.pi)
                    t2 = (-1)**((ni + nj - 2*abs(m)) / 2)
                    t3 = math.sqrt((ni + 1) * (nj + 1)) * gamma((ni + nj - 5/3) / 2)
                    t4 = gamma((ni - nj + 17/3) / 2) * gamma((nj - ni + 17/3) / 2) * gamma((nj + ni + 23/3) / 2)
                    Cz[i-1, j-1] = t1 * t2 * t3 / t4
    return Cz.astype(np.float32)

def ZernikeSerial(D, r0, nZernike, L):
    Cz = ZernikeCovarianceMat(nZernike) * (D / r0)**(5/3)
    U, S, Vh = np.linalg.svd(Cz)
    KL = np.random.randn(nZernike, L).astype(np.float32)
    Zer = U.astype(np.float32) @ np.diag(np.sqrt(S).astype(np.float32)) @ KL
    return Zer.T

# ==========================================
# CORE GPU DATA GENERATION PIPELINE
# ==========================================
def generate_gpu_data_isnet(h5_filename="ISNet_PhaseData_5000.h5", FrameNum=5000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running ISNet data generation on: {device}")
    
    Pixel_Num = 240
    NmodeTotal = 299
    Sub_Num = 12
    Grid_Size = 16  # Padded grid size for ISNet input
    Pixel_Lens = 20
    telediam = 1.8
    r0 = 0.1
    Na = Sub_Num * Pixel_Lens
    
    Pxy_Circle = pupil(Na)
    
    # Precompute Zernike maps
    print("Precomputing Zernike maps...")
    Phase_Aberration_DataSet = np.zeros((Na * Na, NmodeTotal), dtype=np.float32)
    Phase_Aberration_32x32 = np.zeros((32 * 32, NmodeTotal), dtype=np.float32)
    
    for nmode in range(NmodeTotal):
        # Full resolution for SH-WFS simulation
        z_map = zernike(nmode + 2, Na) 
        Phase_Aberration_DataSet[:, nmode] = z_map.ravel()
        # Downsampled resolution for ISNet Target
        z_map_32 = F.interpolate(torch.tensor(z_map).unsqueeze(0).unsqueeze(0), size=(32, 32), mode='bilinear').squeeze().numpy()
        Phase_Aberration_32x32[:, nmode] = z_map_32.ravel()
        
    Phase_Aberration_DataSet_gpu = torch.tensor(Phase_Aberration_DataSet, device=device)
    Phase_Aberration_32x32_gpu = torch.tensor(Phase_Aberration_32x32, device=device)
    Zer_DataSet = ZernikeSerial(telediam, r0, NmodeTotal, FrameNum)

    with h5py.File(h5_filename, 'w') as h5f:
        # X variables are 16x16 grids. Y is a 32x32 phase map.
        X_I = h5f.create_dataset("X_I", shape=(FrameNum, 1, Grid_Size, Grid_Size), dtype='float32')
        X_Sx = h5f.create_dataset("X_Sx", shape=(FrameNum, 1, Grid_Size, Grid_Size), dtype='float32')
        X_Sy = h5f.create_dataset("X_Sy", shape=(FrameNum, 1, Grid_Size, Grid_Size), dtype='float32')
        Y_Phase = h5f.create_dataset("Y_Phase", shape=(FrameNum, 1, 32, 32), dtype='float32')
        
        batch_size = min(250, FrameNum)
        t0 = time.time()
        
        for start_idx in range(0, FrameNum, batch_size):
            end_idx = min(start_idx + batch_size, FrameNum)
            B = end_idx - start_idx
            
            z_batch = torch.tensor(Zer_DataSet[start_idx:end_idx], device=device)
            
            # 1. Generate Target Phase Maps (32x32)
            phase_targets = torch.matmul(Phase_Aberration_32x32_gpu, z_batch.t())
            phase_targets = phase_targets.t().view(B, 1, 32, 32)
            
            # 2. Generate full aberrations for SH-WFS simulation
            phase_aberrations = torch.matmul(Phase_Aberration_DataSet_gpu, z_batch.t())
            phase_aberrations = phase_aberrations.t().view(B, Na, Na)
            
            # 3. Derive slopes directly from the phase gradients
            dx = torch.gradient(phase_aberrations, dim=2)[0]
            dy = torch.gradient(phase_aberrations, dim=1)[0]
            
            # Pool down to 12x12 grid representing the subapertures
            sub_I = F.avg_pool2d(torch.ones_like(phase_aberrations).unsqueeze(1), kernel_size=Pixel_Lens)
            sub_Sx = F.avg_pool2d(dx.unsqueeze(1), kernel_size=Pixel_Lens)
            sub_Sy = F.avg_pool2d(dy.unsqueeze(1), kernel_size=Pixel_Lens)
            
            # Pad the 12x12 physical grid to the 16x16 grid required by ISNet
            pad_size = (Grid_Size - Sub_Num) // 2
            pad_tuple = (pad_size, pad_size, pad_size, pad_size)
            
            grid_I = F.pad(sub_I, pad_tuple, "constant", 0)
            grid_Sx = F.pad(sub_Sx, pad_tuple, "constant", 0)
            grid_Sy = F.pad(sub_Sy, pad_tuple, "constant", 0)
            
            # Save to disk
            X_I[start_idx:end_idx] = grid_I.cpu().numpy()
            X_Sx[start_idx:end_idx] = grid_Sx.cpu().numpy()
            X_Sy[start_idx:end_idx] = grid_Sy.cpu().numpy()
            Y_Phase[start_idx:end_idx] = phase_targets.cpu().numpy()
            
            print(f"Processed frames {start_idx} to {end_idx-1} ({(time.time() - t0):.2f}s elapsed)")

if __name__ == '__main__':
    generate_gpu_data_isnet()