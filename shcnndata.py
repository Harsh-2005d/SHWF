import numpy as np
import h5py
import math
from scipy.special import gamma
import time
import os

# ==========================================
# 1. MATHEMATICAL HELPERS
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
    return (np.sqrt(X**2 + Y**2) <= 1.0).astype(float)

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
            
    return z * pupil(Na)

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
    return Cz

def ZernikeSerial(D, r0, nZernike, L):
    Cz = ZernikeCovarianceMat(nZernike) * (D / r0)**(5/3)
    U, S, Vh = np.linalg.svd(Cz)
    KL = np.random.randn(nZernike, L)
    Zer = U @ np.diag(np.sqrt(S)) @ KL
    return Zer.T

def SumNormalization(Img):
    batch_size, h, w = Img.shape
    block_h, block_w = 20, 20
    grid_h, grid_w = h // block_h, w // block_w
    
    reshaped = Img.reshape(batch_size, grid_h, block_h, grid_w, block_w)
    block_sums = reshaped.sum(axis=(2, 4), keepdims=True)
    
    normalized_reshaped = np.divide(
        reshaped, 
        block_sums, 
        out=np.zeros_like(reshaped), 
        where=block_sums != 0
    )
    return normalized_reshaped.reshape(batch_size, h, w)

# ==========================================
# 2. MAIN OPTICS GENERATOR PIPELINE
# ==========================================

def generate_shcnn_data(h5_filename="Sum_NewData_299_100.h5", FrameNum=20):
    Pixel_Num = 240
    NmodeTotal = 299
    Sub_Num = 12
    Pixel_Lens = 20
    telediam = 1.8
    r0 = 0.1
    Na = Sub_Num * Pixel_Lens
    
    print("Building optical geometry and subaperture masks...")
    Pxy_Circle = pupil(Na)
    PhaseMask_3D = []
    
    for row in range(Sub_Num):
        for col in range(Sub_Num):
            cy = row * Pixel_Lens + Pixel_Lens / 2.0
            cx = col * Pixel_Lens + Pixel_Lens / 2.0
            if math.sqrt((cx - 120)**2 + (cy - 120)**2) <= 120:
                mask = np.zeros((Na, Na))
                mask[row*Pixel_Lens:(row+1)*Pixel_Lens, col*Pixel_Lens:(col+1)*Pixel_Lens] = 1
                PhaseMask_3D.append(mask)
                
    PhaseMask_3D = np.stack(PhaseMask_3D, axis=-1)
    Nsub = PhaseMask_3D.shape[2]
    Pxy = np.sum(PhaseMask_3D, axis=2) * Pxy_Circle

    print("Precomputing Zernike basis maps...")
    Phase_Aberration_DataSet = np.zeros((Na * Na, NmodeTotal))
    for nmode in range(NmodeTotal):
        z_map = zernike(nmode + 2, Na) 
        Phase_Aberration_DataSet[:, nmode] = z_map.ravel()

    print("Generating Zernike Covariance Coefficients...")
    Zer_DataSet = ZernikeSerial(telediam, r0, NmodeTotal, FrameNum)

    print(f"Creating HDF5 Tensor Container: {h5_filename}")
    with h5py.File(h5_filename, 'w') as h5f:
        X_train = h5f.create_dataset("Xtrain", shape=(FrameNum, Na, Na), dtype='float64')
        Y_train = h5f.create_dataset("Ytrain", shape=(FrameNum, NmodeTotal), dtype='float64')
        
        # FIX: Dynamic batch sizing ensures data flushes to disk even on short test runs
        batch_size = min(1000, FrameNum)
        batch_X = np.zeros((batch_size, Na, Na), dtype='float64')
        batch_Y = np.zeros((batch_size, NmodeTotal), dtype='float64')
        
        t0 = time.time()
        for i in range(FrameNum):
            batch_idx = i % batch_size
            
            Phase_Aberration = (Phase_Aberration_DataSet @ Zer_DataSet[i, :]).reshape(Na, Na)
            Uin = Pxy * np.exp(1j * Phase_Aberration)
            
            IFar = np.zeros((Na, Na))
            for j in range(Nsub):
                mask = PhaseMask_3D[:, :, j] == 1
                USubTemp = Uin[mask].reshape(Pixel_Lens, Pixel_Lens)
                ISubFarTemp = np.abs(np.fft.fftshift(np.fft.fft2(USubTemp)))**2
                IFar[mask] = ISubFarTemp.ravel()
                
            batch_X[batch_idx] = IFar
            batch_Y[batch_idx] = Zer_DataSet[i, :]
            
            # Flush to disk when batch is full OR when it hits the final frame
            if (i + 1) % batch_size == 0 or i == FrameNum - 1:
                current_batch_size = batch_idx + 1
                start_idx = i + 1 - current_batch_size
                end_idx = i + 1
                
                print(f"  Writing frames {start_idx} to {end_idx-1} to disk... ({(time.time() - t0):.2f}s elapsed)")
                
                X_train[start_idx:end_idx] = SumNormalization(batch_X[:current_batch_size])
                Y_train[start_idx:end_idx] = batch_Y[:current_batch_size]

if __name__ == '__main__':
    generate_shcnn_data(FrameNum=20) # Safe to test with 20 frames now!