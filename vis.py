import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import subprocess
import math

# ==========================================
# MATHEMATICAL RECONSTRUCTION HELPERS
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

# ==========================================
# VISUALIZATION PIPELINE
# ==========================================

def load_and_plot_shcnn(h5_filename, frame_idx=0):
    try:
        with h5py.File(h5_filename, 'r') as h5f:
            total_frames = h5f['Xtrain'].shape[0]
            if frame_idx >= total_frames:
                print(f"Error: Frame index {frame_idx} out of bounds (max {total_frames-1})")
                return
            
            # Read the SHWFS tensor and the Zernike ground truth vector
            shwfs_img = h5f['Xtrain'][frame_idx]
            z_coeffs = h5f['Ytrain'][frame_idx]
            
    except FileNotFoundError:
        print(f"Error: Could not find target record file {h5_filename}.")
        return

    print("Rebuilding physical phase map from Zernike coefficients...")
    Na = 240
    NmodeTotal = 299
    reconstructed_phase = np.zeros((Na, Na))
    
    # Re-apply the Zernike basis to get the 2D visual wavefront
    for nmode in range(NmodeTotal):
        z_map = zernike(nmode + 2, Na)
        reconstructed_phase += z_map * z_coeffs[nmode]

    # Calculate metrics
    mask = reconstructed_phase != 0
    valid_data = reconstructed_phase[mask] if np.any(mask) else np.array([0])
    pv = np.max(valid_data) - np.min(valid_data)
    rms = np.std(valid_data)

    # -------------------------------------------------------------------------
    # Canvas Layout
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    vmax_phase = np.max(np.abs(reconstructed_phase)) if np.max(np.abs(reconstructed_phase)) > 0 else 1.5

    # Panel A: The Reconstructed Phase
    im_a = axes[0].imshow(reconstructed_phase, cmap='jet', origin='lower', vmin=-vmax_phase, vmax=vmax_phase)
    axes[0].set_title(rf"PV={pv:.4f}$\mu$m  RMS={rms:.4f}$\mu$m", fontsize=11)
    axes[0].set_xlabel("(a) Reconstructed Wavefront ($Y_{train}$)", fontsize=12, labelpad=10)
    fig.colorbar(im_a, ax=axes[0], fraction=0.046, pad=0.04)

    # Panel B: The Normalized Tensor
    im_b = axes[1].imshow(shwfs_img, cmap='jet', origin='lower', vmin=0, vmax=np.max(shwfs_img))
    axes[1].set_title("Sum-Normalized Block Intensity", fontsize=11)
    axes[1].set_xlabel("(b) SHWFS Input Tensor ($X_{train}$)", fontsize=12, labelpad=10)
    fig.colorbar(im_b, ax=axes[1], fraction=0.046, pad=0.04)

    # Formatting cleanup
    for ax in axes.ravel():
        ax.set_xticks([50, 100, 150, 200])
        ax.set_yticks([50, 100, 150, 200])
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(False) 

    plt.tight_layout()
    
    output_filename = f"shcnn_python_frame_{frame_idx}.png"
    plt.savefig(output_filename, dpi=200, bbox_inches='tight')
    plt.close(fig) 
    print(f"Canvas written to: {output_filename}")
    
    # Auto-dispatch viewing instruction
    try:
        subprocess.run(["xdg-open", output_filename], check=True)
    except Exception as e:
        print(f"Could not automatically execute system view call: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate diagnostic plots for purely Pythonic datasets.")
    parser.add_argument("--file", type=str, default="Sum_NewData_299_100.h5")
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()

    load_and_plot_shcnn(args.file, args.frame)