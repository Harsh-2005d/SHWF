import math

import numpy as np


def matlab_round(x):
    return np.floor(x + 0.5)


def nmzern(nz):
    csum = np.cumsum(np.arange(1, nz + 2))
    n = np.sum(csum < nz)
    if n == 0:
        return 0, 0
    if n % 2 == 0:
        m = int(np.fix((nz - csum[n - 1]) / 2.0) * 2)
    else:
        m = int(matlab_round((nz - csum[n - 1]) / 2.0) * 2 - 1)
    return n, m


def pupil(size):
    x = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, x)
    return (np.sqrt(xx**2 + yy**2) <= 1.0).astype(float)


def zernike(mode, size):
    n, m = nmzern(mode)
    x = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, x)
    radius = np.sqrt(xx**2 + yy**2)
    theta = np.arctan2(yy, xx)

    radial = np.zeros_like(radius)
    s = 0
    while s <= (n - abs(m)) / 2:
        numerator = (-1) ** s * math.factorial(n - s)
        denominator = (
            math.factorial(s)
            * math.factorial(int((n + abs(m)) / 2 - s))
            * math.factorial(int((n - abs(m)) / 2 - s))
        )
        radial += (numerator / denominator) * radius ** (n - 2 * s)
        s += 1

    if m == 0:
        values = np.sqrt(n + 1) * radial
    elif mode % 2 == 0:
        values = np.sqrt(2 * (n + 1)) * radial * np.cos(abs(m) * theta)
    else:
        values = np.sqrt(2 * (n + 1)) * radial * np.sin(abs(m) * theta)

    return values * pupil(size)


def zernike_basis(n_modes, size):
    basis = np.zeros((n_modes, size, size), dtype=np.float64)
    for idx in range(n_modes):
        basis[idx] = zernike(idx + 2, size)
    return basis


def reconstruct_phase(coefficients, size=240):
    coeffs = np.asarray(coefficients, dtype=np.float64)
    phase = np.zeros((size, size), dtype=np.float64)
    for idx, coeff in enumerate(coeffs):
        phase += zernike(idx + 2, size) * coeff
    return phase


def phase_metrics(phase):
    mask = phase != 0
    values = phase[mask] if np.any(mask) else np.asarray([0.0])
    return {
        "pv": float(np.max(values) - np.min(values)),
        "rms": float(np.std(values)),
    }
