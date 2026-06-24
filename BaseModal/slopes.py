import math

import numpy as np


def active_subapertures(image_size=240, subapertures=12, lenslet_size=20):
    active = []
    center = image_size / 2.0
    radius = image_size / 2.0

    for row in range(subapertures):
        for col in range(subapertures):
            cy = row * lenslet_size + lenslet_size / 2.0
            cx = col * lenslet_size + lenslet_size / 2.0
            if math.sqrt((cx - center) ** 2 + (cy - center) ** 2) <= radius:
                active.append((row, col))

    return active


def image_to_slopes(image, subapertures=12, lenslet_size=20):
    """Convert one SHWFS intensity matrix into x/y centroid displacements."""
    img = np.asarray(image, dtype=np.float64)
    active = active_subapertures(img.shape[0], subapertures, lenslet_size)
    local_x = np.arange(lenslet_size, dtype=np.float64)
    local_y = np.arange(lenslet_size, dtype=np.float64)
    xx, yy = np.meshgrid(local_x, local_y)
    center = (lenslet_size - 1) / 2.0

    slopes = np.zeros(len(active) * 2, dtype=np.float64)
    for idx, (row, col) in enumerate(active):
        block = img[
            row * lenslet_size : (row + 1) * lenslet_size,
            col * lenslet_size : (col + 1) * lenslet_size,
        ]
        total = block.sum()
        if total <= 0:
            continue

        centroid_x = float((block * xx).sum() / total)
        centroid_y = float((block * yy).sum() / total)
        slopes[idx] = (centroid_x - center) / lenslet_size
        slopes[idx + len(active)] = (centroid_y - center) / lenslet_size

    return slopes


def batch_to_slopes(images, subapertures=12, lenslet_size=20):
    images = np.asarray(images)
    if images.ndim != 3:
        raise ValueError("Expected images with shape (frames, height, width).")

    first = image_to_slopes(images[0], subapertures, lenslet_size)
    slopes = np.zeros((images.shape[0], first.shape[0]), dtype=np.float64)
    slopes[0] = first
    for idx in range(1, images.shape[0]):
        slopes[idx] = image_to_slopes(images[idx], subapertures, lenslet_size)
    return slopes
