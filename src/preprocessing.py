import cv2
import numpy as np


def extract_green_channel(image):
    return image[:, :, 1]


def normalize_image(image):
    image = image.astype(np.float32)
    min_val = image.min()
    max_val = image.max()
    if max_val - min_val == 0:
        return np.zeros_like(image, dtype=np.float32)
    return (image - min_val) / (max_val - min_val)


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    image_uint8 = (image * 255).astype(np.uint8) if image.dtype != np.uint8 else image
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image_uint8)


def to_binary_mask(mask, threshold=127):
    return (mask > threshold).astype(np.uint8) * 255


def resize_image(image, size=(512, 512), interpolation=cv2.INTER_LINEAR):
    return cv2.resize(image, size, interpolation=interpolation)
