import cv2
import numpy as np


def apply_otsu_threshold(image: np.ndarray) -> np.ndarray:
    """Apply Otsu global binarization to a grayscale image."""
    _, result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return result


def apply_adaptive_threshold(image: np.ndarray, block_size: int = 31, constant: int = 10) -> np.ndarray:
    """Apply adaptive Gaussian thresholding to a grayscale image."""
    return cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,block_size,constant)

adaptive = apply_adaptive_threshold
otsu = apply_otsu_threshold

