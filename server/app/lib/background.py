import cv2
import numpy as np

def remove_background(image: np.ndarray, kernel_size: int = 31) -> np.ndarray:
    """Estimate and remove background illumination/shadow variations using morphological closing."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    background = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return cv2.divide(image, background, scale=255)