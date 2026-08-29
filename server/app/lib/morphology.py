import cv2
import numpy as np

def morphological_opening(image: np.ndarray, kernel_size: int = 2) -> np.ndarray:
    """Apply morphological opening (erosion followed by dilation) to remove small noise."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def morphological_closing(image: np.ndarray, kernel_size: int = 2) -> np.ndarray:
    """Apply morphological closing (dilation followed by erosion) to fill small holes."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)