import cv2
import numpy as np

def remove_borders(image: np.ndarray, pixels: int = 10) -> np.ndarray:
    """Mask outer border pixels with white to remove boundary scan artifacts."""
    result = image.copy()

    result[:pixels, :] = 255
    result[-pixels:, :] = 255
    result[:, :pixels] = 255
    result[:, -pixels:] = 255

    return result