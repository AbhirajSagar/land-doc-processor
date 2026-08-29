import cv2
import numpy as np


def deskew_image(image: np.ndarray) -> np.ndarray:
    """Estimate skew angle and deskew (straighten) the document image."""
    points = np.column_stack(np.where(image < 128))

    if len(points) < 10:
        return image

    angle = cv2.minAreaRect(points)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    return cv2.warpAffine(image,matrix,(width, height),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)