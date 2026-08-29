from pathlib import Path
from typing import Union
import cv2
import numpy as np

from app.lib.artifacts import remove_borders
from app.lib.contrast import enhance_contrast
from app.lib.denoising import median_denoise
from app.lib.deskewing import deskew_image
from app.lib.grayscale import to_grayscale
from app.lib.morphology import morphological_opening
from app.lib.thresholding import apply_adaptive_threshold
from app.lib.upscaling import upscale_image

def preprocess(image: np.ndarray, output_dir: Union[str, Path] = "output") -> np.ndarray:

    """
    Execute the document image preprocessing pipeline:
    1. Upscale image resolution
    2. Convert to grayscale
    3. Denoise using median filtering
    4. Enhance contrast using CLAHE
    5. Binarize using adaptive Gaussian thresholding
    6. Correct skew angle (deskew)
    7. Clean small artifacts using morphological opening
    8. Remove border scan artifacts
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    upscaled_image = upscale_image(image)
    cv2.imwrite(str(output_path / "01_upscaled.png"), upscaled_image)

    grayscale_image = to_grayscale(upscaled_image)
    cv2.imwrite(str(output_path / "02_grayscale.png"), grayscale_image)

    denoised_image = median_denoise(grayscale_image)
    cv2.imwrite(str(output_path / "03_denoised.png"), denoised_image)

    contrast_enhanced_image = enhance_contrast(denoised_image)
    cv2.imwrite(str(output_path / "04_contrast_enhanced.png"), contrast_enhanced_image)

    thresholded_image = apply_adaptive_threshold(contrast_enhanced_image)
    cv2.imwrite(str(output_path / "05_thresholded.png"), thresholded_image)

    deskewed_image = deskew_image(thresholded_image)
    cv2.imwrite(str(output_path / "06_deskewed.png"), deskewed_image)

    morphology_cleaned_image = morphological_opening(deskewed_image)
    cv2.imwrite(str(output_path / "07_morphology_cleaned.png"), morphology_cleaned_image)

    border_removed_image = remove_borders(morphology_cleaned_image)
    cv2.imwrite(str(output_path / "08_border_removed.png"), border_removed_image)

    return border_removed_image
