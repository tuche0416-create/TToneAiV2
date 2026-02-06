"""Extract average RGB from tooth regions."""
import numpy as np
import cv2
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


def extract_average_rgb(
    image: np.ndarray,
    tooth_mask: np.ndarray,
    detected_teeth: list,
    total_pixels: int
) -> Tuple[Optional[Tuple[int, int, int]], Dict]:
    """Extract average RGB from masked tooth regions.

    Uses original (pre-CLAHE) image for color accuracy.

    Args:
        image: Original RGB image (H, W, 3), uint8
        tooth_mask: Binary mask (448, 448) of tooth regions
        detected_teeth: List of detected tooth numbers (FDI)
        total_pixels: Total pixel count in mask

    Returns:
        Tuple of:
        - (r_mean, g_mean, b_mean) as integers 0-255, or None if no valid pixels
        - Metadata dict (detectedTeethCount, centralIncisorsMaskPixels, toothNumbers, confidenceScore)
    """
    if tooth_mask is None or total_pixels == 0:
        metadata = {
            "detectedTeethCount": 0,
            "centralIncisorsMaskPixels": 0,
            "toothNumbers": [],
            "confidenceScore": 0.0
        }
        return None, metadata

    # Resize mask to match image dimensions if needed
    if tooth_mask.shape[:2] != image.shape[:2]:
        tooth_mask = cv2.resize(
            tooth_mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    # Extract masked pixels
    masked_pixels = image[tooth_mask > 0]

    if len(masked_pixels) == 0:
        metadata = {
            "detectedTeethCount": len(detected_teeth),
            "centralIncisorsMaskPixels": total_pixels,
            "toothNumbers": detected_teeth,
            "confidenceScore": 0.0
        }
        return None, metadata

    # Calculate mean RGB
    r_mean = int(np.mean(masked_pixels[:, 0]))
    g_mean = int(np.mean(masked_pixels[:, 1]))
    b_mean = int(np.mean(masked_pixels[:, 2]))

    # Calculate confidence score based on pixel count
    # More pixels = higher confidence (up to 1.0)
    confidence = min(1.0, total_pixels / 5000.0)

    metadata = {
        "detectedTeethCount": len(detected_teeth),
        "centralIncisorsMaskPixels": total_pixels,
        "toothNumbers": detected_teeth,
        "confidenceScore": round(confidence, 3)
    }

    logger.info(f"Extracted RGB: ({r_mean}, {g_mean}, {b_mean}), confidence: {confidence:.3f}")

    return (r_mean, g_mean, b_mean), metadata
