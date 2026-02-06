"""Generate tooth region visualization with green overlay."""
import cv2
import numpy as np
import base64
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Green overlay color (#86EFAC in RGB)
OVERLAY_COLOR = (134, 239, 172)
OVERLAY_ALPHA = 0.4

# Graycard overlay color (#EF4444 in RGB, Red)
GRAYCARD_COLOR = (239, 68, 68)
GRAYCARD_ALPHA = 0.3


def generate_visualization(
    image: np.ndarray,
    tooth_mask: np.ndarray,
    graycard_mask: Optional[np.ndarray] = None
) -> Optional[str]:
    """Generate visualization with semi-transparent green overlay on tooth regions.

    Args:
        image: Original RGB image (H, W, 3), uint8
        tooth_mask: Binary mask (448, 448) of tooth regions
        graycard_mask: Optional binary mask of gray card region

    Returns:
        Base64 data URI string (data:image/png;base64,...), or None on error
    """
    if tooth_mask is None:
        logger.warning("No tooth mask provided for visualization")
        return None

    try:
        # Resize mask to match image dimensions if needed
        if tooth_mask.shape[:2] != image.shape[:2]:
            tooth_mask = cv2.resize(
                tooth_mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        # Create overlay image
        overlay = image.copy()

        # Apply green color to masked regions
        overlay[tooth_mask > 0] = (
            image[tooth_mask > 0] * (1 - OVERLAY_ALPHA) +
            np.array(OVERLAY_COLOR) * OVERLAY_ALPHA
        ).astype(np.uint8)

        # Apply red overlay to graycard (if detected)
        if graycard_mask is not None:
            if graycard_mask.shape[:2] != image.shape[:2]:
                graycard_mask = cv2.resize(
                    graycard_mask,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

            overlay[graycard_mask > 0] = (
                image[graycard_mask > 0] * (1 - GRAYCARD_ALPHA) +
                np.array(GRAYCARD_COLOR) * GRAYCARD_ALPHA
            ).astype(np.uint8)

        # Encode as PNG
        success, buffer = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        if not success:
            logger.error("Failed to encode visualization as PNG")
            return None

        # Convert to base64
        png_base64 = base64.b64encode(buffer).decode('utf-8')
        data_uri = f"data:image/png;base64,{png_base64}"

        logger.info(f"Generated visualization: {len(data_uri)} bytes")
        return data_uri

    except Exception as e:
        logger.error(f"Error generating visualization: {e}")
        return None
