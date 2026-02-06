"""Gray card detection and white balance correction.

Detects 18% gray card in image and applies color correction for accurate tooth analysis.
"""
import cv2
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# 18% gray card target values
# In ideal D65 lighting, 18% gray ≈ RGB(119, 119, 119) or L* ≈ 50
TARGET_GRAY_RGB = 119
TARGET_GRAY_L = 50

# Detection parameters
MIN_GRAY_AREA_RATIO = 0.05  # Minimum 5% of image area
MAX_SATURATION = 30  # HSV saturation threshold for "gray"
MIN_VALUE = 40  # Minimum brightness to avoid shadows
MAX_VALUE = 220  # Maximum brightness to avoid highlights


def detect_graycard_region(image: np.ndarray) -> Optional[np.ndarray]:
    """Detect 18% gray card region in image.
    
    Args:
        image: RGB image (H, W, 3), uint8
        
    Returns:
        Binary mask of gray card region, or None if not detected
    """
    h, w = image.shape[:2]
    total_pixels = h * w
    min_pixels = int(total_pixels * MIN_GRAY_AREA_RATIO)
    
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Find low-saturation (gray) pixels with reasonable brightness
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    
    gray_mask = (saturation < MAX_SATURATION) & (value > MIN_VALUE) & (value < MAX_VALUE)
    gray_mask = gray_mask.astype(np.uint8) * 255
    
    # Morphological operations to clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_OPEN, kernel)
    gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gray_mask, connectivity=8)
    
    # Find largest gray region (excluding background label 0)
    best_region_mask = None
    best_area = 0
    
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        
        if area >= min_pixels and area > best_area:
            # Create mask for this region
            region_mask = (labels == label_id).astype(np.uint8) * 255
            
            # Verify it looks like a gray card (consistent color)
            region_pixels = image[labels == label_id]
            std_r = np.std(region_pixels[:, 0])
            std_g = np.std(region_pixels[:, 1])
            std_b = np.std(region_pixels[:, 2])
            
            # Gray card should have low color variance
            if std_r < 30 and std_g < 30 and std_b < 30:
                best_region_mask = region_mask
                best_area = area
    
    if best_region_mask is not None:
        logger.info(f"Gray card detected: {best_area} pixels ({best_area/total_pixels*100:.1f}% of image)")
        return best_region_mask
    
    logger.warning("No gray card region detected in image")
    return None


def compute_correction_factors(image: np.ndarray, graycard_mask: np.ndarray) -> Tuple[float, float, float]:
    """Compute RGB correction factors from gray card region.
    
    Args:
        image: RGB image (H, W, 3), uint8
        graycard_mask: Binary mask of gray card region
        
    Returns:
        Tuple of (r_factor, g_factor, b_factor) correction multipliers
    """
    # Extract gray card pixels
    mask_bool = graycard_mask > 0
    graycard_pixels = image[mask_bool]
    
    # Calculate mean RGB
    mean_r = np.mean(graycard_pixels[:, 0])
    mean_g = np.mean(graycard_pixels[:, 1])
    mean_b = np.mean(graycard_pixels[:, 2])
    
    logger.info(f"Gray card mean RGB: ({mean_r:.1f}, {mean_g:.1f}, {mean_b:.1f})")
    
    # Avoid division by zero
    mean_r = max(mean_r, 1.0)
    mean_g = max(mean_g, 1.0)
    mean_b = max(mean_b, 1.0)
    
    # Compute correction factors to achieve target gray (119, 119, 119)
    r_factor = TARGET_GRAY_RGB / mean_r
    g_factor = TARGET_GRAY_RGB / mean_g
    b_factor = TARGET_GRAY_RGB / mean_b
    
    logger.info(f"Correction factors: R={r_factor:.3f}, G={g_factor:.3f}, B={b_factor:.3f}")
    
    return r_factor, g_factor, b_factor


def apply_white_balance(image: np.ndarray, r_factor: float, g_factor: float, b_factor: float) -> np.ndarray:
    """Apply white balance correction to image.
    
    Args:
        image: RGB image (H, W, 3), uint8
        r_factor, g_factor, b_factor: Correction multipliers
        
    Returns:
        White-balanced RGB image
    """
    # Convert to float for calculation
    corrected = image.astype(np.float32)
    
    # Apply correction factors
    corrected[:, :, 0] = corrected[:, :, 0] * r_factor
    corrected[:, :, 1] = corrected[:, :, 1] * g_factor
    corrected[:, :, 2] = corrected[:, :, 2] * b_factor
    
    # Clamp to valid range and convert back to uint8
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    
    return corrected


def detect_and_correct_graycard(image: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Main function: Detect gray card and apply white balance correction.
    
    Args:
        image: RGB image (H, W, 3), uint8
        
    Returns:
        Tuple of (corrected_image, graycard_detected)
        If gray card not detected, returns original image with False flag
    """
    # Detect gray card region
    graycard_mask = detect_graycard_region(image)
    
    if graycard_mask is None:
        logger.warning("Proceeding without gray card correction")
        return image, False
    
    # Compute correction factors
    r_factor, g_factor, b_factor = compute_correction_factors(image, graycard_mask)
    
    # Apply white balance correction
    corrected_image = apply_white_balance(image, r_factor, g_factor, b_factor)
    
    # Verify correction worked
    corrected_graycard = corrected_image[graycard_mask > 0]
    mean_after = np.mean(corrected_graycard, axis=0)
    logger.info(f"Gray card after correction: RGB({mean_after[0]:.1f}, {mean_after[1]:.1f}, {mean_after[2]:.1f})")
    
    return corrected_image, True
