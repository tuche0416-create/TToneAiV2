"""Mask refinement and tooth region extraction."""
import cv2
import numpy as np
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)

# Central incisors: #11 (class 1), #21 (class 9)
CENTRAL_INCISOR_CLASSES = [1, 9]
# Lateral incisors: #12 (class 2), #22 (class 10)
LATERAL_INCISOR_CLASSES = [2, 10]
# Canines: #13 (class 3), #23 (class 11)
CANINE_CLASSES = [3, 11]
# All front teeth for final fallback (incisors + canines)
ALL_FRONT_TEETH_CLASSES = [1, 2, 3, 9, 10, 11]
# Lowered threshold for better detection of smaller teeth
MIN_TOOTH_PIXELS = 50


def refine_mask(mask: np.ndarray) -> np.ndarray:
    """Refine segmentation mask with morphological operations.

    Steps:
    1. Morphological opening (remove noise)
    2. Morphological closing (fill gaps)
    3. Keep largest connected component per class
    4. Soft erosion (3px) to avoid boundary contamination

    Args:
        mask: Raw segmentation mask (448, 448) with class indices

    Returns:
        Refined mask (448, 448)
    """
    refined = np.zeros_like(mask)
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Process each class separately
    unique_classes = np.unique(mask)
    for class_id in unique_classes:
        if class_id == 0:
            continue

        # Binary mask for this class
        binary = (mask == class_id).astype(np.uint8)

        # Morphological opening
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_morph)

        # Morphological closing
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_morph)

        # Keep largest connected component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels > 1:
            # Skip background (label 0), find largest component
            largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            binary = (labels == largest_idx).astype(np.uint8)

        # Soft erosion to avoid boundary contamination
        binary = cv2.erode(binary, kernel_erode, iterations=1)

        # Add to refined mask
        refined[binary > 0] = class_id

    return refined


def _class_to_fdi(class_id: int) -> int:
    """Convert model class index to FDI tooth number.

    Class 1-8 → tooth 11-18 (Q1), class 9-16 → tooth 21-28 (Q2),
    class 17-24 → tooth 31-38 (Q3), class 25-32 → tooth 41-48 (Q4).
    """
    if class_id <= 8:
        return 10 + class_id
    elif class_id <= 16:
        return 12 + class_id
    elif class_id <= 24:
        return 14 + class_id
    else:
        return 16 + class_id


def extract_tooth_regions(mask: np.ndarray) -> Tuple[Optional[np.ndarray], List[int], int]:
    """Extract priority tooth regions with multiple fallback strategies.

    Priority order:
    1. Central incisors: #11 (class 1), #21 (class 9)
    2. Lateral incisors: #12 (class 2), #22 (class 10)
    3. Canines: #13 (class 3), #23 (class 11)
    4. Any visible front teeth combined

    Args:
        mask: Refined segmentation mask (448, 448)

    Returns:
        Tuple of:
        - Binary mask of selected tooth regions (448, 448), or None if no valid teeth
        - List of detected tooth numbers (FDI notation)
        - Total pixel count
    """
    def try_classes(class_list: List[int], label: str) -> Tuple[np.ndarray, List[int], int]:
        """Try to extract teeth from given class list."""
        tooth_mask = np.zeros_like(mask, dtype=np.uint8)
        detected_teeth = []
        total_pixels = 0

        for class_id in class_list:
            region = (mask == class_id)
            pixel_count = np.sum(region)

            if pixel_count >= MIN_TOOTH_PIXELS:
                tooth_mask[region] = 1
                tooth_number = _class_to_fdi(class_id)
                detected_teeth.append(tooth_number)
                total_pixels += pixel_count

        if total_pixels >= MIN_TOOTH_PIXELS:
            logger.info(f"Detected {label}: {detected_teeth}, {total_pixels} pixels")
        return tooth_mask, detected_teeth, total_pixels

    # Strategy 1: Central incisors (best for accurate WID measurement)
    tooth_mask, detected_teeth, total_pixels = try_classes(
        CENTRAL_INCISOR_CLASSES, "central incisors"
    )
    if total_pixels >= MIN_TOOTH_PIXELS:
        return tooth_mask, detected_teeth, total_pixels

    # Strategy 2: Lateral incisors
    logger.info("Insufficient central incisors, trying lateral incisors")
    tooth_mask, detected_teeth, total_pixels = try_classes(
        LATERAL_INCISOR_CLASSES, "lateral incisors"
    )
    if total_pixels >= MIN_TOOTH_PIXELS:
        return tooth_mask, detected_teeth, total_pixels

    # Strategy 3: Canines
    logger.info("Insufficient lateral incisors, trying canines")
    tooth_mask, detected_teeth, total_pixels = try_classes(
        CANINE_CLASSES, "canines"
    )
    if total_pixels >= MIN_TOOTH_PIXELS:
        return tooth_mask, detected_teeth, total_pixels

    # Strategy 4: Combine all front teeth
    logger.info("Insufficient single class, trying all front teeth combined")
    tooth_mask, detected_teeth, total_pixels = try_classes(
        ALL_FRONT_TEETH_CLASSES, "all front teeth"
    )
    if total_pixels >= MIN_TOOTH_PIXELS:
        return tooth_mask, detected_teeth, total_pixels

    logger.warning("No valid tooth regions found after all strategies")
    return None, [], 0

