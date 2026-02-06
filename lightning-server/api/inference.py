"""AI model loading and inference with TTA."""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple
import logging

from .graycard import detect_and_correct_graycard

logger = logging.getLogger(__name__)

# Global model instance
_model: Optional[torch.nn.Module] = None
_device: Optional[torch.device] = None

# ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# TTA tooth class remapping (FDI numbering)
# Q1 (upper right, teeth 11-18) ↔ Q2 (upper left, teeth 21-28)
# Q3 (lower left, teeth 31-38) ↔ Q4 (lower right, teeth 41-48)
TTA_FLIP_MAP = {
    # Q1 ↔ Q2
    1: 9, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14, 7: 15, 8: 16,
    9: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 6, 15: 7, 16: 8,
    # Q3 ↔ Q4
    17: 25, 18: 26, 19: 27, 20: 28, 21: 29, 22: 30, 23: 31, 24: 32,
    25: 17, 26: 18, 27: 19, 28: 20, 29: 21, 30: 22, 31: 23, 32: 24,
    # Background class 0 unchanged
    0: 0
}


def load_model() -> bool:
    """Load FPN model with EfficientNet-B7 encoder.

    Returns:
        True if model loaded successfully, False otherwise
    """
    global _model, _device

    try:
        import segmentation_models_pytorch as smp

        model_path = Path(__file__).parent.parent / "model" / "tooth_number_saved_weight.pt"

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {_device}")

        if not model_path.exists():
            logger.warning(f"Model file not found at {model_path}. Will use mock inference.")
            _model = None
            return False

        # Create model architecture
        model = smp.FPN(
            encoder_name="efficientnet-b7",
            classes=33,
            activation=None
        )

        # Load weights
        state_dict = torch.load(model_path, map_location=_device)
        model.load_state_dict(state_dict)
        model = model.to(_device)
        model.eval()

        _model = model
        logger.info("Model loaded successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        _model = None
        return False


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE preprocessing to enhance local contrast.

    Args:
        image: RGB image (H, W, 3)

    Returns:
        CLAHE-processed RGB image
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def preprocess_image(image: np.ndarray, apply_graycard: bool = True) -> Tuple[torch.Tensor, bool]:
    """Preprocess image for model inference.

    Steps:
    1. Gray card white balance correction (if detected)
    2. Apply CLAHE
    3. Resize to 448x448
    4. Normalize with ImageNet statistics
    5. Convert to tensor

    Args:
        image: RGB image (H, W, 3), uint8
        apply_graycard: Whether to attempt gray card correction

    Returns:
        Tuple of (preprocessed tensor, graycard_detected flag)
    """
    graycard_detected = False
    
    # Step 1: Gray card white balance correction
    if apply_graycard:
        image, graycard_detected = detect_and_correct_graycard(image)
    
    # Step 2: Apply CLAHE
    image = apply_clahe(image)

    # Resize to 448x448
    image = cv2.resize(image, (448, 448), interpolation=cv2.INTER_LINEAR)

    # Convert to float [0, 1]
    image = image.astype(np.float32) / 255.0

    # Normalize with ImageNet statistics
    for i in range(3):
        image[:, :, i] = (image[:, :, i] - IMAGENET_MEAN[i]) / IMAGENET_STD[i]

    # Convert to tensor (C, H, W)
    tensor = torch.from_numpy(image).permute(2, 0, 1)

    # Add batch dimension
    return tensor.unsqueeze(0), graycard_detected


def remap_classes_tta(mask: np.ndarray) -> np.ndarray:
    """Remap tooth classes for horizontal flip TTA.

    Args:
        mask: Segmentation mask (H, W) with class indices

    Returns:
        Remapped mask with flipped tooth classes
    """
    remapped = np.zeros_like(mask)
    for orig_class, new_class in TTA_FLIP_MAP.items():
        remapped[mask == orig_class] = new_class
    return remapped


def run_inference(image: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Run model inference with test-time augmentation.

    TTA strategy: horizontal flip with tooth class remapping

    Args:
        image: RGB image (H, W, 3), uint8

    Returns:
        Tuple of (segmentation mask (448, 448), graycard_detected flag)
    """
    global _model, _device

    if _model is None:
        logger.warning("Model not loaded. Generating mock mask.")
        # Mock mask for testing: random classes
        return np.random.randint(0, 33, (448, 448), dtype=np.uint8), False

    with torch.no_grad():
        # Original image (with gray card correction)
        tensor, graycard_detected = preprocess_image(image)
        tensor = tensor.to(_device)
        logits = _model(tensor)

        # Flipped image (reuse same corrected image, no need to re-detect graycard)
        tensor_flipped = torch.flip(tensor, dims=[3])
        logits_flipped = _model(tensor_flipped)
        # Flip back spatially
        logits_flipped = torch.flip(logits_flipped, dims=[3])

        # Remap class channels for flipped prediction
        # When image is flipped, Q1↔Q2 and Q3↔Q4 tooth classes swap
        remapped_logits = torch.zeros_like(logits_flipped)
        for orig_class, new_class in TTA_FLIP_MAP.items():
            remapped_logits[:, new_class] = logits_flipped[:, orig_class]

        # Average original + remapped-flipped logits
        logits_avg = (logits + remapped_logits) / 2.0

        # Get class predictions
        mask = torch.argmax(logits_avg, dim=1).squeeze(0).cpu().numpy()

        return mask.astype(np.uint8), graycard_detected

