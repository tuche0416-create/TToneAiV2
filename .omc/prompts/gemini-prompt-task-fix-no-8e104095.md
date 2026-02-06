---
provider: "gemini"
agent_role: "writer"
model: "gemini-3-pro-preview"
files:
  - "/Users/tanpapa/Desktop/develop-b/TToneAiV2/lightning-server/api/main.py"
  - "/Users/tanpapa/Desktop/develop-b/TToneAiV2/lightning-server/api/postprocess.py"
  - "/Users/tanpapa/Desktop/develop-b/TToneAiV2/lightning-server/api/rgb_extraction.py"
  - "/Users/tanpapa/Desktop/develop-b/TToneAiV2/lightning-server/api/visualization.py"
  - "/Users/tanpapa/Desktop/develop-b/TToneAiV2/lightning-server/api/inference.py"
  - "/Users/tanpapa/Desktop/develop-b/TToneAiV2/lib/types.ts"
timestamp: "2026-02-06T19:45:46.746Z"
---

--- File: /Users/tanpapa/Desktop/develop-b/TToneAiV2/lightning-server/api/main.py ---
"""FastAPI application for T-Tone AI V2 Lightning.ai server."""
import asyncio
import base64
import io
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .color_science import compute_wid, rgb_to_lab
from .inference import load_model, run_inference
from .postprocess import extract_tooth_regions, refine_mask
from .rgb_extraction import extract_average_rgb
from .schemas import (
    AiMetadata,
    AnalysisResult,
    AnalyzeResponse,
    HealthResponse,
    JobStatus,
    LabValues,
    Visualization,
)
from .statistics import compute_percentile, estimate_tooth_age, lookup_stats
from .visualization import generate_visualization

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Application state
app = FastAPI(
    title="T-Tone AI V2",
    description="Dental whiteness diagnosis API",
    version="2.0.0"
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS - allow_origin_regex for Vercel wildcard subdomains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
_jobs: Dict[str, JobStatus] = {}
_executor: Optional[ThreadPoolExecutor] = None
_server_start_time: Optional[float] = None
_model_loaded: bool = False

# Image validation
ALLOWED_MAGIC_BYTES = {
    b'\xff\xd8\xff': 'jpeg',
    b'\x89PNG\r\n\x1a\n': 'png',
    b'RIFF': 'webp',
}
MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024  # 5MB

# Quality thresholds
MIN_LAPLACIAN_VARIANCE = 100  # Blur detection
MIN_BRIGHTNESS = 30
MAX_BRIGHTNESS = 225


def validate_image_bytes(content: bytes) -> Optional[str]:
    """Validate image format using magic bytes.

    Returns:
        Format name if valid, None otherwise
    """
    for magic, fmt in ALLOWED_MAGIC_BYTES.items():
        if content.startswith(magic):
            return fmt
    # WebP has RIFF at start, need to check deeper
    if content.startswith(b'RIFF') and b'WEBP' in content[:12]:
        return 'webp'
    return None


def check_image_quality(image_array: np.ndarray) -> list[str]:
    """Check image quality for common issues.

    Returns:
        List of warning messages
    """
    warnings = []

    # Convert to grayscale for blur detection
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # Laplacian variance for blur detection
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < MIN_LAPLACIAN_VARIANCE:
        warnings.append(f"Image may be blurry (variance: {laplacian_var:.1f})")

    # Brightness check
    mean_brightness = np.mean(gray)
    if mean_brightness < MIN_BRIGHTNESS:
        warnings.append(f"Image is too dark (brightness: {mean_brightness:.1f})")
    elif mean_brightness > MAX_BRIGHTNESS:
        warnings.append(f"Image is too bright (brightness: {mean_brightness:.1f})")

    return warnings


async def process_job(
    job_id: str,
    image_bytes: bytes,
    gender: str,
    age: int,
    mouth_info: Optional[str]
):
    """Background job processing function."""
    try:
        # Update progress
        _jobs[job_id].progress = "preprocessing"
        _jobs[job_id].message = "Decoding image and checking quality"

        # Decode image
        image = Image.open(io.BytesIO(image_bytes))

        # Fix EXIF orientation
        image = ImageOps.exif_transpose(image)

        # Convert to RGB numpy array
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_array = np.array(image)

        # Check quality
        quality_warnings = check_image_quality(image_array)

        # Update progress
        _jobs[job_id].progress = "inference"
        _jobs[job_id].message = "Running AI model inference with TTA"

        # Run inference
        start_inference = time.time()
        mask = await asyncio.get_event_loop().run_in_executor(
            _executor,
            run_inference,
            image_array
        )
        inference_time_ms = int((time.time() - start_inference) * 1000)

        # Log raw mask statistics
        unique_classes, counts = np.unique(mask, return_counts=True)
        logger.info(f"Raw mask classes: {dict(zip(unique_classes, counts))}")

        # Update progress
        _jobs[job_id].progress = "postprocessing"
        _jobs[job_id].message = "Refining mask and extracting tooth regions"

        # Refine mask
        refined_mask = refine_mask(mask)

        # Log refined mask statistics
        unique_classes_refined, counts_refined = np.unique(refined_mask, return_counts=True)
        logger.info(f"Refined mask classes: {dict(zip(unique_classes_refined, counts_refined))}")

        # Extract tooth regions
        tooth_mask, detected_teeth, total_pixels = extract_tooth_regions(refined_mask)

        if tooth_mask is None or total_pixels == 0:
            raise ValueError("No valid tooth regions detected in image")

        # Extract RGB
        rgb_result, ai_metadata_dict = extract_average_rgb(
            image_array,
            tooth_mask,
            detected_teeth,
            total_pixels
        )

        if rgb_result is None:
            raise ValueError("Failed to extract RGB values from tooth regions")

        r_mean, g_mean, b_mean = rgb_result

        # Generate visualization
        viz_data_uri = generate_visualization(image_array, tooth_mask)
        if viz_data_uri is None:
            raise ValueError("Failed to generate visualization")

        # Update progress
        _jobs[job_id].progress = "statistics"
        _jobs[job_id].message = "Computing WID and percentile"

        # RGB -> Lab -> WID
        lab = rgb_to_lab(r_mean, g_mean, b_mean)
        wid = compute_wid(lab["l"], lab["a"], lab["b"])

        # Lookup stats and compute percentile
        stats = lookup_stats(gender, age)
        percentile = compute_percentile(wid, stats["wid_mean"], stats["wid_sd"])

        # Estimate tooth age
        tooth_age = estimate_tooth_age(age, percentile)

        # Build result
        result = AnalysisResult(
            wid=round(wid, 2),
            percentile=round(percentile, 1),
            estimatedAge=tooth_age,
            labValues=LabValues(
                l=round(lab["l"], 2),
                a=round(lab["a"], 2),
                b=round(lab["b"], 2)
            ),
            visualization=Visualization(image=viz_data_uri),
            aiMetadata=AiMetadata(
                detectedTeethCount=ai_metadata_dict["detectedTeethCount"],
                processingTimeMs=inference_time_ms,
                centralIncisorsMaskPixels=ai_metadata_dict["centralIncisorsMaskPixels"],
                toothNumbers=ai_metadata_dict["toothNumbers"],
                confidenceScore=ai_metadata_dict["confidenceScore"]
            ),
            qualityWarnings=quality_warnings
        )

        # Mark completed
        _jobs[job_id].status = "completed"
        _jobs[job_id].result = result
        _jobs[job_id].progress = None
        _jobs[job_id].message = "Analysis completed successfully"

        logger.info(f"Job {job_id} completed: WID={wid:.2f}, percentile={percentile:.1f}")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        _jobs[job_id].status = "failed"
        _jobs[job_id].error = str(e)
        _jobs[job_id].message = f"Analysis failed: {str(e)}"


async def cleanup_old_jobs():
    """Periodic task to clean up old jobs (>10 minutes)."""
    while True:
        await asyncio.sleep(60)
        cutoff_time = datetime.now() - timedelta(minutes=10)

        jobs_to_remove = []
        for job_id, job in _jobs.items():
            if job.status in ["completed", "failed"]:
                jobs_to_remove.append(job_id)

        for job_id in jobs_to_remove:
            del _jobs[job_id]

        if jobs_to_remove:
            logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")


@app.on_event("startup")
async def startup_event():
    """Initialize server on startup."""
    global _executor, _server_start_time, _model_loaded

    _server_start_time = time.time()
    _executor = ThreadPoolExecutor(max_workers=2)
    _model_loaded = load_model()

    # Start cleanup task
    asyncio.create_task(cleanup_old_jobs())

    logger.info("Server started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global _executor

    if _executor:
        _executor.shutdown(wait=True)

    logger.info("Server shutdown complete")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - _server_start_time if _server_start_time else 0

    return HealthResponse(
        status="healthy",
        model_loaded=_model_loaded,
        uptime_seconds=round(uptime, 1)
    )


@app.post("/analyze", response_model=AnalyzeResponse)
@limiter.limit("10/minute")
async def analyze_image(
    request: Request,
    image: UploadFile = File(...),
    gender: str = Form(...),
    age: int = Form(...),
    mouth_info: Optional[str] = Form(None)
):
    """Submit image for analysis.

    Returns job_id immediately. Check /status/{job_id} for results.
    """
    # Validate gender
    if gender not in ["male", "female"]:
        raise HTTPException(status_code=400, detail="Gender must be 'male' or 'female'")

    # Validate age
    if not (5 <= age <= 95):
        raise HTTPException(status_code=400, detail="Age must be between 5 and 95")

    # Read image
    image_bytes = await image.read()

    # Validate size
    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Image size exceeds {MAX_IMAGE_SIZE_BYTES / 1024 / 1024:.1f}MB limit"
        )

    # Validate format
    image_format = validate_image_bytes(image_bytes)
    if image_format is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Only JPEG, PNG, and WebP are supported."
        )

    # Create job
    job_id = str(uuid.uuid4())
    _jobs[job_id] = JobStatus(
        status="processing",
        progress="queued",
        message="Job queued for processing"
    )

    # Start background processing
    asyncio.create_task(process_job(job_id, image_bytes, gender, age, mouth_info))

    logger.info(f"Created job {job_id}: gender={gender}, age={age}")

    return AnalyzeResponse(job_id=job_id)


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get job status and result."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return _jobs[job_id]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


--- File: /Users/tanpapa/Desktop/develop-b/TToneAiV2/lightning-server/api/postprocess.py ---
"""Mask refinement and tooth region extraction."""
import cv2
import numpy as np
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)

# Central incisors: #11 (class 1), #21 (class 9), #12 (class 2), #22 (class 10)
CENTRAL_INCISOR_CLASSES = [1, 9]
LATERAL_INCISOR_CLASSES = [2, 10]
MIN_TOOTH_PIXELS = 100


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
    """Extract priority tooth regions (central incisors first).

    Priority order:
    1. #11 (class 1), #21 (class 9)
    2. If insufficient pixels, try #12 (class 2), #22 (class 10)

    Args:
        mask: Refined segmentation mask (448, 448)

    Returns:
        Tuple of:
        - Binary mask of selected tooth regions (448, 448), or None if no valid teeth
        - List of detected tooth numbers (FDI notation)
        - Total pixel count
    """
    tooth_mask = np.zeros_like(mask, dtype=np.uint8)
    detected_teeth = []
    total_pixels = 0

    # Try central incisors first
    for class_id in CENTRAL_INCISOR_CLASSES:
        region = (mask == class_id)
        pixel_count = np.sum(region)

        if pixel_count >= MIN_TOOTH_PIXELS:
            tooth_mask[region] = 1
            tooth_number = _class_to_fdi(class_id)
            detected_teeth.append(tooth_number)
            total_pixels += pixel_count

    # If insufficient pixels, try lateral incisors
    if total_pixels < MIN_TOOTH_PIXELS:
        logger.info("Insufficient central incisor pixels, trying lateral incisors")
        tooth_mask.fill(0)
        detected_teeth.clear()
        total_pixels = 0

        for class_id in LATERAL_INCISOR_CLASSES:
            region = (mask == class_id)
            pixel_count = np.sum(region)

            if pixel_count >= MIN_TOOTH_PIXELS:
                tooth_mask[region] = 1
                tooth_number = _class_to_fdi(class_id)
                detected_teeth.append(tooth_number)
                total_pixels += pixel_count

    if total_pixels == 0:
        logger.warning("No valid tooth regions found")
        return None, [], 0

    logger.info(f"Extracted {len(detected_teeth)} teeth: {detected_teeth}, {total_pixels} pixels")
    return tooth_mask, detected_teeth, total_pixels


--- File: /Users/tanpapa/Desktop/develop-b/TToneAiV2/lightning-server/api/rgb_extraction.py ---
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


--- File: /Users/tanpapa/Desktop/develop-b/TToneAiV2/lightning-server/api/visualization.py ---
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


def generate_visualization(
    image: np.ndarray,
    tooth_mask: np.ndarray
) -> Optional[str]:
    """Generate visualization with semi-transparent green overlay on tooth regions.

    Args:
        image: Original RGB image (H, W, 3), uint8
        tooth_mask: Binary mask (448, 448) of tooth regions

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


--- File: /Users/tanpapa/Desktop/develop-b/TToneAiV2/lightning-server/api/inference.py ---
"""AI model loading and inference with TTA."""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Optional
import logging

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


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """Preprocess image for model inference.

    Steps:
    1. Apply CLAHE
    2. Resize to 448x448
    3. Normalize with ImageNet statistics
    4. Convert to tensor

    Args:
        image: RGB image (H, W, 3), uint8

    Returns:
        Preprocessed tensor (1, 3, 448, 448)
    """
    # Apply CLAHE
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
    return tensor.unsqueeze(0)


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


def run_inference(image: np.ndarray) -> np.ndarray:
    """Run model inference with test-time augmentation.

    TTA strategy: horizontal flip with tooth class remapping

    Args:
        image: RGB image (H, W, 3), uint8

    Returns:
        Segmentation mask (448, 448), uint8 with class indices
    """
    global _model, _device

    if _model is None:
        logger.warning("Model not loaded. Generating mock mask.")
        # Mock mask for testing: random classes
        return np.random.randint(0, 33, (448, 448), dtype=np.uint8)

    with torch.no_grad():
        # Original image
        tensor = preprocess_image(image).to(_device)
        logits = _model(tensor)

        # Flipped image
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

        return mask.astype(np.uint8)


--- File: /Users/tanpapa/Desktop/develop-b/TToneAiV2/lib/types.ts ---
// User info collected in Step 2
export interface UserInfo {
  gender: 'male' | 'female';
  age: number;
}

// MediaPipe mouth landmarks from Step 3
export interface MouthInfo {
  centerX: number;
  centerY: number;
  width: number;
  height: number;
  upperY: number;
  lowerY: number;
  lipPoints: [number, number][];
}

// CIELab color values
export interface LabValues {
  l: number;
  a: number;
  b: number;
}

// Visualization data
export interface Visualization {
  image: string; // data:image/png;base64,...
}

// AI processing metadata
export interface AiMetadata {
  detectedTeethCount: number;
  processingTimeMs: number;
  centralIncisorsMaskPixels: number;
  toothNumbers: number[];
  confidenceScore: number;
}

// Complete analysis result from Lightning.ai
export interface AnalysisResult {
  wid: number;
  percentile: number;
  estimatedAge: number;
  labValues: LabValues;
  visualization: Visualization;
  aiMetadata: AiMetadata;
  qualityWarnings: string[];
}

// Job status response from Lightning.ai
export interface JobStatusResponse {
  status: 'processing' | 'completed' | 'failed';
  progress?: 'preprocessing' | 'inference' | 'postprocessing' | 'statistics';
  result?: AnalysisResult;
  error?: string;
  message?: string;
}

// Analysis flow state
export type AnalysisPhase =
  | { phase: 'idle' }
  | { phase: 'warming' }
  | { phase: 'submitting' }
  | { phase: 'processing'; progress?: string }
  | { phase: 'completed'; result: AnalysisResult }
  | { phase: 'failed'; error: string; canRetry: boolean };


IMPORTANT: Write your complete response to the file: /Users/tanpapa/Desktop/develop-b/TToneAiV2/.tmp-fix-output.md

# Task: Fix "No valid tooth regions detected in image" error

## Bug Description
The server receives `mouth_info` (MediaPipe Face Mesh mouth landmarks) from the frontend but NEVER uses it. The full image (face + background) goes directly to model inference at 448x448. Teeth are a tiny portion of a full face photo, so the model fails to detect them properly.

## What needs to change

### File: `lightning-server/api/main.py`

1. Add `import json` at the top
2. In `process_job()`, after image decoding and quality check, BEFORE inference:
   - Parse `mouth_info` JSON string if provided
   - Crop image to mouth region with 2.0x padding ratio (to include teeth above/below lips)
   - Store the crop bounding box for later visualization mapping
   - If `mouth_info` is not provided or parsing fails, use full image as fallback

3. For RGB extraction: use the cropped image (not original) since the mask (448x448) corresponds to the cropped region

4. For visualization: map the tooth_mask back to original image coordinates so the green overlay shows on the full original image

## Cropping Logic

```python
def crop_mouth_region(image: np.ndarray, mouth_info: dict, padding: float = 2.0):
    """Crop image to mouth region with padding.

    MouthInfo has: centerX, centerY, width, height (in pixel coordinates of the original image).
    Padding 2.0x means the crop box is 2x the mouth bounding box size, centered on the mouth.
    This ensures teeth (above/below lips) are included.

    Returns: (cropped_image, (x1, y1, x2, y2))
    """
```

## Key Constraints
- `mouth_info` is OPTIONAL - if not provided, use full image (current behavior = graceful fallback)
- The crop bbox must be clamped to image boundaries
- Cropped image must be large enough (at least 50x50 pixels) or fall back to full image
- tooth_mask is always 448x448 (model output size)
- `extract_average_rgb()` and `generate_visualization()` both internally resize tooth_mask to match the image they receive
- For RGB extraction: pass cropped image + tooth_mask (mask resized to crop size internally)
- For visualization: create a full-size mask by mapping tooth_mask into original coordinates, then pass original image + full_mask

## Current process_job flow (lines 128-255):
```
1. Decode image, EXIF transpose, convert to RGB
2. Quality check
3. Run inference -> mask (448x448)
4. Refine mask
5. Extract tooth regions -> tooth_mask (448x448 binary)
6. Extract RGB from original_image + tooth_mask
7. Generate visualization from original_image + tooth_mask
8. Compute Lab, WID, stats, etc.
```

## Required new flow:
```
1. Decode image, EXIF transpose, convert to RGB -> image_array (original)
2. Quality check on original
3. **NEW: Parse mouth_info, crop image -> inference_image, crop_bbox**
4. Run inference on inference_image -> mask (448x448)
5. Refine mask
6. Extract tooth regions -> tooth_mask (448x448 binary)
7. Extract RGB from **inference_image** + tooth_mask  (color accuracy on cropped region)
8. **NEW: Map tooth_mask back to original coordinates -> full_mask**
9. Generate visualization from **image_array** (original) + **full_mask**
10. Compute Lab, WID, stats, etc.
```

## Output Format
Please provide the COMPLETE modified `process_job` function and any new helper functions (like `crop_mouth_region` and `map_mask_to_original`). Show the exact code that should go in main.py.

IMPORTANT: Do NOT modify any color science, WID formula, statistics algorithms, or D65 white point values. Only modify the image preprocessing/cropping pipeline in main.py.
