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
