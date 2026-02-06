"""Pydantic models for request/response schemas."""
from pydantic import BaseModel, Field
from typing import Optional, List, Literal


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    uptime_seconds: float


class AnalyzeResponse(BaseModel):
    """Immediate response after submitting analysis job."""
    job_id: str


class LabValues(BaseModel):
    """CIELAB color space values."""
    l: float = Field(..., description="L* (lightness) component")
    a: float = Field(..., description="a* (green-red) component")
    b: float = Field(..., description="b* (yellow-blue) component")


class Visualization(BaseModel):
    """Tooth region visualization data."""
    image: str = Field(..., description="Base64 data URI string (data:image/png;base64,...)")


class AiMetadata(BaseModel):
    """AI model inference metadata."""
    detectedTeethCount: int
    processingTimeMs: int
    centralIncisorsMaskPixels: int
    toothNumbers: List[int]
    confidenceScore: float
    graycardDetected: bool = Field(
        default=False,
        description="Whether 18% gray card was detected and white balance applied"
    )


class AnalysisResult(BaseModel):
    """Complete analysis result."""
    wid: float
    percentile: float
    estimatedAge: int
    labValues: LabValues
    visualization: Visualization
    aiMetadata: AiMetadata
    qualityWarnings: List[str] = Field(default_factory=list)


class JobStatus(BaseModel):
    """Job status and result."""
    status: Literal["processing", "completed", "failed"]
    progress: Optional[str] = None
    result: Optional[AnalysisResult] = None
    error: Optional[str] = None
    message: Optional[str] = None
