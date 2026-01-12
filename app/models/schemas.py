"""
Pydantic schemas for MedExplain AI API.

Defines request/response models for all API endpoints.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# Enums
# =============================================================================

class FileType(str, Enum):
    """Supported file types."""
    PDF = "pdf"
    IMAGE = "image"
    TEXT = "text"


class RiskLevel(str, Enum):
    """Risk classification levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class ConfidenceLevel(str, Enum):
    """Confidence levels for analysis."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# =============================================================================
# Session & Upload Models
# =============================================================================

class UploadResponse(BaseModel):
    """Response after successful file upload."""
    
    session_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique session ID for this upload"
    )
    filename: str = Field(description="Original filename")
    file_type: FileType = Field(description="Detected file type")
    file_size_bytes: int = Field(description="File size in bytes")
    message: str = Field(default="File uploaded successfully")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(from_attributes=True)


class GenerateReportRequest(BaseModel):
    """Request to generate an explanation report."""
    
    session_id: str = Field(description="Session ID from upload")
    tenant_id: Optional[str] = Field(
        default=None,
        description="Clinic/tenant identifier"
    )
    patient_name: Optional[str] = Field(
        default=None,
        description="Optional patient name for report"
    )
    additional_context: Optional[str] = Field(
        default=None,
        description="Additional context from healthcare provider"
    )


# =============================================================================
# Analysis Results
# =============================================================================

class SafetyDisclaimer(BaseModel):
    """Safety disclaimer to include in all responses."""
    
    main_disclaimer: str = Field(
        default="This is NOT a medical diagnosis. This information is for educational purposes only.",
        description="Primary disclaimer text"
    )
    consultation_reminder: str = Field(
        default="Please consult your healthcare provider before making any medical decisions.",
        description="Doctor consultation reminder"
    )
    confidence_note: Optional[str] = Field(
        default=None,
        description="Note about analysis confidence"
    )


class MedicalExplanation(BaseModel):
    """Patient-friendly explanation of medical findings."""
    
    summary: str = Field(
        description="Simple, patient-friendly summary"
    )
    what_this_means: str = Field(
        description="What the result generally indicates"
    )
    common_next_steps: List[str] = Field(
        description="Common follow-up steps (non-prescriptive)"
    )
    key_findings: List[str] = Field(
        default=[],
        description="Key findings from the report"
    )


class AnalysisResult(BaseModel):
    """Complete analysis result with safety information."""
    
    session_id: str = Field(description="Session ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Analysis content
    explanation: MedicalExplanation = Field(
        description="Patient-friendly explanation"
    )
    
    # Risk and confidence
    risk_level: RiskLevel = Field(
        description="Informational risk level"
    )
    confidence: ConfidenceLevel = Field(
        description="Analysis confidence level"
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Numeric confidence score"
    )
    
    # Safety
    disclaimer: SafetyDisclaimer = Field(
        default_factory=SafetyDisclaimer,
        description="Safety disclaimers"
    )
    
    # Metadata
    source_type: FileType = Field(description="Type of source file")
    processing_time_ms: Optional[int] = Field(
        default=None,
        description="Processing time in milliseconds"
    )
    
    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Report Generation
# =============================================================================

class PatientReport(BaseModel):
    """Complete patient report for PDF generation."""
    
    report_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique report ID"
    )
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Patient info (optional)
    patient_name: Optional[str] = Field(default=None)
    
    # Content
    title: str = Field(
        default="Medical Report Explanation",
        description="Report title"
    )
    analysis: AnalysisResult = Field(description="Analysis results")
    
    # Provider info
    tenant_id: Optional[str] = Field(
        default=None,
        description="Clinic/provider identifier"
    )
    
    model_config = ConfigDict(from_attributes=True)


class ReportDownloadResponse(BaseModel):
    """Response for report download endpoint."""
    
    report_id: str = Field(description="Report ID")
    filename: str = Field(description="PDF filename")
    download_url: str = Field(description="Download URL/path")


# =============================================================================
# Health & Status
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(default="healthy")
    version: str = Field(description="Application version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Error Responses
# =============================================================================

class ErrorResponse(BaseModel):
    """Standard error response."""
    
    error: str = Field(description="Error type")
    message: str = Field(description="Human-readable error message")
    error_code: str = Field(description="Machine-readable error code")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = ConfigDict(from_attributes=True)


class LowConfidenceResponse(BaseModel):
    """Response when analysis confidence is too low."""
    
    message: str = Field(
        default="Insufficient clarity for reliable analysis. Please consult your healthcare provider directly.",
        description="Low confidence message"
    )
    reason: str = Field(
        description="Reason for low confidence"
    )
    recommendation: str = Field(
        default="We recommend consulting with your doctor for a professional interpretation.",
        description="Recommended action"
    )
    disclaimer: SafetyDisclaimer = Field(
        default_factory=SafetyDisclaimer
    )
    
    model_config = ConfigDict(from_attributes=True)
