"""
API routes for MedExplain AI.

Defines all REST API endpoints for the medical report explanation system.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, File, UploadFile, HTTPException, Request, Depends, Form
from fastapi.responses import FileResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings
from app.models.schemas import (
    UploadResponse,
    GenerateReportRequest,
    AnalysisResult,
    HealthResponse,
    ErrorResponse,
    FileType,
    PatientReport,
    ReportDownloadResponse
)
from app.utils.file_validators import file_validator
from app.services.report_analyzer import report_analyzer
from app.services.report_generator import get_report_generator
from app.utils.logger import get_logger

logger = get_logger("routes")

# Create router
router = APIRouter()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


def get_tenant_id(request: Request) -> str:
    """Get tenant ID from request state."""
    return getattr(request.state, 'tenant_id', settings.default_tenant_id)


# =============================================================================
# Health Check
# =============================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check endpoint"
)
async def health_check():
    """
    Check if the service is healthy and running.
    
    Returns basic health status and version information.
    """
    return HealthResponse(
        status="healthy",
        version=settings.app_version
    )


# =============================================================================
# File Upload
# =============================================================================

@router.post(
    "/upload-report",
    response_model=UploadResponse,
    tags=["Upload"],
    summary="Upload a medical report (PDF or text)",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file"},
        413: {"model": ErrorResponse, "description": "File too large"}
    }
)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def upload_report(
    request: Request,
    file: UploadFile = File(..., description="PDF or text report file")
):
    """
    Upload a medical report for analysis.
    
    Supports:
    - PDF files (.pdf)
    - Text files (.txt, .doc, .docx)
    
    Returns a session ID to use for generating explanations.
    """
    tenant_id = get_tenant_id(request)
    
    # Read file content
    content = await file.read()
    
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    
    # Validate file
    is_valid, error, file_type = file_validator.validate(content, file.filename)
    
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)
    
    if file_type == 'unknown':
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file.filename}"
        )
    
    # Map file type
    if file_type == 'pdf':
        ftype = FileType.PDF
    elif file_type == 'text':
        ftype = FileType.TEXT
    else:
        raise HTTPException(status_code=400, detail="Invalid report file type")
    
    # Create analysis session
    session = report_analyzer.create_session(
        file_content=content,
        filename=file.filename,
        file_type=ftype,
        tenant_id=tenant_id
    )
    
    logger.info(
        "Report uploaded",
        session_id=session.session_id,
        filename=file.filename,
        file_type=ftype.value,
        tenant_id=tenant_id
    )
    
    return UploadResponse(
        session_id=session.session_id,
        filename=file.filename,
        file_type=ftype,
        file_size_bytes=len(content),
        message="Report uploaded successfully. Use the session_id to generate explanation."
    )


@router.post(
    "/upload-xray",
    response_model=UploadResponse,
    tags=["Upload"],
    summary="Upload an X-ray or medical image",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file"},
        413: {"model": ErrorResponse, "description": "File too large"}
    }
)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def upload_xray(
    request: Request,
    file: UploadFile = File(..., description="X-ray or medical image file")
):
    """
    Upload a medical image (X-ray, CT scan, etc.) for analysis.
    
    Supports:
    - PNG files (.png)
    - JPEG files (.jpg, .jpeg)
    - DICOM files (.dcm, .dicom)
    
    **Important**: This is NOT a diagnostic tool. Image analysis is
    for informational purposes only. Always consult a radiologist.
    
    Returns a session ID to use for generating explanations.
    """
    tenant_id = get_tenant_id(request)
    
    # Read file content
    content = await file.read()
    
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    
    # Validate file
    is_valid, error, file_type = file_validator.validate(content, file.filename)
    
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)
    
    if file_type != 'image':
        raise HTTPException(
            status_code=400, 
            detail=f"Expected image file, got: {file_type}"
        )
    
    # Create analysis session
    session = report_analyzer.create_session(
        file_content=content,
        filename=file.filename,
        file_type=FileType.IMAGE,
        tenant_id=tenant_id
    )
    
    logger.info(
        "X-ray uploaded",
        session_id=session.session_id,
        filename=file.filename,
        tenant_id=tenant_id
    )
    
    return UploadResponse(
        session_id=session.session_id,
        filename=file.filename,
        file_type=FileType.IMAGE,
        file_size_bytes=len(content),
        message="Image uploaded successfully. Use the session_id to generate explanation."
    )


# =============================================================================
# Report Generation
# =============================================================================

@router.post(
    "/generate-report",
    response_model=AnalysisResult,
    tags=["Analysis"],
    summary="Generate patient-friendly explanation",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        404: {"model": ErrorResponse, "description": "Session not found"}
    }
)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def generate_report(
    request: Request,
    body: GenerateReportRequest
):
    """
    Generate a patient-friendly explanation for an uploaded report.
    
    Requires a session_id from a previous upload.
    
    **Safety Features**:
    - All explanations include safety disclaimers
    - Never makes diagnosis claims
    - Encourages professional consultation
    - Shows confidence level
    
    Returns structured explanation with:
    - Patient-friendly summary
    - What results generally indicate
    - Common next steps
    - Risk level (informational)
    - Safety disclaimers
    """
    session_id = body.session_id
    
    # Get session
    session = report_analyzer.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}. Please upload a file first."
        )
    
    # Run analysis
    try:
        result = await report_analyzer.analyze(
            session_id=session_id,
            additional_context=body.additional_context
        )
        
        logger.info(
            "Report generated",
            session_id=session_id,
            confidence=result.confidence.value,
            risk_level=result.risk_level.value
        )
        
        return result
        
    except Exception as e:
        logger.error(
            "Report generation failed",
            session_id=session_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate report: {str(e)}"
        )


# =============================================================================
# PDF Download
# =============================================================================

@router.post(
    "/create-pdf",
    response_model=ReportDownloadResponse,
    tags=["Reports"],
    summary="Create a PDF report"
)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def create_pdf_report(
    request: Request,
    body: GenerateReportRequest
):
    """
    Create a downloadable PDF report.
    
    First generates the explanation (if not already done),
    then creates a professional PDF document.
    """
    session_id = body.session_id
    
    # Get session
    session = report_analyzer.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}"
        )
    
    # Generate analysis if not already done
    if not session.final_result:
        session.final_result = await report_analyzer.analyze(
            session_id=session_id,
            additional_context=body.additional_context
        )
    
    # Create patient report
    patient_report = PatientReport(
        patient_name=body.patient_name,
        analysis=session.final_result,
        tenant_id=body.tenant_id
    )
    
    # Generate PDF
    generator = get_report_generator()
    pdf_path = generator.generate_pdf(patient_report)
    
    return ReportDownloadResponse(
        report_id=patient_report.report_id,
        filename=pdf_path.name,
        download_url=f"/download-pdf/{patient_report.report_id}"
    )


@router.get(
    "/download-pdf/{report_id}",
    tags=["Reports"],
    summary="Download a generated PDF report"
)
async def download_pdf(report_id: str):
    """
    Download a previously generated PDF report.
    
    The report_id comes from the create-pdf endpoint response.
    """
    # Find PDF file
    output_dir = settings.output_path
    pdf_files = list(output_dir.glob(f"*{report_id}*.pdf"))
    
    if not pdf_files:
        raise HTTPException(
            status_code=404,
            detail=f"PDF report not found: {report_id}"
        )
    
    pdf_path = pdf_files[0]
    
    return FileResponse(
        path=str(pdf_path),
        filename=pdf_path.name,
        media_type="application/pdf"
    )


# =============================================================================
# Admin/Debug (only in debug mode)
# =============================================================================

@router.get(
    "/sessions/{session_id}",
    tags=["Debug"],
    summary="Get session details (debug only)"
)
async def get_session(session_id: str, request: Request):
    """Get details about an analysis session. Only available in debug mode."""
    if not settings.debug:
        raise HTTPException(status_code=404, detail="Not found")
    
    session = report_analyzer.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session.session_id,
        "tenant_id": session.tenant_id,
        "file_type": session.file_type.value,
        "filename": session.filename,
        "created_at": session.created_at.isoformat(),
        "has_result": session.final_result is not None
    }
