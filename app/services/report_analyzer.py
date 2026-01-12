"""
Report analyzer service for MedExplain AI.

Orchestrates the analysis pipeline for medical reports and images.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime
from uuid import uuid4

from app.config import settings
from app.core.pdf_extractor import pdf_extractor, PDFExtractionResult
from app.core.text_processor import text_processor, ProcessedText
from app.core.image_processor import image_processor, ImageData
from app.core.vision_model import get_vision_model, VisionAnalysisResult
from app.core.llm_engine import get_llm_engine, LLMResponse
from app.services.safety_checker import safety_checker, SafetyCheckResult
from app.models.schemas import (
    FileType,
    RiskLevel,
    ConfidenceLevel,
    AnalysisResult,
    MedicalExplanation,
    SafetyDisclaimer
)
from app.utils.logger import get_logger

logger = get_logger("report_analyzer")


@dataclass
class AnalysisSession:
    """Tracks an analysis session."""
    
    session_id: str
    tenant_id: str
    file_type: FileType
    filename: str
    file_content: bytes
    created_at: datetime
    
    # Processing results (filled during analysis)
    extraction_result: Optional[PDFExtractionResult] = None
    processed_text: Optional[ProcessedText] = None
    image_data: Optional[ImageData] = None
    vision_result: Optional[VisionAnalysisResult] = None
    llm_response: Optional[LLMResponse] = None
    safety_result: Optional[SafetyCheckResult] = None
    final_result: Optional[AnalysisResult] = None


class ReportAnalyzer:
    """
    Main analysis orchestrator for MedExplain AI.
    
    Coordinates:
    - File type detection and validation
    - PDF/text extraction
    - Image analysis
    - LLM explanation generation
    - Safety checking
    
    All operations are tracked in sessions for auditing.
    """
    
    def __init__(self):
        self._sessions: Dict[str, AnalysisSession] = {}
    
    def create_session(
        self,
        file_content: bytes,
        filename: str,
        file_type: FileType,
        tenant_id: Optional[str] = None
    ) -> AnalysisSession:
        """
        Create a new analysis session.
        
        Args:
            file_content: Raw file bytes
            filename: Original filename
            file_type: Type of file
            tenant_id: Clinic/tenant identifier
            
        Returns:
            AnalysisSession
        """
        session_id = str(uuid4())
        tenant = tenant_id or settings.default_tenant_id
        
        session = AnalysisSession(
            session_id=session_id,
            tenant_id=tenant,
            file_type=file_type,
            filename=filename,
            file_content=file_content,
            created_at=datetime.utcnow()
        )
        
        self._sessions[session_id] = session
        
        logger.info(
            "Analysis session created",
            session_id=session_id,
            tenant_id=tenant,
            file_type=file_type.value,
            filename=filename
        )
        
        return session
    
    def get_session(self, session_id: str) -> Optional[AnalysisSession]:
        """Get a session by ID."""
        return self._sessions.get(session_id)
    
    async def analyze(
        self,
        session_id: str,
        additional_context: Optional[str] = None
    ) -> AnalysisResult:
        """
        Run the full analysis pipeline.
        
        Args:
            session_id: Session ID from create_session
            additional_context: Optional context from healthcare provider
            
        Returns:
            AnalysisResult
        """
        session = self.get_session(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        start_time = time.time()
        
        logger.info(
            "Starting analysis",
            session_id=session_id,
            file_type=session.file_type.value
        )
        
        try:
            if session.file_type == FileType.PDF:
                result = await self._analyze_pdf(session, additional_context)
            elif session.file_type == FileType.TEXT:
                result = await self._analyze_text(session, additional_context)
            elif session.file_type == FileType.IMAGE:
                result = await self._analyze_image(session, additional_context)
            else:
                raise ValueError(f"Unsupported file type: {session.file_type}")
            
            # Calculate processing time
            processing_time = int((time.time() - start_time) * 1000)
            result.processing_time_ms = processing_time
            
            # Store final result
            session.final_result = result
            
            logger.info(
                "Analysis complete",
                session_id=session_id,
                processing_time_ms=processing_time,
                confidence=result.confidence.value
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Analysis failed",
                session_id=session_id,
                error=str(e)
            )
            raise
    
    async def _analyze_pdf(
        self,
        session: AnalysisSession,
        additional_context: Optional[str]
    ) -> AnalysisResult:
        """Analyze a PDF report."""
        # Extract text from PDF
        extraction = pdf_extractor.extract_text(
            session.file_content,
            session.filename
        )
        session.extraction_result = extraction
        
        # Process extracted text
        processed = text_processor.process(extraction.text)
        session.processed_text = processed
        
        # Get context for LLM
        context = text_processor.get_summary_context(processed)
        
        # Generate explanation
        llm_engine = get_llm_engine()
        llm_response = llm_engine.explain_report(context, additional_context)
        session.llm_response = llm_response
        
        # Calculate confidence (based on extraction + LLM)
        base_confidence = extraction.confidence
        if not llm_response.success:
            base_confidence *= 0.5
        
        # Safety check
        safety_result = safety_checker.check_safety(
            llm_response.content,
            base_confidence,
            "report"
        )
        session.safety_result = safety_result
        
        # Build final result
        return self._build_result(
            session_id=session.session_id,
            content=safety_result.modified_content or llm_response.content,
            confidence_score=base_confidence,
            safety_result=safety_result,
            source_type=FileType.PDF,
            key_findings=list(processed.key_values.keys())[:5]
        )
    
    async def _analyze_text(
        self,
        session: AnalysisSession,
        additional_context: Optional[str]
    ) -> AnalysisResult:
        """Analyze a text report."""
        # Decode text
        try:
            text = session.file_content.decode('utf-8')
        except UnicodeDecodeError:
            text = session.file_content.decode('latin-1')
        
        # Process text
        processed = text_processor.process(text)
        session.processed_text = processed
        
        # Get context for LLM
        context = text_processor.get_summary_context(processed)
        
        # Generate explanation
        llm_engine = get_llm_engine()
        llm_response = llm_engine.explain_report(context, additional_context)
        session.llm_response = llm_response
        
        # Calculate confidence
        base_confidence = 0.8 if processed.word_count > 50 else 0.5
        if not llm_response.success:
            base_confidence *= 0.5
        
        # Safety check
        safety_result = safety_checker.check_safety(
            llm_response.content,
            base_confidence,
            "report"
        )
        session.safety_result = safety_result
        
        return self._build_result(
            session_id=session.session_id,
            content=safety_result.modified_content or llm_response.content,
            confidence_score=base_confidence,
            safety_result=safety_result,
            source_type=FileType.TEXT,
            key_findings=list(processed.key_values.keys())[:5]
        )
    
    async def _analyze_image(
        self,
        session: AnalysisSession,
        additional_context: Optional[str]
    ) -> AnalysisResult:
        """Analyze a medical image (X-ray)."""
        # Load and preprocess image
        image_data = image_processor.load_image(
            session.file_content,
            session.filename
        )
        session.image_data = image_data
        
        # Run vision analysis
        vision_model = get_vision_model()
        vision_result = vision_model.analyze_image(image_data)
        session.vision_result = vision_result
        
        # Generate explanation from vision analysis
        llm_engine = get_llm_engine()
        llm_response = llm_engine.explain_image(
            vision_result.feature_summary,
            image_type="X-ray" if "xray" in session.filename.lower() else "Medical Image",
            additional_context=additional_context
        )
        session.llm_response = llm_response
        
        # Calculate confidence
        base_confidence = vision_result.confidence_score
        if not llm_response.success:
            base_confidence *= 0.5
        
        # Safety check
        safety_result = safety_checker.check_safety(
            llm_response.content,
            base_confidence,
            "image"
        )
        session.safety_result = safety_result
        
        return self._build_result(
            session_id=session.session_id,
            content=safety_result.modified_content or llm_response.content,
            confidence_score=base_confidence,
            safety_result=safety_result,
            source_type=FileType.IMAGE,
            key_findings=vision_result.detected_patterns
        )
    
    def _build_result(
        self,
        session_id: str,
        content: str,
        confidence_score: float,
        safety_result: SafetyCheckResult,
        source_type: FileType,
        key_findings: list
    ) -> AnalysisResult:
        """Build the final analysis result."""
        # Parse content into structured explanation
        explanation = self._parse_explanation(content, key_findings)
        
        return AnalysisResult(
            session_id=session_id,
            explanation=explanation,
            risk_level=safety_result.risk_level,
            confidence=safety_result.confidence_level,
            confidence_score=confidence_score,
            disclaimer=safety_result.disclaimer,
            source_type=source_type
        )
    
    def _parse_explanation(
        self,
        content: str,
        key_findings: list
    ) -> MedicalExplanation:
        """Parse LLM content into structured explanation."""
        # Split content into sections
        lines = content.split('\n')
        
        summary = ""
        what_means = ""
        next_steps = []
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect sections
            line_lower = line.lower()
            if 'summary' in line_lower or line.startswith('1.'):
                current_section = 'summary'
                continue
            elif 'indicates' in line_lower or 'means' in line_lower or line.startswith('2.'):
                current_section = 'what_means'
                continue
            elif 'follow' in line_lower or 'next' in line_lower or 'step' in line_lower or line.startswith('3.'):
                current_section = 'next_steps'
                continue
            elif line.startswith('---') or line.startswith('⚠️') or line.startswith('📋'):
                current_section = 'disclaimer'
                continue
            
            # Add content to appropriate section
            if current_section == 'summary' and not summary:
                summary = line
            elif current_section == 'what_means':
                what_means += line + " "
            elif current_section == 'next_steps':
                if line.startswith('- ') or line.startswith('• '):
                    next_steps.append(line[2:])
                elif line and len(next_steps) < 5:
                    next_steps.append(line)
        
        # Defaults if parsing failed
        if not summary:
            summary = "Please review the detailed information below."
        if not what_means:
            what_means = "Your healthcare provider can explain what these results mean for your specific situation."
        if not next_steps:
            next_steps = [
                "Discuss these results with your doctor",
                "Ask any questions you may have",
                "Follow your provider's recommendations"
            ]
        
        return MedicalExplanation(
            summary=summary,
            what_this_means=what_means.strip(),
            common_next_steps=next_steps[:5],
            key_findings=key_findings[:10]
        )
    
    def cleanup_session(self, session_id: str) -> bool:
        """Remove a session from memory."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info("Session cleaned up", session_id=session_id)
            return True
        return False


# Singleton instance
report_analyzer = ReportAnalyzer()
