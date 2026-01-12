"""
Safety and confidence checking for MedExplain AI.

Implements safety rules, confidence thresholds, and disclaimer
injection for all medical explanations.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from enum import Enum

from app.config import settings
from app.models.schemas import (
    RiskLevel,
    ConfidenceLevel,
    SafetyDisclaimer,
    LowConfidenceResponse
)
from app.utils.logger import get_logger

logger = get_logger("safety_checker")


class SafetyFlag(str, Enum):
    """Safety flags for content review."""
    DIAGNOSIS_LANGUAGE = "diagnosis_language"
    PRESCRIPTION_LANGUAGE = "prescription_language"
    ALARMING_LANGUAGE = "alarming_language"
    MISSING_DISCLAIMER = "missing_disclaimer"
    LOW_CONFIDENCE = "low_confidence"
    UNCLEAR_INPUT = "unclear_input"


@dataclass
class SafetyCheckResult:
    """Result of safety check."""
    
    is_safe: bool
    confidence_level: ConfidenceLevel
    confidence_score: float
    risk_level: RiskLevel
    flags: List[SafetyFlag]
    modified_content: Optional[str]  # Content with safety modifications
    disclaimer: SafetyDisclaimer
    requires_human_review: bool


class SafetyChecker:
    """
    Ensures all outputs meet healthcare safety requirements.
    
    Checks for:
    - Diagnosis language that should be avoided
    - Prescription/treatment recommendations to block
    - Alarming words that could cause anxiety
    - Presence of required disclaimers
    - Confidence thresholds
    
    All outputs are modified to include appropriate disclaimers
    and safety language.
    """
    
    # Words/phrases that suggest diagnosis (should be flagged)
    DIAGNOSIS_PHRASES = [
        "you have",
        "you are diagnosed",
        "diagnosis is",
        "confirms that you have",
        "this means you have",
        "indicates disease",
        "indicates cancer",
        "indicates infection",
        "you are suffering from",
        "this test confirms",
    ]
    
    # Prescription/treatment language to block
    PRESCRIPTION_PHRASES = [
        "you should take",
        "take this medication",
        "prescribe",
        "i recommend taking",
        "start treatment",
        "begin therapy",
        "you need surgery",
        "you must undergo",
    ]
    
    # Alarming words to soften
    ALARMING_WORDS = {
        "cancer": "concerning finding",
        "tumor": "abnormal growth",
        "fatal": "serious",
        "terminal": "advanced",
        "dangerous": "concerning",
        "critical": "important",
        "emergency": "urgent attention needed",
        "dying": "requiring immediate care",
        "death": "serious outcome",
    }
    
    # Required disclaimer components
    REQUIRED_DISCLAIMER_PARTS = [
        "not a diagnosis",
        "consult",
        "doctor",
        "healthcare"
    ]
    
    # Standard disclaimers
    DISCLAIMERS = {
        "main": "NOTICE: This analysis does not constitute a medical diagnosis. Information provided is for educational purposes only.",
        "consultation": "Consult a qualified healthcare provider before making any medical decisions based on this information.",
        "low_confidence": "Due to input quality limitations, this analysis may be incomplete. Professional interpretation is recommended.",
        "professional": "These results should be reviewed by a qualified healthcare professional.",
    }
    
    def __init__(self):
        self.low_threshold = settings.confidence_threshold_low
        self.high_threshold = settings.confidence_threshold_high
    
    def check_safety(
        self,
        content: str,
        confidence_score: float,
        source_type: str = "report"
    ) -> SafetyCheckResult:
        """
        Perform comprehensive safety check on content.
        
        Args:
            content: Generated content to check
            confidence_score: Confidence score from analysis
            source_type: Type of source (report, image)
            
        Returns:
            SafetyCheckResult with safety information
        """
        flags = []
        
        # Check for diagnosis language
        if self._contains_diagnosis_language(content):
            flags.append(SafetyFlag.DIAGNOSIS_LANGUAGE)
            logger.warning("Diagnosis language detected in content")
        
        # Check for prescription language
        if self._contains_prescription_language(content):
            flags.append(SafetyFlag.PRESCRIPTION_LANGUAGE)
            logger.warning("Prescription language detected in content")
        
        # Check for alarming language
        alarming_words = self._find_alarming_words(content)
        if alarming_words:
            flags.append(SafetyFlag.ALARMING_LANGUAGE)
            logger.info("Alarming words found", words=list(alarming_words))
        
        # Check for disclaimers
        if not self._has_disclaimer(content):
            flags.append(SafetyFlag.MISSING_DISCLAIMER)
        
        # Check confidence
        confidence_level = self._get_confidence_level(confidence_score)
        if confidence_level == ConfidenceLevel.LOW:
            flags.append(SafetyFlag.LOW_CONFIDENCE)
        
        # Determine risk level
        risk_level = self._assess_risk_level(content, confidence_score, flags)
        
        # Modify content for safety
        modified_content = self._make_safe(content, flags)
        
        # Create disclaimer
        disclaimer = self._create_disclaimer(confidence_level, flags)
        
        # Determine if human review needed
        requires_review = (
            SafetyFlag.DIAGNOSIS_LANGUAGE in flags or
            SafetyFlag.PRESCRIPTION_LANGUAGE in flags or
            confidence_level == ConfidenceLevel.LOW
        )
        
        is_safe = len(flags) == 0 or (
            len(flags) == 1 and 
            SafetyFlag.MISSING_DISCLAIMER in flags
        )
        
        logger.info(
            "Safety check complete",
            is_safe=is_safe,
            flags=[f.value for f in flags],
            confidence=confidence_level.value,
            risk=risk_level.value
        )
        
        return SafetyCheckResult(
            is_safe=is_safe,
            confidence_level=confidence_level,
            confidence_score=confidence_score,
            risk_level=risk_level,
            flags=flags,
            modified_content=modified_content,
            disclaimer=disclaimer,
            requires_human_review=requires_review
        )
    
    def _contains_diagnosis_language(self, content: str) -> bool:
        """Check if content contains diagnosis-like language."""
        content_lower = content.lower()
        return any(phrase in content_lower for phrase in self.DIAGNOSIS_PHRASES)
    
    def _contains_prescription_language(self, content: str) -> bool:
        """Check if content contains prescription-like language."""
        content_lower = content.lower()
        return any(phrase in content_lower for phrase in self.PRESCRIPTION_PHRASES)
    
    def _find_alarming_words(self, content: str) -> set:
        """Find alarming words in content."""
        content_lower = content.lower()
        found = set()
        for word in self.ALARMING_WORDS:
            if word in content_lower:
                found.add(word)
        return found
    
    def _has_disclaimer(self, content: str) -> bool:
        """Check if content has required disclaimers."""
        content_lower = content.lower()
        return any(part in content_lower for part in self.REQUIRED_DISCLAIMER_PARTS)
    
    def _get_confidence_level(self, score: float) -> ConfidenceLevel:
        """Convert confidence score to level."""
        if score < self.low_threshold:
            return ConfidenceLevel.LOW
        elif score < self.high_threshold:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.HIGH
    
    def _assess_risk_level(
        self,
        content: str,
        confidence_score: float,
        flags: List[SafetyFlag]
    ) -> RiskLevel:
        """
        Assess informational risk level.
        
        This is NOT a medical risk assessment. It's an assessment
        of how the information should be presented to the patient.
        """
        # Low confidence = unknown risk
        if SafetyFlag.LOW_CONFIDENCE in flags:
            return RiskLevel.UNKNOWN
        
        content_lower = content.lower()
        
        # Check for concerning terms (informational only)
        high_concern_terms = [
            "abnormal", "elevated", "outside range", "high", "low",
            "concerning", "follow-up", "urgent", "immediate"
        ]
        
        low_concern_terms = [
            "normal", "within range", "healthy", "good", "expected",
            "typical", "routine"
        ]
        
        high_count = sum(1 for term in high_concern_terms if term in content_lower)
        low_count = sum(1 for term in low_concern_terms if term in content_lower)
        
        if high_count > 2 or "urgent" in content_lower or "immediate" in content_lower:
            return RiskLevel.HIGH
        elif high_count > low_count:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _make_safe(self, content: str, flags: List[SafetyFlag]) -> str:
        """
        Modify content to be safe for patient consumption.
        
        Args:
            content: Original content
            flags: Safety flags detected
            
        Returns:
            Modified safe content
        """
        modified = content
        
        # Soften alarming words
        for alarming, replacement in self.ALARMING_WORDS.items():
            # Case-insensitive replacement
            import re
            pattern = re.compile(re.escape(alarming), re.IGNORECASE)
            modified = pattern.sub(replacement, modified)
        
        # Add disclaimers if missing
        if not self._has_disclaimer(modified):
            modified = self._add_disclaimers(modified)
        
        return modified
    
    def _add_disclaimers(self, content: str) -> str:
        """Add required disclaimers to content."""
        disclaimer_block = f"""
---

{self.DISCLAIMERS['main']}

{self.DISCLAIMERS['consultation']}

{self.DISCLAIMERS['professional']}
"""
        return content + disclaimer_block
    
    def _create_disclaimer(
        self,
        confidence_level: ConfidenceLevel,
        flags: List[SafetyFlag]
    ) -> SafetyDisclaimer:
        """Create appropriate disclaimer object."""
        confidence_note = None
        
        if confidence_level == ConfidenceLevel.LOW:
            confidence_note = self.DISCLAIMERS['low_confidence']
        elif confidence_level == ConfidenceLevel.MEDIUM:
            confidence_note = "Analysis confidence is moderate. Professional review recommended."
        
        return SafetyDisclaimer(
            main_disclaimer=self.DISCLAIMERS['main'],
            consultation_reminder=self.DISCLAIMERS['consultation'],
            confidence_note=confidence_note
        )
    
    def create_low_confidence_response(
        self,
        reason: str = "Unable to analyze input with sufficient confidence"
    ) -> LowConfidenceResponse:
        """
        Create a safe response for low-confidence situations.
        
        Used when the analysis confidence is too low to provide
        meaningful information.
        
        Args:
            reason: Reason for low confidence
            
        Returns:
            LowConfidenceResponse
        """
        return LowConfidenceResponse(
            message=(
                "We were unable to provide a reliable analysis of this input. "
                "This may be due to image quality, document clarity, or other factors."
            ),
            reason=reason,
            recommendation=(
                "Please consult with your healthcare provider directly. "
                "They have access to your complete medical history and can "
                "provide personalized guidance."
            ),
            disclaimer=SafetyDisclaimer(
                main_disclaimer=self.DISCLAIMERS['main'],
                consultation_reminder=self.DISCLAIMERS['consultation'],
                confidence_note=self.DISCLAIMERS['low_confidence']
            )
        )
    
    def get_risk_explanation(self, risk_level: RiskLevel) -> str:
        """Get patient-friendly explanation of risk level."""
        explanations = {
            RiskLevel.LOW: (
                "The information in this report generally appears within "
                "expected ranges. Your doctor will discuss this with you."
            ),
            RiskLevel.MEDIUM: (
                "Some findings may warrant discussion with your healthcare "
                "provider to understand what they mean for you personally."
            ),
            RiskLevel.HIGH: (
                "This report contains findings that your healthcare provider "
                "will want to discuss with you. Please ensure you follow up "
                "as recommended."
            ),
            RiskLevel.UNKNOWN: (
                "We were unable to fully assess this report. Please consult "
                "your healthcare provider for a complete interpretation."
            )
        }
        return explanations.get(risk_level, explanations[RiskLevel.UNKNOWN])


# Singleton instance
safety_checker = SafetyChecker()
