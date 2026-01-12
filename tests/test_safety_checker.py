"""
Tests for the safety checker module.
"""

import pytest
from app.services.safety_checker import (
    SafetyChecker,
    SafetyFlag,
    safety_checker
)
from app.models.schemas import RiskLevel, ConfidenceLevel


class TestSafetyChecker:
    """Test suite for SafetyChecker."""
    
    def test_high_confidence_passes(self):
        """High confidence content should pass safety checks."""
        content = "Your blood test results appear normal. Please discuss with your doctor."
        result = safety_checker.check_safety(content, 0.85, "report")
        
        assert result.confidence_level == ConfidenceLevel.HIGH
        assert result.is_safe or SafetyFlag.MISSING_DISCLAIMER in result.flags
    
    def test_low_confidence_flagged(self):
        """Low confidence should be flagged."""
        content = "Some results were found."
        result = safety_checker.check_safety(content, 0.2, "report")
        
        assert result.confidence_level == ConfidenceLevel.LOW
        assert SafetyFlag.LOW_CONFIDENCE in result.flags
    
    def test_diagnosis_language_detected(self):
        """Diagnosis language should be detected and flagged."""
        content = "You have diabetes based on these results."
        result = safety_checker.check_safety(content, 0.8, "report")
        
        assert SafetyFlag.DIAGNOSIS_LANGUAGE in result.flags
    
    def test_prescription_language_detected(self):
        """Prescription language should be detected and flagged."""
        content = "You should take aspirin daily for this condition."
        result = safety_checker.check_safety(content, 0.8, "report")
        
        assert SafetyFlag.PRESCRIPTION_LANGUAGE in result.flags
    
    def test_alarming_words_softened(self):
        """Alarming words should be softened in output."""
        content = "The scan shows a tumor in the lung area."
        result = safety_checker.check_safety(content, 0.8, "report")
        
        # Original alarming word should be replaced
        assert "tumor" not in result.modified_content.lower() or "growth" in result.modified_content.lower()
    
    def test_disclaimer_added_when_missing(self):
        """Disclaimers should be added when missing."""
        content = "Results look normal."
        result = safety_checker.check_safety(content, 0.8, "report")
        
        # Modified content should have disclaimer
        assert "not a diagnosis" in result.modified_content.lower() or \
               "consult" in result.modified_content.lower()
    
    def test_risk_level_assessment(self):
        """Risk levels should be assessed correctly."""
        # Low risk content
        low_content = "All values are normal and within expected range."
        low_result = safety_checker.check_safety(low_content, 0.8, "report")
        assert low_result.risk_level == RiskLevel.LOW
        
        # Higher concern content
        high_content = "Values are abnormal and elevated. Urgent follow-up needed."
        high_result = safety_checker.check_safety(high_content, 0.8, "report")
        assert high_result.risk_level in [RiskLevel.MEDIUM, RiskLevel.HIGH]
    
    def test_low_confidence_response(self):
        """Low confidence response should be properly formatted."""
        response = safety_checker.create_low_confidence_response(
            "Image quality too low"
        )
        
        assert response.reason == "Image quality too low"
        assert "consult" in response.recommendation.lower()
        assert response.disclaimer is not None
    
    def test_risk_explanation(self):
        """Risk explanations should be provided for all levels."""
        for risk_level in RiskLevel:
            explanation = safety_checker.get_risk_explanation(risk_level)
            assert len(explanation) > 0
            assert "doctor" in explanation.lower() or "provider" in explanation.lower()


class TestConfidenceThresholds:
    """Test confidence threshold behavior."""
    
    def test_below_low_threshold(self):
        """Scores below low threshold should be LOW confidence."""
        checker = SafetyChecker()
        result = checker.check_safety("Test content", 0.1, "report")
        assert result.confidence_level == ConfidenceLevel.LOW
    
    def test_between_thresholds(self):
        """Scores between thresholds should be MEDIUM confidence."""
        checker = SafetyChecker()
        result = checker.check_safety("Test content", 0.5, "report")
        assert result.confidence_level == ConfidenceLevel.MEDIUM
    
    def test_above_high_threshold(self):
        """Scores above high threshold should be HIGH confidence."""
        checker = SafetyChecker()
        result = checker.check_safety("Test content", 0.9, "report")
        assert result.confidence_level == ConfidenceLevel.HIGH
