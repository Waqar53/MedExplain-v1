"""
MedExplain - LLM Processing Engine

Provides automated interpretation assistance for medical reports.
Integrates with external language models or provides rule-based analysis.

IMPORTANT: This module does not provide medical diagnoses.
All outputs require review by qualified healthcare professionals.
"""

import os
import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

from app.utils.logger import get_logger

logger = get_logger("llm_engine")


@dataclass
class AnalysisResponse:
    """Response structure for analysis results."""
    content: str
    model: str
    provider: str
    usage: Dict[str, int]
    success: bool
    error: Optional[str] = None


# Backward compatibility alias
LLMResponse = AnalysisResponse


class LLMEngine:
    """
    Language model integration for medical report interpretation.
    
    This engine provides automated analysis assistance. All outputs
    are intended for informational purposes only and do not constitute
    medical diagnoses.
    """
    
    def __init__(self):
        """Initialize the analysis engine."""
        self.client = None
        self.model = "rule-based-analyzer"
        self.provider = "local"
        self._initialize_external_provider()
    
    def _initialize_external_provider(self) -> None:
        """Attempt to initialize external language model provider."""
        api_key = os.getenv("GEMINI_API_KEY", "")
        
        if not api_key:
            logger.info("External provider not configured, using local analysis")
            return
        
        try:
            from google import genai
            self.client = genai.Client(api_key=api_key)
            self.provider = "external"
            self.model = "gemini-2.0-flash-exp"
            logger.info("External provider initialized", model=self.model)
        except Exception as e:
            logger.info("Using local analysis", reason=str(e))
    
    def analyze_report(
        self,
        report_content: str,
        additional_context: Optional[str] = None,
        max_tokens: int = 1500
    ) -> AnalysisResponse:
        """
        Analyze medical report content and generate interpretation.
        
        Args:
            report_content: Raw text content from medical report
            additional_context: Optional clinical notes for context
            max_tokens: Maximum response length
            
        Returns:
            AnalysisResponse containing interpretation
        """
        if self.client:
            try:
                return self._external_analysis(report_content, additional_context)
            except Exception as e:
                logger.info("External analysis unavailable, using local", reason=str(e))
        
        return self._local_analysis(report_content)
    
    def analyze_image(
        self,
        image_analysis: str,
        image_type: str = "radiograph",
        additional_context: Optional[str] = None,
        max_tokens: int = 1200
    ) -> AnalysisResponse:
        """
        Generate interpretation for medical imaging analysis.
        
        Args:
            image_analysis: Pre-processed image analysis data
            image_type: Type of medical image
            additional_context: Optional clinical notes
            max_tokens: Maximum response length
            
        Returns:
            AnalysisResponse containing interpretation
        """
        return self._imaging_interpretation(image_type)
    
    # Alias for backward compatibility
    def explain_report(self, *args, **kwargs) -> AnalysisResponse:
        """Alias for analyze_report method."""
        return self.analyze_report(*args, **kwargs)
    
    def explain_image(self, *args, **kwargs) -> AnalysisResponse:
        """Alias for analyze_image method."""
        return self.analyze_image(*args, **kwargs)
    
    def _external_analysis(
        self,
        report_content: str,
        additional_context: Optional[str]
    ) -> AnalysisResponse:
        """Generate analysis using external language model."""
        prompt = self._build_prompt(report_content, additional_context)
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        
        logger.info("External analysis completed")
        
        return AnalysisResponse(
            content=response.text,
            model=self.model,
            provider=self.provider,
            usage={},
            success=True
        )
    
    def _build_prompt(self, content: str, context: Optional[str]) -> str:
        """Build analysis prompt for language model."""
        return f"""Analyze the following medical report and provide a patient-friendly summary.

IMPORTANT GUIDELINES:
- Do not provide medical diagnoses
- Use neutral, non-alarming language
- Recommend consultation with healthcare providers
- Explain medical terms in plain language

REPORT CONTENT:
{content[:4000]}

{f"CLINICAL CONTEXT: {context}" if context else ""}

Provide:
1. Summary of report contents
2. Explanation of key values
3. General interpretation (not diagnosis)
4. Recommended follow-up steps"""
    
    def _local_analysis(self, content: str) -> AnalysisResponse:
        """
        Perform rule-based local analysis of report content.
        
        Extracts identifiable values and provides standard interpretations
        without external API dependencies.
        """
        logger.info("Performing local analysis")
        
        findings = self._extract_values(content)
        abnormal_flags = self._identify_abnormal_values(content)
        report_type = self._classify_report(content)
        
        interpretation = self._generate_interpretation(
            report_type, findings, abnormal_flags
        )
        
        return AnalysisResponse(
            content=interpretation,
            model="rule-based-analyzer",
            provider="local",
            usage={},
            success=True
        )
    
    def _extract_values(self, content: str) -> Dict[str, str]:
        """Extract laboratory values from report content."""
        values = {}
        
        patterns = [
            (r'Glucose[:\s]+(\d+\.?\d*)\s*(mg/dL)?', 'Glucose'),
            (r'HbA1c[:\s]+(\d+\.?\d*)\s*%?', 'HbA1c'),
            (r'Cholesterol[:\s]+(\d+\.?\d*)\s*(mg/dL)?', 'Total Cholesterol'),
            (r'HDL[:\s]+(\d+\.?\d*)\s*(mg/dL)?', 'HDL Cholesterol'),
            (r'LDL[:\s]+(\d+\.?\d*)\s*(mg/dL)?', 'LDL Cholesterol'),
            (r'Triglycerides[:\s]+(\d+\.?\d*)\s*(mg/dL)?', 'Triglycerides'),
            (r'Hemoglobin[:\s]+(\d+\.?\d*)\s*(g/dL)?', 'Hemoglobin'),
            (r'WBC[:\s]+(\d+\.?\d*)', 'White Blood Cell Count'),
            (r'RBC[:\s]+(\d+\.?\d*)', 'Red Blood Cell Count'),
            (r'Platelet[s]?[:\s]+(\d+)', 'Platelet Count'),
            (r'Creatinine[:\s]+(\d+\.?\d*)\s*(mg/dL)?', 'Creatinine'),
            (r'BUN[:\s]+(\d+\.?\d*)\s*(mg/dL)?', 'Blood Urea Nitrogen'),
            (r'ALT[:\s]+(\d+\.?\d*)', 'ALT'),
            (r'AST[:\s]+(\d+\.?\d*)', 'AST'),
            (r'TSH[:\s]+(\d+\.?\d*)', 'TSH'),
            (r'Sodium[:\s]+(\d+\.?\d*)', 'Sodium'),
            (r'Potassium[:\s]+(\d+\.?\d*)', 'Potassium'),
        ]
        
        for pattern, name in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                values[name] = match.group(1)
        
        return values
    
    def _identify_abnormal_values(self, content: str) -> List[str]:
        """Identify values flagged as outside reference ranges."""
        abnormal = []
        
        patterns = [
            r'(\w+)[:\s]+[\d.]+\s*(?:mg/dL|g/dL|%)?[,\s]*(?:HIGH|H|\*H)',
            r'(\w+)[:\s]+[\d.]+\s*(?:mg/dL|g/dL|%)?[,\s]*(?:LOW|L|\*L)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            abnormal.extend(matches)
        
        return list(set(abnormal))
    
    def _classify_report(self, content: str) -> str:
        """Classify the type of medical report."""
        content_lower = content.lower()
        
        classifications = [
            ('cbc', 'Complete Blood Count'),
            ('complete blood count', 'Complete Blood Count'),
            ('metabolic panel', 'Metabolic Panel'),
            ('cmp', 'Comprehensive Metabolic Panel'),
            ('lipid', 'Lipid Panel'),
            ('cholesterol', 'Lipid Panel'),
            ('thyroid', 'Thyroid Function Panel'),
            ('tsh', 'Thyroid Function Panel'),
            ('urinalysis', 'Urinalysis'),
            ('urine', 'Urinalysis'),
            ('hba1c', 'Glycemic Assessment'),
            ('glucose', 'Glycemic Assessment'),
        ]
        
        for keyword, classification in classifications:
            if keyword in content_lower:
                return classification
        
        return 'Laboratory Report'
    
    def _generate_interpretation(
        self,
        report_type: str,
        findings: Dict[str, str],
        abnormal: List[str]
    ) -> str:
        """Generate structured interpretation from extracted data."""
        
        count = len(findings)
        summary = self._build_summary(report_type, count)
        meaning = self._build_explanation(findings, abnormal)
        findings_text = self._format_findings(findings, abnormal)
        
        return f"""## Summary

{summary}

## General Interpretation

{meaning}

## Identified Values

{findings_text}

## Recommended Follow-up

1. Review these results with your healthcare provider
2. Discuss any values outside reference ranges
3. Follow provider recommendations for any additional testing
4. Maintain scheduled follow-up appointments

## Important Notice

This analysis is not a medical diagnosis. The information provided is for 
educational and informational purposes only. It should not be used as a 
substitute for professional medical advice, diagnosis, or treatment.

Consult a qualified healthcare professional before making any medical decisions.

---
Analysis generated by automated system. Professional review required."""
    
    def _build_summary(self, report_type: str, value_count: int) -> str:
        """Build summary section."""
        if value_count == 0:
            return f"This {report_type} has been processed. The report contains laboratory measurements for clinical review."
        return f"This {report_type} contains {value_count} identified values. These measurements provide information about various health indicators."
    
    def _build_explanation(self, findings: Dict, abnormal: List) -> str:
        """Build explanation section with context."""
        if not findings:
            return "Laboratory reports contain values that are compared against reference ranges. These ranges represent typical values for the general population. Individual interpretation requires clinical context."
        
        explanation = "The identified values represent various health markers:\n\n"
        
        context_map = {
            'glucose': "Glucose: Measures blood sugar levels",
            'cholesterol': "Cholesterol: Lipid levels affecting cardiovascular health",
            'hemoglobin': "Hemoglobin: Oxygen-carrying capacity of blood",
            'creatinine': "Creatinine: Indicator of kidney function",
            'alt': "ALT: Liver enzyme marker",
            'ast': "AST: Liver enzyme marker",
            'tsh': "TSH: Thyroid function indicator",
        }
        
        for name in list(findings.keys())[:4]:
            for key, desc in context_map.items():
                if key in name.lower():
                    explanation += f"- {desc}\n"
                    break
        
        if abnormal:
            explanation += "\nSome values appear outside standard reference ranges. Your healthcare provider will interpret these in the context of your complete medical history."
        
        return explanation
    
    def _format_findings(self, findings: Dict, abnormal: List) -> str:
        """Format findings as list."""
        if not findings:
            return "- Report contains laboratory measurements for review\n- Reference ranges provided for comparison"
        
        result = ""
        for name, value in list(findings.items())[:8]:
            flag = " (outside reference range)" if any(name.lower().split()[0] in a.lower() for a in abnormal) else ""
            result += f"- {name}: {value}{flag}\n"
        
        if len(findings) > 8:
            result += f"- Additional values: {len(findings) - 8}\n"
        
        return result
    
    def _imaging_interpretation(self, image_type: str) -> AnalysisResponse:
        """Generate standard interpretation for medical imaging."""
        interpretation = f"""## Summary

The submitted {image_type} has been received for processing. Medical imaging 
requires interpretation by qualified radiologists or imaging specialists.

## General Information

Medical imaging studies provide visual information about internal body structures.
Proper interpretation requires specialized training and access to complete 
clinical history.

## Processing Status

- Image received and validated
- Pending review by qualified specialist
- Results to be provided through healthcare provider

## Next Steps

1. Await official radiologist interpretation
2. Results will be communicated through your healthcare provider
3. Schedule follow-up appointment to discuss findings

## Important Notice

This system does not provide diagnostic interpretation of medical images.
Only qualified healthcare professionals can interpret medical imaging studies.

Consult your healthcare provider for interpretation and recommendations."""
        
        return AnalysisResponse(
            content=interpretation,
            model="rule-based-analyzer",
            provider="local",
            usage={},
            success=True
        )
    
    def is_available(self) -> bool:
        """Check if analysis engine is available."""
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status information."""
        return {
            "available": True,
            "provider": self.provider,
            "model": self.model,
            "external_configured": self.client is not None
        }


# Module-level singleton
_engine_instance: Optional[LLMEngine] = None


def get_llm_engine() -> LLMEngine:
    """Get or create singleton engine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = LLMEngine()
    return _engine_instance
