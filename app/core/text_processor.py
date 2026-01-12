"""
Text processing utilities for MedExplain AI.

Cleans, normalizes, and structures extracted medical text
for optimal LLM processing.
"""

import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from app.utils.logger import get_logger

logger = get_logger("text_processor")


@dataclass
class ProcessedText:
    """Result of text processing."""
    
    original_text: str
    cleaned_text: str
    sections: Dict[str, str]
    key_values: Dict[str, str]
    word_count: int
    medical_terms_found: List[str]


class TextProcessor:
    """
    Processes and cleans medical report text.
    
    Performs:
    - Noise removal (headers, footers, page numbers)
    - Text normalization
    - Section extraction
    - Key-value pair identification
    - Medical term recognition
    """
    
    # Common medical report section headers
    SECTION_PATTERNS = [
        r"(?i)(patient\s*information|patient\s*details)[:.]?\s*",
        r"(?i)(clinical\s*history|history)[:.]?\s*",
        r"(?i)(chief\s*complaint|cc)[:.]?\s*",
        r"(?i)(diagnosis|impression|conclusion)[:.]?\s*",
        r"(?i)(findings|results|observations)[:.]?\s*",
        r"(?i)(recommendations|plan|follow[- ]?up)[:.]?\s*",
        r"(?i)(laboratory\s*results|lab\s*values)[:.]?\s*",
        r"(?i)(vital\s*signs|vitals)[:.]?\s*",
        r"(?i)(medications|current\s*medications)[:.]?\s*",
        r"(?i)(allergies)[:.]?\s*",
        r"(?i)(radiology|imaging|x[- ]?ray|ct|mri)[:.]?\s*",
    ]
    
    # Common patterns to remove (noise)
    NOISE_PATTERNS = [
        r"page\s*\d+\s*(of\s*\d+)?",  # Page numbers
        r"confidential",  # Confidentiality notices
        r"printed\s*on.*\d{4}",  # Print dates
        r"^\s*[-_=]{3,}\s*$",  # Separator lines
        r"form\s*#?\s*\d+",  # Form numbers
    ]
    
    # Common medical abbreviations to expand
    ABBREVIATIONS = {
        "pt": "patient",
        "hx": "history",
        "dx": "diagnosis",
        "rx": "prescription",
        "prn": "as needed",
        "bid": "twice daily",
        "tid": "three times daily",
        "qid": "four times daily",
        "qd": "once daily",
        "po": "by mouth",
        "iv": "intravenous",
        "im": "intramuscular",
        "wbc": "white blood cell",
        "rbc": "red blood cell",
        "hgb": "hemoglobin",
        "hct": "hematocrit",
        "plt": "platelet",
        "bp": "blood pressure",
        "hr": "heart rate",
        "temp": "temperature",
        "resp": "respiratory",
        "o2": "oxygen",
        "sat": "saturation",
    }
    
    # Common medical terms for highlighting
    MEDICAL_TERMS = [
        "glucose", "cholesterol", "hemoglobin", "platelet", "creatinine",
        "bilirubin", "albumin", "protein", "sodium", "potassium",
        "calcium", "magnesium", "phosphorus", "chloride", "bicarbonate",
        "urea", "nitrogen", "ast", "alt", "alkaline", "phosphatase",
        "thyroid", "tsh", "t3", "t4", "hba1c", "insulin",
        "cortisol", "testosterone", "estrogen", "vitamin",
        "ferritin", "iron", "transferrin", "ldl", "hdl", "triglyceride",
        "esr", "crp", "ana", "rheumatoid", "antibody",
        "opacity", "nodule", "mass", "infiltrate", "consolidation",
        "effusion", "cardiomegaly", "pneumonia", "fracture",
    ]
    
    def __init__(self):
        # Compile patterns for efficiency
        self._noise_patterns = [
            re.compile(p, re.IGNORECASE | re.MULTILINE)
            for p in self.NOISE_PATTERNS
        ]
        self._section_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in self.SECTION_PATTERNS
        ]
    
    def process(
        self,
        text: str,
        expand_abbreviations: bool = False
    ) -> ProcessedText:
        """
        Process and clean medical text.
        
        Args:
            text: Raw extracted text
            expand_abbreviations: Whether to expand medical abbreviations
            
        Returns:
            ProcessedText with cleaned content and extracted data
        """
        logger.info("Processing text", original_length=len(text))
        
        # Step 1: Basic cleaning
        cleaned = self._basic_clean(text)
        
        # Step 2: Remove noise
        cleaned = self._remove_noise(cleaned)
        
        # Step 3: Normalize whitespace
        cleaned = self._normalize_whitespace(cleaned)
        
        # Step 4: Optionally expand abbreviations
        if expand_abbreviations:
            cleaned = self._expand_abbreviations(cleaned)
        
        # Step 5: Extract sections
        sections = self._extract_sections(cleaned)
        
        # Step 6: Extract key-value pairs
        key_values = self._extract_key_values(cleaned)
        
        # Step 7: Find medical terms
        medical_terms = self._find_medical_terms(cleaned)
        
        result = ProcessedText(
            original_text=text,
            cleaned_text=cleaned,
            sections=sections,
            key_values=key_values,
            word_count=len(cleaned.split()),
            medical_terms_found=medical_terms
        )
        
        logger.info(
            "Text processing complete",
            cleaned_length=len(cleaned),
            sections_found=len(sections),
            key_values_found=len(key_values),
            medical_terms=len(medical_terms)
        )
        
        return result
    
    def _basic_clean(self, text: str) -> str:
        """Perform basic text cleaning."""
        # Remove non-printable characters except newlines
        text = re.sub(r'[^\x20-\x7E\n\t]', ' ', text)
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove excessive blank lines (more than 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
    
    def _remove_noise(self, text: str) -> str:
        """Remove noise patterns from text."""
        for pattern in self._noise_patterns:
            text = pattern.sub('', text)
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        # Replace multiple spaces with single space
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Strip whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        
        # Remove empty lines at start and end
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()
        
        return '\n'.join(lines)
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand medical abbreviations."""
        for abbrev, expansion in self.ABBREVIATIONS.items():
            # Match word boundaries
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            text = re.sub(
                pattern,
                f"{abbrev} ({expansion})",
                text,
                flags=re.IGNORECASE
            )
        return text
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract identified sections from text."""
        sections = {}
        
        # Find all section headers and their positions
        section_matches = []
        for pattern in self._section_patterns:
            for match in pattern.finditer(text):
                section_name = match.group(1).lower().strip()
                section_matches.append((match.start(), match.end(), section_name))
        
        # Sort by position
        section_matches.sort(key=lambda x: x[0])
        
        # Extract content between sections
        for i, (start, end, name) in enumerate(section_matches):
            if i < len(section_matches) - 1:
                next_start = section_matches[i + 1][0]
                content = text[end:next_start].strip()
            else:
                content = text[end:end + 500].strip()  # Take next 500 chars
            
            if content:
                # Normalize section name
                name = name.replace('_', ' ').replace('-', ' ')
                sections[name] = content
        
        return sections
    
    def _extract_key_values(self, text: str) -> Dict[str, str]:
        """Extract key-value pairs (like lab results)."""
        key_values = {}
        
        # Pattern: Key: Value or Key = Value
        patterns = [
            r'([A-Za-z][A-Za-z0-9\s]{1,30}):\s*([^\n:]{1,50})',
            r'([A-Za-z][A-Za-z0-9\s]{1,30})\s*=\s*([^\n=]{1,50})',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for key, value in matches:
                key = key.strip()
                value = value.strip()
                
                # Filter out noise
                if len(key) > 2 and len(value) > 0:
                    # Check if it looks like a valid key-value
                    if not re.match(r'^(and|or|the|for|with)$', key, re.I):
                        key_values[key] = value
        
        return key_values
    
    def _find_medical_terms(self, text: str) -> List[str]:
        """Find medical terms in the text."""
        found_terms = []
        text_lower = text.lower()
        
        for term in self.MEDICAL_TERMS:
            if term in text_lower:
                found_terms.append(term)
        
        return found_terms
    
    def get_summary_context(
        self,
        processed: ProcessedText,
        max_length: int = 4000
    ) -> str:
        """
        Generate optimized context for LLM summarization.
        
        Creates a focused summary of the most important content
        suitable for LLM context window.
        
        Args:
            processed: ProcessedText from process()
            max_length: Maximum context length
            
        Returns:
            Optimized context string
        """
        parts = []
        
        # Include key-value pairs first (usually most important)
        if processed.key_values:
            parts.append("KEY FINDINGS:")
            for key, value in list(processed.key_values.items())[:20]:
                parts.append(f"• {key}: {value}")
        
        # Include sections
        if processed.sections:
            for section_name, content in processed.sections.items():
                parts.append(f"\n{section_name.upper()}:")
                # Truncate long sections
                parts.append(content[:500])
        
        # Add remaining text if space allows
        context = "\n".join(parts)
        
        if len(context) < max_length and processed.cleaned_text:
            remaining = max_length - len(context) - 100
            if remaining > 100:
                parts.append("\nADDITIONAL CONTEXT:")
                parts.append(processed.cleaned_text[:remaining])
        
        return "\n".join(parts)[:max_length]


# Singleton instance
text_processor = TextProcessor()
