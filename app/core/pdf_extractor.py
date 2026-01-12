"""
PDF text extraction for MedExplain AI.

Extracts text content from PDF medical reports using:
- pdfplumber for native PDF text
- pytesseract OCR as fallback for scanned documents
"""

import io
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

import pdfplumber
from PIL import Image

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("pdf_extractor")


@dataclass
class PDFExtractionResult:
    """Result of PDF text extraction."""
    
    text: str
    page_count: int
    extraction_method: str  # 'native' or 'ocr'
    confidence: float  # 0.0 to 1.0
    warnings: list[str]


class PDFExtractor:
    """
    Extracts text from PDF medical reports.
    
    Uses pdfplumber for native text extraction and falls back to
    OCR (pytesseract) when native extraction yields insufficient text.
    """
    
    # Minimum text length to consider native extraction successful
    MIN_TEXT_LENGTH = 50
    
    # OCR settings
    OCR_DPI = 300
    
    def __init__(self):
        self._check_ocr_available()
    
    def _check_ocr_available(self) -> bool:
        """Check if OCR (tesseract) is available."""
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self._ocr_available = True
            logger.info("OCR (Tesseract) is available")
        except Exception as e:
            self._ocr_available = False
            logger.warning(
                "OCR (Tesseract) not available, OCR fallback disabled",
                error=str(e)
            )
        return self._ocr_available
    
    def extract_text(
        self, 
        file_content: bytes,
        filename: str = "document.pdf"
    ) -> PDFExtractionResult:
        """
        Extract text from a PDF file.
        
        Attempts native extraction first, then falls back to OCR
        if the native extraction yields insufficient text.
        
        Args:
            file_content: Raw PDF bytes
            filename: Original filename for logging
            
        Returns:
            PDFExtractionResult with extracted text and metadata
        """
        warnings = []
        
        logger.info("Starting PDF extraction", filename=filename)
        
        # Try native extraction first
        try:
            text, page_count = self._extract_native(file_content)
            
            if len(text.strip()) >= self.MIN_TEXT_LENGTH:
                logger.info(
                    "Native PDF extraction successful",
                    filename=filename,
                    text_length=len(text),
                    page_count=page_count
                )
                return PDFExtractionResult(
                    text=text.strip(),
                    page_count=page_count,
                    extraction_method="native",
                    confidence=0.9,  # High confidence for native text
                    warnings=warnings
                )
            else:
                warnings.append(
                    "Native extraction yielded minimal text, attempting OCR"
                )
                logger.info("Native extraction minimal, trying OCR", filename=filename)
                
        except Exception as e:
            warnings.append(f"Native extraction failed: {str(e)}")
            logger.warning(
                "Native PDF extraction failed",
                filename=filename,
                error=str(e)
            )
        
        # Fall back to OCR
        if self._ocr_available:
            try:
                text, page_count = self._extract_ocr(file_content)
                
                if len(text.strip()) >= self.MIN_TEXT_LENGTH:
                    logger.info(
                        "OCR extraction successful",
                        filename=filename,
                        text_length=len(text),
                        page_count=page_count
                    )
                    return PDFExtractionResult(
                        text=text.strip(),
                        page_count=page_count,
                        extraction_method="ocr",
                        confidence=0.7,  # Lower confidence for OCR
                        warnings=warnings
                    )
                else:
                    warnings.append("OCR extraction also yielded minimal text")
                    
            except Exception as e:
                warnings.append(f"OCR extraction failed: {str(e)}")
                logger.warning(
                    "OCR extraction failed",
                    filename=filename,
                    error=str(e)
                )
        else:
            warnings.append("OCR not available for fallback")
        
        # Return whatever we have with low confidence
        logger.warning(
            "PDF extraction yielded poor results",
            filename=filename,
            warnings=warnings
        )
        
        return PDFExtractionResult(
            text=text.strip() if 'text' in dir() else "",
            page_count=page_count if 'page_count' in dir() else 0,
            extraction_method="failed",
            confidence=0.2,
            warnings=warnings
        )
    
    def _extract_native(
        self, 
        file_content: bytes
    ) -> Tuple[str, int]:
        """
        Extract text using pdfplumber (native PDF text).
        
        Args:
            file_content: Raw PDF bytes
            
        Returns:
            Tuple of (extracted_text, page_count)
        """
        all_text = []
        
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            page_count = len(pdf.pages)
            
            for i, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text() or ""
                    
                    # Also try extracting tables
                    tables = page.extract_tables()
                    table_text = self._tables_to_text(tables)
                    
                    if page_text:
                        all_text.append(f"--- Page {i + 1} ---")
                        all_text.append(page_text)
                    
                    if table_text:
                        all_text.append("--- Tables ---")
                        all_text.append(table_text)
                        
                except Exception as e:
                    logger.warning(
                        "Error extracting page",
                        page=i + 1,
                        error=str(e)
                    )
        
        return "\n\n".join(all_text), page_count
    
    def _extract_ocr(
        self, 
        file_content: bytes
    ) -> Tuple[str, int]:
        """
        Extract text using OCR (for scanned PDFs).
        
        Args:
            file_content: Raw PDF bytes
            
        Returns:
            Tuple of (extracted_text, page_count)
        """
        import pytesseract
        from pdf2image import convert_from_bytes
        
        all_text = []
        
        # Convert PDF pages to images
        images = convert_from_bytes(
            file_content,
            dpi=self.OCR_DPI,
            fmt='PNG'
        )
        
        page_count = len(images)
        
        for i, image in enumerate(images):
            try:
                # Run OCR on the image
                page_text = pytesseract.image_to_string(
                    image,
                    lang='eng',  # English - can be made configurable
                    config='--psm 6'  # Assume uniform block of text
                )
                
                if page_text.strip():
                    all_text.append(f"--- Page {i + 1} ---")
                    all_text.append(page_text.strip())
                    
            except Exception as e:
                logger.warning(
                    "OCR error on page",
                    page=i + 1,
                    error=str(e)
                )
        
        return "\n\n".join(all_text), page_count
    
    def _tables_to_text(self, tables: list) -> str:
        """Convert extracted tables to text format."""
        if not tables:
            return ""
        
        text_parts = []
        
        for table in tables:
            if not table:
                continue
                
            for row in table:
                if row:
                    # Clean and join cells
                    cells = [str(cell or "").strip() for cell in row]
                    text_parts.append(" | ".join(cells))
        
        return "\n".join(text_parts)
    
    def extract_from_file(
        self,
        file_path: Path
    ) -> PDFExtractionResult:
        """
        Extract text from a PDF file path.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            PDFExtractionResult
        """
        with open(file_path, 'rb') as f:
            content = f.read()
        
        return self.extract_text(content, filename=file_path.name)


# Singleton instance
pdf_extractor = PDFExtractor()
