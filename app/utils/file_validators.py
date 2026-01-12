"""
File validation utilities for MedExplain AI.

Handles validation of uploaded files including:
- File size limits
- File extension validation
- Content type verification
- Corruption detection
"""

import io
from pathlib import Path
from typing import BinaryIO, Optional, Tuple

from PIL import Image
import pdfplumber

from app.config import settings


class FileValidationError(Exception):
    """Raised when file validation fails."""
    
    def __init__(self, message: str, error_code: str = "VALIDATION_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class FileValidator:
    """
    Validates uploaded files for the MedExplain AI system.
    
    Ensures files are:
    - Within size limits
    - Have allowed extensions
    - Are not corrupt
    - Match declared content type
    """
    
    # MIME types for different file categories
    PDF_MIME_TYPES = {"application/pdf"}
    IMAGE_MIME_TYPES = {
        "image/png",
        "image/jpeg", 
        "image/jpg",
        "image/dicom",
        "application/dicom"
    }
    TEXT_MIME_TYPES = {
        "text/plain",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    }
    
    def __init__(self):
        self.max_file_size = settings.max_file_size_bytes
        self.pdf_extensions = settings.pdf_extensions
        self.image_extensions = settings.image_extensions
        self.text_extensions = settings.text_extensions
    
    def validate_file_size(self, file_content: bytes, filename: str) -> bool:
        """
        Check if file is within size limits.
        
        Args:
            file_content: Raw file bytes
            filename: Original filename for error messages
            
        Returns:
            True if valid
            
        Raises:
            FileValidationError: If file exceeds size limit
        """
        if len(file_content) > self.max_file_size:
            raise FileValidationError(
                f"File '{filename}' exceeds maximum size of {settings.max_file_size_mb}MB",
                error_code="FILE_TOO_LARGE"
            )
        return True
    
    def validate_extension(
        self, 
        filename: str, 
        allowed_extensions: list[str]
    ) -> bool:
        """
        Check if file has an allowed extension.
        
        Args:
            filename: Original filename
            allowed_extensions: List of allowed extensions (with dots)
            
        Returns:
            True if valid
            
        Raises:
            FileValidationError: If extension not allowed
        """
        ext = Path(filename).suffix.lower()
        if ext not in allowed_extensions:
            raise FileValidationError(
                f"File extension '{ext}' not allowed. "
                f"Allowed: {', '.join(allowed_extensions)}",
                error_code="INVALID_EXTENSION"
            )
        return True
    
    def detect_mime_type(self, file_content: bytes) -> str:
        """
        Detect the MIME type of file content using file signatures.
        
        Args:
            file_content: Raw file bytes
            
        Returns:
            Detected MIME type string
        """
        return self._detect_mime_from_header(file_content)
    
    def _detect_mime_from_header(self, content: bytes) -> str:
        """Fallback MIME detection using file signatures."""
        if content[:4] == b'%PDF':
            return 'application/pdf'
        if content[:8] == b'\x89PNG\r\n\x1a\n':
            return 'image/png'
        if content[:2] == b'\xff\xd8':
            return 'image/jpeg'
        return 'application/octet-stream'
    
    def validate_pdf(
        self, 
        file_content: bytes, 
        filename: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a PDF file.
        
        Checks:
        - Extension is .pdf
        - File size is within limits
        - PDF is not corrupt
        - PDF has readable content
        
        Args:
            file_content: Raw PDF bytes
            filename: Original filename
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check extension
            self.validate_extension(filename, self.pdf_extensions)
            
            # Check size
            self.validate_file_size(file_content, filename)
            
            # Try to open and read the PDF
            with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                if len(pdf.pages) == 0:
                    raise FileValidationError(
                        "PDF has no pages",
                        error_code="EMPTY_PDF"
                    )
                
                # Try to extract text from first page to verify readability
                first_page = pdf.pages[0]
                _ = first_page.extract_text()
                
            return True, None
            
        except FileValidationError as e:
            return False, e.message
        except Exception as e:
            return False, f"PDF validation failed: {str(e)}"
    
    def validate_image(
        self, 
        file_content: bytes, 
        filename: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate an image file (X-ray, CT scan, etc.).
        
        Checks:
        - Extension is allowed
        - File size is within limits
        - Image is not corrupt
        - Image has valid dimensions
        
        Args:
            file_content: Raw image bytes
            filename: Original filename
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check extension
            self.validate_extension(filename, self.image_extensions)
            
            # Check size
            self.validate_file_size(file_content, filename)
            
            # Try to open and verify the image
            img = Image.open(io.BytesIO(file_content))
            img.verify()  # Verify image integrity
            
            # Re-open to get dimensions (verify() closes the file)
            img = Image.open(io.BytesIO(file_content))
            width, height = img.size
            
            # Check for reasonable dimensions (not too small)
            if width < 50 or height < 50:
                raise FileValidationError(
                    "Image dimensions too small for analysis",
                    error_code="IMAGE_TOO_SMALL"
                )
            
            # Check for reasonable dimensions (not too large)
            if width > 10000 or height > 10000:
                raise FileValidationError(
                    "Image dimensions too large",
                    error_code="IMAGE_TOO_LARGE"
                )
            
            return True, None
            
        except FileValidationError as e:
            return False, e.message
        except Exception as e:
            return False, f"Image validation failed: {str(e)}"
    
    def validate_text(
        self, 
        file_content: bytes, 
        filename: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a text file.
        
        Checks:
        - Extension is allowed
        - File size is within limits
        - Content is valid text (decodeable)
        
        Args:
            file_content: Raw file bytes  
            filename: Original filename
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check extension
            self.validate_extension(filename, self.text_extensions)
            
            # Check size
            self.validate_file_size(file_content, filename)
            
            # For .txt files, try to decode as text
            ext = Path(filename).suffix.lower()
            if ext == '.txt':
                try:
                    file_content.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        file_content.decode('latin-1')
                    except UnicodeDecodeError:
                        raise FileValidationError(
                            "Text file encoding not supported",
                            error_code="INVALID_ENCODING"
                        )
            
            return True, None
            
        except FileValidationError as e:
            return False, e.message
        except Exception as e:
            return False, f"Text validation failed: {str(e)}"
    
    def get_file_type(self, filename: str) -> str:
        """
        Determine the type of file based on extension.
        
        Args:
            filename: Original filename
            
        Returns:
            One of: 'pdf', 'image', 'text', 'unknown'
        """
        ext = Path(filename).suffix.lower()
        
        if ext in self.pdf_extensions:
            return 'pdf'
        if ext in self.image_extensions:
            return 'image'
        if ext in self.text_extensions:
            return 'text'
        return 'unknown'
    
    def validate(
        self, 
        file_content: bytes, 
        filename: str
    ) -> Tuple[bool, Optional[str], str]:
        """
        Validate any supported file type.
        
        Automatically detects file type and applies appropriate validation.
        
        Args:
            file_content: Raw file bytes
            filename: Original filename
            
        Returns:
            Tuple of (is_valid, error_message, file_type)
        """
        file_type = self.get_file_type(filename)
        
        if file_type == 'pdf':
            is_valid, error = self.validate_pdf(file_content, filename)
        elif file_type == 'image':
            is_valid, error = self.validate_image(file_content, filename)
        elif file_type == 'text':
            is_valid, error = self.validate_text(file_content, filename)
        else:
            return False, f"Unsupported file type: {filename}", "unknown"
        
        return is_valid, error, file_type


# Singleton instance for easy access
file_validator = FileValidator()
