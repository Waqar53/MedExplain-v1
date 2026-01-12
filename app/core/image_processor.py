"""
Image processing for MedExplain AI.

Handles loading, preprocessing, and validation of medical images
including X-rays, CT scans, and other diagnostic images.
"""

import io
from pathlib import Path
from typing import Optional, Tuple, Union
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("image_processor")


@dataclass
class ImageData:
    """Processed image data."""
    
    original_shape: Tuple[int, int, int]  # (height, width, channels)
    processed_shape: Tuple[int, int, int]
    image_array: np.ndarray  # Processed numpy array
    pil_image: Image.Image  # PIL Image for compatibility
    format: str  # Original format (PNG, JPEG, etc.)
    is_grayscale: bool
    preprocessing_applied: list[str]


class ImageProcessor:
    """
    Processes medical images for analysis.
    
    Handles:
    - Loading from bytes or file path
    - Format detection and conversion
    - Resizing to model input size
    - Normalization for neural network input
    - Grayscale conversion for X-rays
    """
    
    # Standard size for vision model input
    MODEL_INPUT_SIZE = (224, 224)  # Standard for ResNet/DenseNet
    
    # Supported formats
    SUPPORTED_FORMATS = {'PNG', 'JPEG', 'JPG', 'BMP', 'TIFF'}
    
    def __init__(self):
        self.input_size = self.MODEL_INPUT_SIZE
    
    def load_image(
        self,
        source: Union[bytes, Path, str],
        filename: Optional[str] = None
    ) -> ImageData:
        """
        Load and preprocess an image.
        
        Args:
            source: Image bytes, file path, or path string
            filename: Optional filename for logging
            
        Returns:
            ImageData with processed image
        """
        preprocessing_steps = []
        
        # Load image based on source type
        if isinstance(source, bytes):
            pil_image = Image.open(io.BytesIO(source))
            logger.info("Loaded image from bytes", size=len(source))
        else:
            path = Path(source) if isinstance(source, str) else source
            pil_image = Image.open(path)
            filename = filename or path.name
            logger.info("Loaded image from file", path=str(path))
        
        # Get original format
        original_format = pil_image.format or "UNKNOWN"
        
        # Convert RGBA to RGB if needed
        if pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB')
            preprocessing_steps.append("RGBA_to_RGB")
        
        # Determine if grayscale
        is_grayscale = pil_image.mode in ('L', '1', 'LA')
        
        # Get original shape
        if is_grayscale:
            original_shape = (pil_image.height, pil_image.width, 1)
        elif pil_image.mode == 'RGB':
            original_shape = (pil_image.height, pil_image.width, 3)
        else:
            pil_image = pil_image.convert('RGB')
            original_shape = (pil_image.height, pil_image.width, 3)
            preprocessing_steps.append(f"{pil_image.mode}_to_RGB")
        
        # Convert to numpy array
        image_array = np.array(pil_image)
        
        logger.info(
            "Image loaded",
            filename=filename,
            original_shape=original_shape,
            format=original_format,
            is_grayscale=is_grayscale
        )
        
        return ImageData(
            original_shape=original_shape,
            processed_shape=original_shape,
            image_array=image_array,
            pil_image=pil_image,
            format=original_format,
            is_grayscale=is_grayscale,
            preprocessing_applied=preprocessing_steps
        )
    
    def preprocess_for_model(
        self,
        image_data: ImageData,
        target_size: Optional[Tuple[int, int]] = None
    ) -> ImageData:
        """
        Preprocess image for neural network input.
        
        Applies:
        - Resizing to target size
        - Grayscale to RGB conversion if needed
        - Normalization
        
        Args:
            image_data: Loaded ImageData
            target_size: (height, width) tuple, defaults to MODEL_INPUT_SIZE
            
        Returns:
            Preprocessed ImageData
        """
        target_size = target_size or self.MODEL_INPUT_SIZE
        preprocessing_steps = image_data.preprocessing_applied.copy()
        
        pil_image = image_data.pil_image.copy()
        
        # Ensure RGB mode for model
        if image_data.is_grayscale:
            pil_image = pil_image.convert('RGB')
            preprocessing_steps.append("grayscale_to_RGB")
        elif pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
            preprocessing_steps.append(f"{pil_image.mode}_to_RGB")
        
        # Resize maintaining aspect ratio with padding
        pil_image = self._resize_with_padding(pil_image, target_size)
        preprocessing_steps.append(f"resize_to_{target_size}")
        
        # Convert to numpy
        image_array = np.array(pil_image)
        
        # Normalize to [0, 1]
        image_array = image_array.astype(np.float32) / 255.0
        preprocessing_steps.append("normalize_0_1")
        
        processed_shape = (target_size[0], target_size[1], 3)
        
        logger.info(
            "Image preprocessed for model",
            original_shape=image_data.original_shape,
            processed_shape=processed_shape,
            steps=preprocessing_steps
        )
        
        return ImageData(
            original_shape=image_data.original_shape,
            processed_shape=processed_shape,
            image_array=image_array,
            pil_image=pil_image,
            format=image_data.format,
            is_grayscale=image_data.is_grayscale,
            preprocessing_applied=preprocessing_steps
        )
    
    def _resize_with_padding(
        self,
        image: Image.Image,
        target_size: Tuple[int, int]
    ) -> Image.Image:
        """
        Resize image maintaining aspect ratio with black padding.
        
        Args:
            image: PIL Image
            target_size: (height, width)
            
        Returns:
            Resized PIL Image
        """
        target_height, target_width = target_size
        
        # Calculate scaling factor
        width_ratio = target_width / image.width
        height_ratio = target_height / image.height
        scale = min(width_ratio, height_ratio)
        
        # Calculate new dimensions
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)
        
        # Resize
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create padded image
        padded = Image.new('RGB', (target_width, target_height), (0, 0, 0))
        
        # Calculate padding
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        
        padded.paste(resized, (paste_x, paste_y))
        
        return padded
    
    def enhance_medical_image(
        self,
        image_data: ImageData
    ) -> ImageData:
        """
        Apply medical image-specific enhancements.
        
        Optimized for X-rays and diagnostic images:
        - Contrast enhancement (CLAHE)
        - Noise reduction
        - Edge enhancement
        
        Args:
            image_data: Loaded ImageData
            
        Returns:
            Enhanced ImageData
        """
        preprocessing_steps = image_data.preprocessing_applied.copy()
        
        # Convert to OpenCV format
        if image_data.is_grayscale:
            cv_image = cv2.cvtColor(
                np.array(image_data.pil_image.convert('RGB')),
                cv2.COLOR_RGB2GRAY
            )
        else:
            cv_image = cv2.cvtColor(
                np.array(image_data.pil_image),
                cv2.COLOR_RGB2GRAY
            )
        preprocessing_steps.append("convert_to_grayscale")
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(cv_image)
        preprocessing_steps.append("CLAHE_contrast")
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        preprocessing_steps.append("denoise")
        
        # Convert back to RGB for model compatibility
        rgb_enhanced = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
        
        # Convert to PIL
        pil_enhanced = Image.fromarray(rgb_enhanced)
        
        logger.info("Medical image enhancement applied", steps=preprocessing_steps)
        
        return ImageData(
            original_shape=image_data.original_shape,
            processed_shape=(cv_image.shape[0], cv_image.shape[1], 3),
            image_array=rgb_enhanced,
            pil_image=pil_enhanced,
            format=image_data.format,
            is_grayscale=True,  # Now converted to grayscale
            preprocessing_applied=preprocessing_steps
        )
    
    def get_image_tensor(
        self,
        image_data: ImageData
    ) -> np.ndarray:
        """
        Get image as tensor ready for PyTorch model.
        
        Returns numpy array in (C, H, W) format normalized.
        
        Args:
            image_data: Preprocessed ImageData
            
        Returns:
            Numpy array in (channels, height, width) format
        """
        # Ensure preprocessed for model
        if image_data.processed_shape != (224, 224, 3):
            image_data = self.preprocess_for_model(image_data)
        
        # Get array and transpose to (C, H, W)
        array = image_data.image_array
        
        # If already normalized (float32)
        if array.dtype != np.float32:
            array = array.astype(np.float32) / 255.0
        
        # Transpose from (H, W, C) to (C, H, W)
        tensor = np.transpose(array, (2, 0, 1))
        
        # Add batch dimension
        tensor = np.expand_dims(tensor, axis=0)
        
        return tensor
    
    def save_processed(
        self,
        image_data: ImageData,
        output_path: Path,
        format: str = "PNG"
    ) -> Path:
        """
        Save processed image to file.
        
        Args:
            image_data: ImageData to save
            output_path: Output file path
            format: Output format (PNG, JPEG)
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        image_data.pil_image.save(output_path, format=format)
        
        logger.info("Saved processed image", path=str(output_path))
        
        return output_path


# Singleton instance
image_processor = ImageProcessor()
