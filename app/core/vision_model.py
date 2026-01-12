"""
Vision model for MedExplain AI.

Provides X-ray and medical image analysis using pre-trained
neural networks with feature extraction for LLM context.
"""

import io
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from app.config import settings
from app.core.image_processor import ImageData, image_processor
from app.utils.logger import get_logger

logger = get_logger("vision_model")


@dataclass
class VisionAnalysisResult:
    """Result of vision model analysis."""
    
    # Feature analysis
    features: np.ndarray  # Extracted feature vector
    feature_summary: str  # Text summary for LLM
    
    # Pattern detection
    detected_patterns: List[str]
    pattern_confidences: Dict[str, float]
    
    # Overall confidence
    confidence_score: float
    analysis_notes: List[str]


class VisionModel:
    """
    Vision model for medical image analysis.
    
    Uses pre-trained models (ResNet/DenseNet) for feature extraction
    and pattern recognition. Designed specifically for X-ray and
    diagnostic image analysis.
    
    NOTE: This is NOT a diagnostic tool. It provides informational
    analysis only and should never be used for medical decisions.
    """
    
    # Image preprocessing for PyTorch models
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    # Common patterns to describe in X-rays (informational only)
    PATTERN_DESCRIPTIONS = {
        "clear": "The image appears relatively clear with expected structures visible.",
        "opacity": "Some opacity or cloudiness may be present in certain areas.",
        "structure": "Normal structural elements are visible in the image.",
        "contrast": "The image shows varying levels of contrast between tissues.",
        "symmetric": "Bilateral structures appear relatively symmetric.",
        "irregular": "Some irregular patterns may be visible and require professional review."
    }
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize vision model.
        
        Args:
            model_name: Model to use ('resnet50', 'densenet121')
        """
        self.model_name = model_name or settings.vision_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.feature_extractor = None
        
        logger.info(
            "Initializing vision model",
            model=self.model_name,
            device=str(self.device)
        )
        
        self._load_model()
        self._setup_preprocessing()
    
    def _load_model(self) -> None:
        """Load the pre-trained model."""
        try:
            if self.model_name == 'resnet50':
                # Load ResNet50 with pretrained weights
                self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                # Remove final classification layer for feature extraction
                self.feature_extractor = nn.Sequential(
                    *list(self.model.children())[:-1]
                )
                self.feature_dim = 2048
                
            elif self.model_name == 'densenet121':
                # Load DenseNet121 - good for medical imaging
                self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
                self.feature_extractor = self.model.features
                self.feature_dim = 1024
                
            else:
                # Default to ResNet50
                logger.warning(f"Unknown model {self.model_name}, using ResNet50")
                self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                self.feature_extractor = nn.Sequential(
                    *list(self.model.children())[:-1]
                )
                self.feature_dim = 2048
            
            # Move to device and set to eval mode
            self.model = self.model.to(self.device)
            self.feature_extractor = self.feature_extractor.to(self.device)
            self.model.eval()
            self.feature_extractor.eval()
            
            logger.info(
                "Vision model loaded successfully",
                model=self.model_name,
                feature_dim=self.feature_dim
            )
            
        except Exception as e:
            logger.error("Failed to load vision model", error=str(e))
            raise
    
    def _setup_preprocessing(self) -> None:
        """Setup image preprocessing pipeline."""
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.IMAGENET_MEAN,
                std=self.IMAGENET_STD
            )
        ])
    
    def analyze_image(
        self,
        image_source: Union[bytes, Path, ImageData],
        filename: Optional[str] = None
    ) -> VisionAnalysisResult:
        """
        Analyze a medical image.
        
        Args:
            image_source: Image bytes, path, or ImageData
            filename: Optional filename for logging
            
        Returns:
            VisionAnalysisResult with analysis
        """
        analysis_notes = []
        
        try:
            # Load image if needed
            if isinstance(image_source, ImageData):
                image_data = image_source
            else:
                image_data = image_processor.load_image(image_source, filename)
            
            # Get PIL image
            pil_image = image_data.pil_image
            
            # Ensure RGB
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Preprocess for model
            input_tensor = self.preprocess(pil_image)
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.feature_extractor(input_batch)
                
                # Flatten features
                if len(features.shape) > 2:
                    features = features.mean([2, 3])  # Global average pooling
                
                features_np = features.cpu().numpy().flatten()
            
            # Analyze patterns
            detected_patterns, pattern_confidences = self._analyze_patterns(
                features_np,
                image_data
            )
            
            # Calculate overall confidence
            confidence_score = self._calculate_confidence(
                features_np,
                image_data,
                detected_patterns
            )
            
            # Generate feature summary for LLM
            feature_summary = self._generate_feature_summary(
                image_data,
                detected_patterns,
                confidence_score
            )
            
            # Add notes about the analysis
            analysis_notes.append("Image analyzed using pre-trained neural network")
            analysis_notes.append(f"Model: {self.model_name}")
            
            if image_data.is_grayscale:
                analysis_notes.append("Image appears to be a grayscale medical image")
            
            logger.info(
                "Vision analysis complete",
                filename=filename,
                confidence=confidence_score,
                patterns=list(detected_patterns)
            )
            
            return VisionAnalysisResult(
                features=features_np,
                feature_summary=feature_summary,
                detected_patterns=detected_patterns,
                pattern_confidences=pattern_confidences,
                confidence_score=confidence_score,
                analysis_notes=analysis_notes
            )
            
        except Exception as e:
            logger.error("Vision analysis failed", error=str(e))
            
            # Return low-confidence result on error
            return VisionAnalysisResult(
                features=np.zeros(self.feature_dim),
                feature_summary="Unable to analyze image due to processing error.",
                detected_patterns=["error"],
                pattern_confidences={"error": 1.0},
                confidence_score=0.1,
                analysis_notes=[f"Analysis error: {str(e)}"]
            )
    
    def _analyze_patterns(
        self,
        features: np.ndarray,
        image_data: ImageData
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Analyze image for common patterns.
        
        This is a simplified pattern detection based on image
        characteristics. NOT for diagnostic use.
        
        Args:
            features: Extracted feature vector
            image_data: Image data
            
        Returns:
            Tuple of (pattern_list, confidence_dict)
        """
        patterns = []
        confidences = {}
        
        # Analyze image statistics
        img_array = np.array(image_data.pil_image.convert('L'))  # Grayscale
        
        mean_intensity = np.mean(img_array)
        std_intensity = np.std(img_array)
        
        # Pattern: Clear/Structure visibility
        if std_intensity > 40:
            patterns.append("structure")
            confidences["structure"] = min(std_intensity / 60, 1.0)
        
        # Pattern: Contrast levels
        if std_intensity > 50:
            patterns.append("contrast")
            confidences["contrast"] = min(std_intensity / 70, 1.0)
        
        # Pattern: Overall clarity
        if mean_intensity > 100 and mean_intensity < 180:
            patterns.append("clear")
            confidences["clear"] = 0.7
        
        # Pattern: Check for very dark or very bright areas
        dark_ratio = np.mean(img_array < 30)
        bright_ratio = np.mean(img_array > 220)
        
        if dark_ratio > 0.3 or bright_ratio > 0.3:
            patterns.append("opacity")
            confidences["opacity"] = max(dark_ratio, bright_ratio)
        
        # Pattern: Symmetry check (simplified)
        h, w = img_array.shape
        left_half = img_array[:, :w//2]
        right_half = np.fliplr(img_array[:, w//2:])
        
        # Handle size mismatch
        min_w = min(left_half.shape[1], right_half.shape[1])
        symmetry_score = 1 - np.mean(np.abs(
            left_half[:, :min_w].astype(float) - 
            right_half[:, :min_w].astype(float)
        )) / 255
        
        if symmetry_score > 0.7:
            patterns.append("symmetric")
            confidences["symmetric"] = symmetry_score
        
        # If nothing detected, note as irregular
        if not patterns:
            patterns.append("irregular")
            confidences["irregular"] = 0.5
        
        return patterns, confidences
    
    def _calculate_confidence(
        self,
        features: np.ndarray,
        image_data: ImageData,
        patterns: List[str]
    ) -> float:
        """
        Calculate overall analysis confidence.
        
        Args:
            features: Feature vector
            image_data: Image data
            patterns: Detected patterns
            
        Returns:
            Confidence score 0.0 to 1.0
        """
        confidence = 0.5  # Base confidence
        
        # Boost for good image quality indicators
        if image_data.processed_shape[0] >= 224 and image_data.processed_shape[1] >= 224:
            confidence += 0.1
        
        # Boost for clear patterns
        if "clear" in patterns or "structure" in patterns:
            confidence += 0.15
        
        # Reduce for unclear patterns
        if "irregular" in patterns:
            confidence -= 0.1
        
        # Reduce for errors or issues
        if "error" in patterns:
            confidence = 0.1
        
        # Feature norm check (very low norm might indicate issues)
        feature_norm = np.linalg.norm(features)
        if feature_norm < 1.0:
            confidence -= 0.2
        
        return max(0.1, min(1.0, confidence))
    
    def _generate_feature_summary(
        self,
        image_data: ImageData,
        patterns: List[str],
        confidence: float
    ) -> str:
        """
        Generate text summary for LLM context.
        
        Args:
            image_data: Image data
            patterns: Detected patterns
            confidence: Confidence score
            
        Returns:
            Text summary string
        """
        summary_parts = [
            "IMAGE ANALYSIS SUMMARY:",
            f"- Image dimensions: {image_data.original_shape[1]}x{image_data.original_shape[0]}",
            f"- Image type: {'Grayscale' if image_data.is_grayscale else 'Color'}",
            f"- Analysis confidence: {confidence:.1%}",
            "",
            "OBSERVED CHARACTERISTICS:"
        ]
        
        for pattern in patterns:
            if pattern in self.PATTERN_DESCRIPTIONS:
                summary_parts.append(f"- {self.PATTERN_DESCRIPTIONS[pattern]}")
        
        summary_parts.extend([
            "",
            "IMPORTANT NOTES:",
            "- This is an automated preliminary analysis only.",
            "- This is NOT a medical diagnosis.",
            "- A qualified healthcare professional must interpret medical images.",
            "- Image quality and acquisition factors may affect this analysis."
        ])
        
        return "\n".join(summary_parts)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "feature_dim": self.feature_dim,
            "device": str(self.device),
            "input_size": "224x224",
            "preprocessing": "ImageNet normalization"
        }


# Lazy-loaded singleton
_vision_model: Optional[VisionModel] = None


def get_vision_model() -> VisionModel:
    """Get or create vision model singleton."""
    global _vision_model
    if _vision_model is None:
        _vision_model = VisionModel()
    return _vision_model
