"""
Configuration management for MedExplain AI.

Uses Pydantic Settings for type-safe environment variable handling.
All configuration is loaded from environment variables or .env file.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # ==========================================================================
    # Application
    # ==========================================================================
    app_name: str = "MedExplain AI"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    
    # ==========================================================================
    # Server
    # ==========================================================================
    host: str = "0.0.0.0"
    port: int = 8000
    
    # ==========================================================================
    # LLM Configuration
    # ==========================================================================
    openai_api_key: str = ""
    openai_model: str = "gpt-4-turbo-preview"
    llm_provider: Literal["openai", "ollama"] = "openai"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama2"
    
    # ==========================================================================
    # Rate Limiting
    # ==========================================================================
    rate_limit_per_minute: int = 30
    rate_limit_per_hour: int = 200
    
    # ==========================================================================
    # File Upload
    # ==========================================================================
    max_file_size_mb: int = 10
    allowed_pdf_extensions: str = ".pdf"
    allowed_image_extensions: str = ".png,.jpg,.jpeg,.dicom,.dcm"
    allowed_text_extensions: str = ".txt,.doc,.docx"
    
    # ==========================================================================
    # Model Configuration
    # ==========================================================================
    vision_model: str = "resnet50"
    confidence_threshold_low: float = 0.3
    confidence_threshold_high: float = 0.7
    
    # ==========================================================================
    # Output Settings
    # ==========================================================================
    output_dir: str = "outputs"
    temp_dir: str = "temp"
    
    # ==========================================================================
    # Multi-Tenant
    # ==========================================================================
    enable_multi_tenant: bool = True
    default_tenant_id: str = "default-clinic"
    
    # ==========================================================================
    # Computed Properties
    # ==========================================================================
    @property
    def max_file_size_bytes(self) -> int:
        """Maximum file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024
    
    @property
    def pdf_extensions(self) -> list[str]:
        """List of allowed PDF extensions."""
        return [ext.strip() for ext in self.allowed_pdf_extensions.split(",")]
    
    @property
    def image_extensions(self) -> list[str]:
        """List of allowed image extensions."""
        return [ext.strip() for ext in self.allowed_image_extensions.split(",")]
    
    @property
    def text_extensions(self) -> list[str]:
        """List of allowed text extensions."""
        return [ext.strip() for ext in self.allowed_text_extensions.split(",")]
    
    @property
    def output_path(self) -> Path:
        """Path to output directory."""
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def temp_path(self) -> Path:
        """Path to temporary directory."""
        path = Path(self.temp_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()


# Convenience access
settings = get_settings()
