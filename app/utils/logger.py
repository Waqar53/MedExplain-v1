"""
Logging configuration for MedExplain AI.

Uses structlog for structured JSON logging suitable for production.
"""

import logging
import sys
from typing import Optional

import structlog
from structlog.types import Processor

from app.config import settings


def configure_logging(
    log_level: Optional[str] = None,
    json_format: bool = True
) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        log_level: Override log level (defaults to settings.log_level)
        json_format: Whether to output JSON (True) or console format (False)
    """
    level = log_level or settings.log_level
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Shared processors for all configurations
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]
    
    if json_format:
        # Production: JSON output
        processors: list[Processor] = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ]
    else:
        # Development: Pretty console output
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Also configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
    )


def get_logger(name: str = "medexplain") -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name for identification
        
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


# Configure logging on module import (can be reconfigured later)
configure_logging(
    log_level=settings.log_level,
    json_format=not settings.debug  # Console format in debug mode
)

# Default logger instance
logger = get_logger()
