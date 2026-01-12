"""
MedExplain AI - FastAPI Application

Production-grade medical report explanation system.
Helps doctors and clinics explain medical reports to patients
in simple, safe, non-alarming language.

IMPORTANT: This is NOT a diagnostic tool.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.api.routes import router
from app.api.middleware import (
    TenantMiddleware,
    RequestLoggingMiddleware,
    ErrorHandlingMiddleware,
    setup_rate_limiting
)
from app.utils.logger import get_logger, configure_logging

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    logger.info(
        "Starting MedExplain AI",
        version=settings.app_version,
        debug=settings.debug
    )
    
    # Ensure output directories exist
    settings.output_path.mkdir(parents=True, exist_ok=True)
    settings.temp_path.mkdir(parents=True, exist_ok=True)
    
    # Configure logging based on mode
    configure_logging(
        log_level=settings.log_level,
        json_format=not settings.debug
    )
    
    logger.info("Application ready")
    
    yield
    
    # Shutdown
    logger.info("Shutting down MedExplain AI")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI instance
    """
    app = FastAPI(
        title=settings.app_name,
        description="""
## MedExplain AI - Medical Report Explanation System

A production-grade AI system that helps doctors and clinics explain medical reports 
and images to patients in simple, safe, non-alarming language.

### ⚠️ Important Disclaimer

**This is NOT a diagnostic tool.** This system:
- Does NOT provide medical diagnoses
- Does NOT replace professional medical advice
- Should NOT be used for medical decisions
- Always recommends consulting healthcare providers

### Features

- **PDF Report Analysis**: Upload lab reports and receive patient-friendly explanations
- **Text Report Processing**: Analyze text-based medical reports
- **X-ray Analysis**: Basic informational analysis of medical images
- **Professional PDF Reports**: Generate downloadable patient reports
- **Safety-First Design**: All outputs include appropriate disclaimers

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload-report` | POST | Upload PDF/text report |
| `/upload-xray` | POST | Upload X-ray image |
| `/generate-report` | POST | Generate explanation |
| `/create-pdf` | POST | Create PDF report |
| `/download-pdf/{id}` | GET | Download PDF |
| `/health` | GET | Health check |
        """,
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else "/docs",
        redoc_url="/redoc" if settings.debug else None,
    )
    
    # Setup middleware (order matters - first added is outermost)
    
    # Error handling (outermost)
    app.add_middleware(ErrorHandlingMiddleware)
    
    # Request logging
    app.add_middleware(RequestLoggingMiddleware)
    
    # Tenant support
    app.add_middleware(TenantMiddleware)
    
    # CORS - Allow all origins for web frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Setup rate limiting
    setup_rate_limiting(app)
    
    # Include API routes
    app.include_router(router, tags=["API"])
    
    # Mount static frontend files
    frontend_path = Path(__file__).parent.parent / "frontend"
    if frontend_path.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")
        
        # Serve index.html at root
        from fastapi.responses import FileResponse
        
        @app.get("/", include_in_schema=False)
        async def serve_frontend():
            return FileResponse(str(frontend_path / "index.html"))
    
    return app


# Create app instance
app = create_app()


# Run with: uvicorn app.main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
