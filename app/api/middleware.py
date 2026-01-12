"""
API middleware for MedExplain AI.

Provides:
- Rate limiting
- Tenant ID validation
- Request logging
"""

import time
from typing import Callable, Optional

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("middleware")


# Rate limiter using client IP
limiter = Limiter(key_func=get_remote_address)


def get_tenant_id(request: Request) -> Optional[str]:
    """Extract tenant ID from request headers."""
    return request.headers.get("X-Tenant-ID")


class TenantMiddleware(BaseHTTPMiddleware):
    """
    Middleware for multi-tenant support.
    
    Validates tenant ID header and adds tenant context to requests.
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        # Skip tenant check for health endpoint and docs
        if request.url.path in ["/health", "/docs", "/openapi.json", "/redoc"]:
            return await call_next(request)
        
        # Get tenant ID
        tenant_id = get_tenant_id(request)
        
        if settings.enable_multi_tenant:
            if not tenant_id:
                # Use default tenant if not provided
                tenant_id = settings.default_tenant_id
            
            # Store tenant in request state
            request.state.tenant_id = tenant_id
        else:
            request.state.tenant_id = settings.default_tenant_id
        
        return await call_next(request)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for request/response logging.
    
    Logs:
    - Request method, path, tenant
    - Response status code
    - Processing time
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        start_time = time.time()
        
        # Get request info
        method = request.method
        path = request.url.path
        tenant_id = getattr(request.state, 'tenant_id', 'unknown')
        client_ip = get_remote_address(request)
        
        # Log request
        logger.info(
            "Request received",
            method=method,
            path=path,
            tenant_id=tenant_id,
            client_ip=client_ip
        )
        
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                "Request completed",
                method=method,
                path=path,
                status_code=response.status_code,
                process_time_ms=int(process_time * 1000),
                tenant_id=tenant_id
            )
            
            # Add processing time header
            response.headers["X-Process-Time"] = f"{process_time:.4f}"
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            logger.error(
                "Request failed",
                method=method,
                path=path,
                error=str(e),
                process_time_ms=int(process_time * 1000),
                tenant_id=tenant_id
            )
            raise


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Global error handling middleware.
    
    Catches unhandled exceptions and returns safe error responses.
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        try:
            return await call_next(request)
            
        except HTTPException:
            # Let FastAPI handle HTTP exceptions normally
            raise
            
        except ValueError as e:
            logger.warning("Validation error", error=str(e))
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Validation Error",
                    "message": str(e),
                    "error_code": "VALIDATION_ERROR"
                }
            )
            
        except Exception as e:
            logger.error("Unhandled exception", error=str(e), exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": "An unexpected error occurred. Please try again.",
                    "error_code": "INTERNAL_ERROR"
                }
            )


def setup_rate_limiting(app) -> None:
    """Setup rate limiting on the application."""
    app.state.limiter = limiter
    
    @app.exception_handler(429)
    async def rate_limit_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate Limit Exceeded",
                "message": "Too many requests. Please wait before trying again.",
                "error_code": "RATE_LIMIT_EXCEEDED"
            }
        )
