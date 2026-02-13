"""
Middleware components for the Honeypot API
Handles authentication, rate limiting, and request logging
"""

import time
import logging
from typing import Callable, Dict
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# API KEY AUTHENTICATION
# ============================================================================

async def verify_api_key(request: Request) -> bool:
    """
    Verify the API key from request headers

    Args:
        request: FastAPI request object

    Returns:
        True if API key is valid

    Raises:
        HTTPException: If API key is missing or invalid
    """
    # Skip authentication for health check and docs
    if request.url.path in ["/", "/health", "/docs", "/openapi.json", "/redoc", "/demo"]:
        return True

    api_key = request.headers.get(settings.API_KEY_HEADER)

    if not api_key:
        logger.warning(f"Missing API key from {request.client.host}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "status": "error",
                "error": "Missing API key",
                "detail": f"Please provide API key in '{settings.API_KEY_HEADER}' header"
            }
        )

    if api_key != settings.API_KEY:
        logger.warning(f"Invalid API key attempt from {request.client.host}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "status": "error",
                "error": "Invalid API key",
                "detail": "The provided API key is not valid"
            }
        )

    return True


# ============================================================================
# RATE LIMITING
# ============================================================================

class RateLimiter:
    """
    Simple in-memory rate limiter
    For production, use Redis-based rate limiting
    """

    def __init__(self, requests_limit: int = 100, window_seconds: int = 60):
        self.requests_limit = requests_limit
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        """
        Check if client is allowed to make request

        Args:
            client_id: Unique identifier for the client (IP or API key)

        Returns:
            True if request is allowed, False if rate limited
        """
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.window_seconds)

        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > window_start
        ]

        # Check limit
        if len(self.requests[client_id]) >= self.requests_limit:
            return False

        # Record this request
        self.requests[client_id].append(now)
        return True

    def get_retry_after(self, client_id: str) -> int:
        """Get seconds until rate limit resets"""
        if not self.requests[client_id]:
            return 0

        oldest_request = min(self.requests[client_id])
        reset_time = oldest_request + timedelta(seconds=self.window_seconds)
        seconds_remaining = (reset_time - datetime.utcnow()).total_seconds()

        return max(0, int(seconds_remaining))


# Global rate limiter instance
rate_limiter = RateLimiter(
    requests_limit=settings.RATE_LIMIT_REQUESTS,
    window_seconds=settings.RATE_LIMIT_WINDOW
)


# ============================================================================
# REQUEST LOGGING MIDDLEWARE
# ============================================================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all incoming requests and their processing time
    """

    async def dispatch(self, request: Request, call_next: Callable):
        # Generate request ID
        request_id = f"{int(time.time() * 1000)}"

        # Log request
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )

        # Process request and measure time
        start_time = time.time()

        try:
            response = await call_next(request)
            process_time = time.time() - start_time

            # Add custom headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.4f}"

            logger.info(
                f"[{request_id}] Completed with status {response.status_code} "
                f"in {process_time:.4f}s"
            )

            return response

        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"[{request_id}] Error after {process_time:.4f}s: {str(e)}"
            )
            raise


# ============================================================================
# AUTHENTICATION MIDDLEWARE
# ============================================================================

class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle API key authentication
    """

    async def dispatch(self, request: Request, call_next: Callable):
        # Skip auth for certain paths
        skip_paths = ["/", "/health", "/docs", "/docs/", "/openapi.json", "/redoc", "/redoc/", "/favicon.ico", "/demo"]

        # Skip auth for OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)

        # Normalize path (strip trailing slash for comparison)
        path = request.url.path.rstrip('/') or '/'
        skip_paths_normalized = [p.rstrip('/') or '/' for p in skip_paths]

        if path.startswith("/static/"):
            return await call_next(request)

        if path not in skip_paths_normalized and request.url.path not in skip_paths:
            try:
                await verify_api_key(request)
            except HTTPException as e:
                return JSONResponse(
                    status_code=e.status_code,
                    content=e.detail
                )

        return await call_next(request)


# ============================================================================
# RATE LIMITING MIDDLEWARE
# ============================================================================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle rate limiting
    """

    async def dispatch(self, request: Request, call_next: Callable):
        # Skip rate limiting for certain paths
        skip_paths = ["/", "/health", "/docs", "/openapi.json", "/redoc", "/demo"]

        if request.url.path in skip_paths or request.url.path.startswith("/static/"):
            return await call_next(request)

        # Use API key or IP as client identifier
        client_id = request.headers.get(settings.API_KEY_HEADER) or \
                    (request.client.host if request.client else "unknown")

        if not rate_limiter.is_allowed(client_id):
            retry_after = rate_limiter.get_retry_after(client_id)
            logger.warning(f"Rate limit exceeded for {client_id}")

            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "status": "error",
                    "error": "Rate limit exceeded",
                    "detail": f"Too many requests. Please retry after {retry_after} seconds"
                },
                headers={"Retry-After": str(retry_after)}
            )

        return await call_next(request)


# ============================================================================
# CORS HEADERS (if needed)
# ============================================================================

def get_cors_config():
    """
    Get CORS configuration for the API
    """
    return {
        "allow_origins": ["*"],  # Configure appropriately for production
        "allow_credentials": True,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }
