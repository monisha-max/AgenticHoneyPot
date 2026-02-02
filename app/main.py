"""
Main FastAPI Application Entry Point
Agentic Honey-Pot for Scam Detection & Intelligence Extraction
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.routes import router
from app.api.middleware import (
    RequestLoggingMiddleware,
    AuthenticationMiddleware,
    RateLimitMiddleware,
    get_cors_config
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# APPLICATION LIFESPAN
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    Handles startup and shutdown events
    """
    # Startup
    logger.info("=" * 60)
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("=" * 60)
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info(f"LLM Provider: {settings.LLM_PROVIDER}")
    logger.info(f"Scam threshold: {settings.SCAM_CONFIDENCE_THRESHOLD}")
    logger.info("=" * 60)

    yield  # Application runs here

    # Shutdown
    logger.info("=" * 60)
    logger.info("Shutting down Honeypot API")
    logger.info("=" * 60)


# ============================================================================
# CREATE APPLICATION
# ============================================================================

app = FastAPI(
    title=settings.APP_NAME,
    description="""
## Agentic Honey-Pot for Scam Detection & Intelligence Extraction

An AI-powered honeypot system that:
- 🔍 Detects scam and fraudulent messages
- 🤖 Activates autonomous AI agents
- 🎭 Maintains believable human-like personas
- 💬 Handles multi-turn conversations
- 🕵️ Extracts scam-related intelligence
- 📊 Returns structured results via API

### Authentication
All API endpoints (except health check) require an API key in the header:
```
x-api-key: YOUR_SECRET_API_KEY
```

### Main Endpoint
`POST /api/honeypot` - Process incoming scam messages

### Available Personas
- **Ramu Uncle** - 62yr retired government clerk (banking/KYC scams)
- **Ananya Student** - 20yr college student (job/lottery scams)
- **Aarti Homemaker** - 38yr homemaker (UPI/bill scams)
- **Vikram IT** - 29yr software developer (tech/investment scams)
- **Sunita Shop** - 45yr kirana shop owner (QR/GST scams)
    """,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# ============================================================================
# MIDDLEWARE SETUP
# ============================================================================

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    **get_cors_config()
)

# Add custom middleware (order matters - first added = last executed)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(AuthenticationMiddleware)


# ============================================================================
# INCLUDE ROUTERS
# ============================================================================

app.include_router(router)


# ============================================================================
# ERROR HANDLERS
# ============================================================================

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with custom format"""
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(x) for x in error["loc"]),
            "message": error["msg"]
        })

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "status": "error",
            "error": "Validation error",
            "detail": errors
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "error": "Internal server error",
            "detail": "An unexpected error occurred"
        }
    )


# ============================================================================
# RUN WITH UVICORN (for development)
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info"
    )
