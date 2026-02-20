"""
Configuration management for the Honeypot API
Loads settings from environment variables with sensible defaults
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # API Settings
    APP_NAME: str = "Agentic Honey-Pot API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Security
    API_KEY: str = "your-secret-api-key-change-in-production"
    API_KEY_HEADER: str = "x-api-key"

    # LLM Configuration
    LLM_PROVIDER: str = "openai"  # openai, anthropic, google
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    LLM_MODEL: str = "gpt-4o-mini"  # Default model
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 500
    ENABLE_LLM_INTEL_ENRICHMENT: bool = True
    LLM_INTEL_TIMEOUT: int = 4
    LLM_INTEL_MIN_CONFIDENCE: float = 0.75
    ENABLE_LLM_CALLBACK_VERIFICATION: bool = True
    LLM_CALLBACK_INTEL_MODEL: Optional[str] = None
    LLM_CALLBACK_INTEL_TIMEOUT: int = 6
    LLM_CALLBACK_INTEL_MIN_CONFIDENCE: float = 0.75

    # Redis Configuration (for session management)
    USE_REDIS: bool = False  # Toggle to switch between Redis and in-memory store
    REDIS_STRICT: bool = False  # When true, fail instead of falling back to memory
    REDIS_URL: str = "redis://localhost:6379/0"
    SESSION_TTL: int = 3600  # 1 hour session timeout

    # GUVI Callback Configuration
    GUVI_CALLBACK_URL: str = "https://hackathon.guvi.in/api/updateHoneyPotFinalResult"
    GUVI_CALLBACK_TIMEOUT: int = 10
    GUVI_CALLBACK_RETRIES: int = 3

    # Detection Thresholds
    SCAM_CONFIDENCE_THRESHOLD: float = 0.6
    HIGH_CONFIDENCE_THRESHOLD: float = 0.85

    # Conversation Limits - optimized for GUVI scoring
    # 12 turns = ~60-72 seconds duration (>60s = +2pts)
    # 12 turns = 24 messages (≥10 = max message points)
    MAX_CONVERSATION_TURNS: int = 18
    ENGAGE_PHASE_TURNS: int = 4
    PROBE_PHASE_TURNS: int = 8
    EXTRACT_PHASE_TURNS: int = 12

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


def validate_settings(s: Settings) -> list:
    """Validate settings and return list of warnings"""
    warnings = []

    # Check API key is not default
    if s.API_KEY == "your-secret-api-key-change-in-production":
        warnings.append("API_KEY is using default value - change in production")

    # Check LLM API key based on provider
    provider = s.LLM_PROVIDER.lower()
    if provider == "openai" and not s.OPENAI_API_KEY:
        warnings.append("OPENAI_API_KEY not set but LLM_PROVIDER is 'openai'")
    elif provider == "anthropic" and not s.ANTHROPIC_API_KEY:
        warnings.append("ANTHROPIC_API_KEY not set but LLM_PROVIDER is 'anthropic'")
    elif provider == "google" and not s.GOOGLE_API_KEY:
        warnings.append("GOOGLE_API_KEY not set but LLM_PROVIDER is 'google'")

    # Check thresholds are valid
    if not 0 <= s.SCAM_CONFIDENCE_THRESHOLD <= 1:
        warnings.append(f"SCAM_CONFIDENCE_THRESHOLD should be 0-1, got {s.SCAM_CONFIDENCE_THRESHOLD}")

    if s.LLM_TEMPERATURE < 0 or s.LLM_TEMPERATURE > 2:
        warnings.append(f"LLM_TEMPERATURE should be 0-2, got {s.LLM_TEMPERATURE}")

    return warnings


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance
    Uses lru_cache to avoid reading .env file on every request
    """
    s = Settings()

    # Log validation warnings
    import logging
    logger = logging.getLogger(__name__)
    warnings = validate_settings(s)
    for warning in warnings:
        logger.warning(f"Config warning: {warning}")

    return s


# Convenience function for accessing settings
settings = get_settings()
