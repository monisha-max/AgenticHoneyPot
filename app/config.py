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

    # Conversation Limits
    MAX_CONVERSATION_TURNS: int = 15  # Increased for deeper engagement scoring
    ENGAGE_PHASE_TURNS: int = 3
    PROBE_PHASE_TURNS: int = 5
    EXTRACT_PHASE_TURNS: int = 7

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance
    Uses lru_cache to avoid reading .env file on every request
    """
    return Settings()


# Convenience function for accessing settings
settings = get_settings()
