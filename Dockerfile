# ============================================================================
# Dockerfile for Agentic Honey-Pot API
# Single-stage build with in-image dependency install
# ============================================================================

FROM python:3.11-slim

WORKDIR /app

# Create non-root user for security
RUN addgroup --system --gid 1001 appgroup \
    && adduser --system --uid 1001 --gid 1001 appuser

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (inline, no requirements.txt copy)
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    pydantic-settings \
    httpx \
    crawl4ai \
    beautifulsoup4 \
    lxml \
    redis \
    openai \
    anthropic \
    google-generativeai \
    pandas \
    scikit-learn \
    numpy \
    python-dotenv

# Copy application code
COPY ./app ./app
COPY ./data ./data

# Set ownership
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]