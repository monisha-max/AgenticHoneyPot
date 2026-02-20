

<h1 align="center">Agentic Honey-Pot</h1>

<p align="center">
  <b>An AI-powered honeypot that fights scammers by becoming their worst nightmare — a convincingly gullible target that secretly extracts their intelligence.</b>
</p>

---

## What is this?

Scammers send thousands of fraudulent messages daily — fake bank alerts, bogus job offers, lottery scams, digital arrest threats. Most people either ignore them or fall victim.

**Agentic Honey-Pot flips the script.** It deploys autonomous AI agents that:

1. **Detect** scam intent using a 4-layer ensemble detection engine
2. **Engage** scammers with realistic Indian personas (Ramu Uncle, Ananya Student, and more)
3. **Extract** intelligence — phone numbers, UPI IDs, bank accounts, phishing links
4. **Waste their time** — keeping scammers busy so they can't target real victims

> *"Ji, mera account block ho jayega? Arey, thoda detail mein samjhao... mera beta abhi ghar pe nahi hai."*
> — Ramu Uncle (62yr retired clerk), while secretly logging the scammer's UPI ID

---

## Table of Contents

- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Configuration Reference](#configuration-reference)
- [API Reference](#api-reference)
- [Scam Types Detected](#scam-types-detected)
- [Final Callback Payload](#final-callback-payload)
- [Project Structure](#project-structure)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Tech Stack](#tech-stack)
- [Contributing](#contributing)

---

## Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │              FastAPI Server                  │
                    │            POST /api/honeypot                │
                    └───────────────────┬──────────────────────────┘
                                        │
                    ┌───────────────────▼──────────────────────────┐
                    │          Conversation Orchestrator           │
                    │     (Session Management + Flow Control)      │
                    └───┬──────────┬──────────┬──────────┬────────┘
                        │          │          │          │
               ┌────────▼───┐ ┌───▼────┐ ┌───▼────┐ ┌───▼──────────┐
               │  Ensemble   │ │Persona │ │Response│ │ Intelligence │
               │  Detection  │ │ Engine │ │  Gen   │ │  Extraction  │
               │  (4-layer)  │ │(5 char)│ │ (LLM)  │ │  (Entities)  │
               └─────────────┘ └────────┘ └────────┘ └──────────────┘
```

### 4-Layer Ensemble Detection Engine

| Layer | Weight | Description |
|-------|:------:|-------------|
| **Rule-Based** | 20% | 400+ keyword patterns across urgency, threats, financial, impersonation |
| **Pattern Matcher** | 15% | Regex extraction of phone numbers, UPI IDs, URLs, bank accounts |
| **ML Classifier** | 35% | TF-IDF + Voting Classifier trained on 10K+ real scam messages |
| **LLM Semantic** | 30% | GPT/Claude analyzes intent, tactics, and context |

### 5 Indian Personas

| Persona | Age | Background | Best Against |
|---------|:---:|------------|--------------|
| **Ramu Uncle** | 62 | Retired government clerk | Banking/KYC scams |
| **Ananya Student** | 21 | College student | Job/lottery scams |
| **Aarti Homemaker** | 38 | Homemaker | UPI/bill scams |
| **Vikram IT** | 29 | Software developer | Tech/investment scams |
| **Sunita Shop** | 45 | Kirana shop owner | QR/GST scams |

---

## Quick Start

### Prerequisites

- Python 3.10 or higher
- An LLM API key (OpenAI, Anthropic, or Google)
- Git

### Step 1: Clone & Install

```bash
# Clone the repository
git clone https://github.com/monisha-max/AgenticHoneyPot.git
cd AgenticHoneyPot

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate        # Linux/macOS
# OR
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your settings
nano .env   # or use any text editor
```

**Minimum required configuration:**

```env
# Choose your LLM provider: openai, anthropic, or google
LLM_PROVIDER=openai

# Add the API key for your chosen provider
OPENAI_API_KEY=sk-your-openai-key-here

# Optional: Set a custom API key for your honeypot endpoint
API_KEY=your-custom-api-key
```

### Step 3: Verify Configuration

```bash
# Start the server - it will validate your configuration on startup
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Look for these startup messages:**

```
✅ Good:
INFO - LLM API key verified successfully
INFO - Starting Agentic Honey-Pot API v1.0.0

⚠️ Warning (will use fallback templates):
WARNING - Config warning: OPENAI_API_KEY not set but LLM_PROVIDER is 'openai'
WARNING - LLM API key verification FAILED - will use fallback templates
```

### Step 4: Test the API

```bash
# Health check
curl http://localhost:8000/health

# Test with a scam message
curl -X POST http://localhost:8000/api/honeypot \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-custom-api-key" \
  -d '{
    "sessionId": "test-123",
    "message": {
      "sender": "scammer",
      "text": "URGENT: Your SBI account will be blocked. Share OTP immediately.",
      "timestamp": "2024-01-01T10:00:00Z"
    }
  }'
```

### Step 5: Interactive Testing

```bash
# Chat with the honeypot as a "scammer" and see detection in real-time
python interactive_honeypot.py
```

---

## Configuration Reference

### Environment Variables

| Variable | Required | Default | Description |
|----------|:--------:|---------|-------------|
| `LLM_PROVIDER` | Yes | `openai` | LLM provider: `openai`, `anthropic`, or `google` |
| `OPENAI_API_KEY` | If using OpenAI | - | Your OpenAI API key |
| `ANTHROPIC_API_KEY` | If using Anthropic | - | Your Anthropic API key |
| `GOOGLE_API_KEY` | If using Google | - | Your Google AI API key |
| `API_KEY` | No | `your-secret-api-key...` | API key for authenticating requests |
| `DEBUG` | No | `false` | Enable debug logging |
| `LLM_MODEL` | No | `gpt-4o-mini` | Model to use for LLM calls |
| `LLM_TEMPERATURE` | No | `0.7` | Temperature for LLM responses (0-2) |
| `SCAM_CONFIDENCE_THRESHOLD` | No | `0.6` | Minimum confidence to flag as scam (0-1) |
| `MAX_CONVERSATION_TURNS` | No | `15` | Maximum turns before ending conversation |
| `USE_REDIS` | No | `false` | Use Redis for session storage |
| `REDIS_URL` | No | `redis://localhost:6379/0` | Redis connection URL |

### LLM Provider Configuration

**OpenAI (Recommended):**
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx
LLM_MODEL=gpt-4o-mini
```

**Anthropic:**
```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx
LLM_MODEL=claude-3-haiku-20240307
```

**Google:**
```env
LLM_PROVIDER=google
GOOGLE_API_KEY=AIzaxxxxxxxxxxxxx
LLM_MODEL=gemini-1.5-flash
```

---

## API Reference

### `POST /api/honeypot`

Process a scam message and get the honeypot's response.

**Headers:**
```
Content-Type: application/json
x-api-key: YOUR_API_KEY
```

**Request Body:**
```json
{
    "sessionId": "unique-session-id",
    "message": {
        "sender": "scammer",
        "text": "Your SBI account will be blocked. Share OTP now.",
        "timestamp": "2024-01-21T10:15:30Z"
    },
    "conversationHistory": [],
    "metadata": {
        "channel": "SMS",
        "language": "English",
        "locale": "IN"
    }
}
```

**Response:**
```json
{
    "status": "success",
    "reply": "Arey, mera SBI account? Kaunsa branch? Mera toh Koramangala mein hai...",
    "scamDetected": true,
    "extractedIntelligence": {
        "phoneNumbers": [],
        "upiIds": [],
        "bankAccounts": [],
        "phishingLinks": [],
        "emailAddresses": []
    },
    "agentNotes": "Phase: PROBE, Emotion: CONFUSED"
}
```

### `GET /health`

Health check endpoint (no authentication required).

**Response:**
```json
{
    "status": "healthy",
    "version": "1.0.0"
}
```

### `GET /api/session/{session_id}`

Get session state and extracted intelligence.

### `POST /api/session/{session_id}/end`

End a session and trigger callback.

---

## Scam Types Detected

| Scam Type | Examples |
|-----------|----------|
| **Banking Fraud** | "Your account is suspended, verify now" |
| **UPI Fraud** | "Scan this QR to receive refund" |
| **KYC Scam** | "Update KYC or account will be blocked" |
| **Job Scam** | "Work from home, earn ₹50K daily" |
| **Lottery Scam** | "You won ₹25 lakh in WhatsApp lucky draw" |
| **Tech Support** | "Your computer is infected, install AnyDesk" |
| **Investment Fraud** | "Double your money in 30 days, guaranteed" |
| **Bill Payment** | "Electricity disconnection in 2 hours" |
| **Delivery Scam** | "Pay ₹49 customs fee for your package" |
| **Digital Arrest** | "This is CBI, you are under digital arrest" |
| **Crypto Scam** | "Invest in Bitcoin, 500% returns guaranteed" |
| **Impersonation** | "Mom, I'm stuck, send money to this number" |

---

## Final Callback Payload

When a conversation ends, the system generates a structured payload:

```json
{
    "sessionId": "abc123-session-id",
    "scamDetected": true,
    "totalMessagesExchanged": 18,
    "engagementDurationSeconds": 180,
    "extractedIntelligence": {
        "bankAccounts": ["1234567890123456"],
        "upiIds": ["scammer@upi"],
        "phishingLinks": ["http://malicious-link.example"],
        "phoneNumbers": ["+919876543210"],
        "emailAddresses": ["scammer@fake.com"]
    },
    "agentNotes": "Scammer used urgency tactics and impersonated SBI officer...",
    "scamType": "BANKING_FRAUD",
    "confidenceLevel": "HIGH",
    "engagementMetrics": {
        "averageResponseTimeMs": 450,
        "conversationDurationSec": 180,
        "engagementScore": 0.85,
        "turnsBeforeScamDetected": 2,
        "intelligenceCompleteness": 0.75
    }
}
```

---

## Project Structure

```
AgenticHoneyPot/
├── app/
│   ├── main.py                  # FastAPI entry point + LLM health check
│   ├── config.py                # Settings, validation & environment config
│   ├── api/
│   │   ├── routes.py            # API endpoints
│   │   ├── schemas.py           # Pydantic models
│   │   └── middleware.py        # Auth, rate limiting, CORS
│   ├── core/
│   │   ├── orchestrator.py      # Conversation flow controller
│   │   ├── session_manager.py   # Session state (in-memory/Redis)
│   │   └── callback_manager.py  # GUVI callback payload builder
│   ├── detection/
│   │   ├── ensemble.py          # 4-layer ensemble detector
│   │   ├── rule_based.py        # 400+ keyword rules
│   │   ├── pattern_matcher.py   # Regex entity extraction
│   │   ├── ml_classifier.py     # Trained ML model
│   │   ├── llm_analyzer.py      # GPT/Claude semantic analysis
│   │   └── scam_taxonomy.py     # Scam type definitions
│   ├── agent/
│   │   ├── persona_engine.py    # 5 Indian persona profiles
│   │   └── response_generator.py# LLM-powered response generation
│   ├── extraction/
│   │   ├── entity_extractor.py  # Intelligence extraction (regex)
│   │   ├── llm_intelligence_enricher.py  # LLM-enhanced extraction
│   │   └── url_analyzer.py      # Phishing URL analysis
│   └── static/                  # Demo UI assets
├── data/                        # ML model files
├── tests/                       # Test suite
├── interactive_honeypot.py      # Interactive CLI for testing
├── Dockerfile                   # Container deployment
├── docker-compose.yml           # Full stack deployment
├── requirements.txt             # Python dependencies
├── .env.example                 # Example environment file
└── README.md                    # This file
```

---

## Deployment

### Local Development

```bash
# Start with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker

```bash
# Build and run
docker build -t honeypot .
docker run -p 8000:8000 --env-file .env honeypot
```

### Docker Compose (with Redis)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f honeypot

# Stop
docker-compose down
```

### Production Checklist

- [ ] Set `DEBUG=false`
- [ ] Change `API_KEY` from default value
- [ ] Configure valid LLM API key
- [ ] Enable Redis for session persistence (`USE_REDIS=true`)
- [ ] Set up HTTPS/TLS termination
- [ ] Configure rate limiting appropriately
- [ ] Set up log aggregation

---

## Troubleshooting

### Common Issues

#### 1. "LLM API key verification FAILED"

**Cause:** Invalid or missing API key for the configured LLM provider.

**Solution:**
```bash
# Check your .env file
cat .env | grep -E "(LLM_PROVIDER|API_KEY)"

# Verify the key works
# For OpenAI:
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

#### 2. "Config warning: OPENAI_API_KEY not set"

**Cause:** LLM_PROVIDER is set to 'openai' but no API key provided.

**Solution:**
```bash
# Add to .env
echo "OPENAI_API_KEY=sk-your-key-here" >> .env
```

#### 3. Repetitive/Generic Responses

**Cause:** LLM is failing, falling back to templates.

**Solution:**
1. Check startup logs for LLM health check status
2. Verify API key is valid and has credits
3. Check `app/agent/response_generator.py` logs for errors

#### 4. Intelligence Not Being Extracted

**Cause:** Regex patterns not matching, or LLM enrichment failing.

**Solution:**
1. Enable debug logging: `DEBUG=true`
2. Check `app/extraction/entity_extractor.py` patterns
3. Verify LLM enrichment is enabled: `ENABLE_LLM_INTEL_ENRICHMENT=true`

#### 5. Session Not Persisting

**Cause:** Using in-memory storage (default), which resets on restart.

**Solution:**
```bash
# Enable Redis
USE_REDIS=true
REDIS_URL=redis://localhost:6379/0

# Start Redis
docker run -d -p 6379:6379 redis
```

### Debug Mode

Enable verbose logging:
```env
DEBUG=true
```

View detailed logs:
```bash
# All logs
python -m uvicorn app.main:app --log-level debug

# Filter specific component
python -m uvicorn app.main:app 2>&1 | grep -E "(detection|extraction|response)"
```

---

## Tech Stack

| Category | Technology |
|----------|------------|
| **Framework** | FastAPI + Uvicorn |
| **LLM** | OpenAI GPT-4o-mini / Anthropic Claude / Google Gemini |
| **ML** | scikit-learn (TF-IDF + Voting Classifier) |
| **Data** | Trained on 10K+ real WhatsApp scam messages |
| **Sessions** | In-memory (dev) / Redis (prod) |
| **Deployment** | Docker + Docker Compose |
| **URL Analysis** | Crawl4AI + BeautifulSoup |
| **Validation** | Pydantic v2 |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/ -v`)
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style

- Use type hints
- Follow PEP 8
- Add docstrings to functions
- Handle errors explicitly (no bare `except:`)

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>Built with by Team AgenticHoneyPot</b><br>
  <i>Fighting scammers, one conversation at a time.</i>
</p>
