<p align="center">
  <h1 align="center">🍯 Agentic Honey-Pot</h1>
  <p align="center">
    <b>An AI-powered honeypot that fights scammers by becoming their worst nightmare, a convincingly gullible target that secretly extracts their intelligence.</b>
  </p>

</p>

---

## 🧠 What is this?

Scammers send thousands of fraudulent messages daily, fake bank alerts, bogus job offers, lottery scams, digital arrest threats. Most people either ignore them or fall victim.

**Agentic Honey-Pot flips the script.** It deploys autonomous AI agents that:

1. 🔍 **Detect** scam intent using a 4-layer ensemble detection engine
2. 🎭 **Engage** scammers with realistic Indian personas (Ramu Uncle, Ananya Student, and more)
3. 🕵️ **Extract** intelligence — phone numbers, UPI IDs, bank accounts, phishing links
4. ⏳ **Waste their time** — keeping scammers busy so they can't target real victims

> *"Ji, mera account block ho jayega? Arey, thoda detail mein samjhao... mera beta abhi ghar pe nahi hai."*
> — Ramu Uncle (62yr retired clerk), while secretly logging the scammer's UPI ID

---

## 🏗️ Architecture

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

### 🔍 4-Layer Ensemble Detection Engine (Parallel)

| Layer | Weight | What it does |
|---|:---:|---|
| **Rule-Based** | 20% | 400+ keyword patterns across urgency, threats, financial, impersonation |
| **Pattern Matcher** | 15% | Regex extraction of phone numbers, UPI IDs, URLs, bank accounts |
| **ML Classifier** | 35% | TF-IDF + Voting Classifier trained on 10K+ real scam messages |
| **LLM Semantic** | 30% | GPT-4o-mini analyzes intent, tactics, and context |

### 🎭 5 Indian Personas

| Persona | Age | Background | Best Against |
|---|:---:|---|---|
| 🧓 **Ramu Uncle** | 62 | Retired government clerk | Banking/KYC scams |
| 👩‍🎓 **Ananya Student** | 21 | College student | Job/lottery scams |
| 👩‍🍳 **Aarti Homemaker** | 38 | Homemaker | UPI/bill scams |
| 👨‍💻 **Vikram IT** | 29 | Software developer | Tech/investment scams |
| 👩‍🏪 **Sunita Shop** | 45 | Kirana shop owner | QR/GST scams |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- OpenAI API key

### 1. Clone & Install

```bash
git clone https://github.com/your-username/AgenticHoneyPot.git
cd AgenticHoneyPot
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=sk-your-key-here
# API_KEY=sk-your-key-here
```

### 3. Run the API Server

```bash
python -m app.main
# Server starts at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### 4. Try the Interactive CLI

```bash
python interactive_honeypot.py
# Chat with the honeypot as a "scammer" and see detection in real-time
```

### 5. Docker (Production)

```bash
docker-compose up -d
```

---

## 📡 API Reference

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
        "timestamp": "2026-01-21T10:15:30Z"
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

## 🎯 Scam Types Detected

| Scam Type | Examples |
|---|---|
| 🏦 Banking Fraud | "Your account is suspended, verify now" |
| 📱 UPI Fraud | "Scan this QR to receive refund" |
| 🪪 KYC Scam | "Update KYC or account will be blocked" |
| 💼 Job Scam | "Work from home, earn ₹50K daily" |
| 🎰 Lottery Scam | "You won ₹25 lakh in WhatsApp lucky draw" |
| 🖥️ Tech Support | "Your computer is infected, install AnyDesk" |
| 📈 Investment Fraud | "Double your money in 30 days, guaranteed" |
| 🧾 Bill Payment | "Electricity disconnection in 2 hours" |
| 📦 Delivery Scam | "Pay ₹49 customs fee for your package" |
| 🏛️ Digital Arrest | "This is CBI, you are under digital arrest" |
| 🪙 Crypto Scam | "Invest in Bitcoin, 500% returns guaranteed" |
| 👤 Impersonation | "Mom, I'm stuck, send money to this number" |

---

## 📊 Final Callback Payload

When a conversation ends, the system sends a structured payload:

```json
{
    "sessionId": "abc123-session-id",
    "scamDetected": true,
    "totalMessagesExchanged": 18,
    "extractedIntelligence": {
        "bankAccounts": ["XXXX-XXXX-XXXX"],
        "upiIds": ["scammer@upi"],
        "phishingLinks": ["http://malicious-link.example"],
        "phoneNumbers": ["+91XXXXXXXXXX"],
        "emailAddresses": ["scammer@gmail.com"]
    },
    "agentNotes": "Scammer used urgency tactics and impersonated SBI officer",
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

## 🗂️ Project Structure

```
AgenticHoneyPot/
├── app/
│   ├── main.py                  # FastAPI entry point
│   ├── config.py                # Settings & environment config
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
│   │   ├── llm_analyzer.py      # GPT semantic analysis
│   │   └── scam_taxonomy.py     # Scam type definitions
│   ├── agent/
│   │   ├── persona_engine.py    # 5 Indian persona profiles
│   │   └── response_generator.py# LLM-powered response generation
│   ├── extraction/
│   │   └── entity_extractor.py  # Intelligence extraction
│   └── static/                  # Demo UI assets
├── data/                        # ML model files
├── tests/                       # Test suite
├── interactive_honeypot.py      # Interactive CLI for testing
├── test_honeypot_api.py         # API test script
├── Dockerfile                   # Container deployment
├── docker-compose.yml           # Full stack deployment
├── requirements.txt             # Python dependencies
└── .env                         # Environment configuration
```

---

## 🧪 Testing

### Interactive CLI Testing
```bash
python interactive_honeypot.py
```

### API Testing
```bash
# Start server first
python -m app.main &

# Run test script
python test_honeypot_api.py
```

### Unit Tests
```bash
pytest tests/ -v
```

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **Framework** | FastAPI + Uvicorn |
| **LLM** | OpenAI GPT-4o-mini |
| **ML** | scikit-learn (TF-IDF + Voting Classifier) |
| **Data** | Trained on 10K+ real WhatsApp scam messages |
| **Sessions** | In-memory (dev) / Redis (prod) |
| **Deployment** | Docker + Docker Compose |
| **URL Analysis** | Crawl4AI + BeautifulSoup |

---

