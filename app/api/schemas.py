"""
Pydantic schemas for API request/response validation
Defines all data structures used in the Honeypot API
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class SenderType(str, Enum):
    """Who sent the message"""
    SCAMMER = "scammer"
    USER = "user"


class ChannelType(str, Enum):
    """Communication channel"""
    SMS = "SMS"
    WHATSAPP = "WhatsApp"
    EMAIL = "Email"
    CHAT = "Chat"


class ScamType(str, Enum):
    """Types of scams we can detect"""
    BANKING_FRAUD = "BANKING_FRAUD"
    UPI_FRAUD = "UPI_FRAUD"
    KYC_SCAM = "KYC_SCAM"
    JOB_SCAM = "JOB_SCAM"
    LOTTERY_SCAM = "LOTTERY_SCAM"
    TECH_SUPPORT = "TECH_SUPPORT"
    INVESTMENT_FRAUD = "INVESTMENT_FRAUD"
    QR_CODE_SCAM = "QR_CODE_SCAM"
    DELIVERY_SCAM = "DELIVERY_SCAM"
    TAX_GST_SCAM = "TAX_GST_SCAM"
    CREDIT_CARD_FRAUD = "CREDIT_CARD_FRAUD"
    CRYPTO_SCAM = "CRYPTO_SCAM"
    BILL_PAYMENT_SCAM = "BILL_PAYMENT_SCAM"
    IMPERSONATION = "IMPERSONATION"
    UNKNOWN = "UNKNOWN"


class ConversationPhase(str, Enum):
    """Phases of honeypot engagement"""
    ENGAGE = "ENGAGE"
    PROBE = "PROBE"
    EXTRACT = "EXTRACT"
    STALL = "STALL"
    COMPLETE = "COMPLETE"


class EmotionalState(str, Enum):
    """Emotional states for persona"""
    INITIAL = "INITIAL"
    CONFUSED = "CONFUSED"
    WORRIED = "WORRIED"
    PANICKED = "PANICKED"
    HESITANT = "HESITANT"
    SKEPTICAL = "SKEPTICAL"
    TRUSTING = "TRUSTING"
    COMPLIANT = "COMPLIANT"


class PersonaType(str, Enum):
    """Available personas"""
    RAMU_UNCLE = "ramu_uncle"
    ANANYA_STUDENT = "ananya_student"
    AARTI_HOMEMAKER = "aarti_homemaker"
    VIKRAM_IT = "vikram_it"
    SUNITA_SHOP = "sunita_shop"
    NEUTRAL_CITIZEN = "neutral_citizen"


# ============================================================================
# REQUEST SCHEMAS
# ============================================================================

class Message(BaseModel):
    """Single message in conversation"""
    sender: SenderType
    text: str = Field(..., min_length=1, max_length=5000)
    timestamp: datetime

    class Config:
        json_schema_extra = {
            "example": {
                "sender": "scammer",
                "text": "Your bank account will be blocked today. Verify immediately.",
                "timestamp": "2026-01-21T10:15:30Z"
            }
        }


class Metadata(BaseModel):
    """Optional metadata about the message"""
    channel: Optional[ChannelType] = ChannelType.SMS
    language: Optional[str] = "English"
    locale: Optional[str] = "IN"

    class Config:
        json_schema_extra = {
            "example": {
                "channel": "SMS",
                "language": "English",
                "locale": "IN"
            }
        }


class HoneypotRequest(BaseModel):
    """
    Main API request schema
    Represents an incoming message from a suspected scammer
    """
    sessionId: str = Field(..., min_length=1, max_length=100, description="Unique session identifier")
    message: Message = Field(..., description="The current incoming message")
    conversationHistory: List[Message] = Field(default=[], description="Previous messages in conversation")
    metadata: Optional[Metadata] = Field(default=None, description="Optional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "sessionId": "wertyu-dfghj-ertyui",
                "message": {
                    "sender": "scammer",
                    "text": "Your bank account will be blocked today. Verify immediately.",
                    "timestamp": "2026-01-21T10:15:30Z"
                },
                "conversationHistory": [],
                "metadata": {
                    "channel": "SMS",
                    "language": "English",
                    "locale": "IN"
                }
            }
        }


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================

class HoneypotResponse(BaseModel):
    """
    Main API response schema
    Returns the agent's reply to the scammer
    """
    status: str = Field(..., description="Status of the request: success or error")
    reply: str = Field(..., description="The agent's response message")
    scamDetected: Optional[bool] = Field(None, description="Whether a scam was detected")
    extractedIntelligence: Optional[Dict[str, Any]] = Field(None, description="Extracted intelligence from the conversation")
    agentNotes: Optional[str] = Field(None, description="Internal notes about the conversation")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "reply": "Why is my account being suspended?",
                "scamDetected": True,
                "extractedIntelligence": {
                    "names": ["Rajesh Kumar"],
                    "phones": ["9876543210"],
                    "upis": ["rajesh@paytm"],
                    "emails": ["rajesh@email.com"],
                    "bankAccounts": []
                },
                "agentNotes": "Victim appears engaged"
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema"""
    status: str = "error"
    error: str
    detail: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "status": "error",
                "error": "Invalid API key",
                "detail": "The provided API key is not valid"
            }
        }


# ============================================================================
# INTERNAL SCHEMAS (used between components)
# ============================================================================

class DetectionResult(BaseModel):
    """Result from scam detection engine"""
    is_scam: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    scam_type: ScamType
    evidence: List[str] = []
    keywords_found: List[str] = []


class URLAnalysisInfo(BaseModel):
    """Analysis result for a single URL"""
    url: str
    risk_level: str = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL
    risk_score: float = 0.0
    is_phishing: bool = False
    brand_impersonation: Optional[str] = None
    findings: List[str] = []
    page_title: str = ""


class ExtractedIntelligence(BaseModel):
    """Intelligence extracted from conversation"""
    bank_accounts: List[str] = []
    upi_ids: List[str] = []
    phishing_links: List[str] = []
    phone_numbers: List[str] = []
    suspicious_keywords: List[str] = []
    email_addresses: List[str] = []
    ifsc_codes: List[str] = []
    scammer_names: List[str] = []
    fake_references: List[str] = []
    # URL Analysis Results
    url_analysis: List[URLAnalysisInfo] = []


class SessionState(BaseModel):
    """State maintained for each conversation session"""
    session_id: str
    persona: PersonaType
    emotional_state: EmotionalState
    conversation_phase: ConversationPhase
    turn_count: int = 0
    scam_detected: bool = False
    scam_type: Optional[ScamType] = None
    confidence_score: float = 0.0
    intelligence: ExtractedIntelligence = Field(default_factory=ExtractedIntelligence)
    conversation_history: List[Dict[str, Any]] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    agent_notes: List[str] = []


# ============================================================================
# CALLBACK SCHEMAS (for GUVI endpoint)
# ============================================================================

class ExtractedIntelligencePayload(BaseModel):
    """Intelligence format for GUVI payload"""
    bankAccounts: List[str] = []
    upiIds: List[str] = []
    phishingLinks: List[str] = []
    phoneNumbers: List[str] = []
    suspiciousKeywords: List[str] = []

class GuviCallbackPayload(BaseModel):
    """
    Payload to send to GUVI evaluation endpoint
    Sent when conversation completes
    """
    sessionId: str
    scamDetected: bool
    totalMessagesExchanged: int
    extractedIntelligence: ExtractedIntelligencePayload
    agentNotes: str

    class Config:
        json_schema_extra = {
            "example": {
                "sessionId": "abc123-session-id",
                "scamDetected": True,
                "totalMessagesExchanged": 18,
                "extractedIntelligence": {
                    "bankAccounts": ["XXXX-XXXX-XXXX"],
                    "upiIds": ["scammer@upi"],
                    "phishingLinks": ["http://malicious-link.example"],
                    "phoneNumbers": ["+91XXXXXXXXXX"],
                    "suspiciousKeywords": ["urgent", "verify now", "account blocked"]
                },
                "agentNotes": "Scammer used urgency tactics and payment redirection"
            }
        }
