from datetime import datetime
import asyncio

import pytest

from app.api.schemas import (
    ConversationPhase,
    EmotionalState,
    ExtractedIntelligence,
    PersonaType,
    SessionState,
)
from app.core.callback_manager import CallbackManager
from app.core.orchestrator import ConversationOrchestrator
from app.extraction.llm_intelligence_enricher import (
    CallbackVerificationResult,
    EnrichmentResult,
    EnrichedCandidate,
)


def _make_state() -> SessionState:
    return SessionState(
        session_id="test-session",
        persona=PersonaType.NEUTRAL_CITIZEN,
        emotional_state=EmotionalState.CONFUSED,
        conversation_phase=ConversationPhase.ENGAGE,
        turn_count=3,
        scam_detected=True,
        confidence_score=0.8,
        intelligence=ExtractedIntelligence(),
        conversation_history=[],
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        agent_notes=[],
    )


def test_callback_payload_keeps_required_five_keys_only():
    state = _make_state()
    state.intelligence.bank_accounts = ["123456789"]
    state.intelligence.upi_ids = ["abc@upi"]
    state.intelligence.phishing_links = ["http://phish.test"]
    state.intelligence.phone_numbers = ["9876543210"]
    state.intelligence.suspicious_keywords = ["urgent"]
    state.intelligence.scammer_names = ["Harsha"]
    state.intelligence.email_addresses = ["harsha@test.com"]

    payload = CallbackManager()._build_payload(state)
    intel_payload = payload.extractedIntelligence.model_dump()

    assert set(intel_payload.keys()) == {
        "bankAccounts",
        "upiIds",
        "phishingLinks",
        "phoneNumbers",
        "suspiciousKeywords",
    }


def test_merge_enriched_intelligence_dedupes_and_adds_valid_llm_fields():
    orch = ConversationOrchestrator.__new__(ConversationOrchestrator)

    existing = ExtractedIntelligence(
        bank_accounts=["111122223333"],
        upi_ids=["base@upi"],
        phone_numbers=["+919999999999"],
        phishing_links=["http://existing.test"],
        ifsc_codes=["SBIN0001234"],
        fake_references=["REF12345"],
        suspicious_keywords=["urgent"],
        scammer_names=["Harsha"],
    )
    enrichment = EnrichmentResult(
        candidates=[
            EnrichedCandidate("scammer_names", "Harsha", 0.95, "im harsha"),
            EnrichedCandidate("scammer_names", "Ravi", 0.85, "my name is ravi"),
            EnrichedCandidate("email_addresses", "ravi@test.com", 0.9, "mail me at ravi@test.com"),
            EnrichedCandidate("phone_numbers", "9999999999", 0.92, "call 9999999999"),
            EnrichedCandidate("phone_numbers", "8888888888", 0.92, "call 8888888888"),
            EnrichedCandidate("upi_ids", "new@upi", 0.8, "pay to new@upi"),
            EnrichedCandidate("bank_accounts", "1234 5678 90123", 0.83, "account 1234 5678 90123"),
            EnrichedCandidate("ifsc_codes", "hdfc0001234", 0.9, "ifsc hdfc0001234"),
            EnrichedCandidate("phishing_links", "www.fake-bank.test/login", 0.88, "visit www.fake-bank.test/login"),
            EnrichedCandidate("fake_references", "AA12BB34", 0.82, "ticket aa12bb34"),
            EnrichedCandidate("suspicious_keywords", "send otp now", 0.86, "send otp now"),
            EnrichedCandidate("scammer_names", "LowConf", 0.3, "name maybe lowconf"),
            EnrichedCandidate("email_addresses", "no-evidence@test.com", 0.95, ""),
            EnrichedCandidate("bank_accounts", "1234567890", 0.99, "acct 1234567890"),
            EnrichedCandidate("ifsc_codes", "bad123", 0.9, "ifsc bad123"),
        ]
    )

    merged = ConversationOrchestrator._merge_enriched_intelligence(orch, existing, enrichment)

    assert "111122223333" in merged.bank_accounts
    assert "base@upi" in merged.upi_ids
    assert "new@upi" in merged.upi_ids
    assert "+919999999999" in merged.phone_numbers
    assert "+918888888888" in merged.phone_numbers
    assert "1234567890123" in merged.bank_accounts
    assert "1234567890" not in merged.bank_accounts
    assert "HDFC0001234" in merged.ifsc_codes
    assert "http://www.fake-bank.test/login" in merged.phishing_links
    assert "AA12BB34" in merged.fake_references
    assert "send otp now" in merged.suspicious_keywords
    assert "Harsha" in merged.scammer_names
    assert "Ravi" in merged.scammer_names
    assert "LowConf" not in merged.scammer_names
    assert "ravi@test.com" in merged.email_addresses
    assert "no-evidence@test.com" not in merged.email_addresses
    assert "urgent" in merged.suspicious_keywords


def test_llm_enrichment_fail_open(monkeypatch):
    monkeypatch.setattr("app.config.settings.ENABLE_LLM_INTEL_ENRICHMENT", True)

    class DummyEnricher:
        async def enrich_recent_messages(self, _messages):
            raise RuntimeError("simulated llm failure")

    class DummySessionManager:
        async def update_intelligence(self, _session_id, intelligence):
            return SessionState(
                session_id="test-session",
                persona=PersonaType.NEUTRAL_CITIZEN,
                emotional_state=EmotionalState.CONFUSED,
                conversation_phase=ConversationPhase.ENGAGE,
                turn_count=1,
                scam_detected=True,
                confidence_score=0.8,
                intelligence=intelligence,
                conversation_history=[],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                agent_notes=[],
            )

        async def add_agent_note(self, _session_id, _note):
            return None

    orch = ConversationOrchestrator.__new__(ConversationOrchestrator)
    orch.llm_intel_enricher = DummyEnricher()
    orch.session_manager = DummySessionManager()

    state = _make_state()
    result = asyncio.run(
        ConversationOrchestrator._run_llm_enrichment(
            orch,
            state,
            ["im harsha", "send 10k"],
            "test",
        )
    )

    assert result.session_id == state.session_id
    assert result.intelligence.model_dump() == state.intelligence.model_dump()


def test_callback_verification_can_prune_and_fill(monkeypatch):
    monkeypatch.setattr("app.config.settings.ENABLE_LLM_CALLBACK_VERIFICATION", True)
    monkeypatch.setattr("app.config.settings.LLM_CALLBACK_INTEL_TIMEOUT", 2)

    state = _make_state()
    state.intelligence.bank_accounts = ["123456789123"]  # victim-owned false positive
    state.intelligence.upi_ids = []
    state.intelligence.phone_numbers = ["+919999999999"]
    state.intelligence.phishing_links = []
    state.intelligence.suspicious_keywords = ["urgent"]
    state.conversation_history = [
        {"sender": "scammer", "text": "Pay to UPI fraudpay@upi now"},
        {"sender": "user", "text": "My account number is 123456789123"},
    ]

    class DummyVerifier:
        async def verify_callback_intelligence(self, conversation_lines, current_intelligence):
            return CallbackVerificationResult(
                bank_accounts=[],
                upi_ids=["fraudpay@upi"],
                phishing_links=[],
                phone_numbers=["+919999999999"],
                suspicious_keywords=["urgent"],
                raw={}
            )

    class DummySessionManager:
        def __init__(self, initial_state):
            self._state = initial_state

        async def get_session(self, _session_id):
            return self._state

        async def replace_intelligence(self, _session_id, intelligence):
            self._state.intelligence = intelligence
            return self._state

        async def add_agent_note(self, _session_id, _note):
            return self._state

    orch = ConversationOrchestrator.__new__(ConversationOrchestrator)
    orch.llm_intel_enricher = DummyVerifier()
    orch.session_manager = DummySessionManager(state)

    result = asyncio.run(ConversationOrchestrator._verify_callback_intelligence(orch, state))

    assert result.intelligence.bank_accounts == []
    assert result.intelligence.upi_ids == ["fraudpay@upi"]
    assert result.intelligence.phone_numbers == ["+919999999999"]


def test_callback_verification_fail_open(monkeypatch):
    monkeypatch.setattr("app.config.settings.ENABLE_LLM_CALLBACK_VERIFICATION", True)
    monkeypatch.setattr("app.config.settings.LLM_CALLBACK_INTEL_TIMEOUT", 2)

    state = _make_state()
    state.intelligence.bank_accounts = ["999988887777"]
    state.conversation_history = [{"sender": "scammer", "text": "pay now"}]

    class DummyVerifier:
        async def verify_callback_intelligence(self, conversation_lines, current_intelligence):
            raise RuntimeError("simulated verifier failure")

    class DummySessionManager:
        def __init__(self, initial_state):
            self._state = initial_state

        async def get_session(self, _session_id):
            return self._state

        async def replace_intelligence(self, _session_id, intelligence):
            self._state.intelligence = intelligence
            return self._state

        async def add_agent_note(self, _session_id, _note):
            return self._state

    orch = ConversationOrchestrator.__new__(ConversationOrchestrator)
    orch.llm_intel_enricher = DummyVerifier()
    orch.session_manager = DummySessionManager(state)

    result = asyncio.run(ConversationOrchestrator._verify_callback_intelligence(orch, state))
    assert result.intelligence.bank_accounts == ["999988887777"]
