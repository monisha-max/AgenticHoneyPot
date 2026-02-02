"""
Callback Manager for GUVI Evaluation Endpoint
Handles sending final intelligence results to the GUVI platform
"""

import logging
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime

import httpx

from app.config import settings
from app.api.schemas import (
    SessionState,
    GuviCallbackPayload,
    ExtractedIntelligence,
    ConversationPhase,
    PersonaType,
    EmotionalState,
    ScamType
)

logger = logging.getLogger(__name__)


class CallbackManager:
    """
    Manages callbacks to the GUVI evaluation endpoint
    Implements retry logic and error handling
    """

    def __init__(
            self,
            callback_url: str = None,
            timeout: int = None,
            max_retries: int = None
    ):
        """
        Initialize callback manager

        Args:
            callback_url: GUVI callback endpoint URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.callback_url = callback_url or settings.GUVI_CALLBACK_URL
        self.timeout = timeout or settings.GUVI_CALLBACK_TIMEOUT
        self.max_retries = max_retries or settings.GUVI_CALLBACK_RETRIES

        # Track callback status
        self._pending_callbacks: Dict[str, Dict[str, Any]] = {}
        self._failed_callbacks: Dict[str, Dict[str, Any]] = {}

        logger.info(f"Initialized CallbackManager with URL: {self.callback_url}")

    def _build_payload(self, state: SessionState) -> GuviCallbackPayload:
        """
        Build callback payload from session state

        Args:
            state: Current session state

        Returns:
            GuviCallbackPayload ready to send
        """
        # Convert intelligence to the expected format
        intelligence_dict = {
            "bankAccounts": state.intelligence.bank_accounts,
            "upiIds": state.intelligence.upi_ids,
            "phishingLinks": state.intelligence.phishing_links,
            "phoneNumbers": state.intelligence.phone_numbers,
            "suspiciousKeywords": state.intelligence.suspicious_keywords,
            "emailAddresses": state.intelligence.email_addresses,
            "ifscCodes": state.intelligence.ifsc_codes,
            "scammerNames": state.intelligence.scammer_names,
            "fakeReferences": state.intelligence.fake_references,
            # Add URL analysis results
            "urlAnalysis": [
                {
                    "url": ua.url,
                    "riskLevel": ua.risk_level,
                    "isPhishing": ua.is_phishing,
                    "brandImpersonation": ua.brand_impersonation,
                    "findings": ua.findings
                }
                for ua in state.intelligence.url_analysis
            ] if state.intelligence.url_analysis else []
        }

        # Build agent notes summary
        agent_notes = self._generate_agent_notes(state)

        return GuviCallbackPayload(
            sessionId=state.session_id,
            scamDetected=state.scam_detected,
            totalMessagesExchanged=state.turn_count * 2,  # Both scammer and agent messages
            extractedIntelligence=intelligence_dict,
            agentNotes=agent_notes
        )

    def _generate_agent_notes(self, state: SessionState) -> str:
        """
        Generate comprehensive summary of agent observations
        Enhanced for better GUVI scoring

        Args:
            state: Session state with notes

        Returns:
            Formatted agent notes string for GUVI scoring
        """
        notes_parts = []

        # =====================================================================
        # SCAM CLASSIFICATION
        # =====================================================================
        if state.scam_type and state.scam_type != ScamType.UNKNOWN:
            scam_descriptions = {
                ScamType.BANKING_FRAUD: "Banking fraud - Scammer impersonated bank official to steal credentials",
                ScamType.UPI_FRAUD: "UPI fraud - Attempted to extract UPI PIN/ID for unauthorized transactions",
                ScamType.KYC_SCAM: "KYC scam - Fake KYC update request to harvest personal data",
                ScamType.JOB_SCAM: "Job scam - Fraudulent job offer requiring upfront payment",
                ScamType.LOTTERY_SCAM: "Lottery scam - Fake prize claim requiring processing fee",
                ScamType.TECH_SUPPORT: "Tech support scam - Fake virus alert to gain remote access",
                ScamType.INVESTMENT_FRAUD: "Investment fraud - Ponzi/pyramid scheme with fake returns",
                ScamType.DELIVERY_SCAM: "Delivery scam - Fake customs/delivery charges",
                ScamType.TAX_GST_SCAM: "Tax scam - Fake tax refund or penalty notice",
                ScamType.CRYPTO_SCAM: "Crypto scam - Fraudulent cryptocurrency investment",
                ScamType.IMPERSONATION: "Impersonation scam - Posing as friend/relative in distress"
            }
            desc = scam_descriptions.get(state.scam_type, state.scam_type.value)
            notes_parts.append(f"[SCAM DETECTED] {desc}")

        # =====================================================================
        # DETECTION CONFIDENCE
        # =====================================================================
        confidence_level = "HIGH" if state.confidence_score > 0.8 else ("MEDIUM" if state.confidence_score > 0.5 else "LOW")
        notes_parts.append(f"[CONFIDENCE: {confidence_level}] Detection score: {state.confidence_score:.0%}")

        # =====================================================================
        # ENGAGEMENT METRICS
        # =====================================================================
        phase_descriptions = {
            ConversationPhase.ENGAGE: "Initial engagement - building trust",
            ConversationPhase.PROBE: "Probing phase - extracting information",
            ConversationPhase.EXTRACT: "Extraction phase - gathering intelligence",
            ConversationPhase.STALL: "Stalling phase - wasting scammer time",
            ConversationPhase.COMPLETE: "Completed - sufficient intelligence gathered"
        }
        phase_desc = phase_descriptions.get(state.conversation_phase, state.conversation_phase.value)
        notes_parts.append(f"[ENGAGEMENT] {state.turn_count} conversation turns. Phase: {phase_desc}")

        # =====================================================================
        # PERSONA & STRATEGY
        # =====================================================================
        persona_names = {
            PersonaType.RAMU_UNCLE: "Ramu Uncle (62yo retired clerk, trusting, low-tech)",
            PersonaType.ANANYA_STUDENT: "Ananya (21yo college student, skeptical but curious)",
            PersonaType.AARTI_HOMEMAKER: "Aarti (38yo homemaker, cautious, family-oriented)",
            PersonaType.VIKRAM_IT: "Vikram (29yo IT professional, tech-savvy, analytical)",
            PersonaType.SUNITA_SHOP: "Sunita (45yo shop owner, business-minded, practical)"
        }
        notes_parts.append(f"[PERSONA] {persona_names.get(state.persona, state.persona.value)}")

        # =====================================================================
        # SCAMMER TACTICS OBSERVED
        # =====================================================================
        tactics_observed = []
        if state.agent_notes:
            for note in state.agent_notes:
                if "Tactics:" in note or "Detected:" in note:
                    tactics_observed.append(note)

        # Analyze message patterns for tactics
        common_tactics = []
        intel = state.intelligence
        if any(kw in str(intel.suspicious_keywords).lower() for kw in ['urgent', 'immediately', 'now']):
            common_tactics.append("Urgency creation")
        if any(kw in str(intel.suspicious_keywords).lower() for kw in ['block', 'suspend', 'legal', 'police']):
            common_tactics.append("Fear/threat tactics")
        if any(kw in str(intel.suspicious_keywords).lower() for kw in ['won', 'prize', 'lottery', 'selected']):
            common_tactics.append("Reward/greed exploitation")
        if intel.scammer_names:
            common_tactics.append("Authority impersonation")
        if intel.phishing_links:
            common_tactics.append("Phishing link distribution")

        if common_tactics or tactics_observed:
            all_tactics = list(set(common_tactics + [t.replace("Tactics:", "").replace("Detected:", "").strip() for t in tactics_observed[:3]]))
            notes_parts.append(f"[TACTICS OBSERVED] {'; '.join(all_tactics[:5])}")

        # =====================================================================
        # INTELLIGENCE EXTRACTED (with actual values)
        # =====================================================================
        intel_details = []

        if intel.upi_ids:
            intel_details.append(f"UPI IDs: {', '.join(intel.upi_ids[:3])}")
        if intel.phone_numbers:
            intel_details.append(f"Phone numbers: {', '.join(intel.phone_numbers[:3])}")
        if intel.bank_accounts:
            intel_details.append(f"Bank accounts: {', '.join(intel.bank_accounts[:2])}")
        if intel.ifsc_codes:
            intel_details.append(f"IFSC codes: {', '.join(intel.ifsc_codes[:2])}")
        if intel.email_addresses:
            intel_details.append(f"Emails: {', '.join(intel.email_addresses[:2])}")
        if intel.scammer_names:
            intel_details.append(f"Names used: {', '.join(intel.scammer_names[:3])}")
        if intel.fake_references:
            intel_details.append(f"Fake references: {', '.join(intel.fake_references[:2])}")
        if intel.phishing_links:
            intel_details.append(f"Malicious URLs: {', '.join(intel.phishing_links[:2])}")

        if intel_details:
            notes_parts.append(f"[INTELLIGENCE EXTRACTED] {' | '.join(intel_details)}")
        else:
            notes_parts.append("[INTELLIGENCE] Limited extraction - scammer was cautious or conversation was short")

        # =====================================================================
        # KEYWORDS DETECTED
        # =====================================================================
        if intel.suspicious_keywords:
            top_keywords = intel.suspicious_keywords[:10]
            clean_keywords = [kw.replace('*', '').strip() for kw in top_keywords if kw]
            if clean_keywords:
                notes_parts.append(f"[KEYWORDS] {', '.join(clean_keywords)}")

        # =====================================================================
        # URL ANALYSIS
        # =====================================================================
        if intel.url_analysis:
            phishing_urls = [ua for ua in intel.url_analysis if ua.is_phishing]
            if phishing_urls:
                notes_parts.append(f"[PHISHING ALERT] {len(phishing_urls)} malicious URL(s) detected")
                for ua in phishing_urls[:2]:
                    url_info = f"URL: {ua.url} | Risk: {ua.risk_level}"
                    if ua.brand_impersonation:
                        url_info += f" | Impersonating: {ua.brand_impersonation}"
                    notes_parts.append(url_info)

        # =====================================================================
        # AGENT EFFECTIVENESS SUMMARY
        # =====================================================================
        effectiveness_score = 0
        if intel.upi_ids: effectiveness_score += 20
        if intel.phone_numbers: effectiveness_score += 15
        if intel.bank_accounts: effectiveness_score += 20
        if intel.phishing_links: effectiveness_score += 15
        if intel.scammer_names: effectiveness_score += 10
        if state.turn_count >= 3: effectiveness_score += 10
        if state.confidence_score > 0.8: effectiveness_score += 10

        effectiveness_level = "EXCELLENT" if effectiveness_score >= 70 else ("GOOD" if effectiveness_score >= 40 else "MODERATE")
        notes_parts.append(f"[AGENT EFFECTIVENESS: {effectiveness_level}] Intelligence score: {effectiveness_score}/100")

        return " || ".join(notes_parts)

    async def send_callback(
            self,
            state: SessionState,
            retry_count: int = 0
    ) -> Dict[str, Any]:
        """
        Send callback to GUVI endpoint

        Args:
            state: Session state to send
            retry_count: Current retry attempt

        Returns:
            Response from callback endpoint
        """
        payload = self._build_payload(state)
        session_id = state.session_id

        logger.info(
            f"Sending callback for session {session_id} "
            f"(attempt {retry_count + 1}/{self.max_retries + 1})"
        )

        # Track as pending
        self._pending_callbacks[session_id] = {
            "payload": payload.model_dump(),
            "timestamp": datetime.utcnow().isoformat(),
            "retry_count": retry_count
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.callback_url,
                    json=payload.model_dump(),
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout
                )

                response_data = {
                    "status_code": response.status_code,
                    "success": response.is_success,
                    "session_id": session_id,
                    "timestamp": datetime.utcnow().isoformat()
                }

                if response.is_success:
                    logger.info(f"Callback successful for session {session_id}")
                    # Remove from pending
                    self._pending_callbacks.pop(session_id, None)

                    try:
                        response_data["response_body"] = response.json()
                    except Exception:
                        response_data["response_body"] = response.text

                    return response_data

                else:
                    logger.warning(
                        f"Callback failed for session {session_id}: "
                        f"Status {response.status_code}"
                    )
                    response_data["error"] = response.text

                    # Retry if attempts remaining
                    if retry_count < self.max_retries:
                        return await self._retry_callback(state, retry_count)

                    # Move to failed callbacks
                    self._failed_callbacks[session_id] = {
                        "payload": payload.model_dump(),
                        "error": response.text,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    self._pending_callbacks.pop(session_id, None)

                    return response_data

        except httpx.TimeoutException as e:
            logger.error(f"Callback timeout for session {session_id}: {str(e)}")

            if retry_count < self.max_retries:
                return await self._retry_callback(state, retry_count)

            self._failed_callbacks[session_id] = {
                "payload": payload.model_dump(),
                "error": f"Timeout: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }
            self._pending_callbacks.pop(session_id, None)

            return {
                "status_code": 0,
                "success": False,
                "session_id": session_id,
                "error": f"Timeout: {str(e)}"
            }

        except Exception as e:
            logger.error(f"Callback error for session {session_id}: {str(e)}")

            if retry_count < self.max_retries:
                return await self._retry_callback(state, retry_count)

            self._failed_callbacks[session_id] = {
                "payload": payload.model_dump(),
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            self._pending_callbacks.pop(session_id, None)

            return {
                "status_code": 0,
                "success": False,
                "session_id": session_id,
                "error": str(e)
            }

    async def _retry_callback(
            self,
            state: SessionState,
            current_retry: int
    ) -> Dict[str, Any]:
        """
        Retry callback with exponential backoff

        Args:
            state: Session state to send
            current_retry: Current retry count

        Returns:
            Response from retry attempt
        """
        # Exponential backoff: 1s, 2s, 4s, ...
        wait_time = 2 ** current_retry
        logger.info(f"Retrying callback in {wait_time}s...")

        await asyncio.sleep(wait_time)
        return await self.send_callback(state, current_retry + 1)

    async def send_callback_async(self, state: SessionState) -> None:
        """
        Send callback in background (fire and forget)

        Args:
            state: Session state to send
        """
        asyncio.create_task(self.send_callback(state))
        logger.info(f"Callback queued for session {state.session_id}")

    def get_pending_callbacks(self) -> Dict[str, Dict[str, Any]]:
        """Get all pending callbacks"""
        return self._pending_callbacks.copy()

    def get_failed_callbacks(self) -> Dict[str, Dict[str, Any]]:
        """Get all failed callbacks"""
        return self._failed_callbacks.copy()

    async def retry_failed_callbacks(self) -> Dict[str, Any]:
        """
        Retry all failed callbacks

        Returns:
            Results of retry attempts
        """
        results = {}

        for session_id, callback_data in list(self._failed_callbacks.items()):
            # Reconstruct state from payload
            payload = callback_data["payload"]
            extracted_intel = payload.get("extractedIntelligence", {})

            # Create minimal state for retry with proper enum types
            state = SessionState(
                session_id=payload["sessionId"],
                persona=PersonaType.RAMU_UNCLE,  # Default enum value
                emotional_state=EmotionalState.INITIAL,  # Proper enum
                conversation_phase=ConversationPhase.COMPLETE,  # Proper enum
                turn_count=payload["totalMessagesExchanged"] // 2,
                scam_detected=payload["scamDetected"],
                intelligence=ExtractedIntelligence(
                    bank_accounts=extracted_intel.get("bankAccounts", []),
                    upi_ids=extracted_intel.get("upiIds", []),
                    phishing_links=extracted_intel.get("phishingLinks", []),
                    phone_numbers=extracted_intel.get("phoneNumbers", []),
                    suspicious_keywords=extracted_intel.get("suspiciousKeywords", []),
                    email_addresses=extracted_intel.get("emailAddresses", []),
                    ifsc_codes=extracted_intel.get("ifscCodes", []),
                    scammer_names=extracted_intel.get("scammerNames", []),
                    fake_references=extracted_intel.get("fakeReferences", [])
                ),
                agent_notes=[payload.get("agentNotes", "")]
            )

            result = await self.send_callback(state)
            results[session_id] = result

            if result.get("success"):
                self._failed_callbacks.pop(session_id, None)

        return results


# ============================================================================
# COMPLETION DETECTOR
# ============================================================================

class CompletionDetector:
    """
    Detects when a conversation should be considered complete
    Triggers callback when appropriate conditions are met
    """

    def __init__(self, callback_manager: CallbackManager = None):
        self.callback_manager = callback_manager or CallbackManager()

    def should_complete(self, state: SessionState) -> bool:
        """
        Check if conversation should be marked complete

        Args:
            state: Current session state

        Returns:
            True if conversation should complete
        """
        # Max turns reached
        if state.turn_count >= settings.MAX_CONVERSATION_TURNS:
            logger.info(f"Session {state.session_id}: Max turns reached")
            return True

        # Sufficient intelligence gathered
        if self._has_sufficient_intelligence(state):
            logger.info(f"Session {state.session_id}: Sufficient intelligence")
            return True

        # Already complete - use enum comparison
        if state.conversation_phase == ConversationPhase.COMPLETE:
            return True

        return False

    def _has_sufficient_intelligence(self, state: SessionState) -> bool:
        """
        Check if we've gathered enough intelligence

        Args:
            state: Current session state

        Returns:
            True if sufficient intelligence gathered
        """
        intel = state.intelligence

        # Consider complete if we have at least one of each key identifier
        has_contact = len(intel.phone_numbers) > 0 or len(intel.email_addresses) > 0
        has_payment = len(intel.upi_ids) > 0 or len(intel.bank_accounts) > 0
        has_keywords = len(intel.suspicious_keywords) >= 3

        # Need contact + payment info, or extended conversation with keywords
        if has_contact and has_payment:
            return True

        if has_keywords and state.turn_count >= 10:
            return True

        return False

    def get_completion_reason(self, state: SessionState) -> Optional[str]:
        """
        Get reason for completion

        Args:
            state: Current session state

        Returns:
            Completion reason string
        """
        if state.turn_count >= settings.MAX_CONVERSATION_TURNS:
            return "max_turns_reached"

        if self._has_sufficient_intelligence(state):
            return "sufficient_intelligence"

        if state.conversation_phase == ConversationPhase.COMPLETE:
            return "manually_completed"

        return None
