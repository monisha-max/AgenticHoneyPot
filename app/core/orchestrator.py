"""
Conversation Orchestrator
The central hub that coordinates all components of the honeypot system
"""

import logging
import re
from typing import Optional, List, Dict, Any
from datetime import datetime
import asyncio

from app.config import settings
from app.api.schemas import (
    Message,
    Metadata,
    SessionState,
    DetectionResult,
    ExtractedIntelligence,
    ScamType,
    PersonaType,
    ConversationPhase,
    EmotionalState
)
from app.core.session_manager import SessionManager
from app.core.callback_manager import CallbackManager, CompletionDetector

# Import all components
from app.detection.ensemble import ScamDetectionEnsemble
from app.detection.scam_taxonomy import ScamTaxonomy
from app.agent.persona_engine import PersonaEngine
from app.agent.emotional_state import EmotionalStateMachine
from app.agent.strategy_selector import StrategySelector
from app.agent.response_generator import ResponseGenerator
from app.extraction.entity_extractor import EntityExtractor, IntelligenceAggregator
from app.extraction.url_analyzer import URLAnalyzer
from app.extraction.llm_intelligence_enricher import (
    LLMIntelligenceEnricher,
    EnrichmentResult
)

logger = logging.getLogger(__name__)


class ConversationOrchestrator:
    """
    Main orchestrator that coordinates the entire honeypot workflow:
    1. Receives incoming messages
    2. Detects scam intent using ensemble detection
    3. Selects appropriate persona based on scam type
    4. Manages emotional state transitions
    5. Selects probing techniques
    6. Generates engaging responses using LLM
    7. Extracts intelligence from conversations
    8. Manages session state
    9. Triggers callbacks when complete
    """

    def __init__(self):
        """Initialize all components"""
        # Core managers
        import sys
        sys.stderr.write(f"DEBUG: orchestrator init: settings.USE_REDIS = {settings.USE_REDIS}\n")
        sys.stderr.flush()
        self.session_manager = SessionManager(use_redis=settings.USE_REDIS)
        self.callback_manager = CallbackManager()
        self.completion_detector = CompletionDetector(self.callback_manager)

        # Detection components
        self.scam_detector = ScamDetectionEnsemble()
        self.scam_taxonomy = ScamTaxonomy()

        # Agent components
        self.persona_engine = PersonaEngine()
        self.emotional_state_machine = EmotionalStateMachine()
        self.strategy_selector = StrategySelector()
        self.response_generator = ResponseGenerator()

        # Extraction components
        self.entity_extractor = EntityExtractor()
        self.intelligence_aggregator = IntelligenceAggregator()
        self.url_analyzer = URLAnalyzer()
        self.llm_intel_enricher = LLMIntelligenceEnricher()

        logger.info("Initialized ConversationOrchestrator with all components")

    async def process_message(
            self,
            session_id: str,
            message: Message,
            conversation_history: List[Message],
            metadata: Optional[Metadata] = None
    ) -> str:
        """
        Main entry point for processing incoming messages

        Args:
            session_id: Unique conversation identifier
            message: Current incoming message
            conversation_history: Previous messages in conversation
            metadata: Optional metadata about the message

        Returns:
            Agent's response string
        """
        # Start timing for response time tracking (engagement metrics)
        import time
        start_time = time.time()

        logger.info(f"Processing message for session {session_id}")

        try:
            # Step 1: Get or create session
            state = await self._get_or_create_session(session_id, message, metadata)
        except Exception as e:
            logger.exception(f"Failed to get/create session {session_id}: {e}")
            # Create a minimal state for fallback
            state = SessionState(
                session_id=session_id,
                persona=PersonaType.NEUTRAL_CITIZEN,
                emotional_state=EmotionalState.CONFUSED,
                conversation_phase=ConversationPhase.ENGAGE,
                turn_count=1
            )

        try:
            # Step 2: Add scammer's message to history
            await self.session_manager.add_message_to_history(
                session_id=session_id,
                sender=message.sender.value,
                text=message.text,
                timestamp=message.timestamp
            )
        except Exception as e:
            logger.warning(f"Failed to add message to history: {e}")

        try:
            # Step 3: Run scam detection
            state = await self._detect_scam(state, message, conversation_history)
        except Exception as e:
            logger.warning(f"Scam detection failed, continuing with defaults: {e}")

        try:
            # Step 4: Extract intelligence from message
            state = await self._extract_intelligence(state, message)
        except Exception as e:
            logger.warning(f"Intelligence extraction failed: {e}")

        try:
            # Step 4b: Conversation-aware LLM enrichment for soft entities (name/email)
            # Run during active scam sessions when these fields are still missing.
            state = await self._enrich_soft_entities_if_needed(state)
        except Exception as e:
            logger.warning(f"LLM soft enrichment skipped due to error: {e}")

        try:
            # Step 5: Update emotional state based on scammer tactics
            state = await self._update_emotional_state(state, message)
        except Exception as e:
            logger.warning(f"Emotional state update failed: {e}")

        try:
            # Step 6: Increment turn and update phase
            state = await self.session_manager.increment_turn(session_id)
        except Exception as e:
            logger.warning(f"Turn increment failed: {e}")
            state.turn_count += 1  # Manual increment

        try:
            # Step 7: Generate response using full pipeline
            response = await self._generate_response(state, message, conversation_history)
            # Persist state updates (e.g., used_techniques) made during response generation
            await self.session_manager.update_session(state.session_id, state)
        except Exception as e:
            logger.exception(
                "Response generation failed for session %s (persona=%s, phase=%s): %s",
                state.session_id,
                state.persona.value,
                state.conversation_phase.value,
                e
            )
            response = self._get_fallback_response(state, message.text)

        try:
            # Step 8: Add agent's response to history
            await self.session_manager.add_message_to_history(
                session_id=session_id,
                sender="user",
                text=response,
                timestamp=datetime.utcnow()
            )
        except Exception as e:
            logger.warning(f"Failed to add response to history: {e}")

        try:
            # Step 9: Check for completion
            await self._check_completion(state)
        except Exception as e:
            logger.warning(f"Completion check failed: {e}")

        # Track response time for engagement metrics
        try:
            response_time_ms = int((time.time() - start_time) * 1000)
            state = await self.session_manager.get_session(session_id)
            if state:
                if not hasattr(state, 'response_times_ms') or state.response_times_ms is None:
                    state.response_times_ms = []
                state.response_times_ms.append(response_time_ms)
                await self.session_manager.update_session(session_id, state)
                logger.debug(f"Response time for session {session_id}: {response_time_ms}ms")
        except Exception as e:
            logger.warning(f"Failed to track response time: {e}")

        logger.info(f"Generated response for session {session_id}")
        return response

    def _get_fallback_response(self, state: SessionState, scammer_message: str) -> str:
        """Generate a simple fallback response when the pipeline fails"""
        import random

        message_lower = scammer_message.lower()

        # Phase-based responses
        if state.turn_count <= 2:
            responses = [
                "Ji haan, main sun raha hoon. Aap kaun bol rahe ho?",
                "Arey? Kya hua? Mujhe samajh nahi aaya.",
                "Haan ji? Thoda detail mein batao."
            ]
        elif state.turn_count <= 5:
            responses = [
                "Accha, aur batao. Aapka naam kya hai?",
                "Theek hai, par aap apna contact number do pehle.",
                "Mujhe trust karna mushkil hai. Koi proof bhejo."
            ]
        else:
            responses = [
                "Ruko, mera beta aa gaya hai. Usse baat karo.",
                "Ek minute, phone ki battery low hai.",
                "Theek hai, baad mein baat karte hain."
            ]

        return random.choice(responses)

    async def _get_or_create_session(
            self,
            session_id: str,
            message: Message,
            metadata: Optional[Metadata]
    ) -> SessionState:
        """Get existing session or create new one"""
        state = await self.session_manager.get_session(session_id)

        if state is None:
            # Run initial detection to determine scam type
            history_texts = []
            detection_result = await self.scam_detector.detect(message.text, history_texts)

            # Check if message is primarily English
            is_english = self._is_primarily_english(message.text)
            
            # Select persona based on scam type and language
            persona = self.persona_engine.select_persona_for_scam(
                scam_type=detection_result.scam_type,
                is_english=is_english,
                message_text=message.text
            )
            
            log_reason = f"scam={detection_result.scam_type.value}, english={is_english}"

            # Create new session with selected persona
            state = await self.session_manager.create_session(session_id, persona)

            # Update with initial detection
            state = await self.session_manager.update_detection(
                session_id=session_id,
                is_scam=detection_result.is_scam,
                scam_type=detection_result.scam_type,
                confidence=detection_result.confidence
            )

            logger.info(
                f"Created new session {session_id} with persona {persona.value} "
                f"due to {log_reason}"
            )

        return state

    def _is_primarily_english(self, text: str) -> bool:
        """
        Check if text is primarily English (ASCII based).
        """
        if not text:
            return False
            
        # Check basic ASCII ratio (English characters, numbers, basic punctuation)
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        ascii_ratio = ascii_chars / len(text)
        
        # If it's mostly ASCII characters, treat it as English context
        # This prevents defaulting to Hinglish personas for English inputs
        return ascii_ratio >= 0.99

    async def _detect_scam(
            self,
            state: SessionState,
            message: Message,
            history: List[Message]
    ) -> SessionState:
        """Run ensemble scam detection"""
        was_scam = state.scam_detected
        # Build history text list
        history_texts = [msg.text for msg in history] if history else []

        # Run ensemble detection
        detection_result = await self.scam_detector.detect(message.text, history_texts)

        # Update session if detection is stronger
        if detection_result.confidence > state.confidence_score:
            state = await self.session_manager.update_detection(
                session_id=state.session_id,
                is_scam=detection_result.is_scam,
                scam_type=detection_result.scam_type,
                confidence=detection_result.confidence
            )

            # Add detection notes
            if detection_result.evidence:
                evidence_summary = ", ".join(detection_result.evidence[:3])
                await self.session_manager.add_agent_note(
                    state.session_id,
                    f"Detected: {evidence_summary}"
                )

            # Add tactics notes
            if detection_result.tactics_identified:
                tactics_summary = ", ".join(detection_result.tactics_identified[:3])
                await self.session_manager.add_agent_note(
                    state.session_id,
                    f"Tactics: {tactics_summary}"
                )

            # Add persuasive intent and reasoning notes
            if detection_result.intent:
                await self.session_manager.add_agent_note(
                    state.session_id,
                    f"Scammer Intent: {detection_result.intent}"
                )

            if detection_result.reasoning:
                await self.session_manager.add_agent_note(
                    state.session_id,
                    f"Analysis: {detection_result.reasoning}"
                )

        # If scam intent just got confirmed, refresh intelligence from recent history
        if not was_scam and state.scam_detected:
            state = await self._refresh_intelligence_from_history(state)
            # Track when scam was first detected (for engagement metrics)
            if state.scam_detected_at_turn is None or state.scam_detected_at_turn == 0:
                state.scam_detected_at_turn = state.turn_count + 1  # +1 because turn increments after detection
                await self.session_manager.update_session(state.session_id, state)

        # CRITICAL: Merge detected keywords into intelligence for callback payload
        if detection_result.keywords_found:
            existing_keywords = set(state.intelligence.suspicious_keywords)
            existing_keywords.update(detection_result.keywords_found)
            state.intelligence.suspicious_keywords = list(existing_keywords)[:20]  # Limit to 20
            await self.session_manager.update_intelligence(state.session_id, state.intelligence)

        return state

    async def _refresh_intelligence_from_history(self, state: SessionState) -> SessionState:
        """
        When scam is detected mid-conversation, re-extract intel from recent history
        so we don't ask for details already shared.
        """
        try:
            latest_state = await self.session_manager.get_session(state.session_id)
            if not latest_state or not latest_state.conversation_history:
                return state

            history_texts = [
                msg.get("text", "")
                for msg in latest_state.conversation_history
                if msg.get("sender") == "scammer"
            ]
            if not history_texts:
                return latest_state

            combined = " ".join(history_texts[-20:])
            if not combined.strip():
                return latest_state

            history_intel = self.entity_extractor.extract_all(combined)
            updated_state = await self.session_manager.update_intelligence(
                latest_state.session_id,
                history_intel
            )
            updated_state = await self._run_llm_enrichment(
                updated_state,
                history_texts,
                reason="scam_flip"
            )
            return updated_state
        except Exception as e:
            logger.warning(f"History re-extraction failed for {state.session_id}: {e}")
            return state

    def _merge_enriched_intelligence(
            self,
            existing: ExtractedIntelligence,
            enrichment: EnrichmentResult
    ) -> ExtractedIntelligence:
        """
        Merge LLM enrichment into intelligence.
        Guardrails:
        - confidence + evidence required
        - never remove existing findings
        - normalize and validate candidate formats per field
        """
        merged = ExtractedIntelligence(**existing.model_dump())
        min_conf = settings.LLM_INTEL_MIN_CONFIDENCE

        def _append_unique(bucket: List[str], value: str) -> None:
            if value and value not in bucket:
                bucket.append(value)

        list_fields = {
            "bank_accounts",
            "upi_ids",
            "phishing_links",
            "phone_numbers",
            "suspicious_keywords",
            "email_addresses",
            "ifsc_codes",
            "scammer_names",
            "fake_references",
        }

        for candidate in enrichment.candidates:
            if candidate.confidence < min_conf:
                continue
            if not candidate.evidence.strip():
                continue

            if candidate.field not in list_fields:
                continue

            normalized_values = self._normalize_enriched_values(candidate.field, candidate.value)
            if not normalized_values:
                continue

            bucket = getattr(merged, candidate.field, None)
            if not isinstance(bucket, list):
                continue
            for normalized in normalized_values:
                _append_unique(bucket, normalized)

        return merged

    def _normalize_enriched_values(self, field: str, value: str) -> List[str]:
        """Validate and normalize LLM enrichment values for safe merge."""
        raw = (value or "").strip()
        if not raw:
            return []

        if field == "scammer_names":
            cleaned = " ".join(raw.split())
            if not (2 <= len(cleaned) <= 60):
                return []
            if any(ch.isdigit() for ch in cleaned):
                return []
            tokens = cleaned.split()
            if len(tokens) > 4:
                return []
            return [cleaned.title()]

        if field == "email_addresses":
            cleaned = raw.lower()
            if re.fullmatch(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}", cleaned):
                return [cleaned]
            return []

        if field == "phone_numbers":
            digits = re.sub(r"\D", "", raw)
            if len(digits) == 10 and digits[0] in "123456789":
                return [f"+91{digits}"]
            if len(digits) == 12 and digits.startswith("91") and digits[2] in "123456789":
                return [f"+{digits}"]
            return []

        if field == "upi_ids":
            cleaned = raw.lower()
            if re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,}@[a-z]{2,15}", cleaned):
                return [cleaned]
            return []

        if field == "bank_accounts":
            digits = re.sub(r"\D", "", raw)
            if len(digits) in {10, 12}:
                return []
            if 9 <= len(digits) <= 18:
                return [digits]
            return []

        if field == "ifsc_codes":
            cleaned = raw.upper().replace(" ", "")
            if re.fullmatch(r"[A-Z]{4}0[A-Z0-9]{6}", cleaned):
                return [cleaned]
            return []

        if field == "phishing_links":
            cleaned = raw.strip()
            if cleaned.startswith("www."):
                cleaned = f"http://{cleaned}"
            if re.match(r"^https?://[^\s/$.?#].[^\s]*$", cleaned, re.IGNORECASE):
                return [cleaned]
            return []

        if field == "fake_references":
            cleaned = re.sub(r"[^A-Za-z0-9]", "", raw).upper()
            if 6 <= len(cleaned) <= 20:
                return [cleaned]
            return []

        if field == "suspicious_keywords":
            cleaned = " ".join(raw.lower().split())
            if 2 <= len(cleaned) <= 40:
                return [cleaned]
            return []

        return []

    async def _run_llm_enrichment(
            self,
            state: SessionState,
            scammer_messages: List[str],
            reason: str
    ) -> SessionState:
        """
        Run LLM enrichment with timeout and fail-open behavior.
        """
        if not settings.ENABLE_LLM_INTEL_ENRICHMENT:
            return state
        if not scammer_messages:
            return state

        try:
            enrichment = await asyncio.wait_for(
                self.llm_intel_enricher.enrich_recent_messages(scammer_messages),
                timeout=settings.LLM_INTEL_TIMEOUT
            )
            merged_intel = self._merge_enriched_intelligence(state.intelligence, enrichment)

            if merged_intel.model_dump() == state.intelligence.model_dump():
                return state

            updated_state = await self.session_manager.update_intelligence(
                state.session_id,
                merged_intel
            )

            tracked_fields = [
                "bank_accounts",
                "upi_ids",
                "phishing_links",
                "phone_numbers",
                "suspicious_keywords",
                "email_addresses",
                "ifsc_codes",
                "scammer_names",
                "fake_references",
            ]
            added_parts = []
            for field in tracked_fields:
                before = set(getattr(state.intelligence, field, []))
                after = set(getattr(updated_state.intelligence, field, []))
                delta = len(after - before)
                if delta > 0:
                    added_parts.append(f"{field}+{delta}")

            await self.session_manager.add_agent_note(
                state.session_id,
                f"LLM enrichment ({reason}): {', '.join(added_parts) if added_parts else 'no new fields'}"
            )
            return updated_state
        except asyncio.TimeoutError:
            logger.warning("LLM enrichment timed out for %s (%s)", state.session_id, reason)
            return state
        except Exception as e:
            logger.warning("LLM enrichment failed for %s (%s): %s", state.session_id, reason, e)
            return state

    async def _enrich_soft_entities_if_needed(self, state: SessionState) -> SessionState:
        """
        Keep intelligence up-to-date using conversation-aware LLM extraction.
        Only runs when scam has been detected and key fields are still missing.
        """
        if not state.scam_detected:
            return state
        if (
            state.intelligence.scammer_names
            and state.intelligence.email_addresses
            and state.intelligence.phone_numbers
            and state.intelligence.upi_ids
        ):
            return state

        latest_state = await self.session_manager.get_session(state.session_id)
        if not latest_state:
            return state

        scammer_messages = [
            m.get("text", "")
            for m in (latest_state.conversation_history or [])[-40:]
            if m.get("sender") == "scammer"
        ]
        if not scammer_messages:
            return latest_state

        return await self._run_llm_enrichment(
            latest_state,
            scammer_messages,
            reason="active_session"
        )

    async def _extract_intelligence(
            self,
            state: SessionState,
            message: Message
    ) -> SessionState:
        """Extract intelligence from message using full extractor"""
        # Extract from current message
        new_intel, new_items = self.intelligence_aggregator.process_message(
            message.text,
            state.intelligence
        )

        # Update session if new intelligence found
        if new_items:
            state = await self.session_manager.update_intelligence(
                state.session_id,
                new_intel
            )

            # Log what was found
            intel_summary = self.intelligence_aggregator.generate_summary(new_intel)
            await self.session_manager.add_agent_note(
                state.session_id,
                f"Intel: {intel_summary}"
            )

        # Analyze any new URLs found (async, don't block response)
        if new_intel.phishing_links:
            # Analyze URLs that haven't been analyzed yet
            analyzed_urls = {ua.url for ua in state.intelligence.url_analysis}
            new_urls = [url for url in new_intel.phishing_links if url not in analyzed_urls]

            if new_urls:
                try:
                    # Run URL analysis (with timeout to not slow down response)
                    url_results = await self._analyze_urls_safely(new_urls)

                    # Update intelligence with URL analysis
                    from app.api.schemas import URLAnalysisInfo
                    for result in url_results:
                        url_info = URLAnalysisInfo(
                            url=result.url,
                            risk_level=result.risk_level,
                            risk_score=result.risk_score,
                            is_phishing=result.is_suspicious,
                            brand_impersonation=result.detected_brand_impersonation,
                            findings=result.findings[:5],  # Limit findings
                            page_title=result.page_title[:100] if result.page_title else ""
                        )
                        new_intel.url_analysis.append(url_info)

                        # Add agent notes for high-risk URLs
                        if result.risk_level in ["HIGH", "CRITICAL"]:
                            await self.session_manager.add_agent_note(
                                state.session_id,
                                f"⚠️ PHISHING URL DETECTED: {result.url} (Risk: {result.risk_level})"
                            )
                            if result.detected_brand_impersonation:
                                await self.session_manager.add_agent_note(
                                    state.session_id,
                                    f"Brand impersonation: {result.detected_brand_impersonation}"
                                )

                    # Update session with URL analysis
                    state = await self.session_manager.update_intelligence(
                        state.session_id,
                        new_intel
                    )

                except Exception as e:
                    logger.warning(f"URL analysis failed (non-critical): {e}")

            logger.info(f"Extracted new intelligence: {new_items}")

        return state

    async def _analyze_urls_safely(self, urls: List[str], timeout: int = 8) -> List:
        """
        Analyze URLs with timeout to avoid slowing down response

        Args:
            urls: List of URLs to analyze
            timeout: Maximum time to wait for all analyses

        Returns:
            List of URLAnalysisResult
        """
        import asyncio

        try:
            # Limit to first 3 URLs to avoid long delays
            urls_to_analyze = urls[:3]

            # Run analysis with timeout
            results = await asyncio.wait_for(
                self.url_analyzer.analyze_multiple_urls(urls_to_analyze, max_concurrent=2),
                timeout=timeout
            )

            return results

        except asyncio.TimeoutError:
            logger.warning(f"URL analysis timed out after {timeout}s")
            return []
        except Exception as e:
            logger.error(f"URL analysis error: {e}")
            return []

    async def _update_emotional_state(
            self,
            state: SessionState,
            message: Message
    ) -> SessionState:
        """Update emotional state based on scammer tactics"""
        # Detect tactics in scammer's message
        tactics = self.emotional_state_machine.detect_scammer_tactics(message.text)

        # Calculate new emotional state
        new_emotion = self.emotional_state_machine.transition(
            current_state=state.emotional_state,
            persona_type=state.persona,
            scammer_tactics=tactics,
            turn_count=state.turn_count
        )

        # Update session state if emotion changed
        if new_emotion != state.emotional_state:
            state.emotional_state = new_emotion
            await self.session_manager.update_session(state.session_id, state)

            logger.debug(
                f"Emotional state transition: {state.emotional_state} -> {new_emotion}"
            )

        return state

    async def _generate_response(
            self,
            state: SessionState,
            message: Message,
            history: List[Message]
    ) -> str:
        """Generate response using full pipeline"""
        # USE SESSION'S STORED HISTORY (has both scammer + agent messages)
        # This is critical because GUVI may send empty conversationHistory
        # and rely on sessionId for state. The session already tracks both sides.
        history_dicts = state.conversation_history[-40:] if state.conversation_history else []

        # Fallback to request history if session history is empty (first message)
        if not history_dicts and history:
            for msg in history:
                history_dicts.append({
                    "sender": msg.sender.value,
                    "text": msg.text,
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None
                })

        # Generate response
        response = await self.response_generator.generate_response(
            state=state,
            scammer_message=message.text,
            conversation_history=history_dicts
        )

        return response

    async def _check_completion(self, state: SessionState) -> None:
        """Check if conversation should be completed and trigger callback"""
        # Refresh state
        state = await self.session_manager.get_session(state.session_id)

        if self.completion_detector.should_complete(state):
            # Final enrichment pass before closing/callback.
            scammer_messages = [
                m.get("text", "")
                for m in (state.conversation_history or [])[-40:]
                if m.get("sender") == "scammer"
            ]
            state = await self._run_llm_enrichment(
                state,
                scammer_messages,
                reason="pre_completion"
            )
            state = await self.session_manager.get_session(state.session_id) or state

            # Mark session as complete
            state = await self.session_manager.end_session(state.session_id)

            # Get completion reason
            reason = self.completion_detector.get_completion_reason(state)
            await self.session_manager.add_agent_note(
                state.session_id,
                f"Session completed: {reason}"
            )

            # Calculate intelligence score
            intel_score = self.intelligence_aggregator.get_intelligence_score(state.intelligence)
            await self.session_manager.add_agent_note(
                state.session_id,
                f"Intelligence score: {intel_score:.2f}"
            )

            # Send callback
            logger.info(f"Triggering callback for session {state.session_id}")
            await self.callback_manager.send_callback_async(state)
            self.session_manager.record_callback_sent()

    async def complete_session(self, session_id: str) -> Dict[str, Any]:
        """Manually complete a session and send callback"""
        state = await self.session_manager.get_session(session_id)
        if not state:
            raise ValueError(f"Session not found: {session_id}")

        # Mark as complete
        state = await self.session_manager.end_session(session_id)

        # Send callback
        result = await self.callback_manager.send_callback(state)
        self.session_manager.record_callback_sent()

        return result

    async def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of a session for debugging"""
        state = await self.session_manager.get_session(session_id)
        if not state:
            return {"error": "Session not found"}

        return {
            "session_id": state.session_id,
            "persona": state.persona.value,
            "emotional_state": state.emotional_state.value,
            "phase": state.conversation_phase.value,
            "turn_count": state.turn_count,
            "scam_detected": state.scam_detected,
            "scam_type": state.scam_type.value if state.scam_type else None,
            "confidence": state.confidence_score,
            "intelligence_summary": self.intelligence_aggregator.generate_summary(state.intelligence),
            "intelligence_score": self.intelligence_aggregator.get_intelligence_score(state.intelligence),
            "agent_notes": state.agent_notes[-5:] if state.agent_notes else []
        }
