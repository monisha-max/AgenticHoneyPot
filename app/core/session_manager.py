"""
Session Manager for the Honeypot API
Handles conversation state persistence and retrieval
Supports both in-memory (dev) and Redis (production) backends
"""

import json
import sys
import logging
import sys
from datetime import datetime
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

from app.config import settings
from app.api.schemas import (
    SessionState,
    ExtractedIntelligence,
    PersonaType,
    ConversationPhase,
    EmotionalState,
    ScamType
)

logger = logging.getLogger(__name__)


# ============================================================================
# ABSTRACT SESSION STORE
# ============================================================================

class SessionStore(ABC):
    """Abstract base class for session storage"""

    @abstractmethod
    async def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data by ID"""
        pass

    @abstractmethod
    async def set(self, session_id: str, data: Dict[str, Any], ttl: int = None) -> bool:
        """Set session data"""
        pass

    @abstractmethod
    async def delete(self, session_id: str) -> bool:
        """Delete session"""
        pass

    @abstractmethod
    async def exists(self, session_id: str) -> bool:
        """Check if session exists"""
        pass


# ============================================================================
# IN-MEMORY SESSION STORE (for development)
# ============================================================================

class InMemorySessionStore(SessionStore):
    """
    Simple in-memory session store
    Use for development/testing only
    """

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}
        self._timestamps: Dict[str, datetime] = {}
        logger.info("Initialized in-memory session store")

    async def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session from memory"""
        # Check TTL
        if session_id in self._timestamps:
            age = (datetime.utcnow() - self._timestamps[session_id]).total_seconds()
            if age > settings.SESSION_TTL:
                await self.delete(session_id)
                return None

        return self._store.get(session_id)

    async def set(self, session_id: str, data: Dict[str, Any], ttl: int = None) -> bool:
        """Store session in memory"""
        self._store[session_id] = data
        self._timestamps[session_id] = datetime.utcnow()
        return True

    async def delete(self, session_id: str) -> bool:
        """Delete session from memory"""
        if session_id in self._store:
            del self._store[session_id]
        if session_id in self._timestamps:
            del self._timestamps[session_id]
        return True

    async def exists(self, session_id: str) -> bool:
        """Check if session exists in memory"""
        return session_id in self._store

    async def get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get all sessions (for stats)"""
        return self._store.copy()


# ============================================================================
# REDIS SESSION STORE (for production)
# ============================================================================

class RedisSessionStore(SessionStore):
    """
    Redis-based session store for production
    Provides persistence and distributed access
    """

    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or settings.REDIS_URL
        self._client = None
        logger.info(f"Initialized Redis session store: {self.redis_url}")

    async def _get_client(self):
        """Get or create Redis client"""
        if self._client is None:
            try:
                import redis.asyncio as redis
                import sys
                sys.stderr.write(f"DEBUG: Creating Redis client for {self.redis_url}\n")
                sys.stderr.flush()
                self._client = redis.from_url(self.redis_url)
            except ImportError:
                import sys
                sys.stderr.write("DEBUG: Redis package 'redis.asyncio' NOT found\n")
                sys.stderr.flush()
                logger.warning("Redis package not installed, falling back to in-memory store")
                raise
        return self._client

    async def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session from Redis"""
        try:
            client = await self._get_client()
            data = await client.get(f"honeypot:session:{session_id}")
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            import sys
            sys.stderr.write(f"DEBUG: Redis GET error: {e}\n")
            sys.stderr.flush()
            logger.error(f"Redis get error: {e}")
            return None

    async def set(self, session_id: str, data: Dict[str, Any], ttl: int = None) -> bool:
        """Store session in Redis"""
        try:
            client = await self._get_client()
            ttl = ttl or settings.SESSION_TTL
            await client.setex(
                f"honeypot:session:{session_id}",
                ttl,
                json.dumps(data, default=str)
            )
            return True
        except Exception as e:
            import sys
            sys.stderr.write(f"DEBUG: Redis SET error for {session_id}: {e}\n")
            sys.stderr.flush()
            logger.error(f"Redis set error: {e}")
            return False

    async def delete(self, session_id: str) -> bool:
        """Delete session from Redis"""
        try:
            client = await self._get_client()
            await client.delete(f"honeypot:session:{session_id}")
            return True
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False

    async def exists(self, session_id: str) -> bool:
        """Check if session exists in Redis"""
        try:
            client = await self._get_client()
            return await client.exists(f"honeypot:session:{session_id}") > 0
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False


# ============================================================================
# SESSION MANAGER
# ============================================================================

class SessionManager:
    """
    High-level session management interface
    Handles session creation, updates, and state transitions
    """

    def __init__(self, use_redis: bool = None, require_redis: bool = None):
        """
        Initialize session manager

        Args:
            use_redis: Whether to use Redis backend (default: False for dev)
        """
        if use_redis:
            try:
                self.store = RedisSessionStore()
                sys.stderr.write("DEBUG: session_manager initialized with RedisSessionStore\n")
                sys.stderr.flush()
            except Exception as e:
                sys.stderr.write(f"DEBUG: Failed to init RedisStore: {e}\n")
                sys.stderr.flush()
                logger.warning("Failed to initialize Redis, using in-memory store")
                self.store = InMemorySessionStore()
        else:
            sys.stderr.write("DEBUG: session_manager initialized with InMemorySessionStore\n")
            sys.stderr.flush()
            self.store = InMemorySessionStore()

        self._stats = {
            "total_sessions": 0,
            "active_sessions": 0,
            "scams_detected": 0,
            "callbacks_sent": 0
        }

    async def get_session(self, session_id: str) -> Optional[SessionState]:
        """
        Get session state by ID

        Args:
            session_id: Unique session identifier

        Returns:
            SessionState if found, None otherwise
        """
        data = await self.store.get(session_id)
        if data:
            return SessionState(**data)
        return None

    async def create_session(
            self,
            session_id: str,
            persona: PersonaType = PersonaType.RAMU_UNCLE
    ) -> SessionState:
        """
        Create new session with initial state

        Args:
            session_id: Unique session identifier
            persona: Initial persona to use

        Returns:
            Newly created SessionState
        """
        state = SessionState(
            session_id=session_id,
            persona=persona,
            emotional_state=EmotionalState.INITIAL,
            conversation_phase=ConversationPhase.ENGAGE,
            turn_count=0,
            scam_detected=False,
            scam_type=None,
            confidence_score=0.0,
            intelligence=ExtractedIntelligence(),
            conversation_history=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            agent_notes=[]
        )

        await self.store.set(session_id, state.model_dump())
        self._stats["total_sessions"] += 1
        self._stats["active_sessions"] += 1

        logger.info(f"Created new session: {session_id} with persona: {persona}")
        return state

    async def update_session(self, session_id: str, state: SessionState) -> bool:
        """
        Update existing session state

        Args:
            session_id: Session identifier
            state: Updated session state

        Returns:
            True if successful
        """
        state.updated_at = datetime.utcnow()
        success = await self.store.set(session_id, state.model_dump())

        if success:
            logger.debug(f"Updated session: {session_id}")
        return success

    async def update_detection(
            self,
            session_id: str,
            is_scam: bool,
            scam_type: ScamType,
            confidence: float
    ) -> SessionState:
        """
        Update session with detection results

        Args:
            session_id: Session identifier
            is_scam: Whether scam was detected
            scam_type: Type of scam detected
            confidence: Confidence score

        Returns:
            Updated SessionState
        """
        state = await self.get_session(session_id)
        if not state:
            raise ValueError(f"Session not found: {session_id}")

        # Store original value BEFORE updating to fix race condition
        was_previously_detected = state.scam_detected

        state.scam_detected = is_scam
        state.scam_type = scam_type
        state.confidence_score = max(state.confidence_score, confidence)

        # Only count new detections (wasn't detected before, but is now)
        if is_scam and not was_previously_detected:
            self._stats["scams_detected"] += 1

        await self.update_session(session_id, state)
        return state

    async def update_intelligence(
            self,
            session_id: str,
            intelligence: ExtractedIntelligence
    ) -> SessionState:
        """
        Update session with extracted intelligence

        Args:
            session_id: Session identifier
            intelligence: Extracted intelligence data

        Returns:
            Updated SessionState
        """
        state = await self.get_session(session_id)
        if not state:
            raise ValueError(f"Session not found: {session_id}")

        # Merge intelligence (deduplicate)
        state.intelligence.bank_accounts = list(set(
            state.intelligence.bank_accounts + intelligence.bank_accounts
        ))
        state.intelligence.upi_ids = list(set(
            state.intelligence.upi_ids + intelligence.upi_ids
        ))
        state.intelligence.phishing_links = list(set(
            state.intelligence.phishing_links + intelligence.phishing_links
        ))
        state.intelligence.phone_numbers = list(set(
            state.intelligence.phone_numbers + intelligence.phone_numbers
        ))
        state.intelligence.suspicious_keywords = list(set(
            state.intelligence.suspicious_keywords + intelligence.suspicious_keywords
        ))
        state.intelligence.email_addresses = list(set(
            state.intelligence.email_addresses + intelligence.email_addresses
        ))
        state.intelligence.ifsc_codes = list(set(
            state.intelligence.ifsc_codes + intelligence.ifsc_codes
        ))
        state.intelligence.scammer_names = list(set(
            state.intelligence.scammer_names + intelligence.scammer_names
        ))
        state.intelligence.fake_references = list(set(
            state.intelligence.fake_references + intelligence.fake_references
        ))
        existing_url_analysis = list(state.intelligence.url_analysis or [])
        incoming_url_analysis = list(intelligence.url_analysis or [])
        state.intelligence.url_analysis = existing_url_analysis + [
            item for item in incoming_url_analysis if item not in existing_url_analysis
        ]

        await self.update_session(session_id, state)
        return state

    async def replace_intelligence(
            self,
            session_id: str,
            intelligence: ExtractedIntelligence
    ) -> SessionState:
        """
        Replace intelligence fields (used by pre-callback verification).
        """
        state = await self.get_session(session_id)
        if not state:
            raise ValueError(f"Session not found: {session_id}")

        state.intelligence = intelligence
        await self.update_session(session_id, state)
        return state

    async def increment_turn(self, session_id: str) -> SessionState:
        """
        Increment conversation turn count and update phase if needed

        Args:
            session_id: Session identifier

        Returns:
            Updated SessionState
        """
        state = await self.get_session(session_id)
        if not state:
            raise ValueError(f"Session not found: {session_id}")

        state.turn_count += 1

        # IMPORTANT: Only advance phase if scam has been detected
        # Stay in ENGAGE until we confirm this is a scam conversation
        # This prevents asking for payment details in non-scam or early conversations
        if not state.scam_detected:
            # Stay in ENGAGE phase until scam is detected
            state.conversation_phase = ConversationPhase.ENGAGE
        else:
            # Scam detected - now advance phases based on turn count AFTER detection
            # Use turns since scam was detected, not total turns
            if state.turn_count <= settings.ENGAGE_PHASE_TURNS:
                state.conversation_phase = ConversationPhase.ENGAGE
            elif state.turn_count <= settings.ENGAGE_PHASE_TURNS + settings.PROBE_PHASE_TURNS:
                state.conversation_phase = ConversationPhase.PROBE
            elif state.turn_count <= settings.ENGAGE_PHASE_TURNS + settings.PROBE_PHASE_TURNS + settings.EXTRACT_PHASE_TURNS:
                state.conversation_phase = ConversationPhase.EXTRACT
            else:
                state.conversation_phase = ConversationPhase.STALL

        await self.update_session(session_id, state)
        return state

    async def add_message_to_history(
            self,
            session_id: str,
            sender: str,
            text: str,
            timestamp: datetime = None
    ) -> SessionState:
        """
        Add message to conversation history

        Args:
            session_id: Session identifier
            sender: Message sender
            text: Message text
            timestamp: Message timestamp

        Returns:
            Updated SessionState
        """
        state = await self.get_session(session_id)
        if not state:
            raise ValueError(f"Session not found: {session_id}")

        state.conversation_history.append({
            "sender": sender,
            "text": text,
            "timestamp": (timestamp or datetime.utcnow()).isoformat()
        })

        await self.update_session(session_id, state)
        return state

    async def add_agent_note(self, session_id: str, note: str) -> SessionState:
        """
        Add note about scammer behavior

        Args:
            session_id: Session identifier
            note: Note to add

        Returns:
            Updated SessionState
        """
        state = await self.get_session(session_id)
        if not state:
            raise ValueError(f"Session not found: {session_id}")

        state.agent_notes.append(note)
        await self.update_session(session_id, state)
        return state

    async def end_session(self, session_id: str) -> SessionState:
        """
        Mark session as complete

        Args:
            session_id: Session identifier

        Returns:
            Final SessionState
        """
        state = await self.get_session(session_id)
        if not state:
            raise ValueError(f"Session not found: {session_id}")

        state.conversation_phase = ConversationPhase.COMPLETE
        await self.update_session(session_id, state)

        self._stats["active_sessions"] -= 1

        logger.info(f"Ended session: {session_id}")
        return state

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete session completely

        Args:
            session_id: Session identifier

        Returns:
            True if successful
        """
        success = await self.store.delete(session_id)
        if success:
            self._stats["active_sessions"] -= 1
        return success

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get session statistics

        Returns:
            Dictionary with stats
        """
        return self._stats.copy()

    def record_callback_sent(self):
        """Record that a callback was sent"""
        self._stats["callbacks_sent"] += 1
