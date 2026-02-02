"""
Emotional State Machine
Manages emotional progression for realistic persona behavior
"""

import logging
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from app.api.schemas import EmotionalState, PersonaType, ConversationPhase

logger = logging.getLogger(__name__)


@dataclass
class EmotionalTransition:
    """Defines a possible emotional transition"""
    from_state: EmotionalState
    to_state: EmotionalState
    trigger: str  # What causes this transition
    probability: float  # Base probability of transition


class EmotionalStateMachine:
    """
    Manages emotional state transitions for personas
    Makes responses feel more human and realistic
    """

    def __init__(self):
        self._init_state_graph()
        self._init_persona_modifiers()

    def _init_state_graph(self):
        """Initialize the state transition graph"""

        # Define valid transitions
        self.transitions: Dict[EmotionalState, Dict[EmotionalState, float]] = {
            EmotionalState.INITIAL: {
                EmotionalState.CONFUSED: 0.5,
                EmotionalState.WORRIED: 0.3,
                EmotionalState.SKEPTICAL: 0.2
            },
            EmotionalState.CONFUSED: {
                EmotionalState.WORRIED: 0.4,
                EmotionalState.SKEPTICAL: 0.2,
                EmotionalState.TRUSTING: 0.3,
                EmotionalState.CONFUSED: 0.1  # Stay confused
            },
            EmotionalState.WORRIED: {
                EmotionalState.PANICKED: 0.3,
                EmotionalState.HESITANT: 0.3,
                EmotionalState.TRUSTING: 0.2,
                EmotionalState.WORRIED: 0.2
            },
            EmotionalState.PANICKED: {
                EmotionalState.COMPLIANT: 0.4,
                EmotionalState.HESITANT: 0.3,
                EmotionalState.TRUSTING: 0.2,
                EmotionalState.PANICKED: 0.1
            },
            EmotionalState.HESITANT: {
                EmotionalState.TRUSTING: 0.4,
                EmotionalState.SKEPTICAL: 0.3,
                EmotionalState.WORRIED: 0.2,
                EmotionalState.HESITANT: 0.1
            },
            EmotionalState.SKEPTICAL: {
                EmotionalState.HESITANT: 0.3,
                EmotionalState.TRUSTING: 0.3,  # Can be convinced
                EmotionalState.CONFUSED: 0.2,
                EmotionalState.SKEPTICAL: 0.2
            },
            EmotionalState.TRUSTING: {
                EmotionalState.COMPLIANT: 0.5,
                EmotionalState.HESITANT: 0.2,
                EmotionalState.TRUSTING: 0.3
            },
            EmotionalState.COMPLIANT: {
                EmotionalState.COMPLIANT: 0.7,
                EmotionalState.HESITANT: 0.2,
                EmotionalState.TRUSTING: 0.1
            }
        }

        # Triggers that influence transitions
        self.triggers = {
            "urgency": [EmotionalState.WORRIED, EmotionalState.PANICKED],
            "threat": [EmotionalState.WORRIED, EmotionalState.PANICKED],
            "authority": [EmotionalState.TRUSTING, EmotionalState.COMPLIANT],
            "technical": [EmotionalState.CONFUSED],
            "reassurance": [EmotionalState.TRUSTING, EmotionalState.HESITANT],
            "pressure": [EmotionalState.PANICKED, EmotionalState.COMPLIANT],
            "reward": [EmotionalState.TRUSTING, EmotionalState.SKEPTICAL]
        }

    def _init_persona_modifiers(self):
        """Initialize persona-specific emotional tendencies"""

        self.persona_modifiers: Dict[PersonaType, Dict[EmotionalState, float]] = {
            PersonaType.RAMU_UNCLE: {
                EmotionalState.CONFUSED: 1.3,  # More likely to be confused
                EmotionalState.TRUSTING: 1.2,  # More trusting of authority
                EmotionalState.SKEPTICAL: 0.7  # Less skeptical
            },
            PersonaType.ANANYA_STUDENT: {
                EmotionalState.SKEPTICAL: 1.3,  # Initially more skeptical
                EmotionalState.TRUSTING: 0.9,
                EmotionalState.WORRIED: 0.8  # Less worried
            },
            PersonaType.AARTI_HOMEMAKER: {
                EmotionalState.WORRIED: 1.3,  # More worried about family
                EmotionalState.HESITANT: 1.2,  # Needs husband's approval
                EmotionalState.COMPLIANT: 0.9
            },
            PersonaType.VIKRAM_IT: {
                EmotionalState.SKEPTICAL: 1.4,  # Very skeptical initially
                EmotionalState.CONFUSED: 0.6,  # Less confused by tech
                EmotionalState.TRUSTING: 0.8
            },
            PersonaType.SUNITA_SHOP: {
                EmotionalState.WORRIED: 1.2,  # Worried about business
                EmotionalState.TRUSTING: 1.1,  # Trusts "company people"
                EmotionalState.CONFUSED: 1.2  # Confused by tech
            }
        }

    def get_current_state_description(self, state: EmotionalState) -> str:
        """Get description of emotional state for LLM context"""

        descriptions = {
            EmotionalState.INITIAL: "neutral and curious, just received a new message",
            EmotionalState.CONFUSED: "confused and trying to understand what's happening",
            EmotionalState.WORRIED: "worried and concerned about the situation",
            EmotionalState.PANICKED: "panicked and desperate to resolve the issue",
            EmotionalState.HESITANT: "hesitant and unsure whether to proceed",
            EmotionalState.SKEPTICAL: "skeptical and questioning the legitimacy",
            EmotionalState.TRUSTING: "trusting and willing to cooperate",
            EmotionalState.COMPLIANT: "compliant and ready to follow instructions"
        }

        return descriptions.get(state, "neutral")

    def transition(
            self,
            current_state: EmotionalState,
            persona_type: PersonaType,
            scammer_tactics: List[str],
            turn_count: int
    ) -> EmotionalState:
        """
        Calculate next emotional state based on various factors

        Args:
            current_state: Current emotional state
            persona_type: The persona being used
            scammer_tactics: Tactics detected in scammer's message
            turn_count: Number of conversation turns

        Returns:
            New emotional state
        """
        # Get base transition probabilities
        possible_transitions = self.transitions.get(
            current_state,
            {EmotionalState.CONFUSED: 1.0}
        )

        # Apply persona modifiers
        adjusted_probs = {}
        persona_mods = self.persona_modifiers.get(persona_type, {})

        for next_state, base_prob in possible_transitions.items():
            modifier = persona_mods.get(next_state, 1.0)
            adjusted_probs[next_state] = base_prob * modifier

        # Apply scammer tactic influence
        for tactic in scammer_tactics:
            if tactic in self.triggers:
                for favored_state in self.triggers[tactic]:
                    if favored_state in adjusted_probs:
                        adjusted_probs[favored_state] *= 1.3

        # Natural progression over time (become more trusting as conversation continues)
        if turn_count > 5:
            if EmotionalState.TRUSTING in adjusted_probs:
                adjusted_probs[EmotionalState.TRUSTING] *= 1.2
            if EmotionalState.COMPLIANT in adjusted_probs:
                adjusted_probs[EmotionalState.COMPLIANT] *= 1.1

        # Normalize probabilities
        total = sum(adjusted_probs.values())
        normalized = {k: v / total for k, v in adjusted_probs.items()}

        # Weighted random selection
        states = list(normalized.keys())
        probs = list(normalized.values())

        new_state = random.choices(states, weights=probs, k=1)[0]

        logger.debug(
            f"Emotional transition: {current_state.value} -> {new_state.value} "
            f"(persona: {persona_type.value}, tactics: {scammer_tactics})"
        )

        return new_state

    def detect_scammer_tactics(self, message: str) -> List[str]:
        """
        Detect tactics used by scammer in message

        Args:
            message: Scammer's message

        Returns:
            List of detected tactics
        """
        message_lower = message.lower()
        detected = []

        tactic_keywords = {
            "urgency": [
                "immediately", "urgent", "now", "today", "expire",
                "last chance", "hurry", "quickly", "jaldi", "abhi"
            ],
            "threat": [
                "block", "suspend", "cancel", "legal", "police",
                "arrest", "fine", "penalty", "action", "terminate"
            ],
            "authority": [
                "bank", "government", "official", "department",
                "rbi", "police", "court", "manager", "officer"
            ],
            "technical": [
                "server", "system", "update", "kyc", "verification",
                "process", "technical", "error", "database"
            ],
            "reassurance": [
                "don't worry", "safe", "secure", "trust",
                "genuine", "real", "authentic", "verified"
            ],
            "pressure": [
                "only way", "must", "have to", "no choice",
                "compulsory", "mandatory", "required", "necessary"
            ],
            "reward": [
                "won", "prize", "cashback", "reward", "free",
                "bonus", "offer", "discount", "gift"
            ]
        }

        for tactic, keywords in tactic_keywords.items():
            if any(kw in message_lower for kw in keywords):
                detected.append(tactic)

        return detected

    def get_response_tone(
            self,
            emotional_state: EmotionalState,
            persona_type: PersonaType
    ) -> Dict[str, any]:
        """
        Get response characteristics based on emotional state

        Args:
            emotional_state: Current emotional state
            persona_type: The persona type

        Returns:
            Dictionary with tone characteristics
        """
        base_tones = {
            EmotionalState.INITIAL: {
                "tone": "curious",
                "verbosity": "medium",
                "question_tendency": 0.7,
                "compliance_tendency": 0.3
            },
            EmotionalState.CONFUSED: {
                "tone": "bewildered",
                "verbosity": "high",
                "question_tendency": 0.9,
                "compliance_tendency": 0.2
            },
            EmotionalState.WORRIED: {
                "tone": "anxious",
                "verbosity": "high",
                "question_tendency": 0.6,
                "compliance_tendency": 0.5
            },
            EmotionalState.PANICKED: {
                "tone": "desperate",
                "verbosity": "high",
                "question_tendency": 0.4,
                "compliance_tendency": 0.8
            },
            EmotionalState.HESITANT: {
                "tone": "uncertain",
                "verbosity": "medium",
                "question_tendency": 0.6,
                "compliance_tendency": 0.4
            },
            EmotionalState.SKEPTICAL: {
                "tone": "doubtful",
                "verbosity": "medium",
                "question_tendency": 0.8,
                "compliance_tendency": 0.2
            },
            EmotionalState.TRUSTING: {
                "tone": "cooperative",
                "verbosity": "medium",
                "question_tendency": 0.4,
                "compliance_tendency": 0.7
            },
            EmotionalState.COMPLIANT: {
                "tone": "agreeable",
                "verbosity": "low",
                "question_tendency": 0.2,
                "compliance_tendency": 0.9
            }
        }

        return base_tones.get(
            emotional_state,
            base_tones[EmotionalState.CONFUSED]
        )
