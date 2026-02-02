"""
Probing Strategy Selector
Manages engagement strategies and probing techniques for each conversation phase
"""

import logging
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from app.api.schemas import (
    ConversationPhase, PersonaType, EmotionalState, ScamType
)

logger = logging.getLogger(__name__)


@dataclass
class ProbingTechnique:
    """Defines a probing technique"""
    name: str
    description: str
    goal: str
    example_prompts: List[str]
    target_info: List[str]  # What we're trying to extract
    persona_effectiveness: Dict[PersonaType, float]  # How effective with each persona


@dataclass
class PhaseStrategy:
    """Strategy for a conversation phase"""
    phase: ConversationPhase
    goal: str
    typical_turns: int
    techniques: List[str]
    success_criteria: List[str]
    transition_conditions: Dict[str, str]


class StrategySelector:
    """
    Selects and manages probing strategies based on conversation phase
    """

    def __init__(self):
        self._init_techniques()
        self._init_phase_strategies()

    def _init_techniques(self):
        """Initialize all probing techniques"""

        self.techniques: Dict[str, ProbingTechnique] = {

            # ================================================================
            # PHASE 1: ENGAGE TECHNIQUES
            # ================================================================

            "confused_echo": ProbingTechnique(
                name="Confused Echo",
                description="Repeat back what scammer said with confusion to appear genuine",
                goal="Build trust, appear like a real victim",
                example_prompts=[
                    "Account block? Matlab kya hua exactly?",
                    "wait what?? my account is blocked??",
                    "Arey! Bill pending hai? Yeh kaise hua?"
                ],
                target_info=[],
                persona_effectiveness={
                    PersonaType.RAMU_UNCLE: 0.95,
                    PersonaType.ANANYA_STUDENT: 0.85,
                    PersonaType.AARTI_HOMEMAKER: 0.90,
                    PersonaType.VIKRAM_IT: 0.60,
                    PersonaType.SUNITA_SHOP: 0.90
                }
            ),

            "family_reference": ProbingTechnique(
                name="Family Reference",
                description="Mention family member who usually helps, adds authenticity",
                goal="Appear genuine, create opportunity for delay tactics",
                example_prompts=[
                    "Mera beta usually yeh sab dekhta hai, woh Bangalore mein hai",
                    "my dad handles all this stuff, should i ask him?",
                    "Mera pati office mein hai, unse poochna padega"
                ],
                target_info=[],
                persona_effectiveness={
                    PersonaType.RAMU_UNCLE: 0.95,
                    PersonaType.ANANYA_STUDENT: 0.80,
                    PersonaType.AARTI_HOMEMAKER: 0.95,
                    PersonaType.VIKRAM_IT: 0.50,
                    PersonaType.SUNITA_SHOP: 0.85
                }
            ),

            "credential_display": ProbingTechnique(
                name="Credential Display",
                description="Show you have something to lose, making you a valuable target",
                goal="Keep scammer engaged by showing you're worth targeting",
                example_prompts=[
                    "Abhi toh pension aayi hai usme, ₹35,000",
                    "i just got my internship stipend yesterday 😭",
                    "Bachche ki school fees bhi usi account se jaati hai"
                ],
                target_info=[],
                persona_effectiveness={
                    PersonaType.RAMU_UNCLE: 0.90,
                    PersonaType.ANANYA_STUDENT: 0.85,
                    PersonaType.AARTI_HOMEMAKER: 0.85,
                    PersonaType.VIKRAM_IT: 0.70,
                    PersonaType.SUNITA_SHOP: 0.85
                }
            ),

            "authority_acceptance": ProbingTechnique(
                name="Authority Acceptance",
                description="Accept their claimed authority to build their confidence",
                goal="Make scammer confident, reveal more of their script",
                example_prompts=[
                    "Aap bank se bol rahe ho na? Theek hai ji",
                    "oh you're from the official support team? ok ok",
                    "Government wale ho na aap? Haan theek hai"
                ],
                target_info=[],
                persona_effectiveness={
                    PersonaType.RAMU_UNCLE: 0.95,
                    PersonaType.ANANYA_STUDENT: 0.70,
                    PersonaType.AARTI_HOMEMAKER: 0.90,
                    PersonaType.VIKRAM_IT: 0.50,
                    PersonaType.SUNITA_SHOP: 0.85
                }
            ),

            # ================================================================
            # PHASE 2: PROBE TECHNIQUES
            # ================================================================

            "verification_request": ProbingTechnique(
                name="Verification Request",
                description="Ask for scammer's details to 'verify' they are legitimate",
                goal="Extract scammer's claimed identity, name, employee ID",
                example_prompts=[
                    "Aap apna naam batao, main note kar loon",
                    "can you share your employee id? for my records",
                    "Aap kaunse branch se bol rahe ho?"
                ],
                target_info=["name", "employee_id", "organization", "branch"],
                persona_effectiveness={
                    PersonaType.RAMU_UNCLE: 0.90,
                    PersonaType.ANANYA_STUDENT: 0.85,
                    PersonaType.AARTI_HOMEMAKER: 0.85,
                    PersonaType.VIKRAM_IT: 0.95,
                    PersonaType.SUNITA_SHOP: 0.80
                }
            ),

            "callback_trap": ProbingTechnique(
                name="Callback Trap",
                description="Ask for their phone number to 'call back'",
                goal="Extract scammer's phone number",
                example_prompts=[
                    "Signal weak hai, aap number do main call karta hoon",
                    "can you give me a number to call back?",
                    "Aap WhatsApp number do, wahan pe baat karte hain"
                ],
                target_info=["phone_number", "whatsapp_number"],
                persona_effectiveness={
                    PersonaType.RAMU_UNCLE: 0.90,
                    PersonaType.ANANYA_STUDENT: 0.75,
                    PersonaType.AARTI_HOMEMAKER: 0.95,
                    PersonaType.VIKRAM_IT: 0.70,
                    PersonaType.SUNITA_SHOP: 0.85
                }
            ),

            "documentation_ask": ProbingTechnique(
                name="Documentation Ask",
                description="Request official proof or documentation",
                goal="Make scammer send fake documents (evidence), waste time",
                example_prompts=[
                    "Email bhej do official, main beta ko dikhata hoon",
                    "send me official email proof, i need to verify",
                    "Koi document hai toh WhatsApp pe bhejo"
                ],
                target_info=["email", "documents", "fake_evidence"],
                persona_effectiveness={
                    PersonaType.RAMU_UNCLE: 0.85,
                    PersonaType.ANANYA_STUDENT: 0.90,
                    PersonaType.AARTI_HOMEMAKER: 0.80,
                    PersonaType.VIKRAM_IT: 0.95,
                    PersonaType.SUNITA_SHOP: 0.75
                }
            ),

            "supervisor_escalation": ProbingTechnique(
                name="Supervisor Escalation",
                description="Ask to speak to their supervisor/manager",
                goal="Test their reaction, potentially get more contacts",
                example_prompts=[
                    "Aapke manager se baat ho sakti hai? Bada amount hai",
                    "can i talk to your senior? this seems serious",
                    "Aapke upar wale ka number do"
                ],
                target_info=["supervisor_name", "supervisor_contact"],
                persona_effectiveness={
                    PersonaType.RAMU_UNCLE: 0.80,
                    PersonaType.ANANYA_STUDENT: 0.70,
                    PersonaType.AARTI_HOMEMAKER: 0.85,
                    PersonaType.VIKRAM_IT: 0.90,
                    PersonaType.SUNITA_SHOP: 0.80
                }
            ),

            "reference_number": ProbingTechnique(
                name="Reference Number",
                description="Ask for complaint/ticket reference number",
                goal="Make scammer create fake reference, evidence of fraud",
                example_prompts=[
                    "Complaint number kya hai iska?",
                    "give me reference number for this case",
                    "Ticket number batao, main note kar leta hoon"
                ],
                target_info=["reference_number", "ticket_number", "complaint_id"],
                persona_effectiveness={
                    PersonaType.RAMU_UNCLE: 0.85,
                    PersonaType.ANANYA_STUDENT: 0.80,
                    PersonaType.AARTI_HOMEMAKER: 0.80,
                    PersonaType.VIKRAM_IT: 0.95,
                    PersonaType.SUNITA_SHOP: 0.75
                }
            ),

            # ================================================================
            # PHASE 3: EXTRACT TECHNIQUES
            # ================================================================

            "willing_victim": ProbingTechnique(
                name="Willing Victim",
                description="Show willingness to comply but ask for payment details",
                goal="Extract UPI ID, bank account, payment details",
                example_prompts=[
                    "Haan bhej deta hoon, kahan bhejoon? UPI ID do",
                    "ok fine ill pay, whats your upi id?",
                    "Theek hai, batao kahan transfer karna hai"
                ],
                target_info=["upi_id", "bank_account", "payment_method"],
                persona_effectiveness={
                    PersonaType.RAMU_UNCLE: 0.90,
                    PersonaType.ANANYA_STUDENT: 0.80,
                    PersonaType.AARTI_HOMEMAKER: 0.90,
                    PersonaType.VIKRAM_IT: 0.75,
                    PersonaType.SUNITA_SHOP: 0.85
                }
            ),

            "amount_confirmation": ProbingTechnique(
                name="Amount Confirmation",
                description="Confirm exact amount to make scammer state it explicitly",
                goal="Get explicit amount demanded (evidence)",
                example_prompts=[
                    "Kitna bhejoon? ₹500 ya ₹5000?",
                    "how much exactly do i need to pay?",
                    "Total kitna paisa hai? Pura batao"
                ],
                target_info=["amount", "fee_breakdown"],
                persona_effectiveness={
                    PersonaType.RAMU_UNCLE: 0.85,
                    PersonaType.ANANYA_STUDENT: 0.85,
                    PersonaType.AARTI_HOMEMAKER: 0.90,
                    PersonaType.VIKRAM_IT: 0.85,
                    PersonaType.SUNITA_SHOP: 0.90
                }
            ),

            "method_confusion": ProbingTechnique(
                name="Method Confusion",
                description="Pretend confusion about payment method to get multiple options",
                goal="Extract multiple payment methods (UPI, bank account, etc.)",
                example_prompts=[
                    "UPI se ya account number pe? Dono de do",
                    "i dont have gpay, can i do neft? give account",
                    "PhonePe se chalega? Ya Paytm? Account number bhi do"
                ],
                target_info=["upi_id", "bank_account", "ifsc", "alternate_payment"],
                persona_effectiveness={
                    PersonaType.RAMU_UNCLE: 0.95,
                    PersonaType.ANANYA_STUDENT: 0.80,
                    PersonaType.AARTI_HOMEMAKER: 0.90,
                    PersonaType.VIKRAM_IT: 0.70,
                    PersonaType.SUNITA_SHOP: 0.90
                }
            ),

            "test_payment": ProbingTechnique(
                name="Test Payment Trap",
                description="Offer to send small test payment first",
                goal="Get scammer to confirm payment details multiple times",
                example_prompts=[
                    "Pehle ₹10 bhejta hoon test ke liye, sahi hai na?",
                    "let me send rs 1 first to verify ur id",
                    "Ek rupya bhejta hoon, confirm ho jaye toh baaki bhejta hoon"
                ],
                target_info=["upi_id_confirmation", "payment_confirmation"],
                persona_effectiveness={
                    PersonaType.RAMU_UNCLE: 0.85,
                    PersonaType.ANANYA_STUDENT: 0.85,
                    PersonaType.AARTI_HOMEMAKER: 0.85,
                    PersonaType.VIKRAM_IT: 0.80,
                    PersonaType.SUNITA_SHOP: 0.80
                }
            ),

            "receipt_request": ProbingTechnique(
                name="Receipt Request",
                description="Ask where to send payment screenshot/receipt",
                goal="Get additional contact (WhatsApp, email)",
                example_prompts=[
                    "Screenshot kahan bhejoon? WhatsApp number do",
                    "where should i send the payment proof?",
                    "Receipt email pe bhejoon ya WhatsApp pe?"
                ],
                target_info=["whatsapp_number", "email", "contact_method"],
                persona_effectiveness={
                    PersonaType.RAMU_UNCLE: 0.80,
                    PersonaType.ANANYA_STUDENT: 0.85,
                    PersonaType.AARTI_HOMEMAKER: 0.85,
                    PersonaType.VIKRAM_IT: 0.75,
                    PersonaType.SUNITA_SHOP: 0.80
                }
            ),

            # ================================================================
            # PHASE 4: STALL TECHNIQUES
            # ================================================================

            "technical_difficulties": ProbingTechnique(
                name="Technical Difficulties",
                description="Pretend phone/app issues to stall",
                goal="Waste scammer's time, keep them engaged",
                example_prompts=[
                    "Arre yeh PhonePe open nahi ho raha, ruko",
                    "my app is lagging badly, one sec",
                    "Network issue hai, 2 minute ruko"
                ],
                target_info=[],
                persona_effectiveness={
                    PersonaType.RAMU_UNCLE: 0.95,
                    PersonaType.ANANYA_STUDENT: 0.85,
                    PersonaType.AARTI_HOMEMAKER: 0.90,
                    PersonaType.VIKRAM_IT: 0.65,
                    PersonaType.SUNITA_SHOP: 0.90
                }
            ),

            "otp_delay": ProbingTechnique(
                name="OTP Delay",
                description="Pretend OTP not received",
                goal="Waste time, potentially get scammer to reveal more",
                example_prompts=[
                    "OTP nahi aaya abhi tak, dobara bhejo",
                    "i didnt get the otp, can you resend?",
                    "OTP nahi aa raha, network problem hai shayad"
                ],
                target_info=[],
                persona_effectiveness={
                    PersonaType.RAMU_UNCLE: 0.90,
                    PersonaType.ANANYA_STUDENT: 0.85,
                    PersonaType.AARTI_HOMEMAKER: 0.90,
                    PersonaType.VIKRAM_IT: 0.70,
                    PersonaType.SUNITA_SHOP: 0.85
                }
            ),

            "battery_excuse": ProbingTechnique(
                name="Battery Excuse",
                description="Phone battery low excuse",
                goal="Create urgency to get scammer to provide alternative contact",
                example_prompts=[
                    "Battery 5% hai, charger dhundhta hoon. Number do",
                    "my phone is about to die, give me whatsapp number quick",
                    "Phone band hone wala hai, jaldi batao"
                ],
                target_info=["phone_number", "whatsapp"],
                persona_effectiveness={
                    PersonaType.RAMU_UNCLE: 0.90,
                    PersonaType.ANANYA_STUDENT: 0.90,
                    PersonaType.AARTI_HOMEMAKER: 0.85,
                    PersonaType.VIKRAM_IT: 0.60,
                    PersonaType.SUNITA_SHOP: 0.80
                }
            ),

            "interruption": ProbingTechnique(
                name="Interruption",
                description="Pretend someone came/distraction",
                goal="Stall conversation, maintain engagement",
                example_prompts=[
                    "Ek minute, darwaze pe koi aaya hai",
                    "hold on someone's at the door",
                    "Ruko, bachcha ro raha hai. 2 minute mein aata hoon"
                ],
                target_info=[],
                persona_effectiveness={
                    PersonaType.RAMU_UNCLE: 0.85,
                    PersonaType.ANANYA_STUDENT: 0.80,
                    PersonaType.AARTI_HOMEMAKER: 0.95,
                    PersonaType.VIKRAM_IT: 0.70,
                    PersonaType.SUNITA_SHOP: 0.90
                }
            ),

            "repetition_request": ProbingTechnique(
                name="Repetition Request",
                description="Ask scammer to repeat everything",
                goal="Waste time, get scammer to repeat evidence",
                example_prompts=[
                    "Ek baar phir se batao, main likh leta hoon",
                    "sorry can you repeat all that? i got confused",
                    "Pura process batao start se, mujhe samajh nahi aaya"
                ],
                target_info=["repeated_instructions"],
                persona_effectiveness={
                    PersonaType.RAMU_UNCLE: 0.95,
                    PersonaType.ANANYA_STUDENT: 0.75,
                    PersonaType.AARTI_HOMEMAKER: 0.85,
                    PersonaType.VIKRAM_IT: 0.60,
                    PersonaType.SUNITA_SHOP: 0.90
                }
            )
        }

    def _init_phase_strategies(self):
        """Initialize strategies for each conversation phase"""

        self.phase_strategies: Dict[ConversationPhase, PhaseStrategy] = {

            ConversationPhase.ENGAGE: PhaseStrategy(
                phase=ConversationPhase.ENGAGE,
                goal="Establish believable victim persona, build scammer's confidence",
                typical_turns=3,
                techniques=[
                    "confused_echo", "family_reference",
                    "credential_display", "authority_acceptance"
                ],
                success_criteria=[
                    "Scammer continues conversation",
                    "Scammer reveals more of their script",
                    "Persona is established believably"
                ],
                transition_conditions={
                    "to_probe": "After 2-3 turns, or when scammer starts asking for info"
                }
            ),

            ConversationPhase.PROBE: PhaseStrategy(
                phase=ConversationPhase.PROBE,
                goal="Extract scammer's identity and contact details",
                typical_turns=5,
                techniques=[
                    "verification_request", "callback_trap",
                    "documentation_ask", "supervisor_escalation",
                    "reference_number"
                ],
                success_criteria=[
                    "Obtained scammer name/ID",
                    "Obtained callback number",
                    "Obtained reference number"
                ],
                transition_conditions={
                    "to_extract": "When scammer pushes for payment"
                }
            ),

            ConversationPhase.EXTRACT: PhaseStrategy(
                phase=ConversationPhase.EXTRACT,
                goal="Get payment details - UPI IDs, bank accounts",
                typical_turns=5,
                techniques=[
                    "willing_victim", "amount_confirmation",
                    "method_confusion", "test_payment",
                    "receipt_request"
                ],
                success_criteria=[
                    "Obtained UPI ID",
                    "Obtained bank account",
                    "Obtained exact amount demanded"
                ],
                transition_conditions={
                    "to_stall": "When sufficient payment info collected"
                }
            ),

            ConversationPhase.STALL: PhaseStrategy(
                phase=ConversationPhase.STALL,
                goal="Waste scammer's time, gather remaining intelligence",
                typical_turns=999,  # Can continue indefinitely
                techniques=[
                    "technical_difficulties", "otp_delay",
                    "battery_excuse", "interruption",
                    "repetition_request"
                ],
                success_criteria=[
                    "Scammer remains engaged",
                    "Additional intel gathered",
                    "Maximum time wasted"
                ],
                transition_conditions={
                    "to_complete": "When scammer gives up or max turns reached"
                }
            ),

            ConversationPhase.COMPLETE: PhaseStrategy(
                phase=ConversationPhase.COMPLETE,
                goal="Session ended, send callback",
                typical_turns=0,
                techniques=[],
                success_criteria=["Callback sent"],
                transition_conditions={}
            )
        }

    def get_phase_strategy(self, phase: ConversationPhase) -> PhaseStrategy:
        """Get strategy for a conversation phase"""
        return self.phase_strategies.get(
            phase,
            self.phase_strategies[ConversationPhase.ENGAGE]
        )

    def select_technique(
            self,
            phase: ConversationPhase,
            persona_type: PersonaType,
            already_used: List[str] = None,
            intelligence_needed: List[str] = None
    ) -> ProbingTechnique:
        """
        Select best probing technique for current situation

        Args:
            phase: Current conversation phase
            persona_type: Active persona
            already_used: Techniques already used this session
            intelligence_needed: What info we still need

        Returns:
            Best ProbingTechnique for the situation
        """
        strategy = self.get_phase_strategy(phase)
        available_techniques = strategy.techniques

        if not available_techniques:
            # Fallback to engage techniques
            available_techniques = self.phase_strategies[ConversationPhase.ENGAGE].techniques

        # Filter out recently used
        if already_used:
            candidates = [t for t in available_techniques if t not in already_used[-3:]]
            if not candidates:
                candidates = available_techniques
        else:
            candidates = available_techniques

        # Score each technique
        scored_techniques = []
        for tech_name in candidates:
            technique = self.techniques.get(tech_name)
            if not technique:
                continue

            score = technique.persona_effectiveness.get(persona_type, 0.5)

            # Boost if technique targets needed intelligence
            if intelligence_needed and technique.target_info:
                overlap = set(technique.target_info) & set(intelligence_needed)
                score += len(overlap) * 0.2

            scored_techniques.append((technique, score))

        if not scored_techniques:
            # Ultimate fallback
            return self.techniques["confused_echo"]

        # Weighted random selection (favor higher scores but allow variety)
        techniques = [t for t, s in scored_techniques]
        weights = [s for t, s in scored_techniques]

        selected = random.choices(techniques, weights=weights, k=1)[0]

        logger.debug(
            f"Selected technique '{selected.name}' for phase {phase.value}, "
            f"persona {persona_type.value}"
        )

        return selected

    def get_technique_prompt(
            self,
            technique: ProbingTechnique,
            persona_type: PersonaType
    ) -> str:
        """
        Get an appropriate prompt for the technique and persona

        Args:
            technique: The probing technique
            persona_type: Active persona

        Returns:
            Example prompt string
        """
        if technique.example_prompts:
            # Filter prompts by language appropriate for persona
            if persona_type in [PersonaType.VIKRAM_IT, PersonaType.ANANYA_STUDENT]:
                # Prefer English prompts
                english_prompts = [p for p in technique.example_prompts
                                   if not any(hindi in p for hindi in ['hai', 'hoon', 'kya', 'mein', 'ka'])]
                if english_prompts:
                    return random.choice(english_prompts)

            return random.choice(technique.example_prompts)

        return ""

    def determine_intelligence_gaps(
            self,
            collected_intel: dict
    ) -> List[str]:
        """
        Determine what intelligence is still needed

        Args:
            collected_intel: Currently collected intelligence

        Returns:
            List of intelligence types still needed
        """
        needed = []

        # Priority intelligence
        if not collected_intel.get("upi_ids"):
            needed.append("upi_id")

        if not collected_intel.get("phone_numbers"):
            needed.append("phone_number")

        if not collected_intel.get("bank_accounts"):
            needed.append("bank_account")

        # Secondary intelligence
        if not collected_intel.get("scammer_names"):
            needed.append("name")

        if not collected_intel.get("email_addresses"):
            needed.append("email")

        return needed
