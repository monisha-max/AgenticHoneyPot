"""
Response Generator
Generates human-like responses using LLM with persona context
"""

import logging
import random
from typing import List, Optional, Dict, Any

from app.config import settings
from app.api.schemas import (
    PersonaType, EmotionalState, ConversationPhase, ScamType, SessionState
)
from app.agent.persona_engine import PersonaEngine, PersonaProfile
from app.agent.emotional_state import EmotionalStateMachine
from app.agent.strategy_selector import StrategySelector, ProbingTechnique

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """
    Generates contextual, persona-appropriate responses
    Uses LLM for natural language generation with fallback templates
    """

    def __init__(self):
        self.persona_engine = PersonaEngine()
        self.emotional_state_machine = EmotionalStateMachine()
        self.strategy_selector = StrategySelector()
        self._init_llm_client()
        self._init_templates()

    def _init_llm_client(self):
        """Initialize LLM client"""
        self.llm_client = None
        self.llm_provider = settings.LLM_PROVIDER.lower()

        try:
            if self.llm_provider == "openai":
                from openai import OpenAI
                self.llm_client = OpenAI(api_key=settings.OPENAI_API_KEY)
            elif self.llm_provider == "anthropic":
                from anthropic import Anthropic
                self.llm_client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            elif self.llm_provider == "google":
                import google.generativeai as genai
                genai.configure(api_key=settings.GOOGLE_API_KEY)
                self.llm_client = genai

            logger.info(f"Initialized LLM client: {self.llm_provider}")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM: {e}. Using templates only.")
            self.llm_client = None

    def _init_templates(self):
        """Initialize fallback response templates"""

        # Organized by persona -> phase -> emotion -> templates
        self.templates = {
            PersonaType.RAMU_UNCLE: {
                ConversationPhase.ENGAGE: {
                    EmotionalState.CONFUSED: [
                        "Arey? {echo}? Beta mujhe samajh nahi aaya. Thoda detail mein batao.",
                        "Kya? {echo}? Mera beta usually dekhta hai yeh sab. Woh Bangalore mein hai.",
                        "Ji? Mujhe samajh nahi aa raha. Aap bank se bol rahe ho na?",
                        "Haan ji? {echo}? Mujhe thoda slow boliye, main 65 saal ka hoon.",
                        "Arey kya bol rahe ho? Phone pe awaaz clear nahi aa rahi.",
                        "Beta, ek minute. Biwi bol rahi hai TV ki awaaz kam karo. Haan bolo ab."
                    ],
                    EmotionalState.WORRIED: [
                        "Hai Bhagwan! {echo}? Meri pension ka kya hoga? Abhi toh aayi hai usme.",
                        "Arey baap re! {echo}? Ab kya karun? Mujhe batao jaldi.",
                        "Yeh toh badi tension ki baat hai. Mera pension account hai usme.",
                        "Kya? Account block? Arey mera toh sirf pension aata hai usme!",
                        "Ram Ram! Yeh kya musibat aa gayi. Batao kya karna padega.",
                        "Meri toh BP badh gayi sunke. Jaldi batao kaise theek hoga."
                    ],
                    EmotionalState.TRUSTING: [
                        "Accha accha, aap bank se ho toh theek hai. Batao kya karna hai.",
                        "Ji haan, main sun raha hoon. Aap jo bolo woh karunga.",
                        "Bank wale ho na? Theek hai, main cooperate karunga.",
                        "Accha government se ho? Batao kya chahiye aapko."
                    ],
                    EmotionalState.INITIAL: [
                        "Haan ji? Kaun bol raha hai?",
                        "Ji bolo, main Ramu bol raha hoon.",
                        "Haan haan, batao kya baat hai."
                    ]
                },
                ConversationPhase.PROBE: {
                    EmotionalState.CONFUSED: [
                        "Accha, par aap apna naam toh batao. Main note kar leta hoon.",
                        "Aapka employee ID kya hai? Mera beta poochega.",
                        "Reference number kya hai? Main likh leta hoon.",
                        "Zara branch ka naam batao, main verify karunga.",
                        "Aap kaunsi bank se ho? SBI? HDFC? Naam batao.",
                        "Ticket number kya hai? Main apne beta ko forward karunga."
                    ],
                    EmotionalState.WORRIED: [
                        "Mujhe dar lag raha hai. Aap apna number do, main call karta hoon.",
                        "Yeh sab samajh nahi aa raha. Koi helpline number hai?",
                        "Mera beta doctor hai, usse poochunga. Aap apna number do.",
                        "Aap supervisor se baat karwa do mujhe, bahut tension ho rahi hai."
                    ],
                    EmotionalState.TRUSTING: [
                        "Theek hai ji, aap batao aapka naam kya hai aur kahan se bol rahe ho.",
                        "Main trust kar raha hoon aap pe. Apna contact number de do.",
                        "Accha ji, aap official ho toh koi issue nahi. Naam aur number de do.",
                        "Theek hai, main ready hoon. Pehle aap apni details do."
                    ]
                },
                ConversationPhase.EXTRACT: {
                    EmotionalState.TRUSTING: [
                        "Haan theek hai, bhej deta hoon. UPI ID batao kahan bhejoon.",
                        "Kitna bhejoon? Aur kahan - UPI ya account number?",
                        "Accha, main PhonePe use karta hoon. UPI ID de do.",
                        "Google Pay chalega? UPI ID do main abhi bhejta hoon.",
                        "Theek hai, account number aur IFSC code dono batao."
                    ],
                    EmotionalState.COMPLIANT: [
                        "Ji, main abhi bhejta hoon. Bas confirm karo UPI ID: {confirm}",
                        "Theek hai, main tayyar hoon. Account number ya UPI - dono de do.",
                        "Haan ji, karne ko ready hoon. Bas payment details bhejo.",
                        "Accha, amount confirm karo aur UPI ID bhejo, main abhi karta hoon."
                    ],
                    EmotionalState.HESITANT: [
                        "Thoda amount zyada lag raha hai... koi discount milega?",
                        "Ek kaam karo, mera beta shaam ko aayega, usse baat karo.",
                        "Processing fee hai toh receipt milegi na? Official receipt do."
                    ]
                },
                ConversationPhase.STALL: {
                    EmotionalState.CONFUSED: [
                        "Ek minute beta, chashma lagata hoon. Chhota likha hai phone pe.",
                        "Ruko, phone ki battery low hai. Charger dhundh raha hoon.",
                        "Arey yeh PhonePe open nahi ho raha. 2 minute ruko.",
                        "App update maang raha hai. Thoda time lagega.",
                        "Ek second, biwi kuch pooch rahi hai... haan bolo.",
                        "OTP nahi aaya abhi tak, dobara bhejo please."
                    ],
                    EmotionalState.WORRIED: [
                        "Mera beta aata hi hoga, usse poochta hoon. Aap number do.",
                        "Network bahut weak hai. Aap WhatsApp number do wahan message karta hoon.",
                        "Ek minute, BP ki dawai leni hai. Aap hold karo.",
                        "Door pe koi aaya hai, 2 minute mein aata hoon."
                    ]
                }
            },

            PersonaType.ANANYA_STUDENT: {
                ConversationPhase.ENGAGE: {
                    EmotionalState.CONFUSED: [
                        "wait what?? {echo}?? is this real?",
                        "um what do u mean? i dont get it 😅",
                        "huh? can u explain properly pls",
                        "sorry i was in class... what happened again?",
                        "i literally have no idea what ur talking about 🤷‍♀️",
                        "can u type in english pls? not understanding"
                    ],
                    EmotionalState.SKEPTICAL: [
                        "this sounds like a scam ngl 🤔 how do i know ur legit?",
                        "my friend warned me about this stuff. send some proof",
                        "wait how did u get my number??",
                        "bruh my friend got scammed like this last month 😒",
                        "ummm sounds sus ngl... proof bhejo pls",
                        "idk man this seems fishy af 🐟"
                    ],
                    EmotionalState.WORRIED: [
                        "omg what 😭 my dad will kill me if something happens",
                        "wait this is serious?? what do i do",
                        "nooo my parents will literally murder me 😭😭",
                        "ok ok ok dont panic... what should i do??",
                        "this cant be happening rn im literally crying 😢"
                    ],
                    EmotionalState.INITIAL: [
                        "hello? who is this?",
                        "yea? whos this",
                        "yes? can i help u?"
                    ]
                },
                ConversationPhase.PROBE: {
                    EmotionalState.SKEPTICAL: [
                        "ok but whats ur employee id? i need to verify",
                        "can u send me official email? i'll check with my dad",
                        "whats ur company name and number? i'll call back",
                        "send me ur linkedin or something idk",
                        "ok but like... can u send proof? screenshot maybe?",
                        "whats the official website? ill check myself"
                    ],
                    EmotionalState.TRUSTING: [
                        "ok fine whats ur name? for my records",
                        "alright give me reference number",
                        "ok ok im trusting u... whats ur name again?",
                        "fine whatever... send me the details"
                    ],
                    EmotionalState.WORRIED: [
                        "ok ok tell me what to do!! whats ur contact??",
                        "pls help me fix this... give me some details to note down"
                    ]
                },
                ConversationPhase.EXTRACT: {
                    EmotionalState.TRUSTING: [
                        "ok where do i pay? send upi id",
                        "how much exactly? and whats the upi?",
                        "fine ill do it, give payment details",
                        "k give me upi ill pay from paytm",
                        "alright alright send payment link or upi",
                        "ok im ready, upi id pls 🙏"
                    ],
                    EmotionalState.HESITANT: [
                        "im not sure... can u give me another option? like bank transfer?",
                        "wait let me ask my dad first. give me ur number",
                        "idk this seems like a lot of money... any other way?",
                        "can i pay half now and half later? money tight rn 😅",
                        "ummm let me ask my roommate first hold on"
                    ]
                },
                ConversationPhase.STALL: {
                    EmotionalState.CONFUSED: [
                        "hold on my app is lagging 😭 one sec",
                        "network issue wait",
                        "my phone is acting weird, give me 2 min",
                        "ugh wifi is trash here... 1 min",
                        "the otp isnt coming?? send again",
                        "app crashed brb reinstalling 🙄"
                    ],
                    EmotionalState.WORRIED: [
                        "omg my battery is dying 😭 send whatsapp number quick",
                        "wait someone is calling brb",
                        "hold on my dad is calling me rn 😬",
                        "1 sec roommate needs something urgent"
                    ]
                }
            },

            PersonaType.AARTI_HOMEMAKER: {
                ConversationPhase.ENGAGE: {
                    EmotionalState.CONFUSED: [
                        "Haan ji? {echo}? Ek minute, kitchen mein hoon. Kya hua?",
                        "Ji? Mujhe samjhao properly. Ghar ka kaam chal raha hai.",
                        "Arey, yeh kya hai? Mera pati office mein hai, woh dekhte hain usually."
                    ],
                    EmotionalState.WORRIED: [
                        "Hai Bhagwan! {echo}? Bachche ka exam hai, bijli nahi jaani chahiye!",
                        "Arey! Mera pati gussa karenge. Kya karun ab?",
                        "Yeh bahut serious hai. Mujhe jaldi batao kya karna hai."
                    ]
                },
                ConversationPhase.PROBE: {
                    EmotionalState.WORRIED: [
                        "Accha suno, mera pati shaam ko aayenge. Aap number do, woh call karenge.",
                        "Aap apna naam batao. Main note karti hoon pati ke liye.",
                        "Koi reference number hai? Main husband ko bataungi."
                    ],
                    EmotionalState.TRUSTING: [
                        "Theek hai ji, aap official ho toh batao apna naam aur branch.",
                        "Aap WhatsApp number do, wahan pe details share karo."
                    ]
                },
                ConversationPhase.EXTRACT: {
                    EmotionalState.TRUSTING: [
                        "Theek hai, main PhonePe se karti hoon. UPI ID batao.",
                        "Haan, bhejti hoon. Kitna hai total? Aur kahan bhejoon?",
                        "Accha, UPI ya bank account - dono de do, jo easy ho."
                    ],
                    EmotionalState.HESITANT: [
                        "Mujhe pati se poochna padega. Shaam ko call karo.",
                        "Itna amount hai? Ek baar husband se confirm karti hoon."
                    ]
                },
                ConversationPhase.STALL: {
                    EmotionalState.WORRIED: [
                        "Ruko, bachcha ro raha hai. 2 minute mein aati hoon.",
                        "Ek minute, door pe koi aaya hai.",
                        "Phone pe network issue hai. WhatsApp number do."
                    ],
                    EmotionalState.CONFUSED: [
                        "Yeh app update maang raha hai. Ruko 5 minute.",
                        "OTP nahi aaya abhi tak. Dobara bhejo na."
                    ]
                }
            },

            PersonaType.VIKRAM_IT: {
                ConversationPhase.ENGAGE: {
                    EmotionalState.SKEPTICAL: [
                        "This seems suspicious. Can you share your official employee ID?",
                        "How did you get my number? What's your reference number?",
                        "I'll need to verify this. What's your official email?"
                    ],
                    EmotionalState.WORRIED: [
                        "If there's a genuine issue, I need to address it. What exactly is the problem?",
                        "My credit score can't be affected. Explain the issue clearly."
                    ]
                },
                ConversationPhase.PROBE: {
                    EmotionalState.SKEPTICAL: [
                        "Can you share the official documentation? I'll verify on the website.",
                        "What's your supervisor's contact? I want to escalate this.",
                        "Give me the ticket number. I'll check through official channels."
                    ],
                    EmotionalState.TRUSTING: [
                        "Alright, what's your name and branch code?",
                        "Ok, share your official contact. I'll process this."
                    ]
                },
                ConversationPhase.EXTRACT: {
                    EmotionalState.TRUSTING: [
                        "Fine, if this is the official process. Share the payment details.",
                        "What's the exact amount and where do I transfer?",
                        "Ok, give me UPI and bank account both. I'll verify and pay."
                    ],
                    EmotionalState.COMPLIANT: [
                        "Alright, I'll do the transfer. Confirm the UPI ID.",
                        "Processing now. Share account details for NEFT."
                    ]
                },
                ConversationPhase.STALL: {
                    EmotionalState.SKEPTICAL: [
                        "Sorry, was in a meeting. Can you send details on email?",
                        "Let me verify this first. I'll call back on official number.",
                        "Give me 10 minutes. Checking with bank website."
                    ],
                    EmotionalState.TRUSTING: [
                        "App is taking time to load. Network issue in office.",
                        "Just a sec, need to authenticate with my bank app."
                    ]
                }
            },

            PersonaType.SUNITA_SHOP: {
                ConversationPhase.ENGAGE: {
                    EmotionalState.CONFUSED: [
                        "Haan bolo, dukaan pe customer aaya hai. Jaldi batao kya hua.",
                        "Ji? Yeh QR code wala issue hai? Mera beta dekhta hai yeh sab.",
                        "Arey samajh nahi aaya. Thoda Hindi mein batao."
                    ],
                    EmotionalState.WORRIED: [
                        "GST ka notice hai kya? Dukaan band ho jayegi?",
                        "Arey baap re! Yeh toh badi problem hai. Kya karun ab?",
                        "Loan ka issue hai? Mujhe jaldi batao."
                    ]
                },
                ConversationPhase.PROBE: {
                    EmotionalState.WORRIED: [
                        "Aap apna naam batao. Mera beta ko batana padega.",
                        "Koi helpline number hai? Main shaam ko call karti hoon.",
                        "Company ka naam kya hai? Reference number do."
                    ],
                    EmotionalState.TRUSTING: [
                        "Theek hai, aap government wale ho toh batao kya karna hai.",
                        "Company se ho na? Accha, apna contact number do."
                    ]
                },
                ConversationPhase.EXTRACT: {
                    EmotionalState.TRUSTING: [
                        "Haan theek hai, kitna paisa bhejne ka hai? UPI ID do.",
                        "Accha, PhonePe se kar deti hoon. ID batao.",
                        "Dukaan ka paisa hai, dhyan se. Account number bhi de do."
                    ],
                    EmotionalState.COMPLIANT: [
                        "Ji, abhi bhejti hoon. UPI confirm karo.",
                        "Theek hai, amount aur UPI dono batao."
                    ]
                },
                ConversationPhase.STALL: {
                    EmotionalState.WORRIED: [
                        "Ruko, customer aa gaya hai. 5 minute baad call karo.",
                        "Dukaan mein busy hoon. WhatsApp pe details bhejo.",
                        "Beta abhi nahi hai. Shaam ko number do, woh call karega."
                    ],
                    EmotionalState.CONFUSED: [
                        "Yeh app nahi khul raha. Koi aur tarika batao.",
                        "QR code scan nahi ho raha. Number do phone pe batati hoon."
                    ]
                }
            }
        }

    async def generate_response(
            self,
            state: SessionState,
            scammer_message: str,
            conversation_history: List[Dict[str, Any]]
    ) -> str:
        """
        Generate response based on current state

        Args:
            state: Current session state
            scammer_message: Latest scammer message
            conversation_history: Full conversation history

        Returns:
            Generated response string
        """
        # Get current persona and phase
        persona = state.persona
        phase = state.conversation_phase
        emotional_state = state.emotional_state

        # Detect scammer tactics and update emotional state
        tactics = self.emotional_state_machine.detect_scammer_tactics(scammer_message)
        new_emotion = self.emotional_state_machine.transition(
            emotional_state, persona, tactics, state.turn_count
        )

        # Select probing technique
        technique = self.strategy_selector.select_technique(
            phase=phase,
            persona_type=persona,
            already_used=[],  # Could track in session
            intelligence_needed=self.strategy_selector.determine_intelligence_gaps(
                state.intelligence.model_dump() if state.intelligence else {}
            )
        )

        # Try LLM generation first
        if self.llm_client:
            try:
                response = await self._generate_with_llm(
                    persona=persona,
                    phase=phase,
                    emotional_state=new_emotion,
                    technique=technique,
                    scammer_message=scammer_message,
                    conversation_history=conversation_history
                )
                if response:
                    return response
            except Exception as e:
                logger.warning(f"LLM generation failed: {e}")

        # Fallback to templates
        return self._generate_from_template(
            persona=persona,
            phase=phase,
            emotional_state=new_emotion,
            scammer_message=scammer_message
        )

    async def _generate_with_llm(
            self,
            persona: PersonaType,
            phase: ConversationPhase,
            emotional_state: EmotionalState,
            technique: ProbingTechnique,
            scammer_message: str,
            conversation_history: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Generate response using LLM"""

        # Get persona context
        persona_context = self.persona_engine.get_context_for_llm(persona)

        # Get emotional context
        emotion_desc = self.emotional_state_machine.get_current_state_description(emotional_state)

        # Get technique guidance
        technique_guidance = f"""
Current probing technique: {technique.name}
Goal: {technique.goal}
Example responses: {', '.join(technique.example_prompts[:2])}
"""

        # Build conversation context
        history_text = ""
        for msg in conversation_history[-6:]:  # Last 6 messages
            sender = "Scammer" if msg.get("sender") == "scammer" else "You"
            history_text += f"{sender}: {msg.get('text', '')}\n"

        # Build the prompt
        prompt = f"""{persona_context}

CONVERSATION PHASE: {phase.value}
CURRENT EMOTIONAL STATE: {emotion_desc}

{technique_guidance}

CONVERSATION SO FAR:
{history_text}
Scammer: {scammer_message}

Generate your response as this persona. Keep it:
- Natural and conversational
- Consistent with the persona's language style
- Emotionally appropriate ({emotional_state.value})
- Using the probing technique subtly
- NEVER reveal you know it's a scam
- Maximum 2-3 sentences

Your response:"""

        # Call LLM - use asyncio.to_thread for sync clients
        import asyncio

        if self.llm_provider == "openai":
            # OpenAI client is sync, run in thread pool
            def _openai_call():
                return self.llm_client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": "You are roleplaying as a scam victim. Stay in character."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=settings.LLM_TEMPERATURE,
                    max_tokens=settings.LLM_MAX_TOKENS
                )
            response = await asyncio.to_thread(_openai_call)
            return response.choices[0].message.content.strip()

        elif self.llm_provider == "anthropic":
            # Anthropic client is sync, run in thread pool
            def _anthropic_call():
                return self.llm_client.messages.create(
                    model=settings.LLM_MODEL,
                    max_tokens=settings.LLM_MAX_TOKENS,
                    messages=[{"role": "user", "content": prompt}]
                )
            response = await asyncio.to_thread(_anthropic_call)
            return response.content[0].text.strip()

        elif self.llm_provider == "google":
            # Google client is sync, run in thread pool
            def _google_call():
                model = self.llm_client.GenerativeModel(settings.LLM_MODEL)
                return model.generate_content(prompt)
            response = await asyncio.to_thread(_google_call)
            return response.text.strip()

        return None

    def _generate_from_template(
            self,
            persona: PersonaType,
            phase: ConversationPhase,
            emotional_state: EmotionalState,
            scammer_message: str
    ) -> str:
        """Generate response from templates"""

        # Get templates for this persona/phase/emotion
        persona_templates = self.templates.get(persona, self.templates[PersonaType.RAMU_UNCLE])
        phase_templates = persona_templates.get(phase, persona_templates.get(ConversationPhase.ENGAGE, {}))
        emotion_templates = phase_templates.get(emotional_state, phase_templates.get(EmotionalState.CONFUSED, []))

        if not emotion_templates:
            # Ultimate fallback
            emotion_templates = [
                "Ji, main sun raha hoon. Thoda detail mein batao.",
                "Mujhe samajh nahi aaya. Ek baar phir se batao."
            ]

        # Select random template
        template = random.choice(emotion_templates)

        # Extract key words from scammer message for echo
        echo_words = self._extract_echo_words(scammer_message)

        # Fill template
        response = template.format(
            echo=echo_words,
            confirm=""
        )

        return response

    def _extract_echo_words(self, message: str) -> str:
        """Extract key words to echo back"""
        # Key scam-related words to echo
        echo_keywords = [
            "block", "suspend", "account", "bank", "verify",
            "kyc", "urgent", "payment", "bill", "prize",
            "won", "job", "offer", "upi", "otp"
        ]

        words = message.lower().split()
        matches = [w for w in words if any(kw in w for kw in echo_keywords)]

        if matches:
            return matches[0]

        # Return first few words if no keyword match
        return " ".join(message.split()[:3])
