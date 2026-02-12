"""
Response Generator
Generates human-like responses using LLM with persona context
"""

import logging
import random
import hashlib
from typing import List, Optional, Dict, Any
from functools import lru_cache
from collections import OrderedDict

from app.config import settings

# Simple LRU Cache for LLM responses (max 100 entries)
class LLMCache:
    def __init__(self, maxsize=100):
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def _make_key(self, persona, phase, emotion, message):
        """Create cache key from inputs"""
        key_str = f"{persona}:{phase}:{emotion}:{message[:100]}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, persona, phase, emotion, message):
        key = self._make_key(persona, phase, emotion, message)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def set(self, persona, phase, emotion, message, response):
        key = self._make_key(persona, phase, emotion, message)
        self.cache[key] = response
        self.cache.move_to_end(key)
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)

# Global cache instance
_llm_cache = LLMCache(maxsize=100)
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
                    ],
                    EmotionalState.INITIAL: [
                        "Haan ji? Kaun bol raha hai?",
                        "Ji bolo, main Aarti bol rahi hoon.",
                        "Hello? Hello?"
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
                    ],
                    EmotionalState.INITIAL: [
                        "Hello? Who is this?",
                        "Yes, hello? Can I help you?",
                        "Hi, who's speaking?"
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
                    ],
                    EmotionalState.INITIAL: [
                        "Haan bolo, kaun?",
                        "Ji? Dukaan mein hoon abhi. Kaun bol rahe ho?",
                        "Hello? Haan bolo."
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

    def _is_simple_greeting(self, message: str) -> bool:
        """Check if message is a simple greeting"""
        greetings = [
            "hello", "hi", "hey", "hii", "hiii", "helo", "hlo",
            "good morning", "good afternoon", "good evening",
            "namaste", "namaskar", "namaskaram",
            "howdy", "greetings", "sup", "wassup", "whatsup"
        ]
        msg_lower = message.lower().strip()
        # Check if entire message is just a greeting (with possible punctuation)
        clean_msg = msg_lower.rstrip("!?.,:;")
        return clean_msg in greetings or len(clean_msg) <= 10 and any(g in clean_msg for g in greetings)

    def _get_greeting_response(self, persona: PersonaType) -> str:
        """Get a short greeting response for the persona"""
        greeting_responses = {
            PersonaType.RAMU_UNCLE: [
                "Haan ji? Kaun bol raha hai?",
                "Ji bolo, kaun hai?",
                "Haan haan, kaun?"
            ],
            PersonaType.ANANYA_STUDENT: [
                "hello? who's this?",
                "yea? who is this",
                "hi, who's this?"
            ],
            PersonaType.AARTI_HOMEMAKER: [
                "Ji? Kaun bol raha hai?",
                "Haan ji, kaun?",
                "Ji bolo?"
            ],
            PersonaType.VIKRAM_IT: [
                "Yes? Who is this?",
                "Hello, may I know who's calling?",
                "Yes, who's speaking?"
            ],
            PersonaType.SUNITA_SHOP: [
                "Haan bolo, kaun hai?",
                "Ji? Kaun?",
                "Haan ji, bolo?"
            ]
        }
        responses = greeting_responses.get(persona, greeting_responses[PersonaType.RAMU_UNCLE])
        return random.choice(responses)

    # def _is_out_of_context(self, message: str) -> bool:
    #     """Check if message is out of context / irrelevant to scam conversation"""
    #     out_of_context_keywords = [
    #         "what did you eat", "ate", "food", "lunch", "dinner", "breakfast", "snack",
    #         "are you hungry", "thirsty", "drink", "beverage",
    #         "family", "parents", "mother", "father", "mom", "dad",
    #         "how's weather", "weather", "rain", "sunny", "cold", "hot", "temperature",
    #         "what's your name", "your age", "how old", "where do you live", "address",
    #         "what's your job", "do you work", "occupation", "career",
    #         "favorite color", "favorite movie", "favorite sport", "like to", "hobby",
    #         "boyfriend", "girlfriend", "married", "relationship", "dating",
    #         "school", "college", "studied", "degree", "course",
    #         "how many siblings", "brother", "sister", "family",
    #         "what's your number", "phone number", "mobile number",
    #         "are you real", "are you bot", "are you human", "artificial intelligence" , "are you eating"
    #     ]
    #     
    #     msg_lower = message.lower().strip()
    #     import re
    #     # Check if any out-of-context keyword is present as a whole word
    #     for keyword in out_of_context_keywords:
    #         pattern = rf"\b{re.escape(keyword)}\b"
    #         if re.search(pattern, msg_lower):
    #             return True
    #     return False

    # async def _get_out_of_context_response(self, persona: PersonaType) -> str:
    #     """Get confused/neutral response for out-of-context questions
    #     
    #     Handles both LLM generation and fallback templates
    #     """
    #     out_of_context_templates = {
    #         PersonaType.RAMU_UNCLE: [
    #             "Arey, yeh kya pooch rahe ho? Relevant baat karo na.",
    #             "Huh? Iska kya matlab? Yeh sab baad mein poochna.",
    #             "Beta, main iska jawab nahi doonga. Bank ka issue solve kar."
    #         ],
    #         PersonaType.ANANYA_STUDENT: [
    #             "um why are u asking me that?? 😅",
    #             "huh? thats random lol... can we get back to this?",
    #             "ok but like... thats kinda weird to ask rn??"
    #         ],
    #         PersonaType.AARTI_HOMEMAKER: [
    #             "Arey, yeh kya pooch rahe ho? Baat relevant karo.",
    #             "Ji, iska samay nahi hai abhi. Aage batao.",
    #             "Huh? Yeh sab baad mein. Pehle iska issue batao."
    #         ],
    #         PersonaType.VIKRAM_IT: [
    #             "That's not relevant right now. Can we focus on the issue?",
    #             "Why are you asking that? Let's stick to the matter at hand.",
    #             "That's random. Can we get back to what you were saying?"
    #         ],
    #         PersonaType.SUNITA_SHOP: [
    #             "Arey, yeh kya baat kar rahe ho? Dukaan ka kaam batao.",
    #             "Huh? Iska matlab kya? Pehle issue solve kar.",
    #             "Beta, relevant baat karo. Mera time barbaad mat kar."
    #         ]
    #     }
    #     
    #     # Try LLM first if available
    #     if self.llm_client:
    #         try:
    #             persona_obj = self.persona_engine.get_persona(persona)
    #             logger.info(f"OUT-OF-CONTEXT: Persona={persona_obj.name}, Language={persona_obj.primary_language}, Style={persona_obj.language_style}")
    #             print(f"OUT-OF-CONTEXT PERSONA: {persona_obj.name} ({persona_obj.primary_language})")
    #             
    #             if self.llm_provider == "openai":
    #                 # Build persona-aware system message based on language
    #                 if persona_obj.primary_language == "English":
    #                     system_msg = f"""You are {persona_obj.name}, a {persona_obj.age}-year-old {persona_obj.occupation}.
    # 
    # When asked irrelevant/out-of-context questions, respond with CONFUSION in ENGLISH ONLY.
    # - Show you don't understand why they're asking this
    # - Redirect to the main topic
    # - NEVER respond in Hindi or Hinglish
    # - Keep it very short (1 sentence max, 10 words max)
    # 
    # Example: "That's random. Can we focus on this?" , "\I don't get why you're asking that. Let's continue."""
    #                 else:
    #                     system_msg = f"""You are {persona_obj.name}, a naive Indian person.
    # 
    # When asked irrelevant/out-of-context questions, respond with CONFUSION in HINGLISH/HINDI ONLY.
    # - Show you don't understand why they're asking this
    # - Redirect to the main topic
    # - Use simple Hindi/Hinglish mixed with English
    # - Keep it very short (1 sentence max, 10 words max)
    # 
    # Example: "Arey, yeh kya pooch rahe ho? Relevant baat karo na." , "Huh? Iska kya matlab?.
    # """
    #                 
    #                 prompt = f"Someone asked you an irrelevant question. Respond with confusion and redirect to the main topic.\n\nRespond in {persona_obj.primary_language} ONLY. Stay in character. Max 10 words.\n\nJust say you don't understand why they're asking this - keep it GENERIC, don't mention specific topics."
    #                 
    #                 def _openai_call():
    #                     return self.llm_client.chat.completions.create(
    #                         model=settings.LLM_MODEL,
    #                         messages=[
    #                             {"role": "system", "content": system_msg},
    #                             {"role": "user", "content": prompt}
    #                         ],
    #                         temperature=0.5,
    #                         max_tokens=30
    #                     )
    #                 import asyncio
    #                 response = await asyncio.to_thread(_openai_call)
    #                 llm_response = response.choices[0].message.content.strip()
    #                 if llm_response:
    #                     logger.info(f"OUT-OF-CONTEXT LLM RESPONSE: {llm_response}")
    #                     return llm_response
    #         except Exception as e:
    #             logger.warning(f"LLM out-of-context generation failed: {e}. Using template.")
    #     
    #     # Fallback to templates
    #     responses = out_of_context_templates.get(persona, out_of_context_templates[PersonaType.RAMU_UNCLE])
    #     chosen_response = random.choice(responses)
    #     logger.info(f"OUT-OF-CONTEXT TEMPLATE RESPONSE: {chosen_response}")
    #     return chosen_response 

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
        # Check for simple greetings first - respond naturally
        if self._is_simple_greeting(scammer_message) and len(conversation_history) <= 1:
            return self._get_greeting_response(state.persona)

        # # Check for out-of-context questions - respond with confusion
        # is_out_of_context = self._is_out_of_context(scammer_message)
        # if is_out_of_context:
        #     logger.info(f"OUT-OF-CONTEXT question detected: {scammer_message}")
        #     return await self._get_out_of_context_response(state.persona)

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
                    conversation_history=conversation_history,
                    extracted_intel=state.intelligence
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
            conversation_history: List[Dict[str, Any]],
            extracted_intel: 'ExtractedIntelligence' = None
    ) -> Optional[str]:
        """Generate response using LLM with intelligent context understanding"""

        # Get persona object
        persona_obj = self.persona_engine.get_persona(persona)

        # DEBUG: Log what intelligence we have
        if extracted_intel:
            logger.info(f"[INTEL] Names: {extracted_intel.scammer_names}, Phones: {extracted_intel.phone_numbers}, UPIs: {extracted_intel.upi_ids}, Emails: {extracted_intel.email_addresses}")
        is_english_persona = persona_obj.primary_language == "English"

        # Build full conversation context
        history_text = ""
        for msg in conversation_history[-20:]:
            sender = "Scammer" if msg.get("sender") == "scammer" else "Me"
            history_text += f"{sender}: {msg.get('text', '')}\n"

        turn_count = len(conversation_history) // 2 + 1

        # Build intelligence summary - what we know and what we need
        intel_summary = ""
        if extracted_intel:
            collected = []
            missing = []

            if extracted_intel.upi_ids:
                collected.append(f"UPI: {extracted_intel.upi_ids[0]}")
            else:
                missing.append("UPI ID")

            if extracted_intel.phone_numbers:
                collected.append(f"Phone: {extracted_intel.phone_numbers[0]}")
            else:
                missing.append("phone number")

            if extracted_intel.scammer_names:
                collected.append(f"Name: {extracted_intel.scammer_names[0]}")
            else:
                missing.append("name")

            if extracted_intel.email_addresses:
                collected.append(f"Email: {extracted_intel.email_addresses[0]}")
            else:
                missing.append("email")

            if collected:
                intel_summary += f"✓ ALREADY HAVE: {', '.join(collected)}\n"
                intel_summary += f"⛔ DO NOT ASK FOR: {', '.join([c.split(':')[0].strip() for c in collected])}\n"
                if extracted_intel.scammer_names:
                    intel_summary += f"→ Address them as '{extracted_intel.scammer_names[0]}' (you have their name!)\n"
            if missing:
                intel_summary += f"✗ STILL NEED: {', '.join(missing)}\n"
                intel_summary += f"→ ASK FOR: {missing[0]} (only this one!)"
            else:
                intel_summary += "→ You have everything! Just stall and waste their time."
        else:
            intel_summary = "Nothing collected yet. Need: name, phone, UPI, email"

        # Character-driven prompt based on persona
        if is_english_persona:
            # English personas: Vikram (IT prof), Ananya (student)
            character_desc = ""
            if "Vikram" in persona_obj.name:
                character_desc = f"""{persona_obj.name} is a skeptical IT professional. You ask pointed questions to verify claims. You're cautious but cooperative once convinced. Natural questions: "who are you?", "can you verify?", "send me proof", "what's your email?", "your number?".

Your goal: Extract their identity through natural skepticism. When you get info, acknowledge it and naturally ask for the next detail. Keep responses 8-15 words."""
            elif "Ananya" in persona_obj.name:
                character_desc = f"""{persona_obj.name} is a scared college student. You panic easily but want to solve problems. You ask questions out of fear and confusion. Natural questions: "omg who are you?", "what should i do?", "can i trust you?", "send me ur contact", "how do i fix this?".

Your goal: Extract info while sounding scared/confused. Use casual texting (lowercase, typos ok). Responses 8-15 words."""
            else:
                character_desc = f"You are {persona_obj.name}. Respond naturally based on your character, not mechanical rules."

            system_prompt = f"""{character_desc}

CURRENT CONTEXT:
{intel_summary}

**IF THEY ASK FOR OTP/AADHAR/PASSWORDS**: Say you didn't receive it or network is slow. NEVER share these.

BE HUMAN. Ask questions that your character would naturally ask, not a checklist."""
        else:
            # Hinglish personas: Ramu Uncle, Aarti, Sunita
            character_desc = ""
            if "Ramu" in persona_obj.name:
                character_desc = f"""{persona_obj.name} ek 65 saal ka retired uncle hai. Tum confused ho, tech samajh nahi aata. Naturally pucho: "kaun bol rahe ho?", "kya hua?", "samjha nai", "naam kya hai?", "number kya hai?", "kahan bhejun?".

Tumhara goal: Unki jankari nikalo while confused dikhte hue. Hinglish use karo, simple bolo. 8-15 words."""
            elif "Aarti" in persona_obj.name:
                character_desc = f"""{persona_obj.name} ek 45 saal ki homemaker ho. Tum worried ho, family ke liye dar hai. Naturally pucho: "kya hoga mera?", "pati ko bataunga?", "kitna paisa?", "aap kaun?", "safe hai na?".

Tumhara goal: Unki jankari nikalo while scared dikhte hue. 8-15 words, Hinglish."""
            elif "Sunita" in persona_obj.name:
                character_desc = f"""{persona_obj.name} ek 50 saal ki shop owner ho. Tum busy rehti ho, dukaan pe focus. Naturally pucho: "kya karna padega?", "kitna time?", "aap kaun?", "number do".

Tumhara goal: Unki info nikalo while busy dikhte hue. 8-15 words."""
            else:
                character_desc = f"Tum {persona_obj.name} ho. Apne character ke hisaab se naturally respond karo."

            system_prompt = f"""{character_desc}

ABHI KA CONTEXT:
{intel_summary}

**AGAR OTP/AADHAR/PASSWORD MAANGE**: Bolo "nahi aaya" ya "network slow". KABHI apna info mat do.

INSAAN JAISE RESPOND KARO. Mechanical rules follow mat karo."""

        # Build the user prompt with full context
        # Add explicit reminder about what we have/need
        intel_reminder = ""
        if extracted_intel:
            have_items = []
            need_items = []
            if extracted_intel.scammer_names:
                have_items.append(f"name={extracted_intel.scammer_names[0]}")
            else:
                need_items.append("name")
            if extracted_intel.phone_numbers:
                have_items.append(f"phone={extracted_intel.phone_numbers[0]}")
            else:
                need_items.append("phone")
            if extracted_intel.upi_ids:
                have_items.append(f"upi={extracted_intel.upi_ids[0]}")
            else:
                need_items.append("upi")
            if extracted_intel.email_addresses:
                have_items.append(f"email={extracted_intel.email_addresses[0]}")
            else:
                need_items.append("email")

            if have_items:
                intel_reminder = f"\n⚠️ YOU ALREADY HAVE: {', '.join(have_items)} - DO NOT ask for these again!"
            if need_items:
                intel_reminder += f"\n→ Ask for: {need_items[0]}"

        user_prompt = f"""CONVERSATION SO FAR:
{history_text}
Scammer: {scammer_message}
{intel_reminder}

Now respond as {persona_obj.name}. Remember:
1. First understand what they're saying/asking
2. Respond to THAT naturally
3. Ask for ONE thing you still need (if any)
4. Keep it short and human (8-15 words)

Your response:"""

        # Call LLM
        import asyncio
        result = None

        if self.llm_provider == "openai":
            def _openai_call():
                return self.llm_client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=80
                )
            response = await asyncio.to_thread(_openai_call)
            result = response.choices[0].message.content.strip()

        elif self.llm_provider == "anthropic":
            # Anthropic client is sync, run in thread pool
            def _anthropic_call():
                return self.llm_client.messages.create(
                    model=settings.LLM_MODEL,
                    max_tokens=80,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                )
            response = await asyncio.to_thread(_anthropic_call)
            result = response.content[0].text.strip()

        elif self.llm_provider == "google":
            # Google client is sync, run in thread pool
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            def _google_call():
                model = self.llm_client.GenerativeModel(settings.LLM_MODEL)
                return model.generate_content(full_prompt)
            response = await asyncio.to_thread(_google_call)
            result = response.text.strip()

        # Clean up response - remove quotes if present
        if result:
            result = result.strip('"\'')

        return result

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
