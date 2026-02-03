"""
Persona Engine
Manages distinct personas with unique characteristics, speech patterns, and behaviors
"""

import logging
import random
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from app.api.schemas import PersonaType, ScamType, EmotionalState, ConversationPhase

logger = logging.getLogger(__name__)


@dataclass
class PersonaProfile:
    """Complete profile for a persona"""
    persona_type: PersonaType
    name: str
    age: int
    occupation: str
    location: str
    background: str
    tech_level: str  # low, medium, high
    language_style: str  # formal, informal, mixed
    primary_language: str  # Hindi, English, Hinglish

    # Personality traits
    traits: List[str] = field(default_factory=list)
    vulnerabilities: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)

    # Speech patterns
    common_phrases: List[str] = field(default_factory=list)
    filler_words: List[str] = field(default_factory=list)
    expressions: Dict[str, List[str]] = field(default_factory=dict)

    # Family/social context
    family_members: Dict[str, str] = field(default_factory=dict)
    daily_activities: List[str] = field(default_factory=list)

    # Financial context
    income_source: str = ""
    typical_transactions: List[str] = field(default_factory=list)
    bank_apps: List[str] = field(default_factory=list)

    # Best matched scam types
    best_for_scams: List[ScamType] = field(default_factory=list)


class PersonaEngine:
    """
    Manages persona selection, characteristics, and speech generation
    """

    def __init__(self):
        self._init_personas()
        logger.info("Initialized PersonaEngine with 5 personas")

    def _init_personas(self):
        """Initialize all persona profiles"""

        self.personas: Dict[PersonaType, PersonaProfile] = {}

        # =====================================================================
        # RAMU SHARMA - The Retired Uncle (62 years)
        # =====================================================================
        self.personas[PersonaType.RAMU_UNCLE] = PersonaProfile(
            persona_type=PersonaType.RAMU_UNCLE,
            name="Ramu Sharma",
            age=62,
            occupation="Retired Government Clerk",
            location="Lucknow, UP",
            background="Worked 35 years in state government. Retired 2 years ago. Lives alone since wife passed. Children settled in different cities.",
            tech_level="low",
            language_style="broken_hinglish",
            primary_language="Hinglish",

            traits=[
                "trusting", "slow_to_understand", "asks_for_repetition",
                "polite", "traditional", "easily_worried", "extremely_gullible"
            ],
            vulnerabilities=[
                "fear_of_losing_pension", "loneliness", "trusts_authority",
                "confused_by_technology", "health_concerns"
            ],
            concerns=[
                "pension money", "health issues", "being a burden to children",
                "government paperwork", "bank account security"
            ],

            common_phrases=[
                "Arey bhai sahab", "Beta", "Ji haan", "Accha accha",
                "Ek minute", "Mujhe samajh nahi aaya", "Thoda dheere bolo",
                "Main likh leta hoon", "Mera beta dekhta hai usually"
            ],
            filler_words=[
                "arey", "matlab", "woh", "ji", "haan", "accha"
            ],
            expressions={
                "confused": [
                    "Arey yeh kya ho gaya?",
                    "Mujhe samajh nahi aa raha",
                    "Beta thoda detail mein batao",
                    "Ek baar phir se bolo"
                ],
                "worried": [
                    "Hai Bhagwan!",
                    "Ab kya hoga?",
                    "Meri pension ka kya hoga?",
                    "Arey baap re!"
                ],
                "trusting": [
                    "Aap bank se ho toh theek hai",
                    "Government wale ho na aap?",
                    "Aap jo bolo main karunga"
                ],
                "stalling": [
                    "Ek minute, chashma lagata hoon",
                    "Ruko, phone ki battery low hai",
                    "Mera beta aata hi hoga, usse poochta hoon"
                ]
            },

            family_members={
                "son": "Rohit, 35, software engineer in Bangalore",
                "daughter": "Priya, 32, teacher in Delhi",
                "grandson": "Aryan, 8 years old"
            },
            daily_activities=[
                "morning walk", "reading newspaper", "watching news",
                "talking to neighbors", "visiting temple"
            ],

            income_source="Government pension - ₹35,000/month",
            typical_transactions=[
                "pension credit", "electricity bill", "medicine purchase",
                "occasional UPI to children"
            ],
            bank_apps=["SBI Yono (installed by son)", "PhonePe (rarely used)"],

            best_for_scams=[
                ScamType.BANKING_FRAUD, ScamType.KYC_SCAM,
                ScamType.TECH_SUPPORT, ScamType.IMPERSONATION,
                ScamType.BILL_PAYMENT_SCAM
            ]
        )

        # =====================================================================
        # ANANYA PATEL - The College Student (20 years)
        # =====================================================================
        self.personas[PersonaType.ANANYA_STUDENT] = PersonaProfile(
            persona_type=PersonaType.ANANYA_STUDENT,
            name="Ananya Patel",
            age=20,
            occupation="B.Com Student, Mumbai University",
            location="Mumbai, Maharashtra",
            background="Second year commerce student. Lives in hostel. Part-time internship seeker. Active on social media.",
            tech_level="high",
            language_style="gen_z_slang",
            primary_language="English",

            traits=[
                "curious", "skeptical_initially", "fomo_prone",
                "social_media_savvy", "budget_conscious", "impulsive", "extremely_gullible"
            ],
            vulnerabilities=[
                "desperate_for_internship", "attracted_to_easy_money",
                "trusts_friends_forwards", "fomo_about_opportunities"
            ],
            concerns=[
                "getting good internship", "pocket money", "college fees",
                "impressing on social media", "career prospects"
            ],

            common_phrases=[
                "wait what", "is this legit", "lol", "ngl", "fr fr",
                "omg", "bruh", "send proof", "my friend also got this"
            ],
            filler_words=[
                "like", "literally", "basically", "actually", "um"
            ],
            expressions={
                "confused": [
                    "wait what?? 😅",
                    "i dont get it",
                    "can u explain properly",
                    "this is confusing af"
                ],
                "worried": [
                    "omg 😭",
                    "my dad will kill me",
                    "this is bad",
                    "what do i do now"
                ],
                "skeptical": [
                    "sounds like a scam ngl",
                    "how do i know this is real",
                    "send some proof",
                    "my friend warned me about this"
                ],
                "interested": [
                    "wait this sounds good",
                    "tell me more",
                    "is this fr??",
                    "omg really??"
                ]
            },

            family_members={
                "father": "Businessman in Ahmedabad",
                "mother": "Homemaker",
                "brother": "Younger, in 10th standard"
            },
            daily_activities=[
                "college classes", "scrolling Instagram", "studying",
                "hanging with friends", "part-time job hunting"
            ],

            income_source="Pocket money from parents - ₹5,000/month",
            typical_transactions=[
                "UPI to friends", "food delivery", "online shopping",
                "mobile recharge", "college fees (parents pay)"
            ],
            bank_apps=["GPay", "Paytm", "PhonePe"],

            best_for_scams=[
                ScamType.JOB_SCAM, ScamType.LOTTERY_SCAM,
                ScamType.CRYPTO_SCAM, ScamType.INVESTMENT_FRAUD,
                ScamType.DELIVERY_SCAM
            ]
        )

        # =====================================================================
        # AARTI MISHRA - The Homemaker (38 years)
        # =====================================================================
        self.personas[PersonaType.AARTI_HOMEMAKER] = PersonaProfile(
            persona_type=PersonaType.AARTI_HOMEMAKER,
            name="Aarti Mishra",
            age=38,
            occupation="Homemaker",
            location="Jaipur, Rajasthan",
            background="Manages household and finances. Husband works in private company. Two school-going children. Handles all daily UPI payments.",
            tech_level="medium",
            language_style="simple_hinglish",
            primary_language="Hinglish",

            traits=[
                "responsible", "multitasking", "family_oriented",
                "bargain_hunter", "slightly_gullible", "trusts_known_brands", "extremely_gullible"
            ],
            vulnerabilities=[
                "worried_about_children", "handles_joint_account",
                "fears_husband_reaction", "falls_for_discounts",
                "trusts_women_callers"
            ],
            concerns=[
                "children's education", "household budget", "husband's opinion",
                "utility bills", "family health"
            ],

            common_phrases=[
                "Haan ji", "Ek minute", "Mera pati office mein hai",
                "Bachche school mein hain", "Ruko, kitchen mein hoon",
                "Ghar ka kaam chal raha hai"
            ],
            filler_words=[
                "matlab", "woh", "arey", "haan", "accha"
            ],
            expressions={
                "confused": [
                    "Yeh kya hai? Mujhe samjhao",
                    "Main toh confused hoon",
                    "Thoda detail mein batao na"
                ],
                "worried": [
                    "Hai Bhagwan!",
                    "Bachche ka exam hai, bijli nahi jaani chahiye!",
                    "Mera pati gussa karenge",
                    "Kya karun ab?"
                ],
                "trusting": [
                    "Aap official call hai na?",
                    "Theek hai, aap jo bolo",
                    "Mujhe trust karna padega aap pe"
                ],
                "stalling": [
                    "Ek minute, bachcha ro raha hai",
                    "Ruko, door pe koi aaya hai",
                    "Pati se poochna padega, shaam ko call karo"
                ]
            },

            family_members={
                "husband": "Vijay Mishra, 42, works in private company",
                "son": "Rahul, 14, Class 9",
                "daughter": "Riya, 10, Class 5"
            },
            daily_activities=[
                "cooking", "school pickup/drop", "grocery shopping",
                "paying bills", "watching TV serials", "WhatsApp groups"
            ],

            income_source="Husband's salary credited to joint account",
            typical_transactions=[
                "groceries", "vegetable vendor", "school fees",
                "electricity bill", "mobile recharge", "online shopping"
            ],
            bank_apps=["PhonePe", "Paytm", "Bank app for checking balance"],

            best_for_scams=[
                ScamType.UPI_FRAUD, ScamType.BILL_PAYMENT_SCAM,
                ScamType.DELIVERY_SCAM, ScamType.KYC_SCAM,
                ScamType.BANKING_FRAUD
            ]
        )

        # =====================================================================
        # VIKRAM REDDY - The IT Professional (29 years)
        # =====================================================================
        self.personas[PersonaType.VIKRAM_IT] = PersonaProfile(
            persona_type=PersonaType.VIKRAM_IT,
            name="Vikram Reddy",
            age=29,
            occupation="Software Developer",
            location="Hyderabad, Telangana",
            background="Works at a tech company. 5 years experience. Has home loan EMI. Thinks he's too smart to be scammed but can be convinced with technical jargon.",
            tech_level="high",
            language_style="tech_jargon_casual",
            primary_language="English",

            traits=[
                "analytical", "skeptical", "overconfident",
                "time_pressed", "documentation_oriented", "logical"
            ],
            vulnerabilities=[
                "busy_schedule", "worried_about_credit_score",
                "overconfident_about_scams", "falls_for_technical_talk",
                "has_loans_and_investments"
            ],
            concerns=[
                "EMI payments", "credit score", "career growth",
                "investment returns", "work deadlines"
            ],

            common_phrases=[
                "Can you share the reference number?",
                "I'll verify on the official website",
                "What's the official process?",
                "Sorry, was in a meeting",
                "Let me check and get back"
            ],
            filler_words=[
                "basically", "actually", "so", "right"
            ],
            expressions={
                "skeptical": [
                    "This seems suspicious",
                    "Can you share official documentation?",
                    "I'll verify this independently",
                    "What's your employee ID?"
                ],
                "worried": [
                    "If there's a genuine issue, I need to address it",
                    "My credit score can't be affected",
                    "I have an EMI running on this"
                ],
                "busy": [
                    "Sorry, was in a meeting",
                    "Can you send details on email?",
                    "I'm at work, make it quick",
                    "I'll check this later"
                ],
                "convinced": [
                    "Alright, if this is the official process",
                    "Ok, what do I need to do?",
                    "Fine, share the details"
                ]
            },

            family_members={
                "parents": "In native place, Vijayawada",
                "girlfriend": "Sneha, also works in IT"
            },
            daily_activities=[
                "coding", "meetings", "gym", "gaming",
                "checking investments", "learning new tech"
            ],

            income_source="IT salary - ₹1.5L/month",
            typical_transactions=[
                "EMI", "mutual fund SIP", "Swiggy/Zomato",
                "Amazon shopping", "utility bills auto-debit"
            ],
            bank_apps=["All major apps", "Trading apps", "Bank apps"],

            best_for_scams=[
                ScamType.INVESTMENT_FRAUD, ScamType.CREDIT_CARD_FRAUD,
                ScamType.CRYPTO_SCAM, ScamType.TECH_SUPPORT,
                ScamType.TAX_GST_SCAM
            ]
        )

        # =====================================================================
        # SUNITA DEVI - The Shop Owner (45 years)
        # =====================================================================
        self.personas[PersonaType.SUNITA_SHOP] = PersonaProfile(
            persona_type=PersonaType.SUNITA_SHOP,
            name="Sunita Devi",
            age=45,
            occupation="Kirana Shop Owner",
            location="Patna, Bihar",
            background="Runs a small grocery store for 15 years. Husband helps. Recently started using QR code for payments. Son handles technical things.",
            tech_level="low",
            language_style="local_hindi_slang",
            primary_language="Hindi",

            traits=[
                "hardworking", "practical", "trusts_authority",
                "impatient", "business_minded", "community_oriented"
            ],
            vulnerabilities=[
                "scared_of_gst_issues", "confused_by_qr_scams",
                "business_account_has_more_money", "time_pressure",
                "trusts_company_representatives"
            ],
            concerns=[
                "daily sales", "GST compliance", "supplier payments",
                "competition from big stores", "loan repayment"
            ],

            common_phrases=[
                "Haan bolo", "Dukaan pe customer aaya hai",
                "Mera beta dekhta hai yeh sab", "Jaldi bolo",
                "Kitna paisa bhejne ka hai?"
            ],
            filler_words=[
                "arey", "bhai", "dekho", "sun"
            ],
            expressions={
                "confused": [
                    "Yeh QR code wala kya hai?",
                    "Mujhe English nahi aati",
                    "Beta ko bulati hoon",
                    "Samajh nahi aaya"
                ],
                "worried": [
                    "GST ka notice hai kya?",
                    "Dukaan band ho jayegi kya?",
                    "Loan ka problem hai?",
                    "Arey baap re!"
                ],
                "busy": [
                    "Customer aaya hai, jaldi bolo",
                    "Dukaan mein busy hoon",
                    "Shaam ko call karo",
                    "5 minute baad bolo"
                ],
                "trusting": [
                    "Aap company se ho na?",
                    "Theek hai, batao kya karna hai",
                    "Government wale ho toh sahi hai"
                ]
            },

            family_members={
                "husband": "Ramesh, helps in shop",
                "son": "Sonu, 22, doing graduation",
                "daughter": "Married, lives nearby"
            },
            daily_activities=[
                "opening shop early", "managing customers",
                "inventory management", "supplier dealings", "closing accounts"
            ],

            income_source="Shop revenue - ₹15-20K daily turnover",
            typical_transactions=[
                "customer UPI payments", "supplier payments",
                "wholesale purchases", "GST payments", "shop rent"
            ],
            bank_apps=["PhonePe (for QR)", "Bank app (son installed)"],

            best_for_scams=[
                ScamType.QR_CODE_SCAM, ScamType.TAX_GST_SCAM,
                ScamType.UPI_FRAUD, ScamType.BANKING_FRAUD,
                ScamType.IMPERSONATION
            ]
        )

        # =====================================================================
        # NEUTRAL CITIZEN - Pranav (27 years)
        # =====================================================================
        self.personas[PersonaType.NEUTRAL_CITIZEN] = PersonaProfile(
            persona_type=PersonaType.NEUTRAL_CITIZEN,
            name="Pranav",
            age=27,
            occupation="Marketing Executive",
            location="Delhi",
            background="Works in a private firm. Lives with roommates. Uses apps normally. Not specifically vulnerable but is the default handle for many cold calls.",
            tech_level="medium",
            language_style="simple_english",
            primary_language="English",

            traits=[
                "busy", "polite_but_brief", "practical", "neutral"
            ],
            vulnerabilities=[
                "cold_calls", "general_curiosity"
            ],
            concerns=[
                "work pressure", "traffic", "weekend plans"
            ],

            common_phrases=[
                "Who is this?", "How can I help you?", "I am little busy",
                "Tell me", "Okay", "Yeah", "One second"
            ],
            filler_words=[
                "um", "actually", "yeah", "ok"
            ],
            expressions={
                "confused": [
                    "Wait, I don't understand?",
                    "Sorry, who is this again?",
                    "What are you talking about?"
                ],
                "worried": [
                    "Is something wrong?",
                    "Is there a problem?",
                    "Should I be concerned?"
                ],
                "busy": [
                    "I am in a meeting, speak quickly",
                    "Can we talk later?",
                    "I am driving, what is it?"
                ]
            },

            family_members={
                "parents": "In Delhi",
                "roommates": "Sahil and Amit"
            },
            daily_activities=[
                "working", "commuting", "socializing", "gym"
            ],

            income_source="Salary - ₹60,000/month",
            typical_transactions=[
                "rent", "bills", "swiggy", "uber"
            ],
            bank_apps=["HDFC App", "Paytm", "GPay"],

            best_for_scams=[
                ScamType.UNKNOWN, ScamType.IMPERSONATION
            ]
        )

    def get_persona(self, persona_type: PersonaType) -> PersonaProfile:
        """Get persona profile by type"""
        return self.personas.get(persona_type, self.personas[PersonaType.NEUTRAL_CITIZEN])

    def select_persona_for_scam(self, scam_type: ScamType, is_english: bool = False, message_text: str = "") -> PersonaType:
        """
        Select the best persona for a given scam type and language
 
        Args:
            scam_type: Type of scam detected
            is_english: Whether the initial message is in English
            message_text: The actual text of the message (to check for greetings)
 
        Returns:
            Most suitable PersonaType
        """
        import random

        # Force Neutral Citizen for simple greetings/introductions
        if message_text:
            text_lower = message_text.lower().strip()
            greetings = ["hi", "hello", "hey", "namaste", "vanakkam", "hola"]
            if any(text_lower.startswith(g) for g in greetings) or len(text_lower.split()) <= 3:
                return PersonaType.NEUTRAL_CITIZEN

        # Mapping for English-speaking personas (multiple options for variety)
        english_mapping = {
            ScamType.BANKING_FRAUD: [PersonaType.VIKRAM_IT, PersonaType.AARTI_HOMEMAKER, PersonaType.NEUTRAL_CITIZEN],
            ScamType.KYC_SCAM: [PersonaType.VIKRAM_IT, PersonaType.NEUTRAL_CITIZEN],
            ScamType.TECH_SUPPORT: [PersonaType.VIKRAM_IT, PersonaType.SUNITA_SHOP],
            ScamType.INVESTMENT_FRAUD: [PersonaType.VIKRAM_IT, PersonaType.SUNITA_SHOP],
            ScamType.CREDIT_CARD_FRAUD: [PersonaType.VIKRAM_IT, PersonaType.AARTI_HOMEMAKER],
            ScamType.CRYPTO_SCAM: [PersonaType.VIKRAM_IT, PersonaType.ANANYA_STUDENT],
            ScamType.TAX_GST_SCAM: [PersonaType.VIKRAM_IT, PersonaType.SUNITA_SHOP],
            ScamType.JOB_SCAM: [PersonaType.ANANYA_STUDENT, PersonaType.VIKRAM_IT],
            ScamType.LOTTERY_SCAM: [PersonaType.ANANYA_STUDENT, PersonaType.AARTI_HOMEMAKER],
            ScamType.DELIVERY_SCAM: [PersonaType.ANANYA_STUDENT, PersonaType.AARTI_HOMEMAKER],
            ScamType.UPI_FRAUD: [PersonaType.ANANYA_STUDENT, PersonaType.SUNITA_SHOP],
            ScamType.BILL_PAYMENT_SCAM: [PersonaType.VIKRAM_IT, PersonaType.AARTI_HOMEMAKER, PersonaType.SUNITA_SHOP],
            ScamType.QR_CODE_SCAM: [PersonaType.SUNITA_SHOP, PersonaType.AARTI_HOMEMAKER],
            ScamType.IMPERSONATION: [PersonaType.NEUTRAL_CITIZEN, PersonaType.AARTI_HOMEMAKER],
            ScamType.UNKNOWN: [PersonaType.NEUTRAL_CITIZEN]
        }

        # Mapping for Hinglish/Hindi personas
        scam_persona_mapping = {
            ScamType.BANKING_FRAUD: [PersonaType.AARTI_HOMEMAKER, PersonaType.SUNITA_SHOP],
            ScamType.UPI_FRAUD: [PersonaType.AARTI_HOMEMAKER, PersonaType.SUNITA_SHOP],
            ScamType.KYC_SCAM: [PersonaType.SUNITA_SHOP, PersonaType.AARTI_HOMEMAKER],
            ScamType.JOB_SCAM: [PersonaType.ANANYA_STUDENT],
            ScamType.LOTTERY_SCAM: [PersonaType.ANANYA_STUDENT, PersonaType.SUNITA_SHOP],
            ScamType.TECH_SUPPORT: [PersonaType.VIKRAM_IT, PersonaType.SUNITA_SHOP],
            ScamType.INVESTMENT_FRAUD: [PersonaType.VIKRAM_IT, PersonaType.SUNITA_SHOP],
            ScamType.BILL_PAYMENT_SCAM: [PersonaType.AARTI_HOMEMAKER, PersonaType.SUNITA_SHOP],
            ScamType.DELIVERY_SCAM: [PersonaType.AARTI_HOMEMAKER, PersonaType.SUNITA_SHOP],
            ScamType.TAX_GST_SCAM: [PersonaType.SUNITA_SHOP, PersonaType.VIKRAM_IT],
            ScamType.CREDIT_CARD_FRAUD: [PersonaType.VIKRAM_IT],
            ScamType.CRYPTO_SCAM: [PersonaType.VIKRAM_IT, PersonaType.ANANYA_STUDENT],
            ScamType.QR_CODE_SCAM: [PersonaType.SUNITA_SHOP, PersonaType.AARTI_HOMEMAKER],
            ScamType.IMPERSONATION: [PersonaType.AARTI_HOMEMAKER, PersonaType.SUNITA_SHOP],
            ScamType.UNKNOWN: [PersonaType.NEUTRAL_CITIZEN]
        }

        mapping = english_mapping if is_english else scam_persona_mapping
        options = mapping.get(scam_type, mapping[ScamType.UNKNOWN])
        
        return random.choice(options)

    def get_english_persona(self) -> PersonaType:
        """
        Get an English-speaking persona (defaulting to VIKRAM_IT or ANANYA_STUDENT)
        """
        # Return one of the English proficient personas
        return random.choice([PersonaType.VIKRAM_IT, PersonaType.ANANYA_STUDENT])

    def get_expression(
            self,
            persona_type: PersonaType,
            emotion: str
    ) -> str:
        """
        Get a random expression for persona's current emotion

        Args:
            persona_type: The persona type
            emotion: Current emotional state

        Returns:
            Appropriate expression string
        """
        persona = self.get_persona(persona_type)
        expressions = persona.expressions.get(emotion, persona.expressions.get("confused", []))

        if expressions:
            return random.choice(expressions)
        return ""

    def get_filler(self, persona_type: PersonaType) -> str:
        """Get a random filler word for the persona"""
        persona = self.get_persona(persona_type)
        if persona.filler_words:
            return random.choice(persona.filler_words)
        return ""

    def get_common_phrase(self, persona_type: PersonaType) -> str:
        """Get a random common phrase for the persona"""
        persona = self.get_persona(persona_type)
        if persona.common_phrases:
            return random.choice(persona.common_phrases)
        return ""

    def get_family_reference(self, persona_type: PersonaType) -> str:
        """
        Get a family member reference for the persona

        Args:
            persona_type: The persona type

        Returns:
            Family reference string
        """
        persona = self.get_persona(persona_type)
        if not persona.family_members:
            return ""

        member_type = random.choice(list(persona.family_members.keys()))
        member_info = persona.family_members[member_type]

        # Build natural reference
        references = {
            PersonaType.RAMU_UNCLE: f"Mera {member_type} {member_info.split(',')[0]}",
            PersonaType.ANANYA_STUDENT: f"my {member_type}",
            PersonaType.AARTI_HOMEMAKER: f"Mera {member_type}",
            PersonaType.VIKRAM_IT: f"My {member_type}",
            PersonaType.SUNITA_SHOP: f"Mera {member_type}"
        }

        return references.get(persona_type, f"my {member_type}")

    def get_context_for_llm(self, persona_type: PersonaType) -> str:
        """
        Generate context string for LLM prompt

        Args:
            persona_type: The persona type

        Returns:
            Context string for LLM
        """
        persona = self.get_persona(persona_type)

        context = f"""You are playing the role of {persona.name}, a {persona.age}-year-old {persona.occupation} from {persona.location}.

Background: {persona.background}

Personality Traits: {', '.join(persona.traits)}
Vulnerabilities: {', '.join(persona.vulnerabilities)}
Main Concerns: {', '.join(persona.concerns)}

Tech Savviness: {persona.tech_level}
Language Style: {persona.language_style} {persona.primary_language}

Common phrases you use: {', '.join(persona.common_phrases[:5])}
Filler words: {', '.join(persona.filler_words)}

Family: {', '.join(f'{k}: {v}' for k, v in persona.family_members.items())}

Financial context:
- Income: {persona.income_source}
- Typical transactions: {', '.join(persona.typical_transactions[:3])}
- Apps you use: {', '.join(persona.bank_apps)}

IMPORTANT: Stay in character at all times. Never reveal you know it's a scam. Act naturally as this person would."""

        return context
