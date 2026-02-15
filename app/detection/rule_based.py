"""
Rule-Based Scam Detector
Uses predefined rules and keyword patterns to detect scam intent
"""

import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

from app.api.schemas import ScamType

logger = logging.getLogger(__name__)


@dataclass
class RuleMatch:
    """Represents a rule match"""
    rule_name: str
    category: str
    score: float
    matched_text: str


@dataclass
class RuleBasedResult:
    """Result from rule-based detection"""
    score: float
    scam_type: ScamType
    matches: List[RuleMatch] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    keywords_found: List[str] = field(default_factory=list)


class RuleBasedDetector:
    """
    Rule-based scam detection using keyword matching and pattern rules
    Weight: 0.25 in ensemble
    """

    def __init__(self, weight: float = 0.25):
        self.weight = weight
        self._init_rules()

    def _init_rules(self):
        """Initialize detection rules and keyword lists"""

        # =================================================================
        # URGENCY KEYWORDS (create panic) - 50+ patterns
        # =================================================================
        self.urgency_keywords = {
            "high": [
                "immediately", "urgent", "urgently", "right now", "today only",
                "last chance", "final warning", "expires today",
                "within 24 hours", "abhi", "turant", "jaldi",
                "now or never", "deadline", "expires in", "only today",
                "act immediately", "do it now", "without delay",
                "time is running out", "hours left", "minutes left",
                "before midnight", "before 6pm", "closing soon",
                "fauran", "abhi ke abhi", "turant karo", "jaldi karo",
                "der mat karo", "waqt nahi hai", "aaj hi", "kal tak",
                "2 ghante mein", "1 hour mein", "30 minutes mein",
                "need money urgently", "send money", "transfer money",
                "very bad place", "emergency", "stuck", "stranded",
                "help me", "please help", "in trouble", "bad situation"
            ],
            "medium": [
                "soon", "quickly", "hurry", "limited time",
                "don't delay", "act fast", "time sensitive",
                "as soon as possible", "at the earliest", "promptly",
                "without further delay", "rush", "expedite",
                "limited period", "offer ending", "stock limited"
            ],
            "low": [
                "asap", "priority", "important", "attention required",
                "action needed", "pending", "awaiting"
            ]
        }

        # =================================================================
        # THREAT KEYWORDS (create fear) - 60+ patterns
        # =================================================================
        self.threat_keywords = {
            "high": [
                "blocked", "suspended", "terminated", "cancelled",
                "legal action", "police", "arrest", "court",
                "prosecution", "jail", "fine", "penalty",
                "blacklisted", "band ho jayega", "block ho jayega",
                "fir", "complaint filed", "case registered",
                "warrant", "summons", "notice", "seizure",
                "criminal charges", "cyber crime", "fraud case",
                "account seized", "assets frozen", "jail term",
                "permanent ban", "service terminated", "access revoked",
                "giraftar", "kaid", "jurmana", "saza",
                "khatarnak", "illegal activity", "violation detected",
                "security threat", "unauthorized access", "breach detected",
                "lose everything", "all your money", "data stolen",
                # SIM/Telecom threats
                "sim blocked", "sim deactivated", "sim suspended", "sim cancelled",
                "sim will be blocked", "sim band", "number blocked", "number deactivated",
                "aadhaar not linked", "aadhaar verification", "verify aadhaar",
                "trai", "telecom", "mobile number blocked", "outgoing blocked"
            ],
            "medium": [
                "deactivated", "restricted", "frozen", "hold",
                "investigation", "fraud alert", "security breach",
                "suspicious activity", "unusual transaction", "risk detected",
                "verification failed", "identity mismatch", "flagged",
                "under review", "compliance issue", "regulatory action",
                "suspicious", "account suspicious", "activity suspicious"
            ],
            "low": [
                "issue", "problem", "concern", "attention needed",
                "alert", "warning", "notice", "reminder"
            ]
        }

        # =================================================================
        # REQUEST KEYWORDS (information seeking) - 70+ patterns
        # =================================================================
        self.request_keywords = {
            "high": [
                "otp", "pin", "password", "cvv", "mpin", "atm pin",
                "share your", "send your", "provide your",
                "enter your", "verify your", "confirm your",
                "bhejo", "batao", "de do", "bata do",
                "tell me your", "give me your", "need your",
                "send otp", "share otp", "otp bhejo", "otp batao",
                "card number", "expiry date", "security code",
                "bank details", "account details", "login details",
                "username password", "internet banking password",
                "mobile banking pin", "upi pin batao", "mpin share",
                "aadhar number", "pan number", "date of birth",
                "mother maiden name", "secret question", "security answer",
                "remote access", "screen share", "install anydesk",
                "install teamviewer", "download app", "scan qr code"
            ],
            "medium": [
                "click here", "click link", "tap here", "click below",
                "download", "install", "update required", "upgrade now",
                "verify", "confirm", "authenticate", "validate",
                "fill form", "submit details", "complete verification",
                "open link", "visit website", "go to url",
                "call this number", "whatsapp", "telegram join"
            ],
            "low": [
                "check", "review", "respond", "reply", "contact us"
            ]
        }

        # =================================================================
        # FINANCIAL KEYWORDS (money/banking context) - 80+ patterns
        # =================================================================
        self.financial_keywords = {
            "high": [
                "bank account", "upi", "paytm", "phonepe", "gpay",
                "google pay", "credit card", "debit card",
                "net banking", "transfer", "payment",
                "account number", "ifsc", "kyc", "pan card", "aadhar",
                "bhim upi", "amazon pay", "mobikwik", "freecharge",
                "savings account", "current account", "fixed deposit",
                "neft", "rtgs", "imps", "upi transfer",
                "bank transfer", "wire_transfer", "money transfer",
                "processing fee", "registration fee", "advance payment",
                "security deposit", "refundable deposit", "token amount",
                "gst", "tds", "income tax", "tax refund", "tax due",
                "loan approved", "loan sanction", "emi", "credit limit",
                "insurance premium", "policy payment", "claim amount",
                "pay this number", "send to this number", "transfer money",
                "pay to me", "send me the money", "account mein bhejo",
                "money bhejo", "rupees send", "pay kardo", "payment kardo"
            ],
            "medium": [
                "transaction", "balance", "amount", "rupees",
                "rs", "₹", "money", "funds", "refund", "cashback",
                "paisa", "rupe", "lakh", "crore", "thousand",
                "payment pending", "payment due", "outstanding",
                "wallet balance", "bank balance", "account balance",
                "bhejo", "bhejiye", "daal do", "transfer karo"
            ],
            "low": [
                "wallet", "pay", "receive", "send", "credit", "debit",
                "cost", "price", "charge", "fee"
            ]
        }

        # =================================================================
        # IMPERSONATION KEYWORDS (fake authority) - 90+ patterns
        # =================================================================
        self.impersonation_keywords = {
            "high": [
                "sbi", "hdfc", "icici", "axis", "rbi", "reserve bank",
                "income tax", "it department", "customs", "cyber cell",
                "police department", "government", "ministry",
                "amazon", "flipkart", "customer care", "customer support",
                "pnb", "bob", "canara", "kotak", "yes bank", "idbi",
                "union bank", "indian bank", "central bank", "boi",
                "lic", "insurance company", "mutual fund",
                "microsoft", "google", "apple", "facebook", "whatsapp",
                "reliance jio", "airtel", "vodafone", "bsnl",
                "electricity board", "gas company", "water department",
                "passport office", "rti", "noc", "municipality",
                "dgca", "rto", "traffic police", "cbi", "enforcement directorate", "nia", "digital arrest",
                "sebi", "irda", "rbi circular", "government scheme",
                "pm kisan", "ayushman bharat", "jan dhan", "mudra loan",
                "your friend", "this is your", "i am your", "its me",
                "your son", "your daughter", "your brother", "your sister",
                "your uncle", "your cousin", "your relative", "your boss",
                "stuck abroad", "stuck overseas", "stuck in", "arrested",
                "hospital", "accident", "lost wallet", "lost phone",
                "tera dost", "tera bhai", "main hoon", "mujhe pehchana"
            ],
            "medium": [
                "official", "authorized", "representative",
                "department", "head office", "branch manager",
                "executive", "officer", "inspector", "sub inspector", "commissioner",
                "senior official", "nodal officer", "zonal manager",
                "regional head", "verification team", "fraud department",
                "security team", "compliance team", "audit team",
                "crime branch", "cyber cell", "cyber crime"
            ],
            "low": [
                "team", "support", "helpdesk", "service", "desk", "cell"
            ]
        }

        # =================================================================
        # REWARD/LURE KEYWORDS (too good to be true) - 70+ patterns
        # =================================================================
        self.reward_keywords = {
            "high": [
                "won", "winner", "lottery", "prize", "jackpot",
                "lakh", "crore", "million", "free gift",
                "congratulations", "selected", "lucky",
                "you have won", "you are selected", "you are the winner",
                "claim your prize", "collect your reward", "gift voucher",
                "free iphone", "free car", "free bike", "free laptop",
                "100% cashback", "200% return", "300% return", "500% return",
                "double your money", "triple your money", "guaranteed profit",
                "risk free", "no loss", "assured returns", "fixed income",
                "daily income", "weekly income", "monthly income",
                "earn from home", "earn lakhs", "earn crores",
                "shortlisted", "pre-approved", "pre-selected", "eligible",
                "exclusive offer", "limited offer", "one time offer",
                "badhai ho", "mubarakho", "jeeta", "inaam", "tohfa"
            ],
            "medium": [
                "offer", "discount", "cashback", "reward",
                "bonus", "benefit", "guaranteed income",
                "special price", "reduced price", "lowest price",
                "best deal", "bumper offer", "dhamaka offer",
                "festival offer", "clearance sale", "flash sale"
            ],
            "low": [
                "opportunity", "chance", "special", "exclusive", "vip"
            ]
        }

        # =================================================================
        # SCAM TYPE SPECIFIC KEYWORDS - 150+ patterns
        # =================================================================
        self.scam_type_keywords = {
            ScamType.BANKING_FRAUD: [
                "bank", "account", "netbanking", "statement",
                "debit", "credit", "transaction failed",
                "account blocked", "account suspended", "unauthorized transaction",
                "suspicious transaction", "card blocked", "atm blocked",
                "internet banking", "mobile banking", "sms banking",
                "passbook", "cheque book", "bank manager", "branch"
            ],
            ScamType.UPI_FRAUD: [
                "upi", "paytm", "phonepe", "gpay", "bhim",
                "upi id", "upi pin", "scan qr", "request money",
                "collect request", "pay request", "upi collect",
                "wrong upi", "upi refund", "upi cashback",
                "@ybl", "@paytm", "@oksbi", "@okaxis", "@upi"
            ],
            ScamType.KYC_SCAM: [
                "kyc", "re-kyc", "update kyc", "kyc expired",
                "pan", "aadhar", "verification pending",
                "kyc update", "kyc verification", "ekyc",
                "video kyc", "kyc link", "complete kyc",
                "kyc mandatory", "kyc deadline", "kyc expiry"
            ],
            ScamType.JOB_SCAM: [
                "job offer", "work from home", "earn money",
                "part time", "data entry", "typing job",
                "no experience", "registration fee", "salary",
                "amazon job", "flipkart job", "google job",
                "online job", "home based job", "simple task",
                "copy paste job", "ad posting job", "form filling",
                "guaranteed job", "interview call", "offer letter",
                "wfh", "remote job", "freelance", "hiring urgently"
            ],
            ScamType.LOTTERY_SCAM: [
                "lottery", "lucky draw", "prize money",
                "winner", "jackpot", "claim prize",
                "whatsapp lottery", "kbc lottery", "bbc lottery",
                "international lottery", "online lottery",
                "bumper prize", "mega prize", "first prize",
                "consolation prize", "lucky number", "ticket number"
            ],
            ScamType.TECH_SUPPORT: [
                "virus", "malware", "hacked", "compromised",
                "remote access", "anydesk", "teamviewer",
                "computer", "laptop", "windows", "microsoft",
                "tech support", "technical support", "it support",
                "antivirus", "security software", "firewall",
                "system infected", "data at risk", "device hacked",
                "quicksupport", "ultraviewer", "ammyy admin"
            ],
            ScamType.INVESTMENT_FRAUD: [
                "investment", "trading", "stock", "forex",
                "bitcoin", "crypto", "guaranteed returns",
                "double money", "high returns", "mutual fund",
                "ipo", "share market", "nifty", "sensex",
                "cryptocurrency", "ethereum", "dogecoin",
                "trading platform", "investment scheme",
                "ponzi", "mlm", "network marketing", "referral income"
            ],
            ScamType.BILL_PAYMENT_SCAM: [
                "bill", "electricity", "overdue", "disconnection",
                "gas", "water", "pending payment", "due amount",
                "electricity bill", "bijli bill", "gas bill",
                "water bill", "phone bill", "broadband bill",
                "last date", "final notice", "disconnection notice",
                "supply cut", "meter reading", "outstanding amount"
            ],
            ScamType.DELIVERY_SCAM: [
                "delivery", "package", "courier", "parcel",
                "tracking", "customs", "hold", "shipping fee"
            ],
            ScamType.TAX_GST_SCAM: [
                "tax", "gst", "it return", "refund",
                "tax notice", "it department", "compliance"
            ],
            ScamType.IMPERSONATION: [
                # Friend-in-trouble patterns
                "your friend", "this is your", "i am your friend",
                "stuck abroad", "stuck overseas", "stranded",
                "need money", "send money", "lend me", "borrow",
                "emergency", "hospital", "accident", "arrested",
                "lost wallet", "lost phone", "help me out",
                "will repay", "pay you back", "return the money",
                "gpay", "paytm", "phonepe", "upi", "transfer",
                "in trouble", "bad situation", "very bad place",
                "your son", "your daughter", "your brother",
                "tera dost", "mera accident", "hospital mein",
                # Authority impersonation (Digital Arrest) patterns
                "digital arrest", "cyber arrest", "video call arrest",
                "police", "inspector", "sub inspector", "cbi", "nia",
                "customs", "enforcement directorate", "court order",
                "warrant", "fir", "case registered", "under arrest",
                "crime branch", "cyber cell", "cyber crime",
                "do not tell anyone", "confidential", "secret operation",
                "stay on video call", "do not disconnect", "being monitored"
            ]
        }

    def detect(self, text: str, history: List[str] = None) -> RuleBasedResult:
        """
        Analyze text for scam indicators using rules

        Args:
            text: Message text to analyze
            history: Previous messages for context

        Returns:
            RuleBasedResult with score and matches
        """
        text_lower = text.lower()
        all_text = text_lower
        if history:
            all_text += " " + " ".join(h.lower() for h in history)

        matches = []
        keywords_found = []

        # Check each keyword category
        urgency_score = self._check_keywords(
            text_lower, self.urgency_keywords, "urgency", matches, keywords_found
        )
        threat_score = self._check_keywords(
            text_lower, self.threat_keywords, "threat", matches, keywords_found
        )
        request_score = self._check_keywords(
            text_lower, self.request_keywords, "request", matches, keywords_found
        )
        financial_score = self._check_keywords(
            text_lower, self.financial_keywords, "financial", matches, keywords_found
        )
        impersonation_score = self._check_keywords(
            text_lower, self.impersonation_keywords, "impersonation", matches, keywords_found
        )
        reward_score = self._check_keywords(
            text_lower, self.reward_keywords, "reward", matches, keywords_found
        )

        # Calculate weighted score
        # Urgency + Threat + Request = classic scam pattern
        base_score = (
                urgency_score * 0.20 +
                threat_score * 0.25 +
                request_score * 0.25 +
                financial_score * 0.15 +
                impersonation_score * 0.10 +
                reward_score * 0.05
        )

        # Boost if multiple categories present
        categories_present = sum([
            urgency_score > 0,
            threat_score > 0,
            request_score > 0,
            financial_score > 0
        ])

        if categories_present >= 3:
            base_score = min(1.0, base_score * 1.3)
        elif categories_present >= 2:
            base_score = min(1.0, base_score * 1.15)

        # SPECIAL BOOST: Friend-in-trouble scam pattern (urgency + impersonation + financial)
        # This is a very common scam where someone pretends to be a friend needing money urgently
        friend_scam_detected = False
        if urgency_score > 0 and impersonation_score > 0 and financial_score > 0:
            base_score = min(1.0, base_score + 0.5)  # Strong boost for this pattern
            friend_scam_detected = True

        # Also boost if impersonation + money request
        if impersonation_score > 0 and (financial_score > 0 or "need money" in text_lower or "send money" in text_lower or "gpay" in text_lower):
            base_score = min(1.0, base_score + 0.3)

        # Boost for impersonation + threat (classic scam pattern: "I am from RBI, your account is suspicious")
        if impersonation_score > 0 and threat_score > 0:
            base_score = min(1.0, base_score + 0.45)  # Strong boost

        # Boost for SIM/telecom scam patterns
        if any(kw in text_lower for kw in ["sim", "aadhaar", "trai", "telecom"]) and threat_score > 0:
            base_score = min(1.0, base_score + 0.3)

        # Direct pattern match for common scam openings (high confidence)
        import re
        scam_opening_detected = False
        scam_opening_patterns = [
            r"(i am|this is|calling) from (rbi|sbi|hdfc|icici|axis|bank|police|customs)",
            r"(your|ur) (account|a/c|ac).*(suspicious|blocked|fraud|illegal|freeze)",
            r"(rbi|sbi|bank|police).*(calling|notice|alert)",
        ]
        for pattern in scam_opening_patterns:
            if re.search(pattern, text_lower):
                base_score = min(1.0, base_score + 0.4)
                scam_opening_detected = True
                break

        # CRITICAL: Direct pattern match for "Digital Arrest" scam (very high confidence)
        # This is a common authority impersonation scam in India
        digital_arrest_patterns = [
            r"digital arrest",
            r"cyber arrest",
            r"under arrest",
            r"you are arrested",
            r"(i am|this is).*(inspector|police|cbi|customs|officer)",
            r"(police|cbi|customs|enforcement).*(case|complaint|warrant|fir)",
            r"do not (tell|inform|contact) anyone",
            r"stay on.*(call|video|line)",
        ]
        digital_arrest_detected = False
        for pattern in digital_arrest_patterns:
            if re.search(pattern, text_lower):
                base_score = min(1.0, base_score + 0.75)  # INCREASED: Must exceed 0.7 threshold
                matches.append(RuleMatch(
                    rule_name="digital_arrest_pattern",
                    category="impersonation",
                    score=0.75,
                    matched_text=pattern
                ))
                logger.info(f"Digital arrest pattern detected: {pattern}")
                digital_arrest_detected = True
                break

        # Determine scam type
        scam_type = self._determine_scam_type(all_text)

        # FORCE IMPERSONATION for digital arrest patterns (override other scam types)
        if digital_arrest_detected:
            scam_type = ScamType.IMPERSONATION
            logger.info("Forcing scam_type=IMPERSONATION due to digital arrest pattern")

        # Build evidence list
        evidence = []
        if friend_scam_detected:
            evidence.append("friend_in_trouble_pattern")
        # Check if digital arrest pattern was detected
        if any(m.rule_name == "digital_arrest_pattern" for m in matches):
            evidence.append("digital_arrest_scam_detected")
        if urgency_score > 0.3:
            evidence.append("high_urgency_tactics")
        if threat_score > 0.3:
            evidence.append("threat_intimidation")
        if request_score > 0.3:
            evidence.append("sensitive_info_request")
        if financial_score > 0.3:
            evidence.append("financial_context")
        if impersonation_score > 0.3:
            evidence.append("authority_impersonation")
        if reward_score > 0.3:
            evidence.append("too_good_to_be_true")

        return RuleBasedResult(
            score=min(1.0, base_score),
            scam_type=scam_type,
            matches=matches,
            evidence=evidence,
            keywords_found=keywords_found
        )

    def _check_keywords(
            self,
            text: str,
            keyword_dict: Dict[str, List[str]],
            category: str,
            matches: List[RuleMatch],
            keywords_found: List[str]
    ) -> float:
        """
        Check text against keyword dictionary

        Args:
            text: Text to check
            keyword_dict: Dictionary with priority levels
            category: Category name
            matches: List to append matches
            keywords_found: List to append found keywords

        Returns:
            Score for this category (0-1)
        """
        score = 0.0

        for priority, keywords in keyword_dict.items():
            weight = {"high": 0.4, "medium": 0.25, "low": 0.1}[priority]

            for keyword in keywords:
                if keyword in text:
                    score += weight
                    keywords_found.append(keyword)
                    matches.append(RuleMatch(
                        rule_name=f"{category}_{priority}",
                        category=category,
                        score=weight,
                        matched_text=keyword
                    ))

        return min(1.0, score)

    def _determine_scam_type(self, text: str) -> ScamType:
        """
        Determine the most likely scam type based on keywords

        Args:
            text: Text to analyze

        Returns:
            Most likely ScamType
        """
        type_scores = {}

        for scam_type, keywords in self.scam_type_keywords.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                type_scores[scam_type] = score

        if not type_scores:
            return ScamType.UNKNOWN

        return max(type_scores, key=type_scores.get)
