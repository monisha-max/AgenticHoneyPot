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
        # URGENCY KEYWORDS (create panic)
        # =================================================================
        self.urgency_keywords = {
            "high": [
                "immediately", "urgent", "right now", "today only",
                "last chance", "final warning", "expires today",
                "within 24 hours", "abhi", "turant", "jaldi"
            ],
            "medium": [
                "soon", "quickly", "hurry", "limited time",
                "don't delay", "act fast", "time sensitive"
            ],
            "low": [
                "asap", "priority", "important"
            ]
        }

        # =================================================================
        # THREAT KEYWORDS (create fear)
        # =================================================================
        self.threat_keywords = {
            "high": [
                "blocked", "suspended", "terminated", "cancelled",
                "legal action", "police", "arrest", "court",
                "prosecution", "jail", "fine", "penalty",
                "blacklisted", "band ho jayega", "block ho jayega"
            ],
            "medium": [
                "deactivated", "restricted", "frozen", "hold",
                "investigation", "fraud alert", "security breach"
            ],
            "low": [
                "issue", "problem", "concern", "attention needed"
            ]
        }

        # =================================================================
        # REQUEST KEYWORDS (information seeking)
        # =================================================================
        self.request_keywords = {
            "high": [
                "otp", "pin", "password", "cvv", "mpin",
                "share your", "send your", "provide your",
                "enter your", "verify your", "confirm your",
                "bhejo", "batao", "de do"
            ],
            "medium": [
                "click here", "click link", "tap here",
                "download", "install", "update required",
                "verify", "confirm", "authenticate"
            ],
            "low": [
                "check", "review", "respond"
            ]
        }

        # =================================================================
        # FINANCIAL KEYWORDS (money/banking context)
        # =================================================================
        self.financial_keywords = {
            "high": [
                "bank account", "upi", "paytm", "phonepe", "gpay",
                "google pay", "credit card", "debit card",
                "net banking", "transfer", "payment",
                "account number", "ifsc", "kyc", "pan card", "aadhar"
            ],
            "medium": [
                "transaction", "balance", "amount", "rupees",
                "rs", "₹", "money", "funds", "refund", "cashback"
            ],
            "low": [
                "wallet", "pay", "receive"
            ]
        }

        # =================================================================
        # IMPERSONATION KEYWORDS (fake authority)
        # =================================================================
        self.impersonation_keywords = {
            "high": [
                "sbi", "hdfc", "icici", "axis", "rbi", "reserve bank",
                "income tax", "it department", "customs", "cyber cell",
                "police department", "government", "ministry",
                "amazon", "flipkart", "customer care", "customer support"
            ],
            "medium": [
                "official", "authorized", "representative",
                "department", "head office", "branch manager",
                "executive", "officer"
            ],
            "low": [
                "team", "support", "helpdesk", "service"
            ]
        }

        # =================================================================
        # REWARD/LURE KEYWORDS (too good to be true)
        # =================================================================
        self.reward_keywords = {
            "high": [
                "won", "winner", "lottery", "prize", "jackpot",
                "lakh", "crore", "million", "free gift",
                "congratulations", "selected", "lucky"
            ],
            "medium": [
                "offer", "discount", "cashback", "reward",
                "bonus", "benefit", "guaranteed income"
            ],
            "low": [
                "opportunity", "chance", "special"
            ]
        }

        # =================================================================
        # SCAM TYPE SPECIFIC KEYWORDS
        # =================================================================
        self.scam_type_keywords = {
            ScamType.BANKING_FRAUD: [
                "bank", "account", "netbanking", "statement",
                "debit", "credit", "transaction failed"
            ],
            ScamType.UPI_FRAUD: [
                "upi", "paytm", "phonepe", "gpay", "bhim",
                "upi id", "upi pin", "scan qr", "request money"
            ],
            ScamType.KYC_SCAM: [
                "kyc", "re-kyc", "update kyc", "kyc expired",
                "pan", "aadhar", "verification pending"
            ],
            ScamType.JOB_SCAM: [
                "job offer", "work from home", "earn money",
                "part time", "data entry", "typing job",
                "no experience", "registration fee", "salary"
            ],
            ScamType.LOTTERY_SCAM: [
                "lottery", "lucky draw", "prize money",
                "winner", "jackpot", "claim prize"
            ],
            ScamType.TECH_SUPPORT: [
                "virus", "malware", "hacked", "compromised",
                "remote access", "anydesk", "teamviewer",
                "computer", "laptop", "windows"
            ],
            ScamType.INVESTMENT_FRAUD: [
                "investment", "trading", "stock", "forex",
                "bitcoin", "crypto", "guaranteed returns",
                "double money", "high returns"
            ],
            ScamType.BILL_PAYMENT_SCAM: [
                "bill", "electricity", "overdue", "disconnection",
                "gas", "water", "pending payment", "due amount"
            ],
            ScamType.DELIVERY_SCAM: [
                "delivery", "package", "courier", "parcel",
                "tracking", "customs", "hold", "shipping fee"
            ],
            ScamType.TAX_GST_SCAM: [
                "tax", "gst", "it return", "refund",
                "tax notice", "it department", "compliance"
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

        # Determine scam type
        scam_type = self._determine_scam_type(all_text)

        # Build evidence list
        evidence = []
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
