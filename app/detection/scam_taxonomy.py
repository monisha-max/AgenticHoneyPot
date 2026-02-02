"""
Scam Taxonomy Classifier
Categorizes scams into specific types with detailed characteristics
"""

import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

from app.api.schemas import ScamType, PersonaType

logger = logging.getLogger(__name__)


@dataclass
class ScamCategory:
    """Detailed information about a scam category"""
    scam_type: ScamType
    name: str
    description: str
    common_tactics: List[str]
    keywords: List[str]
    typical_requests: List[str]
    risk_level: str  # low, medium, high, critical
    best_personas: List[PersonaType]
    engagement_strategy: str


class ScamTaxonomy:
    """
    Comprehensive scam classification and mapping
    Maps scam types to appropriate personas and strategies
    """

    def __init__(self):
        self._init_categories()

    def _init_categories(self):
        """Initialize scam category definitions"""

        self.categories: Dict[ScamType, ScamCategory] = {

            ScamType.BANKING_FRAUD: ScamCategory(
                scam_type=ScamType.BANKING_FRAUD,
                name="Banking Fraud",
                description="Fake bank representatives claiming account issues",
                common_tactics=[
                    "Account suspension threats",
                    "Urgent verification requests",
                    "Fake security alerts",
                    "KYC update demands"
                ],
                keywords=[
                    "bank", "account", "suspended", "blocked", "verify",
                    "netbanking", "debit card", "credit card", "transaction",
                    "unauthorized", "security", "alert"
                ],
                typical_requests=[
                    "OTP", "PIN", "Password", "Card details",
                    "Account number", "Net banking credentials"
                ],
                risk_level="critical",
                best_personas=[PersonaType.RAMU_UNCLE, PersonaType.AARTI_HOMEMAKER],
                engagement_strategy="confused_elderly"
            ),

            ScamType.UPI_FRAUD: ScamCategory(
                scam_type=ScamType.UPI_FRAUD,
                name="UPI Payment Fraud",
                description="Tricks to make victims send money via UPI",
                common_tactics=[
                    "Request money instead of send",
                    "QR code manipulation",
                    "Fake refund processes",
                    "Customer care impersonation"
                ],
                keywords=[
                    "upi", "paytm", "phonepe", "gpay", "google pay",
                    "scan", "qr code", "request", "payment", "refund",
                    "upi pin", "upi id"
                ],
                typical_requests=[
                    "UPI PIN", "Scan QR", "Accept request",
                    "UPI ID", "Payment confirmation"
                ],
                risk_level="critical",
                best_personas=[PersonaType.AARTI_HOMEMAKER, PersonaType.SUNITA_SHOP],
                engagement_strategy="confused_about_upi"
            ),

            ScamType.KYC_SCAM: ScamCategory(
                scam_type=ScamType.KYC_SCAM,
                name="KYC Update Scam",
                description="Fake KYC verification to steal personal data",
                common_tactics=[
                    "KYC expiry warnings",
                    "Service suspension threats",
                    "Link to fake forms",
                    "Aadhar/PAN collection"
                ],
                keywords=[
                    "kyc", "re-kyc", "update", "expired", "verify",
                    "aadhar", "pan", "documents", "link", "form"
                ],
                typical_requests=[
                    "Aadhar number", "PAN card", "Personal details",
                    "Document photos", "OTP for verification"
                ],
                risk_level="high",
                best_personas=[PersonaType.RAMU_UNCLE, PersonaType.SUNITA_SHOP],
                engagement_strategy="document_confused"
            ),

            ScamType.JOB_SCAM: ScamCategory(
                scam_type=ScamType.JOB_SCAM,
                name="Job/Work From Home Scam",
                description="Fake job offers requiring upfront payment",
                common_tactics=[
                    "High salary promises",
                    "Work from home opportunity",
                    "Registration fee demands",
                    "Training fee requests"
                ],
                keywords=[
                    "job", "offer", "work from home", "wfh", "salary",
                    "earn", "income", "part time", "data entry",
                    "typing", "registration", "training"
                ],
                typical_requests=[
                    "Registration fee", "Training fee", "Kit charges",
                    "Processing fee", "Security deposit"
                ],
                risk_level="medium",
                best_personas=[PersonaType.ANANYA_STUDENT, PersonaType.VIKRAM_IT],
                engagement_strategy="interested_but_cautious"
            ),

            ScamType.LOTTERY_SCAM: ScamCategory(
                scam_type=ScamType.LOTTERY_SCAM,
                name="Lottery/Prize Scam",
                description="Fake lottery winnings requiring processing fees",
                common_tactics=[
                    "Unsolicited winning notification",
                    "Processing fee demands",
                    "Tax payment requests",
                    "Urgency to claim prize"
                ],
                keywords=[
                    "lottery", "winner", "won", "prize", "jackpot",
                    "lucky", "draw", "claim", "congratulations",
                    "selected", "lakh", "crore"
                ],
                typical_requests=[
                    "Processing fee", "Tax amount", "Bank details",
                    "ID proof", "Claim form fee"
                ],
                risk_level="medium",
                best_personas=[PersonaType.ANANYA_STUDENT, PersonaType.RAMU_UNCLE],
                engagement_strategy="excited_but_verify"
            ),

            ScamType.TECH_SUPPORT: ScamCategory(
                scam_type=ScamType.TECH_SUPPORT,
                name="Tech Support Scam",
                description="Fake tech support to gain remote access",
                common_tactics=[
                    "Virus/malware alerts",
                    "Remote access requests",
                    "Fake error messages",
                    "Microsoft/Google impersonation"
                ],
                keywords=[
                    "virus", "malware", "hacked", "computer", "laptop",
                    "windows", "microsoft", "remote", "anydesk",
                    "teamviewer", "support", "error", "fix"
                ],
                typical_requests=[
                    "Remote access", "Software download",
                    "Payment for fix", "Personal information"
                ],
                risk_level="high",
                best_personas=[PersonaType.RAMU_UNCLE, PersonaType.AARTI_HOMEMAKER],
                engagement_strategy="tech_confused"
            ),

            ScamType.INVESTMENT_FRAUD: ScamCategory(
                scam_type=ScamType.INVESTMENT_FRAUD,
                name="Investment/Trading Fraud",
                description="Fake investment schemes with guaranteed returns",
                common_tactics=[
                    "Guaranteed high returns",
                    "Ponzi scheme structure",
                    "Crypto investment promises",
                    "Trading tips scam"
                ],
                keywords=[
                    "investment", "trading", "stock", "forex", "crypto",
                    "bitcoin", "returns", "profit", "guaranteed",
                    "double", "multiply", "scheme"
                ],
                typical_requests=[
                    "Investment amount", "Trading account details",
                    "Crypto wallet", "Initial deposit"
                ],
                risk_level="high",
                best_personas=[PersonaType.VIKRAM_IT, PersonaType.ANANYA_STUDENT],
                engagement_strategy="interested_investor"
            ),

            ScamType.BILL_PAYMENT_SCAM: ScamCategory(
                scam_type=ScamType.BILL_PAYMENT_SCAM,
                name="Bill Payment Scam",
                description="Fake utility bill payment demands",
                common_tactics=[
                    "Service disconnection threats",
                    "Overdue bill claims",
                    "Immediate payment pressure",
                    "Fake bill details"
                ],
                keywords=[
                    "bill", "electricity", "water", "gas", "overdue",
                    "pending", "disconnection", "payment", "due",
                    "meter", "connection"
                ],
                typical_requests=[
                    "Immediate payment", "UPI transfer",
                    "Bill payment link", "Consumer number"
                ],
                risk_level="medium",
                best_personas=[PersonaType.AARTI_HOMEMAKER, PersonaType.RAMU_UNCLE],
                engagement_strategy="worried_about_service"
            ),

            ScamType.DELIVERY_SCAM: ScamCategory(
                scam_type=ScamType.DELIVERY_SCAM,
                name="Delivery/Courier Scam",
                description="Fake delivery issues requiring payment",
                common_tactics=[
                    "Package held claims",
                    "Customs fee demands",
                    "Delivery rescheduling fees",
                    "Address verification scam"
                ],
                keywords=[
                    "delivery", "courier", "package", "parcel", "held",
                    "customs", "shipping", "tracking", "amazon",
                    "flipkart", "order"
                ],
                typical_requests=[
                    "Customs fee", "Delivery charges",
                    "Address confirmation", "OTP for delivery"
                ],
                risk_level="medium",
                best_personas=[PersonaType.AARTI_HOMEMAKER, PersonaType.ANANYA_STUDENT],
                engagement_strategy="expecting_delivery"
            ),

            ScamType.TAX_GST_SCAM: ScamCategory(
                scam_type=ScamType.TAX_GST_SCAM,
                name="Tax/GST Scam",
                description="Fake tax notices and GST compliance threats",
                common_tactics=[
                    "Tax evasion accusations",
                    "GST non-compliance notices",
                    "Penalty threats",
                    "Legal action warnings"
                ],
                keywords=[
                    "tax", "gst", "income tax", "it department",
                    "notice", "penalty", "compliance", "return",
                    "refund", "evasion", "raid"
                ],
                typical_requests=[
                    "Penalty payment", "Tax settlement",
                    "Document submission", "Account details"
                ],
                risk_level="high",
                best_personas=[PersonaType.SUNITA_SHOP, PersonaType.VIKRAM_IT],
                engagement_strategy="tax_worried"
            ),

            ScamType.CREDIT_CARD_FRAUD: ScamCategory(
                scam_type=ScamType.CREDIT_CARD_FRAUD,
                name="Credit Card Fraud",
                description="Fake credit card issues or offers",
                common_tactics=[
                    "Unauthorized transaction alerts",
                    "Credit limit increase offers",
                    "Card blocking threats",
                    "Reward point scams"
                ],
                keywords=[
                    "credit card", "card", "transaction", "unauthorized",
                    "cvv", "expiry", "limit", "reward", "points",
                    "statement", "payment due"
                ],
                typical_requests=[
                    "CVV", "Card number", "Expiry date",
                    "OTP", "Card verification"
                ],
                risk_level="critical",
                best_personas=[PersonaType.VIKRAM_IT, PersonaType.AARTI_HOMEMAKER],
                engagement_strategy="card_concerned"
            ),

            ScamType.CRYPTO_SCAM: ScamCategory(
                scam_type=ScamType.CRYPTO_SCAM,
                name="Cryptocurrency Scam",
                description="Fake crypto investment or giveaway scams",
                common_tactics=[
                    "Crypto giveaway scams",
                    "Fake exchange platforms",
                    "Investment doubling schemes",
                    "Celebrity impersonation"
                ],
                keywords=[
                    "crypto", "bitcoin", "ethereum", "wallet",
                    "blockchain", "coin", "token", "nft",
                    "giveaway", "airdrop", "exchange"
                ],
                typical_requests=[
                    "Wallet address", "Seed phrase",
                    "Initial deposit", "Verification payment"
                ],
                risk_level="high",
                best_personas=[PersonaType.VIKRAM_IT, PersonaType.ANANYA_STUDENT],
                engagement_strategy="crypto_curious"
            ),

            ScamType.IMPERSONATION: ScamCategory(
                scam_type=ScamType.IMPERSONATION,
                name="Authority Impersonation",
                description="Impersonating officials to extract money/info",
                common_tactics=[
                    "Police/CBI impersonation",
                    "Court notice threats",
                    "Arrest warrants",
                    "Digital arrest scam"
                ],
                keywords=[
                    "police", "cbi", "customs", "court", "arrest",
                    "warrant", "investigation", "case", "crime",
                    "digital arrest", "video call"
                ],
                typical_requests=[
                    "Fine payment", "Settlement amount",
                    "Personal details", "Bank statements"
                ],
                risk_level="critical",
                best_personas=[PersonaType.RAMU_UNCLE, PersonaType.AARTI_HOMEMAKER],
                engagement_strategy="frightened_citizen"
            ),

            ScamType.UNKNOWN: ScamCategory(
                scam_type=ScamType.UNKNOWN,
                name="Unknown/Other Scam",
                description="Unclassified suspicious activity",
                common_tactics=["Various"],
                keywords=[],
                typical_requests=[],
                risk_level="medium",
                best_personas=[PersonaType.RAMU_UNCLE],
                engagement_strategy="general_confused"
            )
        }

    def classify(self, text: str, keywords_found: List[str] = None) -> Tuple[ScamType, float]:
        """
        Classify text into a scam type

        Args:
            text: Message text
            keywords_found: Pre-identified keywords

        Returns:
            Tuple of (ScamType, confidence)
        """
        text_lower = text.lower()
        scores = {}

        for scam_type, category in self.categories.items():
            if scam_type == ScamType.UNKNOWN:
                continue

            score = 0
            for keyword in category.keywords:
                if keyword in text_lower:
                    score += 1

            if keywords_found:
                for kw in keywords_found:
                    if kw in category.keywords:
                        score += 0.5

            if score > 0:
                # Normalize by keyword count
                scores[scam_type] = score / len(category.keywords)

        if not scores:
            return ScamType.UNKNOWN, 0.0

        best_type = max(scores, key=scores.get)
        return best_type, min(1.0, scores[best_type])

    def get_category(self, scam_type: ScamType) -> ScamCategory:
        """Get category details for a scam type"""
        return self.categories.get(scam_type, self.categories[ScamType.UNKNOWN])

    def get_best_persona(self, scam_type: ScamType) -> PersonaType:
        """Get the best persona for engaging with a scam type"""
        category = self.get_category(scam_type)
        if category.best_personas:
            return category.best_personas[0]
        return PersonaType.RAMU_UNCLE

    def get_engagement_strategy(self, scam_type: ScamType) -> str:
        """Get the recommended engagement strategy"""
        category = self.get_category(scam_type)
        return category.engagement_strategy

    def get_risk_level(self, scam_type: ScamType) -> str:
        """Get risk level for a scam type"""
        category = self.get_category(scam_type)
        return category.risk_level
