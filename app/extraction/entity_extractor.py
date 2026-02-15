"""
Entity Extractor for Intelligence Extraction
Extracts structured intelligence from scam conversations
"""

import re
import logging
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field
from urllib.parse import urlparse

from app.api.schemas import ExtractedIntelligence

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result from entity extraction"""
    entity_type: str
    value: str
    confidence: float
    source_text: str
    is_validated: bool = False


class EntityExtractor:
    """
    Extracts intelligence entities from text using regex and validation
    """

    def __init__(self):
        self._compile_patterns()
        self._init_validators()

    def _compile_patterns(self):
        """Compile regex patterns for entity extraction"""

        # UPI ID pattern - enhanced to catch more variants
        self.upi_pattern = re.compile(
            r'([a-zA-Z0-9._-]+@[a-zA-Z]{2,15})',
            re.IGNORECASE
        )

        # Indian phone number patterns - enhanced with strict boundaries
        self.phone_patterns = [
            # With +91 prefix (high confidence)
            re.compile(r'\+91[\s.-]?([1-9]\d{9})\b'),
            # With context keywords (high confidence)
            re.compile(r'(?:call|contact|phone|mobile|whatsapp|wa|number|no\.|num)[\s:.-]*(?:\+?91)?[\s.-]?([1-9]\d{9})\b', re.IGNORECASE),
            # Standalone 10-digit with strict word boundary (must not be part of longer number)
            re.compile(r'(?:^|[^\d])([1-9]\d{9})(?:[^\d]|$)'),
            # With 91 prefix (no plus)
            re.compile(r'\b91[\s.-]?([1-9]\d{9})\b'),
            # Split format like 98765-43210
            re.compile(r'\b([1-9]\d{4})[\s.-](\d{5})\b'),
        ]

        # Bank account number - with context
        self.bank_account_pattern = re.compile(r'\b(\d{9,18})\b')
        self.bank_account_context_pattern = re.compile(
            r'(?:account|a/c|ac|acct)[\s.#:-]*(?:no|num|number)?[\s.#:-]*(\d{9,18})',
            re.IGNORECASE
        )

        # IFSC code - enhanced
        self.ifsc_pattern = re.compile(r'\b([A-Z]{4}0[A-Z0-9]{6})\b')
        self.ifsc_context_pattern = re.compile(
            r'(?:ifsc|branch)[\s.#:-]*(?:code)?[\s.#:-]*([A-Z]{4}0[A-Z0-9]{6})',
            re.IGNORECASE
        )

        # URL pattern - enhanced for short URLs and suspicious domains
        self.url_pattern = re.compile(
            r'(https?://[^\s<>\"\']+|www\.[^\s<>\"\']+|bit\.ly/[^\s]+|tinyurl\.com/[^\s]+|t\.co/[^\s]+)',
            re.IGNORECASE
        )

        # Email pattern
        self.email_pattern = re.compile(
            r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
            re.IGNORECASE
        )

        # PAN Card
        self.pan_pattern = re.compile(r'\b([A-Z]{5}\d{4}[A-Z])\b')

        # Aadhar (12 digits with optional spaces)
        self.aadhar_pattern = re.compile(
            r'\b(\d{4}[\s-]?\d{4}[\s-]?\d{4})\b'
        )

        # Amount pattern - enhanced
        self.amount_pattern = re.compile(
            r'(?:rs\.?|₹|inr|rupees?)\s*([\d,]+(?:\.\d{2})?)|'
            r'([\d,]+(?:\.\d{2})?)\s*(?:rs\.?|₹|rupees?)|'
            r'(?:amount|pay|transfer|send)\s*(?:of)?\s*(?:rs\.?|₹)?\s*([\d,]+)',
            re.IGNORECASE
        )

        # Name patterns (for scammer names) - enhanced
        self.name_patterns = [
            re.compile(r"(?:my name is|i am|i'm|im|this is|mera naam|mai|main)\s+([A-Za-z]{2,}(?:\s+[A-Za-z]{2,})?)", re.IGNORECASE),
            re.compile(r'(?:Mr\.|Mrs\.|Ms\.|Shri|Smt\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', re.IGNORECASE),
            re.compile(r'(?:contact|speak to|ask for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', re.IGNORECASE),
            re.compile(r'(?:from|officer|manager|executive)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', re.IGNORECASE),
        ]

        # Reference/Ticket number - enhanced
        self.reference_pattern = re.compile(
            r'(?:ref|reference|ticket|complaint|case|transaction|txn|order)[\s.#:-]*(?:no|num|number|id)?[\s.#:-]*([A-Z0-9]{6,20})',
            re.IGNORECASE
        )

        # WhatsApp/Telegram specific patterns
        self.messenger_pattern = re.compile(
            r'(?:whatsapp|wa|telegram|tg)[\s:.-]*(?:\+?91)?[\s.-]?([1-9]\d{9})',
            re.IGNORECASE
        )

        # Google Pay/PhonePe specific patterns
        self.payment_app_pattern = re.compile(
            r'(?:gpay|google pay|phonepe|paytm|bhim)[\s:.-]*([a-zA-Z0-9._-]+@[a-zA-Z]{2,15})',
            re.IGNORECASE
        )

    def _init_validators(self):
        """Initialize validation functions"""

        # Valid UPI providers - comprehensive list
        self.valid_upi_providers = {
            # Major banks
            'okicici', 'okhdfcbank', 'okaxis', 'oksbi',
            'ybl', 'paytm', 'apl', 'upi', 'ibl',
            'axisbank', 'sbi', 'icici', 'hdfcbank',
            'kotak', 'indus', 'federal', 'rbl', 'fbl',
            'barodampay', 'aubank', 'citi', 'hsbc',
            # WhatsApp payments
            'waicici', 'wahdfcbank', 'waaxis', 'wasbi',
            # Payment apps
            'pingpay', 'gpay', 'googlepay', 'amazonpay',
            'phonepe', 'postbank', 'pnb', 'bob', 'canara',
            # Other banks
            'bandhan', 'idbi', 'idfc', 'yes', 'ubi',
            'syndicate', 'allbank', 'centralbank',
            'indianbank', 'iob', 'pnb', 'dbs', 'sc',
            # Fintech
            'freecharge', 'mobikwik', 'slice', 'jupiter',
            'fi', 'niyo', 'razorpay', 'cashfree',
            # Common short forms
            'axis', 'hdfc', 'icic', 'pnb', 'bob', 'boi',
            'canara', 'union', 'baroda', 'syndicate'
        }

        # Known suspicious domains
        self.suspicious_domains = {
            '.tk', '.ml', '.ga', '.cf', '.gq',
            'bit.ly', 'tinyurl', 'shorturl'
        }

    def extract_all(
            self,
            text: str,
            conversation_history: List[str] = None
    ) -> ExtractedIntelligence:
        """
        Extract all intelligence from text

        Args:
            text: Current message text
            conversation_history: Previous messages

        Returns:
            ExtractedIntelligence with all extracted entities
        """
        # Combine all text for extraction
        all_text = text
        if conversation_history:
            all_text += " " + " ".join(conversation_history)

        intelligence = ExtractedIntelligence()

        # Extract each entity type
        intelligence.upi_ids = self._extract_upi_ids(all_text)
        intelligence.phone_numbers = self._extract_phone_numbers(all_text)
        intelligence.bank_accounts = self._extract_bank_accounts(all_text, intelligence.phone_numbers)
        intelligence.ifsc_codes = self._extract_ifsc_codes(all_text)
        intelligence.phishing_links = self._extract_urls(all_text)
        # Email extraction disabled — low value, causes UPI/email confusion
        intelligence.email_addresses = self._extract_emails(all_text)
        # Name extraction is handled by LLM enricher (conversation-aware)
        intelligence.scammer_names = self._extract_names(all_text)
        intelligence.fake_references = self._extract_references(all_text)
        intelligence.suspicious_keywords = self._extract_keywords(all_text)

        return intelligence

    def _extract_upi_ids(self, text: str) -> List[str]:
        """Extract and validate UPI IDs"""
        matches = self.upi_pattern.findall(text)

        # Also check payment app specific patterns
        payment_matches = self.payment_app_pattern.findall(text)
        matches.extend(payment_matches)

        validated = []

        for upi in matches:
            upi_lower = upi.lower()

            # Skip email-like addresses
            email_domains = ['gmail', 'yahoo', 'outlook', 'hotmail', 'mail', 'email', 'rediffmail', 'proton']
            if any(domain in upi_lower for domain in email_domains):
                continue

            # Skip if it looks like a website domain
            if any(ext in upi_lower for ext in ['.com', '.in', '.org', '.net', '.co']):
                continue

            # Check for valid UPI provider
            if '@' in upi:
                provider = upi.split('@')[1].lower()
                # Accept if it's a known provider or short provider name (likely UPI)
                if provider in self.valid_upi_providers or (len(provider) <= 12 and provider.isalpha()):
                    validated.append(upi.lower())

        return list(set(validated))

    def _extract_phone_numbers(self, text: str) -> List[str]:
        """Extract and format phone numbers"""
        numbers = set()

        # Standard patterns
        for pattern in self.phone_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    # Handle split patterns
                    number = ''.join(match)
                else:
                    number = match

                # Clean and format
                clean_number = re.sub(r'[\s.-]', '', number)
                if len(clean_number) == 10 and clean_number[0] in '123456789':
                    numbers.add(f"+91{clean_number}")
                elif len(clean_number) == 12 and clean_number.startswith('91') and clean_number[2] in '123456789':
                    numbers.add(f"+{clean_number}")

        # WhatsApp/Telegram specific patterns
        messenger_matches = self.messenger_pattern.findall(text)
        for match in messenger_matches:
            clean_number = re.sub(r'[\s.-]', '', match)
            if len(clean_number) == 10 and clean_number[0] in '123456789':
                numbers.add(f"+91{clean_number}")

        return list(numbers)

    def _extract_bank_accounts(
            self,
            text: str,
            phone_numbers: List[str]
    ) -> List[str]:
        """Extract bank account numbers (filtering out phone numbers)"""
        # Get matches from both patterns
        matches = self.bank_account_pattern.findall(text)
        context_matches = self.bank_account_context_pattern.findall(text)
        all_matches = list(set(matches + context_matches))

        validated = []

        # Convert phone numbers to set for O(1) lookup
        phone_set = set()
        for phone in phone_numbers:
            clean = re.sub(r'[\s.+-]', '', phone)[-10:]
            phone_set.add(clean)

        # Also exclude Aadhaar-like numbers (12 digits starting with certain patterns)
        for account in all_matches:
            clean_account = re.sub(r'[\s-]', '', account)

            # Skip if it's a phone number
            if clean_account in phone_set or clean_account[-10:] in phone_set:
                continue

            # Skip 10 digit numbers (likely phone numbers)
            if len(clean_account) == 10:
                continue

            # Skip 12 digit numbers that look like Aadhaar
            if len(clean_account) == 12:
                continue

            # Validate length (Indian bank accounts typically 9-18 digits, but not 10 or 12)
            if 9 <= len(clean_account) <= 18:
                validated.append(clean_account)

        return list(set(validated))

    def _extract_ifsc_codes(self, text: str) -> List[str]:
        """Extract IFSC codes"""
        matches = self.ifsc_pattern.findall(text)
        context_matches = self.ifsc_context_pattern.findall(text)
        all_matches = list(set(matches + context_matches))
        # Validate IFSC format (4 letters + 0 + 6 alphanumeric)
        validated = [ifsc.upper() for ifsc in all_matches if len(ifsc) == 11]
        return list(set(validated))

    def _extract_urls(self, text: str) -> List[str]:
        """Extract and categorize URLs"""
        matches = self.url_pattern.findall(text)
        urls = []

        for url in matches:
            # Add http if missing
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url

            urls.append(url)

        return list(set(urls))

    def _extract_emails(self, text: str) -> List[str]:
        """Extract email addresses"""
        matches = self.email_pattern.findall(text)

        # Filter out UPI IDs (which look like emails)
        emails = []
        for email in matches:
            if '@' in email:
                domain = email.split('@')[1].lower()
                # Common email domains
                if any(d in domain for d in ['gmail', 'yahoo', 'outlook', 'hotmail', 'mail', '.com', '.in', '.org']):
                    emails.append(email.lower())

        return list(set(emails))

    def _extract_names(self, text: str) -> List[str]:
        """Name extraction intentionally disabled (LLM enricher handles this)."""
        return []

    def _extract_references(self, text: str) -> List[str]:
        """Extract reference/ticket numbers"""
        matches = self.reference_pattern.findall(text)
        return list(set(matches))

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract suspicious keywords - enhanced categorization"""
        text_lower = text.lower()

        suspicious_keywords = {
            # Urgency - high signal
            "urgent": 2, "immediately": 2, "expire": 2, "last chance": 2,
            "jaldi": 1, "abhi": 1, "turant": 2, "deadline": 2,

            # Threats - high signal
            "block": 2, "suspend": 2, "terminate": 2, "legal action": 3,
            "police": 2, "arrest": 3, "fine": 2, "penalty": 2,
            "court": 2, "fir": 3, "complaint": 1,

            # Sensitive data requests - very high signal
            "otp": 3, "pin": 2, "password": 3, "cvv": 3,
            "aadhaar": 2, "pan": 1, "card number": 3,

            # Action requests
            "verify": 1, "confirm": 1, "click": 2, "link": 1,
            "download": 2, "install": 2, "update": 1,

            # Financial terms
            "transfer": 1, "payment": 1, "fee": 2, "charge": 1,
            "refund": 2, "cashback": 2, "prize": 3, "won": 3,
            "lottery": 3, "jackpot": 3, "winner": 2,

            # Authority impersonation
            "bank": 1, "government": 2, "official": 1, "department": 1,
            "rbi": 2, "customs": 2, "income tax": 2, "gst": 1,
            "ministry": 2, "cyber cell": 2,

            # Job scam keywords
            "job offer": 2, "shortlisted": 2, "selected": 1,
            "registration fee": 3, "work from home": 2, "part time job": 2,

            # KYC scam keywords
            "kyc": 2, "kyc update": 3, "kyc expire": 3,
            "sim block": 3, "sim deactivate": 3,

            # Remote access - very high signal
            "teamviewer": 3, "anydesk": 3, "remote access": 3,

            # Cryptocurrency
            "bitcoin": 2, "crypto": 2, "invest": 1, "guaranteed returns": 3,
            "double your money": 3, "trading": 1
        }

        found = []
        for kw, weight in suspicious_keywords.items():
            if kw in text_lower:
                # Add keyword with weight indicator for higher signal keywords
                if weight >= 2:
                    found.append(f"{kw}*")  # Mark high-signal keywords
                else:
                    found.append(kw)

        return found

    def extract_incremental(
            self,
            new_text: str,
            existing_intelligence: ExtractedIntelligence
    ) -> ExtractedIntelligence:
        """
        Extract from new text and merge with existing intelligence

        Args:
            new_text: New message text
            existing_intelligence: Previously extracted intelligence

        Returns:
            Merged ExtractedIntelligence
        """
        new_intel = self.extract_all(new_text)

        # Merge (deduplicate)
        merged = ExtractedIntelligence(
            bank_accounts=list(set(existing_intelligence.bank_accounts + new_intel.bank_accounts)),
            upi_ids=list(set(existing_intelligence.upi_ids + new_intel.upi_ids)),
            phishing_links=list(set(existing_intelligence.phishing_links + new_intel.phishing_links)),
            phone_numbers=list(set(existing_intelligence.phone_numbers + new_intel.phone_numbers)),
            suspicious_keywords=list(set(existing_intelligence.suspicious_keywords + new_intel.suspicious_keywords)),
            email_addresses=list(set(existing_intelligence.email_addresses + new_intel.email_addresses),  # Email extraction disabled
            ifsc_codes=list(set(existing_intelligence.ifsc_codes + new_intel.ifsc_codes)),
            scammer_names=list(set(existing_intelligence.scammer_names + new_intel.scammer_names)),
            fake_references=list(set(existing_intelligence.fake_references + new_intel.fake_references))
        ))

        return merged


class IntelligenceAggregator:
    """
    Aggregates and validates intelligence across conversation
    """

    def __init__(self):
        self.extractor = EntityExtractor()

    def process_message(
            self,
            message: str,
            existing_intelligence: ExtractedIntelligence
    ) -> Tuple[ExtractedIntelligence, List[str]]:
        """
        Process new message and update intelligence

        Args:
            message: New message text
            existing_intelligence: Current accumulated intelligence

        Returns:
            Tuple of (updated intelligence, list of new items found)
        """
        new_intel = self.extractor.extract_incremental(message, existing_intelligence)

        # Find what's new
        new_items = []

        if set(new_intel.upi_ids) - set(existing_intelligence.upi_ids):
            new_items.append("upi_id")
        if set(new_intel.phone_numbers) - set(existing_intelligence.phone_numbers):
            new_items.append("phone_number")
        if set(new_intel.bank_accounts) - set(existing_intelligence.bank_accounts):
            new_items.append("bank_account")
        if set(new_intel.phishing_links) - set(existing_intelligence.phishing_links):
            new_items.append("url")
        if set(new_intel.suspicious_keywords) - set(existing_intelligence.suspicious_keywords):
            new_items.append("keywords")
        if set(new_intel.scammer_names) - set(existing_intelligence.scammer_names):
            new_items.append("name")
        # Email extraction disabled
        if set(new_intel.email_addresses) - set(existing_intelligence.email_addresses):
            new_items.append("email")

        return new_intel, new_items

    def generate_summary(self, intelligence: ExtractedIntelligence) -> str:
        """Generate human-readable summary of intelligence"""

        parts = []

        if intelligence.upi_ids:
            parts.append(f"UPI IDs: {', '.join(intelligence.upi_ids)}")
        if intelligence.phone_numbers:
            parts.append(f"Phone numbers: {', '.join(intelligence.phone_numbers)}")
        if intelligence.bank_accounts:
            parts.append(f"Bank accounts: {', '.join(intelligence.bank_accounts)}")
        if intelligence.phishing_links:
            parts.append(f"Suspicious links: {len(intelligence.phishing_links)} found")
        if intelligence.scammer_names:
            parts.append(f"Names mentioned: {', '.join(intelligence.scammer_names)}")
        if intelligence.email_addresses:
            parts.append(f"Email addresses: {', '.join(intelligence.email_addresses)}")

        return "; ".join(parts) if parts else "No significant intelligence extracted yet"

    def get_intelligence_score(self, intelligence: ExtractedIntelligence) -> float:
        """
        Calculate a score for intelligence quality

        Args:
            intelligence: Extracted intelligence

        Returns:
            Score between 0 and 1
        """
        score = 0.0

        # High value items
        if intelligence.upi_ids:
            score += 0.20
        if intelligence.phone_numbers:
            score += 0.20
        if intelligence.bank_accounts:
            score += 0.20

        # Medium value items
        if intelligence.phishing_links:
            score += 0.10
        if intelligence.scammer_names:
            score += 0.10  # Bumped since email removed
        if intelligence.email_addresses:
            score += 0.10

        # Low value items
        if len(intelligence.suspicious_keywords) >= 3:
            score += 0.05
        if intelligence.fake_references:
            score += 0.05

        return min(1.0, score)
