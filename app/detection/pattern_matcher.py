"""
Pattern Matcher for Scam Detection
Uses regex patterns to identify suspicious entities and formats
"""

import re
import logging
from typing import List, Dict, Set
from dataclasses import dataclass, field
from urllib.parse import urlparse

from app.api.schemas import ScamType

logger = logging.getLogger(__name__)


@dataclass
class PatternMatch:
    """Represents a pattern match"""
    pattern_type: str
    value: str
    is_suspicious: bool
    reason: str = ""


@dataclass
class PatternMatcherResult:
    """Result from pattern matching"""
    score: float
    matches: List[PatternMatch] = field(default_factory=list)
    entities_found: Dict[str, List[str]] = field(default_factory=dict)
    suspicious_count: int = 0


class PatternMatcher:
    """
    Pattern-based detection using regex
    Identifies suspicious entities like fake UPIs, phishing URLs, etc.
    Weight: 0.15 in ensemble
    """

    def __init__(self, weight: float = 0.15):
        self.weight = weight
        self._compile_patterns()
        self._init_blacklists()

    def _compile_patterns(self):
        """Compile regex patterns for efficiency"""

        # UPI ID pattern (name@provider)
        self.upi_pattern = re.compile(
            r'[a-zA-Z0-9._-]+@[a-zA-Z]{2,}',
            re.IGNORECASE
        )

        # Indian phone number patterns
        self.phone_patterns = [
            re.compile(r'\+91[\s-]?[6-9]\d{9}'),  # +91 format
            re.compile(r'0?[6-9]\d{9}'),  # 10 digit
            re.compile(r'[6-9]\d{4}[\s-]?\d{5}'),  # Split format
        ]

        # Bank account number (9-18 digits)
        self.bank_account_pattern = re.compile(r'\b\d{9,18}\b')

        # IFSC code pattern
        self.ifsc_pattern = re.compile(r'\b[A-Z]{4}0[A-Z0-9]{6}\b')

        # URL patterns
        self.url_pattern = re.compile(
            r'https?://[^\s<>\"\']+|www\.[^\s<>\"\']+',
            re.IGNORECASE
        )

        # Shortened URL services
        self.short_url_pattern = re.compile(
            r'(bit\.ly|tinyurl|goo\.gl|t\.co|is\.gd|cli\.gs|'
            r'pic\.gd|DwarfURL|ow\.ly|yfrog|migre\.me|ff\.im|'
            r'tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl|'
            r'short\.to|BudURL|ping\.fm|post\.ly|Just\.as|bkite|'
            r'snipr|fic\.kr|loopt|doiop|short\.ie|kl\.am|wp\.me|'
            r'rubyurl|om\.ly|to\.ly|bit\.do|cutt\.ly|rb\.gy)[\/\w]*',
            re.IGNORECASE
        )

        # Email pattern
        self.email_pattern = re.compile(
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            re.IGNORECASE
        )

        # PAN Card pattern
        self.pan_pattern = re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b')

        # Aadhar pattern (12 digits, usually in groups)
        self.aadhar_pattern = re.compile(
            r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
        )

        # OTP pattern (4-8 digits)
        self.otp_pattern = re.compile(r'\b\d{4,8}\b')

        # Amount pattern (Indian currency)
        self.amount_pattern = re.compile(
            r'(?:rs\.?|₹|inr)\s*[\d,]+(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:rs\.?|₹|rupees?)',
            re.IGNORECASE
        )

    def _init_blacklists(self):
        """Initialize blacklists for suspicious entities"""

        # Known legitimate UPI providers
        self.legitimate_upi_providers = {
            'okicici', 'okhdfcbank', 'okaxis', 'oksbi',
            'ybl', 'paytm', 'apl', 'upi', 'ibl',
            'axisbank', 'sbi', 'icici', 'hdfcbank',
            'kotak', 'indus', 'federal', 'rbl'
        }

        # Suspicious UPI patterns (personal accounts disguised as official)
        self.suspicious_upi_patterns = [
            r'customer.*care',
            r'support',
            r'helpdesk',
            r'refund',
            r'verify',
            r'official',
            r'service',
            r'help',
            r'secure'
        ]

        # Known phishing domain patterns
        self.phishing_domain_patterns = [
            r'.*-secure\..*',
            r'.*\.tk$',
            r'.*\.ml$',
            r'.*\.ga$',
            r'.*\.cf$',
            r'.*login.*\..*',
            r'.*verify.*\..*',
            r'.*update.*\..*',
            r'.*account.*\.(?!com|in|org)',
            r'.*bank.*\.(?!com|in|org|co\.in)',
        ]

        # Legitimate domains (whitelist)
        self.legitimate_domains = {
            'sbi.co.in', 'onlinesbi.com',
            'hdfcbank.com', 'netbanking.hdfcbank.com',
            'icicibank.com', 'axisbank.com',
            'paytm.com', 'phonepe.com',
            'gpay.google.com', 'pay.google.com',
            'amazon.in', 'flipkart.com',
            'gov.in', 'incometax.gov.in'
        }

    def match(self, text: str, history: List[str] = None) -> PatternMatcherResult:
        """
        Find patterns in text and assess suspiciousness

        Args:
            text: Message text to analyze
            history: Previous messages for context

        Returns:
            PatternMatcherResult with findings
        """
        all_text = text
        if history:
            all_text += " " + " ".join(history)

        matches = []
        entities = {
            "upi_ids": [],
            "phone_numbers": [],
            "urls": [],
            "bank_accounts": [],
            "emails": [],
            "amounts": []
        }
        suspicious_count = 0

        # Check UPI IDs
        upi_matches = self.upi_pattern.findall(all_text)
        for upi in upi_matches:
            is_suspicious, reason = self._check_upi_suspicious(upi)
            matches.append(PatternMatch(
                pattern_type="upi_id",
                value=upi,
                is_suspicious=is_suspicious,
                reason=reason
            ))
            entities["upi_ids"].append(upi)
            if is_suspicious:
                suspicious_count += 1

        # Check phone numbers
        for pattern in self.phone_patterns:
            phone_matches = pattern.findall(all_text)
            for phone in phone_matches:
                # Clean the phone number
                clean_phone = re.sub(r'[\s-]', '', phone)
                if clean_phone not in entities["phone_numbers"]:
                    matches.append(PatternMatch(
                        pattern_type="phone_number",
                        value=clean_phone,
                        is_suspicious=True,  # Unsolicited phone numbers are suspicious
                        reason="Unsolicited contact number"
                    ))
                    entities["phone_numbers"].append(clean_phone)
                    suspicious_count += 1

        # Check URLs
        url_matches = self.url_pattern.findall(all_text)
        for url in url_matches:
            is_suspicious, reason = self._check_url_suspicious(url)
            matches.append(PatternMatch(
                pattern_type="url",
                value=url,
                is_suspicious=is_suspicious,
                reason=reason
            ))
            entities["urls"].append(url)
            if is_suspicious:
                suspicious_count += 2  # URLs are high risk

        # Check shortened URLs (always suspicious in unsolicited messages)
        short_url_matches = self.short_url_pattern.findall(all_text)
        for short_url in short_url_matches:
            matches.append(PatternMatch(
                pattern_type="shortened_url",
                value=short_url,
                is_suspicious=True,
                reason="Shortened URL hides destination"
            ))
            suspicious_count += 2

        # Check bank account numbers
        account_matches = self.bank_account_pattern.findall(all_text)
        # Filter out phone numbers
        phone_set = set(entities["phone_numbers"])
        for account in account_matches:
            if account not in phone_set and len(account) >= 9:
                matches.append(PatternMatch(
                    pattern_type="bank_account",
                    value=account,
                    is_suspicious=True,
                    reason="Unsolicited bank account number"
                ))
                entities["bank_accounts"].append(account)
                suspicious_count += 1

        # Check amounts (requesting specific amounts is suspicious)
        amount_matches = self.amount_pattern.findall(all_text)
        for amount in amount_matches:
            matches.append(PatternMatch(
                pattern_type="amount",
                value=amount,
                is_suspicious=True,
                reason="Specific amount mentioned"
            ))
            entities["amounts"].append(amount)
            suspicious_count += 1

        # Calculate score
        score = self._calculate_score(matches, suspicious_count)

        return PatternMatcherResult(
            score=score,
            matches=matches,
            entities_found=entities,
            suspicious_count=suspicious_count
        )

    def _check_upi_suspicious(self, upi: str) -> tuple:
        """
        Check if UPI ID appears suspicious

        Args:
            upi: UPI ID to check

        Returns:
            Tuple of (is_suspicious, reason)
        """
        upi_lower = upi.lower()

        # Extract provider
        if '@' in upi:
            _, provider = upi.split('@', 1)
            provider_lower = provider.lower()

            # Check for suspicious patterns in username
            username = upi.split('@')[0].lower()
            for pattern in self.suspicious_upi_patterns:
                if re.search(pattern, username):
                    return True, f"Suspicious keyword in UPI: {pattern}"

            # Unknown/unusual provider
            if provider_lower not in self.legitimate_upi_providers:
                return True, f"Unusual UPI provider: {provider}"

        # Personal-looking UPIs claiming to be official
        if any(word in upi_lower for word in ['official', 'support', 'care', 'help', 'verify']):
            return True, "Personal UPI disguised as official"

        return False, "Appears normal"

    def _check_url_suspicious(self, url: str) -> tuple:
        """
        Check if URL appears suspicious/phishing

        Args:
            url: URL to check

        Returns:
            Tuple of (is_suspicious, reason)
        """
        try:
            # Parse URL
            if not url.startswith(('http://', 'https://')):
                url = 'http://' + url

            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Check against whitelist
            for legit_domain in self.legitimate_domains:
                if domain == legit_domain or domain.endswith('.' + legit_domain):
                    return False, "Known legitimate domain"

            # Check for IP address instead of domain
            if re.match(r'\d+\.\d+\.\d+\.\d+', domain):
                return True, "IP address instead of domain name"

            # Check for suspicious TLDs
            suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq']
            if any(domain.endswith(tld) for tld in suspicious_tlds):
                return True, "Free/suspicious TLD commonly used in phishing"

            # Check for phishing patterns
            for pattern in self.phishing_domain_patterns:
                if re.match(pattern, domain):
                    return True, "Domain matches phishing pattern"

            # Check for typosquatting (common brand misspellings)
            typosquat_patterns = [
                r'.*payt[i1]m.*', r'.*ph[o0]nep[e3].*',
                r'.*g[o0]{2}gle.*', r'.*amaz[o0]n.*',
                r'.*hdfc.*(?!hdfcbank)', r'.*sbi.*(?!sbi\.co\.in)',
            ]
            for pattern in typosquat_patterns:
                if re.match(pattern, domain):
                    return True, "Possible typosquatting/impersonation"

            # HTTP instead of HTTPS for sensitive sites
            if parsed.scheme == 'http' and any(
                    word in domain for word in ['bank', 'pay', 'login', 'secure']
            ):
                return True, "Non-secure HTTP for sensitive site"

            # Long subdomain chains (common in phishing)
            if domain.count('.') > 3:
                return True, "Excessive subdomains"

            # URL contains credentials
            if '@' in url.split('?')[0]:
                return True, "URL contains embedded credentials"

            # Check path for suspicious keywords
            path = parsed.path.lower()
            suspicious_path_keywords = [
                'login', 'verify', 'update', 'secure', 'account',
                'confirm', 'suspend', 'unlock', 'validate'
            ]
            if any(kw in path for kw in suspicious_path_keywords):
                return True, "Suspicious keywords in URL path"

        except Exception as e:
            logger.warning(f"Error parsing URL {url}: {e}")
            return True, "Malformed URL"

        # Default: unsolicited URLs are suspicious
        return True, "Unsolicited link in message"

    def _calculate_score(
            self,
            matches: List[PatternMatch],
            suspicious_count: int
    ) -> float:
        """
        Calculate suspiciousness score based on matches

        Args:
            matches: List of pattern matches
            suspicious_count: Number of suspicious items

        Returns:
            Score between 0 and 1
        """
        if not matches:
            return 0.0

        # Base score from suspicious count
        base_score = min(1.0, suspicious_count * 0.15)

        # Bonus for certain high-risk patterns
        high_risk_types = {'shortened_url', 'url', 'bank_account'}
        high_risk_count = sum(
            1 for m in matches
            if m.pattern_type in high_risk_types and m.is_suspicious
        )
        bonus = min(0.3, high_risk_count * 0.1)

        return min(1.0, base_score + bonus)
