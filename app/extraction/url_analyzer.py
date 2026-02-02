"""
URL Analyzer for Phishing Detection
Uses crawl4ai to scrape suspicious URLs and analyze content for scam indicators
"""

import re
import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class URLAnalysisResult:
    """Result from URL analysis"""
    url: str
    is_suspicious: bool
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    risk_score: float  # 0.0 to 1.0
    findings: List[str] = field(default_factory=list)
    page_title: str = ""
    detected_brand_impersonation: Optional[str] = None
    form_fields_found: List[str] = field(default_factory=list)
    external_links: List[str] = field(default_factory=list)
    ssl_valid: bool = True
    domain_age_suspicious: bool = False
    content_snippet: str = ""
    error: Optional[str] = None


class URLAnalyzer:
    """
    Analyzes URLs for phishing/scam indicators
    Uses crawl4ai for scraping and custom analysis for detection
    """

    def __init__(self):
        self._init_patterns()
        self._init_brand_keywords()
        self.crawler = None

    def _init_patterns(self):
        """Initialize detection patterns"""

        # Suspicious TLDs (free/cheap domains often used for scams)
        self.suspicious_tlds = {
            '.tk', '.ml', '.ga', '.cf', '.gq',  # Free domains
            '.xyz', '.top', '.work', '.click',  # Cheap domains
            '.loan', '.win', '.bid', '.review'  # Scam-associated
        }

        # URL shorteners (hide real destination)
        self.url_shorteners = {
            'bit.ly', 'tinyurl.com', 't.co', 'goo.gl',
            'ow.ly', 'is.gd', 'buff.ly', 'rebrand.ly',
            'shorte.st', 'adf.ly', 'cutt.ly'
        }

        # Sensitive form fields that indicate credential harvesting
        self.sensitive_fields = {
            'password', 'pwd', 'pass', 'otp', 'pin', 'cvv', 'cvc',
            'card', 'cardnumber', 'card_number', 'credit', 'debit',
            'account', 'accountnumber', 'account_number', 'acc_no',
            'aadhaar', 'aadhar', 'pan', 'pan_number', 'ssn',
            'atm', 'atmpin', 'netbanking', 'upi', 'upipin',
            'ifsc', 'swift', 'routing'
        }

        # Urgency keywords in page content
        self.urgency_patterns = [
            r'account.*(?:suspend|block|limit|restrict)',
            r'(?:verify|confirm|update).*(?:immediate|urgent|now)',
            r'(?:expire|expir).*(?:today|hour|minute|soon)',
            r'(?:unauthorized|suspicious).*(?:activity|transaction)',
            r'(?:secure|protect).*account.*(?:now|immediate)',
            r'last.*(?:chance|warning|notice)',
            r'(?:act|respond).*(?:immediate|urgent|now)'
        ]

    def _init_brand_keywords(self):
        """Initialize brand impersonation detection"""

        # Major Indian banks and services commonly impersonated
        self.brand_keywords = {
            'sbi': ['state bank', 'sbi', 'onlinesbi', 'sbiyono'],
            'hdfc': ['hdfc', 'hdfcbank', 'hdfc bank'],
            'icici': ['icici', 'icicibank', 'icici bank'],
            'axis': ['axis', 'axisbank', 'axis bank'],
            'kotak': ['kotak', 'kotakbank', 'kotak mahindra'],
            'pnb': ['pnb', 'punjab national', 'pnbindia'],
            'paytm': ['paytm', 'paytm bank', 'paytmmall'],
            'phonepe': ['phonepe', 'phone pe'],
            'gpay': ['google pay', 'gpay', 'googlepay'],
            'amazon': ['amazon', 'amazonpay', 'amazn'],
            'flipkart': ['flipkart', 'flpkrt'],
            'rbi': ['rbi', 'reserve bank', 'reservebank'],
            'incometax': ['income tax', 'incometax', 'efiling', 'itr'],
            'customs': ['customs', 'indian customs', 'cbic'],
            'police': ['cyber cell', 'cybercrime', 'police']
        }

    async def _init_crawler(self):
        """Initialize crawl4ai crawler lazily"""
        if self.crawler is None:
            try:
                from crawl4ai import AsyncWebCrawler
                self.crawler = AsyncWebCrawler(verbose=False)
                # Try to warmup if method exists (API varies by version)
                if hasattr(self.crawler, 'awarmup'):
                    await self.crawler.awarmup()
                elif hasattr(self.crawler, 'start'):
                    await self.crawler.start()
            except ImportError:
                logger.warning("crawl4ai not installed, using fallback HTTP client")
                self.crawler = "fallback"
            except Exception as e:
                logger.warning(f"crawl4ai init issue, using fallback: {e}")
                self.crawler = "fallback"

    async def analyze_url(self, url: str, timeout: int = 10) -> URLAnalysisResult:
        """
        Analyze a single URL for phishing indicators

        Args:
            url: URL to analyze
            timeout: Request timeout in seconds

        Returns:
            URLAnalysisResult with analysis findings
        """
        # Normalize URL
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url

        result = URLAnalysisResult(
            url=url,
            is_suspicious=False,
            risk_level="LOW",
            risk_score=0.0
        )

        try:
            # Step 1: Analyze URL structure
            url_findings, url_score = self._analyze_url_structure(url)
            result.findings.extend(url_findings)
            result.risk_score += url_score

            # Step 2: Fetch and analyze page content
            await self._init_crawler()
            content_result = await self._fetch_and_analyze(url, timeout)

            if content_result.get("error"):
                result.error = content_result["error"]
                # If we can't fetch, increase suspicion
                result.findings.append("Unable to fetch URL content - may be temporary scam page")
                result.risk_score += 0.2
            else:
                # Merge content analysis
                result.findings.extend(content_result.get("findings", []))
                result.risk_score += content_result.get("score", 0)
                result.page_title = content_result.get("title", "")
                result.form_fields_found = content_result.get("form_fields", [])
                result.external_links = content_result.get("external_links", [])
                result.content_snippet = content_result.get("snippet", "")
                result.detected_brand_impersonation = content_result.get("brand_impersonation")

            # Step 3: Calculate final risk level
            result.risk_score = min(1.0, result.risk_score)
            result.is_suspicious = result.risk_score > 0.3
            result.risk_level = self._calculate_risk_level(result.risk_score)

        except Exception as e:
            logger.error(f"Error analyzing URL {url}: {e}")
            result.error = str(e)
            result.findings.append(f"Analysis error: {str(e)}")
            result.is_suspicious = True
            result.risk_level = "MEDIUM"
            result.risk_score = 0.5

        return result

    def _analyze_url_structure(self, url: str) -> tuple:
        """Analyze URL structure for suspicious patterns"""
        findings = []
        score = 0.0

        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path.lower()

            # Check for suspicious TLD
            for tld in self.suspicious_tlds:
                if domain.endswith(tld):
                    findings.append(f"Suspicious TLD detected: {tld}")
                    score += 0.25
                    break

            # Check for URL shortener
            for shortener in self.url_shorteners:
                if shortener in domain:
                    findings.append(f"URL shortener detected: {shortener} (hiding real destination)")
                    score += 0.3
                    break

            # Check for IP address instead of domain
            ip_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
            if re.match(ip_pattern, domain):
                findings.append("URL uses IP address instead of domain name")
                score += 0.35

            # Check for brand name in subdomain (common phishing tactic)
            for brand, keywords in self.brand_keywords.items():
                for kw in keywords:
                    if kw in domain and not domain.endswith(f'{kw}.com') and not domain.endswith(f'{kw}.in'):
                        findings.append(f"Brand name '{brand}' found in suspicious domain position")
                        score += 0.3
                        break

            # Check for excessive subdomains
            subdomain_count = domain.count('.')
            if subdomain_count > 3:
                findings.append(f"Excessive subdomains ({subdomain_count}) - common in phishing")
                score += 0.15

            # Check for suspicious path keywords
            suspicious_paths = ['login', 'signin', 'verify', 'secure', 'update', 'confirm', 'account', 'banking']
            for sp in suspicious_paths:
                if sp in path:
                    findings.append(f"Suspicious path keyword: '{sp}'")
                    score += 0.1
                    break

            # Check for HTTP (not HTTPS)
            if parsed.scheme == 'http':
                findings.append("URL uses HTTP instead of HTTPS (insecure)")
                score += 0.2

            # Check for unusual port
            if parsed.port and parsed.port not in [80, 443]:
                findings.append(f"Unusual port number: {parsed.port}")
                score += 0.15

            # Check for typosquatting patterns
            typo_patterns = [
                ('googel', 'google'), ('gogle', 'google'),
                ('faceboook', 'facebook'), ('amaz0n', 'amazon'),
                ('paytim', 'paytm'), ('phonpe', 'phonepe'),
                ('sbibank', 'sbi'), ('hdfcbanking', 'hdfc')
            ]
            for typo, brand in typo_patterns:
                if typo in domain:
                    findings.append(f"Possible typosquatting detected: '{typo}' (mimicking {brand})")
                    score += 0.4
                    break

        except Exception as e:
            findings.append(f"URL parsing error: {str(e)}")
            score += 0.1

        return findings, score

    async def _fetch_and_analyze(self, url: str, timeout: int) -> Dict[str, Any]:
        """Fetch URL content and analyze for phishing indicators"""
        result = {
            "findings": [],
            "score": 0.0,
            "title": "",
            "form_fields": [],
            "external_links": [],
            "snippet": "",
            "brand_impersonation": None,
            "error": None
        }

        try:
            html_content = await self._fetch_content(url, timeout)

            if not html_content:
                result["error"] = "Empty or failed response"
                return result

            # Parse HTML
            soup = BeautifulSoup(html_content, 'lxml')

            # Get title
            title_tag = soup.find('title')
            result["title"] = title_tag.get_text().strip() if title_tag else ""

            # Get text content for analysis
            text_content = soup.get_text(separator=' ', strip=True).lower()
            result["snippet"] = text_content[:500] if text_content else ""

            # Analyze forms for credential harvesting
            forms = soup.find_all('form')
            for form in forms:
                inputs = form.find_all('input')
                for inp in inputs:
                    field_name = (inp.get('name', '') + inp.get('id', '') + inp.get('placeholder', '')).lower()
                    field_type = inp.get('type', '').lower()

                    for sensitive in self.sensitive_fields:
                        if sensitive in field_name or field_type == 'password':
                            result["form_fields"].append(field_name or field_type)
                            break

            if result["form_fields"]:
                result["findings"].append(f"Credential harvesting form detected with fields: {', '.join(set(result['form_fields']))}")
                result["score"] += 0.4

            # Check for brand impersonation
            detected_brand = self._detect_brand_impersonation(text_content, result["title"].lower())
            if detected_brand:
                result["brand_impersonation"] = detected_brand
                result["findings"].append(f"Possible impersonation of: {detected_brand}")
                result["score"] += 0.3

            # Check for urgency language
            for pattern in self.urgency_patterns:
                if re.search(pattern, text_content):
                    result["findings"].append("Urgency/fear tactics detected in page content")
                    result["score"] += 0.2
                    break

            # Check for suspicious JavaScript
            scripts = soup.find_all('script')
            suspicious_js = ['document.cookie', 'localStorage', 'keylogger', 'formgrabber']
            for script in scripts:
                script_text = script.string or ''
                for sus in suspicious_js:
                    if sus in script_text.lower():
                        result["findings"].append(f"Suspicious JavaScript detected: {sus}")
                        result["score"] += 0.25
                        break

            # Check for external form submission
            for form in forms:
                action = form.get('action', '')
                if action.startswith('http') and urlparse(url).netloc not in action:
                    result["findings"].append(f"Form submits data to external domain: {urlparse(action).netloc}")
                    result["score"] += 0.35

            # Check meta tags for suspicious content
            meta_refresh = soup.find('meta', attrs={'http-equiv': 'refresh'})
            if meta_refresh:
                result["findings"].append("Page uses meta refresh redirect")
                result["score"] += 0.15

            # Check for hidden iframes
            iframes = soup.find_all('iframe')
            for iframe in iframes:
                style = iframe.get('style', '')
                if 'display:none' in style or 'visibility:hidden' in style:
                    result["findings"].append("Hidden iframe detected (possible clickjacking)")
                    result["score"] += 0.25

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Error fetching/analyzing {url}: {e}")

        return result

    async def _fetch_content(self, url: str, timeout: int) -> Optional[str]:
        """Fetch URL content using crawl4ai or fallback"""
        try:
            if self.crawler and self.crawler != "fallback":
                from crawl4ai import AsyncWebCrawler
                result = await self.crawler.arun(
                    url=url,
                    timeout=timeout * 1000  # ms
                )
                return result.html if result.success else None
            else:
                # Fallback to httpx
                import httpx
                async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                    response = await client.get(url, headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    })
                    return response.text if response.status_code == 200 else None

        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    def _detect_brand_impersonation(self, text: str, title: str) -> Optional[str]:
        """Detect if page is impersonating a known brand"""
        combined_text = f"{title} {text}"

        for brand, keywords in self.brand_keywords.items():
            matches = sum(1 for kw in keywords if kw in combined_text)
            if matches >= 2:  # Multiple brand keywords found
                return brand.upper()

        return None

    def _calculate_risk_level(self, score: float) -> str:
        """Calculate risk level from score"""
        if score >= 0.7:
            return "CRITICAL"
        elif score >= 0.5:
            return "HIGH"
        elif score >= 0.3:
            return "MEDIUM"
        return "LOW"

    async def analyze_multiple_urls(self, urls: List[str], max_concurrent: int = 3) -> List[URLAnalysisResult]:
        """
        Analyze multiple URLs concurrently

        Args:
            urls: List of URLs to analyze
            max_concurrent: Maximum concurrent requests

        Returns:
            List of URLAnalysisResult
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_with_limit(url: str) -> URLAnalysisResult:
            async with semaphore:
                return await self.analyze_url(url)

        tasks = [analyze_with_limit(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        final_results = []
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                final_results.append(URLAnalysisResult(
                    url=url,
                    is_suspicious=True,
                    risk_level="MEDIUM",
                    risk_score=0.5,
                    error=str(result),
                    findings=["Analysis failed due to error"]
                ))
            else:
                final_results.append(result)

        return final_results

    def generate_report(self, result: URLAnalysisResult) -> str:
        """Generate human-readable report from analysis"""
        lines = [
            f"URL Analysis Report",
            f"=" * 50,
            f"URL: {result.url}",
            f"Risk Level: {result.risk_level}",
            f"Risk Score: {result.risk_score:.0%}",
            f"Suspicious: {'YES' if result.is_suspicious else 'NO'}",
            ""
        ]

        if result.page_title:
            lines.append(f"Page Title: {result.page_title}")

        if result.detected_brand_impersonation:
            lines.append(f"⚠️  BRAND IMPERSONATION DETECTED: {result.detected_brand_impersonation}")

        if result.findings:
            lines.append("")
            lines.append("Findings:")
            for finding in result.findings:
                lines.append(f"  • {finding}")

        if result.form_fields_found:
            lines.append("")
            lines.append(f"⚠️  Sensitive form fields: {', '.join(set(result.form_fields_found))}")

        if result.error:
            lines.append("")
            lines.append(f"Error: {result.error}")

        return "\n".join(lines)
