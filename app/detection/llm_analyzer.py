"""
LLM-Based Semantic Analyzer for Scam Detection
Uses large language models to understand intent and context
"""

import logging
import json
from typing import List, Optional
from dataclasses import dataclass, field

from app.config import settings
from app.api.schemas import ScamType

logger = logging.getLogger(__name__)


@dataclass
class LLMAnalysisResult:
    """Result from LLM analysis"""
    score: float
    scam_type: ScamType
    is_scam: bool
    intent: str
    tactics_identified: List[str] = field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.0


class LLMSemanticAnalyzer:
    """
    LLM-based semantic analysis for deep intent detection
    Weight: 0.30 in ensemble
    """

    def __init__(self, weight: float = 0.30):
        self.weight = weight
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initialize LLM client based on provider"""
        provider = settings.LLM_PROVIDER.lower()

        try:
            if provider == "openai":
                from openai import OpenAI
                self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
                self.provider = "openai"

            elif provider == "anthropic":
                from anthropic import Anthropic
                self.client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
                self.provider = "anthropic"

            elif provider == "google":
                import google.generativeai as genai
                genai.configure(api_key=settings.GOOGLE_API_KEY)
                self.client = genai
                self.provider = "google"

            else:
                logger.warning(f"Unknown LLM provider: {provider}, using fallback")
                self.client = None
                self.provider = None

        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            self.client = None
            self.provider = None

    async def analyze(
            self,
            text: str,
            history: List[str] = None
    ) -> LLMAnalysisResult:
        """
        Analyze message using LLM for semantic understanding

        Args:
            text: Message text to analyze
            history: Conversation history

        Returns:
            LLMAnalysisResult with analysis
        """
        if self.client is None:
            return self._fallback_analysis(text)

        # Build prompt
        prompt = self._build_analysis_prompt(text, history)

        try:
            # Call LLM
            response = await self._call_llm(prompt)

            # Parse response
            return self._parse_response(response)

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return self._fallback_analysis(text)

    def _build_analysis_prompt(
            self,
            text: str,
            history: List[str] = None
    ) -> str:
        """Build the analysis prompt for LLM"""

        context = ""
        if history:
            context = "Previous messages:\n" + "\n".join(
                f"- {msg}" for msg in history[-5:]  # Last 5 messages
            ) + "\n\n"

        return f"""You are a scam detection expert. Analyze the following message for signs of fraud or scam.

{context}Current message to analyze:
"{text}"

Analyze for these scam indicators:
1. Urgency tactics (creating panic about deadlines)
2. Authority impersonation (pretending to be bank/government)
3. Request for sensitive info (OTP, PIN, passwords)
4. Threats (account suspension, legal action)
5. Too-good-to-be-true offers (lottery, prizes)
6. Suspicious payment requests
7. Phishing attempts
8. Authority impersonation (fake police, CBI, customs, "digital arrest", "do not tell anyone", court orders)

Respond in JSON format:
{{
    "is_scam": true/false,
    "confidence": 0.0-1.0,
    "scam_type": "BANKING_FRAUD|UPI_FRAUD|KYC_SCAM|JOB_SCAM|LOTTERY_SCAM|TECH_SUPPORT|INVESTMENT_FRAUD|BILL_PAYMENT_SCAM|DELIVERY_SCAM|IMPERSONATION|UNKNOWN",
    "intent": "brief description of sender's intent",
    "tactics": ["list", "of", "tactics", "used"],
    "reasoning": "brief explanation"
}}

Only respond with the JSON, no other text."""

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM API - uses asyncio.to_thread for sync clients"""
        import asyncio

        if self.provider == "openai":
            def _openai_call():
                return self.client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a scam detection expert. Respond only in JSON format."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
            response = await asyncio.to_thread(_openai_call)
            return response.choices[0].message.content

        elif self.provider == "anthropic":
            def _anthropic_call():
                return self.client.messages.create(
                    model=settings.LLM_MODEL,
                    max_tokens=500,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
            response = await asyncio.to_thread(_anthropic_call)
            return response.content[0].text

        elif self.provider == "google":
            def _google_call():
                model = self.client.GenerativeModel(settings.LLM_MODEL)
                return model.generate_content(prompt)
            response = await asyncio.to_thread(_google_call)
            return response.text

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _parse_response(self, response: str) -> LLMAnalysisResult:
        """Parse LLM response into structured result"""

        try:
            # Clean response (remove markdown code blocks if present)
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            cleaned = cleaned.strip()

            data = json.loads(cleaned)

            # Map scam type
            scam_type_str = data.get("scam_type", "UNKNOWN").upper()
            try:
                scam_type = ScamType(scam_type_str)
            except ValueError:
                scam_type = ScamType.UNKNOWN

            return LLMAnalysisResult(
                score=data.get("confidence", 0.5),
                scam_type=scam_type,
                is_scam=data.get("is_scam", False),
                intent=data.get("intent", "Unknown intent"),
                tactics_identified=data.get("tactics", []),
                reasoning=data.get("reasoning", ""),
                confidence=data.get("confidence", 0.5)
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return self._fallback_analysis("")

    def _fallback_analysis(self, text: str) -> LLMAnalysisResult:
        """
        Fallback analysis when LLM is unavailable
        Uses simple heuristics
        """
        text_lower = text.lower()

        # Simple keyword-based fallback
        scam_indicators = [
            ("urgent", 0.15), ("immediately", 0.15), ("blocked", 0.2),
            ("suspended", 0.2), ("verify", 0.15), ("otp", 0.25),
            ("click", 0.15), ("link", 0.1), ("bank", 0.1),
            ("account", 0.1), ("kyc", 0.15), ("prize", 0.2),
            ("won", 0.2), ("lottery", 0.25)
        ]

        score = 0.0
        tactics = []

        for keyword, weight in scam_indicators:
            if keyword in text_lower:
                score += weight
                tactics.append(f"keyword:{keyword}")

        score = min(1.0, score)

        return LLMAnalysisResult(
            score=score,
            scam_type=ScamType.UNKNOWN,
            is_scam=score > 0.5,
            intent="Unable to determine (LLM unavailable)",
            tactics_identified=tactics,
            reasoning="Fallback keyword analysis",
            confidence=score * 0.8  # Lower confidence for fallback
        )
