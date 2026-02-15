"""
LLM Intelligence Enricher
Adds intelligence candidates from recent conversation text.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class EnrichedCandidate:
    field: str
    value: str
    confidence: float
    evidence: str


@dataclass
class EnrichmentResult:
    candidates: List[EnrichedCandidate] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CallbackVerificationResult:
    bank_accounts: List[str] = field(default_factory=list)
    upi_ids: List[str] = field(default_factory=list)
    phishing_links: List[str] = field(default_factory=list)
    phone_numbers: List[str] = field(default_factory=list)
    suspicious_keywords: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


class LLMIntelligenceEnricher:
    """Extracts intelligence candidates from recent messages via LLM."""

    def __init__(self, enabled: Optional[bool] = None):
        self.enabled = (
            settings.ENABLE_LLM_INTEL_ENRICHMENT or settings.ENABLE_LLM_CALLBACK_VERIFICATION
        ) if enabled is None else enabled
        self.provider = settings.LLM_PROVIDER.lower()
        self.client = None
        self._init_client()

    def _init_client(self) -> None:
        if not self.enabled:
            return

        try:
            if self.provider == "openai":
                from openai import OpenAI
                self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            elif self.provider == "anthropic":
                from anthropic import Anthropic
                self.client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            elif self.provider == "google":
                import google.generativeai as genai
                genai.configure(api_key=settings.GOOGLE_API_KEY)
                self.client = genai
            else:
                logger.warning("LLM enricher disabled: unknown provider '%s'", self.provider)
                self.client = None
        except Exception as e:
            logger.warning("Failed to initialize LLM enricher client: %s", e)
            self.client = None

    async def enrich_recent_messages(self, messages: List[str]) -> EnrichmentResult:
        """Return intelligence candidates with confidence and evidence."""
        if not self.enabled or not self.client or not messages:
            return EnrichmentResult()

        prompt = self._build_prompt(messages[-20:])
        response = await self._call_llm(prompt)
        if not response:
            return EnrichmentResult()

        return self._parse_response(response)

    async def verify_callback_intelligence(
            self,
            conversation_lines: List[str],
            current_intelligence: Dict[str, List[str]]
    ) -> CallbackVerificationResult:
        """
        Reconcile callback intelligence from full conversation.
        Can prune wrong ownership and fill missing fields.
        """
        if not self.enabled or not self.client or not conversation_lines:
            return CallbackVerificationResult()

        prompt = self._build_callback_prompt(conversation_lines[-80:], current_intelligence)
        callback_model = settings.LLM_CALLBACK_INTEL_MODEL or settings.LLM_MODEL
        response = await self._call_llm(prompt, model=callback_model)
        if not response:
            return CallbackVerificationResult()

        return self._parse_callback_response(response, conversation_lines)

    def _build_callback_prompt(
            self,
            conversation_lines: List[str],
            current_intelligence: Dict[str, List[str]]
    ) -> str:
        transcript = "\n".join(f"- {line}" for line in conversation_lines if line and line.strip())
        current_json = json.dumps(current_intelligence, ensure_ascii=True)
        return f"""You are verifying final scam intelligence before callback.

Conversation transcript (speaker-tagged):
{transcript}

Current extracted intelligence:
{current_json}

Task:
1) Remove victim-owned or unrelated entities from callback fields.
2) Keep only values that are explicitly present in transcript.
3) If a callback field value is missing but clearly present in transcript, add it.
4) Prefer entities given or controlled by scammer; exclude victim's own details.

Return STRICT JSON only in this shape:
{{
  "bank_accounts": [{{"value": "string", "ownership": "scammer|victim|unknown", "confidence": 0.0, "evidence": "short quote"}}],
  "upi_ids": [{{"value": "string", "ownership": "scammer|victim|unknown", "confidence": 0.0, "evidence": "short quote"}}],
  "phishing_links": [{{"value": "string", "ownership": "scammer|victim|unknown", "confidence": 0.0, "evidence": "short quote"}}],
  "phone_numbers": [{{"value": "string", "ownership": "scammer|victim|unknown", "confidence": 0.0, "evidence": "short quote"}}],
  "suspicious_keywords": [{{"value": "string", "ownership": "scammer|victim|unknown", "confidence": 0.0, "evidence": "short quote"}}]
}}

Rules:
- Do not infer values not present in transcript.
- Use "victim" ownership if the number/account belongs to user/victim context.
- Use "scammer" only when the detail appears provided by scammer for payment/contact/link.
- Confidence must be between 0 and 1.
- Keep evidence short and specific."""

    def _build_prompt(self, messages: List[str]) -> str:
        transcript = "\n".join(f"- {m}" for m in messages if m and m.strip())
        return f"""Extract scammer intelligence from this chat transcript.

Transcript:
{transcript}

Return STRICT JSON only with this shape:
{{
  "scammer_names": [{{"value": "string", "confidence": 0.0, "evidence": "short quote"}}],
  "email_addresses": [{{"value": "string", "confidence": 0.0, "evidence": "short quote"}}],
  "phone_numbers": [{{"value": "string", "confidence": 0.0, "evidence": "short quote"}}],
  "upi_ids": [{{"value": "string", "confidence": 0.0, "evidence": "short quote"}}],
  "bank_accounts": [{{"value": "string", "confidence": 0.0, "evidence": "short quote"}}],
  "ifsc_codes": [{{"value": "string", "confidence": 0.0, "evidence": "short quote"}}],
  "phishing_links": [{{"value": "string", "confidence": 0.0, "evidence": "short quote"}}],
  "fake_references": [{{"value": "string", "confidence": 0.0, "evidence": "short quote"}}],
  "suspicious_keywords": [{{"value": "string", "confidence": 0.0, "evidence": "short quote"}}]
}}

Rules:
- Only extract values explicitly present in transcript.
- Confidence must be between 0 and 1.
- Provide short evidence snippet per extracted value.
- If no values for a field, return empty list.
- Do not infer or hallucinate."""

    async def _call_llm(self, prompt: str, model: Optional[str] = None) -> str:
        import asyncio
        chosen_model = model or settings.LLM_MODEL

        if self.provider == "openai":
            def _openai_call():
                return self.client.chat.completions.create(
                    model=chosen_model,
                    messages=[
                        {"role": "system", "content": "You extract structured entities. Reply with JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=400,
                )

            response = await asyncio.to_thread(_openai_call)
            return (response.choices[0].message.content or "").strip()

        if self.provider == "anthropic":
            def _anthropic_call():
                return self.client.messages.create(
                    model=chosen_model,
                    max_tokens=400,
                    messages=[{"role": "user", "content": prompt}],
                )

            response = await asyncio.to_thread(_anthropic_call)
            return (response.content[0].text if response.content else "").strip()

        if self.provider == "google":
            def _google_call():
                selected_model = self.client.GenerativeModel(chosen_model)
                return selected_model.generate_content(prompt)

            response = await asyncio.to_thread(_google_call)
            return (response.text or "").strip()

        return ""

    def _parse_response(self, response: str) -> EnrichmentResult:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except Exception:
            logger.warning("LLM enricher JSON parse failed")
            return EnrichmentResult()

        candidates: List[EnrichedCandidate] = []
        field_map = {
            "scammer_names": "scammer_names",
            "email_addresses": "email_addresses",
            "phone_numbers": "phone_numbers",
            "upi_ids": "upi_ids",
            "bank_accounts": "bank_accounts",
            "ifsc_codes": "ifsc_codes",
            "phishing_links": "phishing_links",
            "fake_references": "fake_references",
            "suspicious_keywords": "suspicious_keywords",
        }

        for response_key, target_field in field_map.items():
            items = data.get(response_key, [])
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                value = str(item.get("value", "")).strip()
                evidence = str(item.get("evidence", "")).strip()
                try:
                    confidence = float(item.get("confidence", 0.0))
                except Exception:
                    confidence = 0.0
                confidence = max(0.0, min(1.0, confidence))
                if not value:
                    continue
                if target_field in {"email_addresses", "upi_ids", "phishing_links"}:
                    value = value.lower()
                value = value.rstrip(".,;:")

                candidates.append(
                    EnrichedCandidate(
                        field=target_field,
                        value=value,
                        confidence=confidence,
                        evidence=evidence,
                    )
                )

        return EnrichmentResult(candidates=candidates, raw=data)

    def _parse_callback_response(
            self,
            response: str,
            conversation_lines: List[str]
    ) -> CallbackVerificationResult:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except Exception:
            logger.warning("LLM callback verification JSON parse failed")
            return CallbackVerificationResult()

        transcript = "\n".join(conversation_lines).lower()
        min_conf = settings.LLM_CALLBACK_INTEL_MIN_CONFIDENCE
        allowed_ownership = {"scammer", "likely_scammer", "unknown"}

        result = CallbackVerificationResult(raw=data)

        field_to_attr = {
            "bank_accounts": "bank_accounts",
            "upi_ids": "upi_ids",
            "phishing_links": "phishing_links",
            "phone_numbers": "phone_numbers",
            "suspicious_keywords": "suspicious_keywords",
        }

        for field, attr_name in field_to_attr.items():
            items = data.get(field, [])
            if not isinstance(items, list):
                continue

            accepted: List[str] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                value = str(item.get("value", "")).strip()
                ownership = str(item.get("ownership", "")).strip().lower()
                evidence = str(item.get("evidence", "")).strip()
                try:
                    confidence = float(item.get("confidence", 0.0))
                except Exception:
                    confidence = 0.0
                confidence = max(0.0, min(1.0, confidence))

                if not value or confidence < min_conf or not evidence:
                    continue
                if ownership not in allowed_ownership:
                    continue

                normalized = self._normalize_callback_value(field, value)
                if not normalized:
                    continue
                if not self._value_exists_in_transcript(field, normalized, transcript):
                    continue
                if normalized not in accepted:
                    accepted.append(normalized)

            setattr(result, attr_name, accepted)

        return result

    def _normalize_callback_value(self, field: str, value: str) -> str:
        raw = value.strip()
        if not raw:
            return ""

        if field == "bank_accounts":
            digits = re.sub(r"\D", "", raw)
            if len(digits) in {10, 12}:
                return ""
            if 9 <= len(digits) <= 18:
                return digits
            return ""

        if field == "upi_ids":
            cleaned = raw.lower().rstrip(".,;:")
            if re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,}@[a-z]{2,15}", cleaned):
                return cleaned
            return ""

        if field == "phishing_links":
            cleaned = raw.strip().rstrip(".,;:")
            if cleaned.startswith("www."):
                cleaned = f"http://{cleaned}"
            if re.match(r"^https?://[^\s/$.?#].[^\s]*$", cleaned, re.IGNORECASE):
                return cleaned
            return ""

        if field == "phone_numbers":
            digits = re.sub(r"\D", "", raw)
            if len(digits) == 10 and digits[0] in "123456789":
                return f"+91{digits}"
            if len(digits) == 12 and digits.startswith("91") and digits[2] in "123456789":
                return f"+{digits}"
            return ""

        if field == "suspicious_keywords":
            cleaned = " ".join(raw.lower().split())
            if 2 <= len(cleaned) <= 40:
                return cleaned
            return ""

        return ""

    def _value_exists_in_transcript(self, field: str, value: str, transcript: str) -> bool:
        if field in {"bank_accounts", "phone_numbers"}:
            transcript_digits = re.sub(r"\D", "", transcript)
            target_digits = re.sub(r"\D", "", value)
            return bool(target_digits and target_digits in transcript_digits)

        if field == "suspicious_keywords":
            return value.lower() in transcript

        if field == "phishing_links":
            raw = value.lower()
            without_scheme = raw.replace("http://", "").replace("https://", "")
            return raw in transcript or without_scheme in transcript

        return value.lower() in transcript
