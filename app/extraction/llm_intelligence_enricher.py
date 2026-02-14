"""
LLM Intelligence Enricher
Adds intelligence candidates from recent conversation text.
"""

import json
import logging
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


class LLMIntelligenceEnricher:
    """Extracts intelligence candidates from recent messages via LLM."""

    def __init__(self, enabled: Optional[bool] = None):
        self.enabled = settings.ENABLE_LLM_INTEL_ENRICHMENT if enabled is None else enabled
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

    async def _call_llm(self, prompt: str) -> str:
        import asyncio

        if self.provider == "openai":
            def _openai_call():
                return self.client.chat.completions.create(
                    model=settings.LLM_MODEL,
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
                    model=settings.LLM_MODEL,
                    max_tokens=400,
                    messages=[{"role": "user", "content": prompt}],
                )

            response = await asyncio.to_thread(_anthropic_call)
            return (response.content[0].text if response.content else "").strip()

        if self.provider == "google":
            def _google_call():
                model = self.client.GenerativeModel(settings.LLM_MODEL)
                return model.generate_content(prompt)

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
