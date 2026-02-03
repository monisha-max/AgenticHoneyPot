"""
Ensemble Scam Detection
Combines multiple detection methods for robust scam identification
"""

import logging
from typing import List, Optional
from dataclasses import dataclass, field

from app.config import settings
from app.api.schemas import ScamType, DetectionResult
from app.detection.rule_based import RuleBasedDetector, RuleBasedResult
from app.detection.pattern_matcher import PatternMatcher, PatternMatcherResult
from app.detection.llm_analyzer import LLMSemanticAnalyzer, LLMAnalysisResult
from app.detection.ml_classifier import ScamMLClassifier, MLClassifierResult

logger = logging.getLogger(__name__)


@dataclass
class EnsembleResult:
    """Complete result from ensemble detection"""
    is_scam: bool
    confidence: float
    scam_type: ScamType
    evidence: List[str] = field(default_factory=list)
    keywords_found: List[str] = field(default_factory=list)
    entities_found: dict = field(default_factory=dict)
    tactics_identified: List[str] = field(default_factory=list)
    component_scores: dict = field(default_factory=dict)
    ml_scam_type: str = ""  # Original ML classification
    reasoning: str = ""     # LLM reasoning
    intent: str = ""        # LLM inferred intent


class ScamDetectionEnsemble:
    """
    Ensemble detector combining multiple detection methods:
    1. Rule-based detection (keywords, patterns) - 20%
    2. Pattern matching (entities, URLs) - 15%
    3. ML Classifier (trained on 10K scam dataset) - 35%
    4. LLM Semantic Analysis - 30%

    Uses weighted voting to determine final verdict
    """

    def __init__(self):
        """Initialize all detection components"""
        self.rule_detector = RuleBasedDetector(weight=0.20)
        self.pattern_matcher = PatternMatcher(weight=0.15)
        self.ml_classifier = ScamMLClassifier(weight=0.35)
        self.llm_analyzer = LLMSemanticAnalyzer(weight=0.30)

        logger.info("Initialized ScamDetectionEnsemble with ML classifier")

    async def detect(
            self,
            text: str,
            history: List[str] = None
    ) -> EnsembleResult:
        """
        Run ensemble detection on message

        Args:
            text: Current message text
            history: Previous message texts

        Returns:
            EnsembleResult with combined verdict
        """
        logger.debug(f"Running ensemble detection on: {text[:50]}...")

        # Run all detectors IN PARALLEL for faster response
        import asyncio

        # Wrap sync detectors in async
        async def run_rule_detector():
            return await asyncio.to_thread(self.rule_detector.detect, text, history)

        async def run_pattern_matcher():
            return await asyncio.to_thread(self.pattern_matcher.match, text, history)

        async def run_ml_classifier():
            return await asyncio.to_thread(self.ml_classifier.predict, text)

        # Run all 4 detectors in parallel
        rule_result, pattern_result, ml_result, llm_result = await asyncio.gather(
            run_rule_detector(),
            run_pattern_matcher(),
            run_ml_classifier(),
            self.llm_analyzer.analyze(text, history)
        )

        # Calculate weighted ensemble score
        weighted_score = (
                rule_result.score * self.rule_detector.weight +
                pattern_result.score * self.pattern_matcher.weight +
                ml_result.score * self.ml_classifier.weight +
                llm_result.score * self.llm_analyzer.weight
        )

        # Determine scam type using ML as primary, LLM as secondary
        scam_type = self._determine_scam_type(
            rule_result, ml_result, llm_result, weighted_score
        )

        # Collect evidence from all sources
        evidence = self._collect_evidence(rule_result, pattern_result, ml_result, llm_result)

        # Collect all keywords and tactics
        keywords = list(set(rule_result.keywords_found))
        tactics = list(set(
            llm_result.tactics_identified +
            [m.rule_name for m in rule_result.matches]
        ))

        # Add ML top features as evidence
        if ml_result.top_features:
            tactics.extend([f"ml:{f}" for f in ml_result.top_features[:3]])

        # Component scores for debugging
        component_scores = {
            "rule_based": round(rule_result.score, 3),
            "pattern_matcher": round(pattern_result.score, 3),
            "ml_classifier": round(ml_result.score, 3),
            "llm_analyzer": round(llm_result.score, 3),
            "weighted_total": round(weighted_score, 3)
        }

        # Determine if scam (with threshold)
        is_scam = bool(weighted_score >= 0.55) # Lowered from 0.60

        # Boost for (Financial Request + Contact Entity) combo - classic scam signal
        has_financial = any("financial" in m.category for m in rule_result.matches)
        has_contact = len(pattern_result.entities_found.get("upi_ids", [])) > 0 or \
                      len(pattern_result.entities_found.get("phone_numbers", [])) > 0
        
        if has_financial and has_contact:
            weighted_score = max(weighted_score, 0.75) # Forced high confidence
            is_scam = True

        # High confidence override from Rule-based (friend-in-trouble, impersonation patterns)
        if rule_result.score > 0.7:
            is_scam = True
            weighted_score = max(weighted_score, 0.85)
            logger.info(f"Rule-based high confidence override: score={rule_result.score}")

        # High confidence override from ML (trained on real data)
        if ml_result.confidence > 0.85:
            is_scam = True
            weighted_score = max(weighted_score, ml_result.confidence)

        # High confidence override from LLM
        if llm_result.confidence > settings.HIGH_CONFIDENCE_THRESHOLD:
            is_scam = bool(llm_result.is_scam)
            weighted_score = max(weighted_score, llm_result.confidence)

        logger.info(
            f"Ensemble result: is_scam={is_scam}, "
            f"confidence={weighted_score:.2f}, type={scam_type}, "
            f"ml_type={ml_result.original_type}"
        )

        return EnsembleResult(
            is_scam=is_scam,
            confidence=min(1.0, weighted_score),
            scam_type=scam_type,
            evidence=evidence,
            keywords_found=keywords,
            entities_found=pattern_result.entities_found,
            tactics_identified=tactics,
            component_scores=component_scores,
            ml_scam_type=ml_result.original_type,
            reasoning=llm_result.reasoning,
            intent=llm_result.intent
        )

    def _determine_scam_type(
            self,
            rule_result: RuleBasedResult,
            ml_result: MLClassifierResult,
            llm_result: LLMAnalysisResult,
            confidence: float
    ) -> ScamType:
        """
        Determine the most likely scam type

        Priority: ML (high confidence) > LLM (high confidence) > Rule-based > ML > LLM > UNKNOWN

        Args:
            rule_result: Rule-based detection result
            ml_result: ML classifier result
            llm_result: LLM analysis result
            confidence: Overall confidence score

        Returns:
            Most likely ScamType
        """
        # ML with high confidence (trained on real data)
        if ml_result.confidence > 0.75 and ml_result.scam_type != ScamType.UNKNOWN:
            return ml_result.scam_type

        # LLM with high confidence
        if llm_result.confidence > 0.7 and llm_result.scam_type != ScamType.UNKNOWN:
            return llm_result.scam_type

        # Rule-based if it found something specific
        if rule_result.scam_type != ScamType.UNKNOWN:
            return rule_result.scam_type

        # ML result
        if ml_result.scam_type != ScamType.UNKNOWN:
            return ml_result.scam_type

        # LLM result
        if llm_result.scam_type != ScamType.UNKNOWN:
            return llm_result.scam_type

        return ScamType.UNKNOWN

    def _collect_evidence(
            self,
            rule_result: RuleBasedResult,
            pattern_result: PatternMatcherResult,
            ml_result: MLClassifierResult,
            llm_result: LLMAnalysisResult
    ) -> List[str]:
        """
        Collect evidence from all detection sources
        """
        evidence = []

        # Add rule-based evidence
        evidence.extend(rule_result.evidence)

        # Add pattern evidence
        suspicious_patterns = [
            m for m in pattern_result.matches if m.is_suspicious
        ]
        for match in suspicious_patterns[:5]:
            evidence.append(f"{match.pattern_type}: {match.reason}")

        # Add ML classification
        if ml_result.original_type and ml_result.original_type != "fallback":
            evidence.append(f"ML detected: {ml_result.original_type} ({ml_result.confidence:.0%})")

        # Add ML top features
        if ml_result.top_features:
            evidence.append(f"ML features: {', '.join(ml_result.top_features[:3])}")

        # Add LLM reasoning
        if llm_result.reasoning:
            evidence.append(f"LLM: {llm_result.reasoning[:100]}")

        # Add identified tactics
        if llm_result.tactics_identified:
            evidence.append(f"Tactics: {', '.join(llm_result.tactics_identified[:3])}")

        return list(set(evidence))

    def to_detection_result(self, ensemble_result: EnsembleResult) -> DetectionResult:
        """Convert EnsembleResult to standard DetectionResult"""
        return DetectionResult(
            is_scam=ensemble_result.is_scam,
            confidence=ensemble_result.confidence,
            scam_type=ensemble_result.scam_type,
            evidence=ensemble_result.evidence,
            keywords_found=ensemble_result.keywords_found
        )


# Convenience function for one-off detection
async def detect_scam(text: str, history: List[str] = None) -> DetectionResult:
    """
    Quick scam detection function

    Args:
        text: Message to analyze
        history: Previous messages

    Returns:
        DetectionResult
    """
    ensemble = ScamDetectionEnsemble()
    result = await ensemble.detect(text, history)
    return ensemble.to_detection_result(result)
