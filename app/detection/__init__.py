"""
Scam Detection Module
Multi-layer detection system for identifying scam messages
"""

from app.detection.rule_based import RuleBasedDetector, RuleBasedResult
from app.detection.pattern_matcher import PatternMatcher, PatternMatcherResult
from app.detection.llm_analyzer import LLMSemanticAnalyzer, LLMAnalysisResult
from app.detection.ml_classifier import ScamMLClassifier, MLClassifierResult, ScamSimilarityMatcher
from app.detection.ensemble import ScamDetectionEnsemble, EnsembleResult, detect_scam
from app.detection.scam_taxonomy import ScamTaxonomy, ScamCategory

__all__ = [
    # Rule-based
    "RuleBasedDetector",
    "RuleBasedResult",

    # Pattern matching
    "PatternMatcher",
    "PatternMatcherResult",

    # LLM analysis
    "LLMSemanticAnalyzer",
    "LLMAnalysisResult",

    # ML Classifier
    "ScamMLClassifier",
    "MLClassifierResult",
    "ScamSimilarityMatcher",

    # Ensemble
    "ScamDetectionEnsemble",
    "EnsembleResult",
    "detect_scam",

    # Taxonomy
    "ScamTaxonomy",
    "ScamCategory",
]
