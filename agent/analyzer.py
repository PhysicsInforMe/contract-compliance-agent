"""
Contract Analyzer for Contract Compliance Agent.

Orchestrates the full analysis pipeline: clause extraction,
scoring, and result aggregation.
"""

import json
import time
import uuid
from typing import Optional

from tools.llm_client import OllamaClient
from agent.extractor import ClauseExtractor, KeywordPreFilter
from agent.scorer import ComplianceScorer
from models.schemas import (
    ClauseAnalysis,
    ClauseRule,
    ClauseStatus,
    RiskLevel,
    AnalysisResult,
    DocumentMetadata,
    Recommendation,
    ContractType
)


# =============================================================================
# Similarity Scoring
# =============================================================================

SIMILARITY_SYSTEM_PROMPT = """You are a legal language analyst. Your task is to evaluate how well contract language matches expected professional standards.

Rate the language quality and compliance-appropriateness on a scale of 0.0 to 1.0."""


SIMILARITY_PROMPT = """Evaluate the following extracted contract clause text for:
1. Clarity and precision of language
2. Completeness of coverage
3. Professional legal drafting quality

CLAUSE TYPE: {clause_name}
EXPECTED CONTENT: {description}

EXTRACTED TEXT:
---
{extracted_text}
---

Respond with a JSON object:
{{
    "similarity_score": 0.0-1.0,
    "language_quality": "poor" | "fair" | "good" | "excellent",
    "issues": ["list of any language issues"],
    "reasoning": "brief explanation of score"
}}"""


# Ordering used when prioritising recommendations
_RISK_ORDER = {RiskLevel.HIGH: 0, RiskLevel.MEDIUM: 1, RiskLevel.LOW: 2}


# =============================================================================
# Contract Analyzer
# =============================================================================

class ContractAnalyzer:
    """
    Main analyzer class that orchestrates contract compliance analysis.

    Combines clause extraction and scoring to produce a comprehensive
    compliance assessment.
    """

    def __init__(
        self,
        llm_client: OllamaClient,
        clause_rules: list[ClauseRule],
        clause_weights: Optional[dict] = None,
        component_weights: Optional[dict] = None,
        risk_thresholds: Optional[dict] = None,
        max_text_length: int = 8000
    ):
        """
        Initialize the contract analyzer.

        Args:
            llm_client: Configured Ollama client
            clause_rules: List of clause rules to analyze
            clause_weights: Optional dict of clause_id -> importance weight
            component_weights: Optional presence/similarity/completeness weights
            risk_thresholds: Optional ``low_risk`` / ``medium_risk`` thresholds
            max_text_length: Maximum characters of contract text sent per clause
        """
        self.llm = llm_client
        self.clause_rules = clause_rules
        self.extractor = ClauseExtractor(llm_client, max_text_length=max_text_length)
        self.scorer = ComplianceScorer(
            weights=component_weights,
            thresholds=risk_thresholds,
            clause_weights=clause_weights or {}
        )
        self.prefilter = KeywordPreFilter()

    def analyze(
        self,
        contract_text: str,
        document_metadata: DocumentMetadata,
        contract_type: ContractType = ContractType.GENERAL
    ) -> AnalysisResult:
        """
        Perform full compliance analysis on a contract.

        Args:
            contract_text: The contract text to analyze
            document_metadata: Metadata about the source document
            contract_type: Type of contract for specialized handling

        Returns:
            Complete AnalysisResult with scores and recommendations
        """
        start_time = time.time()
        total_tokens = 0

        # Step 1: Extract and analyze each clause
        clause_analyses = []

        for rule in self.clause_rules:
            extraction = self.extractor.extract_clause(contract_text, rule)
            total_tokens += extraction.get("tokens_used", 0)

            presence_score = self._calculate_presence_score(extraction)

            similarity_score = 0.0
            if extraction["extracted_text"] and extraction["status"] != ClauseStatus.NOT_APPLICABLE:
                similarity_score, used = self._calculate_similarity_score(
                    extraction["extracted_text"],
                    rule
                )
                total_tokens += used

            completeness_score = self._calculate_completeness_score(
                extraction["status"],
                extraction["found_elements"],
                extraction["missing_elements"]
            )

            analysis = ClauseAnalysis(
                clause_id=rule.id,
                clause_name=rule.name,
                status=extraction["status"],
                presence_score=presence_score,
                similarity_score=similarity_score,
                completeness_score=completeness_score,
                overall_score=0.0,  # Will be calculated by scorer
                risk_level=RiskLevel.HIGH,  # Will be updated
                extracted_text=extraction["extracted_text"],
                found_elements=extraction["found_elements"],
                missing_elements=extraction["missing_elements"],
                issues=extraction["issues"],
                recommendations=[],
                confidence=extraction["confidence"]
            )

            analysis = self.scorer.score_clause_analysis(analysis)
            analysis.recommendations = self._generate_clause_recommendations(analysis, rule)
            clause_analyses.append(analysis)

        # Step 2: Calculate document-level scores
        compliance_score = self.scorer.calculate_document_score(clause_analyses)

        # Step 3: Generate prioritized recommendations
        recommendations = self._generate_document_recommendations(clause_analyses)

        # Step 4: Build final result
        processing_time = time.time() - start_time

        return AnalysisResult(
            analysis_id=str(uuid.uuid4()),
            model_used=self.llm.model,
            document=document_metadata,
            contract_type=contract_type,
            compliance_score=compliance_score,
            clause_analyses=clause_analyses,
            recommendations=recommendations,
            processing_time_seconds=round(processing_time, 2),
            tokens_used=total_tokens
        )

    # -------------------------------------------------------------------------
    # Score Calculation Methods
    # -------------------------------------------------------------------------

    def _calculate_presence_score(self, extraction: dict) -> float:
        """
        Calculate presence score from extraction result.

        Args:
            extraction: Extraction result dict

        Returns:
            Presence score (0-1)
        """
        status = extraction["status"]
        confidence = extraction.get("confidence", 0.5)

        if status == ClauseStatus.PRESENT:
            return min(1.0, 0.8 + (confidence * 0.2))
        elif status == ClauseStatus.PARTIAL:
            return 0.4 + (confidence * 0.3)
        elif status == ClauseStatus.NOT_APPLICABLE:
            return 1.0  # N/A clauses don't penalize score
        else:  # MISSING
            return 0.0

    def _calculate_similarity_score(
        self,
        extracted_text: str,
        rule: ClauseRule
    ) -> tuple[float, int]:
        """
        Calculate similarity score using LLM evaluation.

        Args:
            extracted_text: Extracted clause text
            rule: Clause rule for context

        Returns:
            Tuple of (similarity score in [0, 1], tokens used)
        """
        if not extracted_text or len(extracted_text.strip()) < 20:
            return 0.0, 0

        prompt = SIMILARITY_PROMPT.format(
            clause_name=rule.name,
            description=rule.description,
            extracted_text=extracted_text[:2000]  # Limit length
        )

        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt=SIMILARITY_SYSTEM_PROMPT,
                temperature=0.1,
                format_json=True
            )
            result = json.loads(response.content)
            if not isinstance(result, dict):
                raise ValueError("similarity response is not a JSON object")
            score = float(result.get("similarity_score", 0.5))
            if score != score:  # NaN guard
                raise ValueError("similarity score is NaN")
            return max(0.0, min(1.0, score)), response.tokens_used

        except Exception:
            # Fallback: simple heuristic based on text length and keyword presence
            return self._heuristic_similarity(extracted_text, rule), 0

    def _heuristic_similarity(self, text: str, rule: ClauseRule) -> float:
        """
        Fallback heuristic similarity calculation.

        Args:
            text: Extracted text
            rule: Clause rule

        Returns:
            Heuristic similarity score
        """
        if not text:
            return 0.0

        keyword_score = self.prefilter.keyword_score(text, rule.keywords)
        length_score = min(1.0, len(text) / 500)

        return (keyword_score * 0.6) + (length_score * 0.4)

    def _calculate_completeness_score(
        self,
        status: ClauseStatus,
        found_elements: list[str],
        missing_elements: list[str]
    ) -> float:
        """
        Calculate completeness score based on found vs. missing elements.

        Args:
            status: Clause status from extraction
            found_elements: Elements that were found
            missing_elements: Elements that were missing

        Returns:
            Completeness score (0-1)
        """
        if status == ClauseStatus.NOT_APPLICABLE:
            return 1.0
        if status == ClauseStatus.MISSING:
            return 0.0

        total = len(found_elements) + len(missing_elements)
        if total == 0:
            return 0.5  # No expectations defined

        return len(found_elements) / total

    # -------------------------------------------------------------------------
    # Recommendation Generation
    # -------------------------------------------------------------------------

    def _generate_clause_recommendations(
        self,
        analysis: ClauseAnalysis,
        rule: ClauseRule
    ) -> list[str]:
        """
        Generate recommendations for a specific clause.

        Args:
            analysis: Clause analysis result
            rule: Clause rule

        Returns:
            List of recommendation strings
        """
        recommendations = []

        if analysis.status == ClauseStatus.MISSING:
            recommendations.append(f"Add {rule.name} clause to the contract")

        elif analysis.status == ClauseStatus.PARTIAL:
            for element in analysis.missing_elements[:3]:  # Top 3
                recommendations.append(f"Add {element} to {rule.name}")

        for issue in analysis.issues[:2]:  # Top 2 issues
            recommendations.append(f"Address: {issue}")

        return recommendations

    def _generate_document_recommendations(
        self,
        analyses: list[ClauseAnalysis]
    ) -> list[Recommendation]:
        """
        Generate prioritized document-level recommendations.

        Args:
            analyses: All clause analyses

        Returns:
            Prioritized list of Recommendation objects
        """
        recommendations = []

        sorted_analyses = sorted(
            analyses,
            key=lambda a: (_RISK_ORDER.get(a.risk_level, 3), a.overall_score)
        )

        for analysis in sorted_analyses:
            if analysis.status == ClauseStatus.NOT_APPLICABLE:
                continue
            if analysis.risk_level not in (RiskLevel.HIGH, RiskLevel.MEDIUM):
                continue

            if analysis.status == ClauseStatus.MISSING:
                recommendations.append(Recommendation(
                    priority=analysis.risk_level,
                    clause_id=analysis.clause_id,
                    title=f"Add {analysis.clause_name}",
                    description=(
                        f"The contract is missing a {analysis.clause_name} clause, "
                        f"which is a {analysis.risk_level.value} risk item."
                    ),
                    action_items=[
                        f"Draft a {analysis.clause_name} clause",
                        "Review with legal counsel",
                        "Negotiate inclusion with counterparty"
                    ]
                ))

            elif analysis.missing_elements:
                recommendations.append(Recommendation(
                    priority=analysis.risk_level,
                    clause_id=analysis.clause_id,
                    title=f"Complete {analysis.clause_name}",
                    description=(
                        f"The {analysis.clause_name} clause is incomplete. "
                        f"Missing elements: {', '.join(analysis.missing_elements[:3])}"
                    ),
                    action_items=[
                        f"Add: {elem}" for elem in analysis.missing_elements[:3]
                    ]
                ))

        return recommendations[:10]  # Top 10 recommendations


# =============================================================================
# Convenience Functions
# =============================================================================

def analyze_contract(
    contract_text: str,
    metadata: DocumentMetadata,
    llm_client: OllamaClient,
    rules: list[ClauseRule],
    weights: Optional[dict] = None
) -> AnalysisResult:
    """
    Convenience function for one-shot contract analysis.

    Args:
        contract_text: Contract text to analyze
        metadata: Document metadata
        llm_client: LLM client
        rules: Clause rules
        weights: Optional clause weights

    Returns:
        Analysis result
    """
    analyzer = ContractAnalyzer(llm_client, rules, weights)
    return analyzer.analyze(contract_text, metadata)
