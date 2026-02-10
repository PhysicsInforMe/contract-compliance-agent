"""
Compliance Scoring Engine for Contract Compliance Agent.

Implements quantitative compliance scoring based on:
- CUAD (Contract Understanding Atticus Dataset) benchmark methodology
- ISO 37301:2021 compliance management framework
- ISO 37302:2025 maturity level assessment
- Industry best practices from legal tech vendors

The scoring system produces objective, reproducible metrics for
contract compliance assessment.
"""

from typing import Optional

from models.schemas import (
    ClauseAnalysis,
    ComplianceScore,
    RiskLevel,
    MaturityLevel,
    ClauseStatus,
    determine_risk_level,
    determine_maturity_level
)


# =============================================================================
# Scoring Configuration
# =============================================================================

# Default scoring weights (sum to 1.0)
DEFAULT_WEIGHTS = {
    "presence": 0.30,      # Is the clause present?
    "similarity": 0.40,    # Language similarity to expected
    "completeness": 0.30,  # All expected elements present?
}

# Risk level thresholds
DEFAULT_THRESHOLDS = {
    "low_risk": 0.70,      # Score >= this = Green
    "medium_risk": 0.40,   # Score >= this (< low) = Yellow
    # Score < medium_risk = Red
}


# =============================================================================
# Compliance Scorer
# =============================================================================

class ComplianceScorer:
    """
    Calculates compliance scores for contract analysis.
    
    The scoring methodology is based on established academic benchmarks
    (CUAD, LexGLUE) and industry frameworks (ISO 37301/37302).
    
    Scoring Formula:
        clause_score = (presence × w_p) + (similarity × w_s) + (completeness × w_c)
        document_score = Σ(clause_score × clause_weight) / Σ(clause_weight)
    
    Where:
        - w_p, w_s, w_c are component weights (default: 0.3, 0.4, 0.3)
        - clause_weight is the importance weight for each clause category
    """
    
    def __init__(
        self,
        weights: Optional[dict] = None,
        thresholds: Optional[dict] = None,
        clause_weights: Optional[dict] = None
    ):
        """
        Initialize the compliance scorer.
        
        Args:
            weights: Component weights (presence, similarity, completeness)
            thresholds: Risk level thresholds
            clause_weights: Per-clause importance weights
        """
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.thresholds = thresholds or DEFAULT_THRESHOLDS.copy()
        self.clause_weights = clause_weights or {}
        
        # Validate weights sum to 1.0
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            # Normalize weights
            self.weights = {k: v / weight_sum for k, v in self.weights.items()}
    
    # -------------------------------------------------------------------------
    # Clause-Level Scoring
    # -------------------------------------------------------------------------
    
    def calculate_clause_score(
        self,
        presence_score: float,
        similarity_score: float,
        completeness_score: float
    ) -> float:
        """
        Calculate overall score for a single clause.
        
        Uses weighted combination of component scores.
        
        Args:
            presence_score: Score for clause presence (0-1)
            similarity_score: Score for language similarity (0-1)
            completeness_score: Score for element completeness (0-1)
            
        Returns:
            Weighted overall score (0-1)
        """
        return (
            presence_score * self.weights["presence"] +
            similarity_score * self.weights["similarity"] +
            completeness_score * self.weights["completeness"]
        )
    
    def score_clause_analysis(self, analysis: ClauseAnalysis) -> ClauseAnalysis:
        """
        Calculate and update scores for a clause analysis.
        
        Args:
            analysis: ClauseAnalysis with component scores set
            
        Returns:
            Updated ClauseAnalysis with overall score and risk level
        """
        # Calculate overall score
        analysis.overall_score = self.calculate_clause_score(
            analysis.presence_score,
            analysis.similarity_score,
            analysis.completeness_score
        )
        
        # Determine risk level
        analysis.risk_level = determine_risk_level(
            analysis.overall_score,
            self.thresholds
        )
        
        return analysis
    
    def get_clause_weight(self, clause_id: str) -> float:
        """
        Get the importance weight for a clause.
        
        Args:
            clause_id: Clause identifier
            
        Returns:
            Weight value (0-1), defaults to 0.5 if not specified
        """
        return self.clause_weights.get(clause_id, 0.5)
    
    # -------------------------------------------------------------------------
    # Document-Level Scoring
    # -------------------------------------------------------------------------
    
    def calculate_document_score(
        self,
        clause_analyses: list[ClauseAnalysis]
    ) -> ComplianceScore:
        """
        Calculate overall compliance score for a document.
        
        Uses weighted average of clause scores, where weights
        reflect clause importance.
        
        Args:
            clause_analyses: List of clause analysis results
            
        Returns:
            ComplianceScore with overall metrics
        """
        if not clause_analyses:
            return self._empty_compliance_score()
        
        # Calculate weighted average
        weighted_sum = 0.0
        weight_total = 0.0
        
        # Track critical clauses separately
        critical_weighted_sum = 0.0
        critical_weight_total = 0.0
        
        # Count statistics
        present = 0
        partial = 0
        missing = 0
        high_risk = 0
        medium_risk = 0
        low_risk = 0
        
        for analysis in clause_analyses:
            # Get clause weight
            weight = self.get_clause_weight(analysis.clause_id)
            
            # Add to weighted sum
            weighted_sum += analysis.overall_score * weight
            weight_total += weight
            
            # Track critical clauses (weight >= 0.85)
            if weight >= 0.85:
                critical_weighted_sum += analysis.overall_score * weight
                critical_weight_total += weight
            
            # Update statistics
            if analysis.status == ClauseStatus.PRESENT:
                present += 1
            elif analysis.status == ClauseStatus.PARTIAL:
                partial += 1
            else:
                missing += 1
            
            # Count by risk level
            if analysis.risk_level == RiskLevel.HIGH:
                high_risk += 1
            elif analysis.risk_level == RiskLevel.MEDIUM:
                medium_risk += 1
            else:
                low_risk += 1
        
        # Calculate final scores
        overall_score = weighted_sum / weight_total if weight_total > 0 else 0.0
        
        critical_score = (
            critical_weighted_sum / critical_weight_total 
            if critical_weight_total > 0 else 1.0
        )
        
        total_clauses = len(clause_analyses)
        coverage = ((present + partial * 0.5) / total_clauses * 100) if total_clauses > 0 else 0.0
        
        return ComplianceScore(
            overall_score=round(overall_score, 4),
            risk_level=determine_risk_level(overall_score, self.thresholds),
            maturity_level=determine_maturity_level(overall_score),
            critical_clause_score=round(critical_score, 4),
            coverage_percentage=round(coverage, 2),
            clauses_analyzed=total_clauses,
            clauses_present=present,
            clauses_partial=partial,
            clauses_missing=missing,
            high_risk_issues=high_risk,
            medium_risk_issues=medium_risk,
            low_risk_issues=low_risk
        )
    
    def _empty_compliance_score(self) -> ComplianceScore:
        """Create an empty compliance score for edge cases."""
        return ComplianceScore(
            overall_score=0.0,
            risk_level=RiskLevel.HIGH,
            maturity_level=MaturityLevel.BASIC,
            critical_clause_score=0.0,
            coverage_percentage=0.0,
            clauses_analyzed=0,
            clauses_present=0,
            clauses_partial=0,
            clauses_missing=0
        )
    
    # -------------------------------------------------------------------------
    # Scoring Metrics
    # -------------------------------------------------------------------------
    
    @staticmethod
    def calculate_precision_at_recall(
        predictions: list[bool],
        ground_truth: list[bool],
        target_recall: float = 0.8
    ) -> float:
        """
        Calculate precision at a target recall level.
        
        This is the primary metric used in CUAD benchmark evaluation.
        
        Args:
            predictions: List of predicted positive/negative
            ground_truth: List of actual positive/negative
            target_recall: Target recall level (default 0.8 = 80%)
            
        Returns:
            Precision at the target recall level
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        # Calculate true positives and actual positives
        true_positives = sum(p and g for p, g in zip(predictions, ground_truth))
        actual_positives = sum(ground_truth)
        predicted_positives = sum(predictions)
        
        if actual_positives == 0:
            return 1.0 if not any(predictions) else 0.0
        
        recall = true_positives / actual_positives
        
        # If we haven't reached target recall, return 0
        if recall < target_recall:
            return 0.0
        
        # Calculate precision at this point
        precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
        
        return precision
    
    @staticmethod
    def calculate_f1_score(precision: float, recall: float) -> float:
        """
        Calculate F1 score from precision and recall.
        
        F1 = 2 × (precision × recall) / (precision + recall)
        
        Args:
            precision: Precision value (0-1)
            recall: Recall value (0-1)
            
        Returns:
            F1 score (0-1)
        """
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def calculate_jaccard_similarity(set_a: set, set_b: set) -> float:
        """
        Calculate Jaccard similarity between two sets.
        
        J(A,B) = |A ∩ B| / |A ∪ B|
        
        Used for measuring overlap between extracted and expected elements.
        
        Args:
            set_a: First set
            set_b: Second set
            
        Returns:
            Jaccard similarity (0-1)
        """
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
        
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        
        return intersection / union if union > 0 else 0.0


# =============================================================================
# Convenience Functions
# =============================================================================

def create_scorer(clause_weights: Optional[dict] = None) -> ComplianceScorer:
    """
    Create a compliance scorer with optional clause weights.
    
    Args:
        clause_weights: Optional dict of clause_id -> weight
        
    Returns:
        Configured ComplianceScorer instance
    """
    return ComplianceScorer(clause_weights=clause_weights)


def quick_score(
    presence: float,
    similarity: float,
    completeness: float
) -> tuple[float, RiskLevel]:
    """
    Quick scoring for a single clause.
    
    Args:
        presence: Presence score (0-1)
        similarity: Similarity score (0-1)
        completeness: Completeness score (0-1)
        
    Returns:
        Tuple of (overall_score, risk_level)
    """
    scorer = ComplianceScorer()
    score = scorer.calculate_clause_score(presence, similarity, completeness)
    risk = determine_risk_level(score)
    return score, risk
