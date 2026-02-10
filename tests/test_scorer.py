"""
Tests for the Compliance Scoring Engine.

Tests the scoring calculations based on CUAD and ISO 37301 methodologies.
"""

import pytest
from agent.scorer import ComplianceScorer, quick_score
from models.schemas import (
    ClauseAnalysis,
    ClauseStatus,
    RiskLevel,
    MaturityLevel,
    determine_risk_level,
    determine_maturity_level
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def scorer():
    """Create a scorer with default settings."""
    return ComplianceScorer()


@pytest.fixture
def weighted_scorer():
    """Create a scorer with custom clause weights."""
    weights = {
        "termination": 0.95,
        "ip_assignment": 0.90,
        "confidentiality": 0.80,
        "payment": 0.70,
    }
    return ComplianceScorer(clause_weights=weights)


@pytest.fixture
def sample_clause_analysis():
    """Create a sample clause analysis."""
    return ClauseAnalysis(
        clause_id="termination",
        clause_name="Termination Rights",
        status=ClauseStatus.PRESENT,
        presence_score=0.9,
        similarity_score=0.8,
        completeness_score=0.7,
        overall_score=0.0,  # Will be calculated
        risk_level=RiskLevel.HIGH,  # Will be updated
    )


# =============================================================================
# Clause Score Calculation Tests
# =============================================================================

class TestClauseScoring:
    """Tests for individual clause scoring."""
    
    def test_calculate_clause_score_perfect(self, scorer):
        """Test perfect scores produce 1.0."""
        score = scorer.calculate_clause_score(1.0, 1.0, 1.0)
        assert score == 1.0
    
    def test_calculate_clause_score_zero(self, scorer):
        """Test zero scores produce 0.0."""
        score = scorer.calculate_clause_score(0.0, 0.0, 0.0)
        assert score == 0.0
    
    def test_calculate_clause_score_weighted(self, scorer):
        """Test weighted calculation is correct."""
        # Default weights: presence=0.3, similarity=0.4, completeness=0.3
        score = scorer.calculate_clause_score(1.0, 0.5, 0.0)
        # Expected: 1.0*0.3 + 0.5*0.4 + 0.0*0.3 = 0.3 + 0.2 + 0.0 = 0.5
        assert score == pytest.approx(0.5, rel=0.01)
    
    def test_calculate_clause_score_bounds(self, scorer):
        """Test scores stay within 0-1 bounds."""
        score = scorer.calculate_clause_score(0.5, 0.5, 0.5)
        assert 0.0 <= score <= 1.0
    
    def test_score_clause_analysis(self, scorer, sample_clause_analysis):
        """Test scoring updates analysis object correctly."""
        analysis = scorer.score_clause_analysis(sample_clause_analysis)
        
        # Check score was calculated
        assert analysis.overall_score > 0
        
        # Check risk level was set
        assert analysis.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]
    
    def test_score_clause_analysis_high_score(self, scorer):
        """Test high-scoring clause gets LOW risk."""
        analysis = ClauseAnalysis(
            clause_id="test",
            clause_name="Test Clause",
            status=ClauseStatus.PRESENT,
            presence_score=1.0,
            similarity_score=0.9,
            completeness_score=0.85,
            overall_score=0.0,
            risk_level=RiskLevel.HIGH,
        )
        
        result = scorer.score_clause_analysis(analysis)
        assert result.risk_level == RiskLevel.LOW
    
    def test_score_clause_analysis_low_score(self, scorer):
        """Test low-scoring clause gets HIGH risk."""
        analysis = ClauseAnalysis(
            clause_id="test",
            clause_name="Test Clause",
            status=ClauseStatus.MISSING,
            presence_score=0.0,
            similarity_score=0.0,
            completeness_score=0.0,
            overall_score=0.0,
            risk_level=RiskLevel.LOW,
        )
        
        result = scorer.score_clause_analysis(analysis)
        assert result.risk_level == RiskLevel.HIGH


# =============================================================================
# Document Score Calculation Tests
# =============================================================================

class TestDocumentScoring:
    """Tests for document-level scoring."""
    
    def test_calculate_document_score_empty(self, scorer):
        """Test empty analysis list produces empty score."""
        score = scorer.calculate_document_score([])
        assert score.overall_score == 0.0
        assert score.clauses_analyzed == 0
    
    def test_calculate_document_score_single_clause(self, weighted_scorer):
        """Test single clause document scoring."""
        analysis = ClauseAnalysis(
            clause_id="termination",
            clause_name="Termination Rights",
            status=ClauseStatus.PRESENT,
            presence_score=1.0,
            similarity_score=0.8,
            completeness_score=0.7,
            overall_score=0.83,  # Pre-calculated
            risk_level=RiskLevel.LOW,
        )
        
        score = weighted_scorer.calculate_document_score([analysis])
        
        assert score.clauses_analyzed == 1
        assert score.clauses_present == 1
        assert score.clauses_missing == 0
    
    def test_calculate_document_score_multiple_clauses(self, weighted_scorer):
        """Test multiple clause document scoring."""
        analyses = [
            ClauseAnalysis(
                clause_id="termination",
                clause_name="Termination Rights",
                status=ClauseStatus.PRESENT,
                presence_score=1.0,
                similarity_score=0.9,
                completeness_score=0.8,
                overall_score=0.89,
                risk_level=RiskLevel.LOW,
            ),
            ClauseAnalysis(
                clause_id="confidentiality",
                clause_name="Confidentiality",
                status=ClauseStatus.PARTIAL,
                presence_score=0.5,
                similarity_score=0.6,
                completeness_score=0.4,
                overall_score=0.51,
                risk_level=RiskLevel.MEDIUM,
            ),
            ClauseAnalysis(
                clause_id="payment",
                clause_name="Payment Terms",
                status=ClauseStatus.MISSING,
                presence_score=0.0,
                similarity_score=0.0,
                completeness_score=0.0,
                overall_score=0.0,
                risk_level=RiskLevel.HIGH,
            ),
        ]
        
        score = weighted_scorer.calculate_document_score(analyses)
        
        assert score.clauses_analyzed == 3
        assert score.clauses_present == 1
        assert score.clauses_partial == 1
        assert score.clauses_missing == 1
        assert score.high_risk_issues == 1
        assert score.medium_risk_issues == 1
        assert score.low_risk_issues == 1
    
    def test_document_score_coverage_calculation(self, scorer):
        """Test coverage percentage calculation."""
        analyses = [
            ClauseAnalysis(
                clause_id="a",
                clause_name="A",
                status=ClauseStatus.PRESENT,
                presence_score=1.0,
                similarity_score=1.0,
                completeness_score=1.0,
                overall_score=1.0,
                risk_level=RiskLevel.LOW,
            ),
            ClauseAnalysis(
                clause_id="b",
                clause_name="B",
                status=ClauseStatus.PARTIAL,
                presence_score=0.5,
                similarity_score=0.5,
                completeness_score=0.5,
                overall_score=0.5,
                risk_level=RiskLevel.MEDIUM,
            ),
        ]
        
        score = scorer.calculate_document_score(analyses)
        
        # Coverage: (1 present + 0.5 partial) / 2 total = 0.75 = 75%
        assert score.coverage_percentage == pytest.approx(75.0, rel=0.01)


# =============================================================================
# Risk Level Determination Tests
# =============================================================================

class TestRiskLevelDetermination:
    """Tests for risk level determination."""
    
    def test_determine_risk_level_low(self):
        """Test low risk determination."""
        assert determine_risk_level(0.8) == RiskLevel.LOW
        assert determine_risk_level(0.7) == RiskLevel.LOW
        assert determine_risk_level(1.0) == RiskLevel.LOW
    
    def test_determine_risk_level_medium(self):
        """Test medium risk determination."""
        assert determine_risk_level(0.5) == RiskLevel.MEDIUM
        assert determine_risk_level(0.4) == RiskLevel.MEDIUM
        assert determine_risk_level(0.69) == RiskLevel.MEDIUM
    
    def test_determine_risk_level_high(self):
        """Test high risk determination."""
        assert determine_risk_level(0.39) == RiskLevel.HIGH
        assert determine_risk_level(0.0) == RiskLevel.HIGH
        assert determine_risk_level(0.1) == RiskLevel.HIGH
    
    def test_determine_risk_level_custom_thresholds(self):
        """Test risk determination with custom thresholds."""
        custom = {"low_risk": 0.9, "medium_risk": 0.5}
        
        assert determine_risk_level(0.95, custom) == RiskLevel.LOW
        assert determine_risk_level(0.7, custom) == RiskLevel.MEDIUM
        assert determine_risk_level(0.3, custom) == RiskLevel.HIGH


# =============================================================================
# Maturity Level Tests
# =============================================================================

class TestMaturityLevel:
    """Tests for ISO 37302-style maturity level determination."""
    
    def test_determine_maturity_level_basic(self):
        """Test basic maturity level."""
        assert determine_maturity_level(0.0) == MaturityLevel.BASIC
        assert determine_maturity_level(0.39) == MaturityLevel.BASIC
    
    def test_determine_maturity_level_developing(self):
        """Test developing maturity level."""
        assert determine_maturity_level(0.4) == MaturityLevel.DEVELOPING
        assert determine_maturity_level(0.59) == MaturityLevel.DEVELOPING
    
    def test_determine_maturity_level_established(self):
        """Test established maturity level."""
        assert determine_maturity_level(0.6) == MaturityLevel.ESTABLISHED
        assert determine_maturity_level(0.79) == MaturityLevel.ESTABLISHED
    
    def test_determine_maturity_level_advanced(self):
        """Test advanced maturity level."""
        assert determine_maturity_level(0.8) == MaturityLevel.ADVANCED
        assert determine_maturity_level(0.94) == MaturityLevel.ADVANCED
    
    def test_determine_maturity_level_optimized(self):
        """Test optimized maturity level."""
        assert determine_maturity_level(0.95) == MaturityLevel.OPTIMIZED
        assert determine_maturity_level(1.0) == MaturityLevel.OPTIMIZED


# =============================================================================
# Academic Metrics Tests
# =============================================================================

class TestAcademicMetrics:
    """Tests for CUAD-style academic metrics."""
    
    def test_precision_at_recall_perfect(self):
        """Test perfect predictions."""
        predictions = [True, True, False, False]
        ground_truth = [True, True, False, False]
        
        precision = ComplianceScorer.calculate_precision_at_recall(
            predictions, ground_truth, target_recall=0.8
        )
        assert precision == 1.0
    
    def test_f1_score_calculation(self):
        """Test F1 score calculation."""
        f1 = ComplianceScorer.calculate_f1_score(0.8, 0.8)
        assert f1 == pytest.approx(0.8, rel=0.01)
        
        f1_zero = ComplianceScorer.calculate_f1_score(0.0, 0.0)
        assert f1_zero == 0.0
    
    def test_jaccard_similarity_identical(self):
        """Test Jaccard similarity for identical sets."""
        set_a = {"a", "b", "c"}
        set_b = {"a", "b", "c"}
        
        similarity = ComplianceScorer.calculate_jaccard_similarity(set_a, set_b)
        assert similarity == 1.0
    
    def test_jaccard_similarity_disjoint(self):
        """Test Jaccard similarity for disjoint sets."""
        set_a = {"a", "b"}
        set_b = {"c", "d"}
        
        similarity = ComplianceScorer.calculate_jaccard_similarity(set_a, set_b)
        assert similarity == 0.0
    
    def test_jaccard_similarity_partial(self):
        """Test Jaccard similarity for overlapping sets."""
        set_a = {"a", "b", "c"}
        set_b = {"b", "c", "d"}
        
        # Intersection: {b, c} = 2
        # Union: {a, b, c, d} = 4
        # Jaccard: 2/4 = 0.5
        similarity = ComplianceScorer.calculate_jaccard_similarity(set_a, set_b)
        assert similarity == pytest.approx(0.5, rel=0.01)


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for scoring convenience functions."""
    
    def test_quick_score(self):
        """Test quick scoring function."""
        score, risk = quick_score(0.9, 0.8, 0.7)
        
        assert 0.0 <= score <= 1.0
        assert risk in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]
    
    def test_quick_score_high_values(self):
        """Test quick score with high values produces LOW risk."""
        score, risk = quick_score(1.0, 0.9, 0.85)
        
        assert score >= 0.7
        assert risk == RiskLevel.LOW
    
    def test_quick_score_low_values(self):
        """Test quick score with low values produces HIGH risk."""
        score, risk = quick_score(0.0, 0.1, 0.2)
        
        assert score < 0.4
        assert risk == RiskLevel.HIGH
