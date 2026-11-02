"""
Tests for clause extraction, pre-filtering and the analysis pipeline.

These tests use a fake LLM client so they run without Ollama.
"""

import json

import pytest

from agent.analyzer import ContractAnalyzer
from agent.extractor import ClauseExtractor, KeywordPreFilter
from agent.scorer import ComplianceScorer, apply_contract_type_modifiers
from agent import ComplianceAgent
from models.schemas import (
    ClauseRule,
    ClauseStatus,
    ContractType,
    DocumentMetadata,
    LLMResponse,
    RiskLevel,
)
from tools.llm_client import model_matches


# =============================================================================
# Fakes and fixtures
# =============================================================================

class FakeLLM:
    """Minimal stand-in for OllamaClient returning canned JSON."""

    def __init__(self, responses):
        self.model = "fake-model"
        self._responses = list(responses)
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        content = self._responses.pop(0) if self._responses else "{}"
        if not isinstance(content, str):
            content = json.dumps(content)
        return LLMResponse(
            content=content,
            model=self.model,
            tokens_used=120,
            prompt_tokens=100,
            completion_tokens=20,
        )


@pytest.fixture
def rule():
    return ClauseRule(
        name="Termination Rights",
        id="termination",
        description="Conditions under which either party can terminate",
        keywords=["termination", "terminate", "notice period"],
        expected_elements=["Notice period", "Effect of termination"],
    )


@pytest.fixture
def metadata():
    return DocumentMetadata(
        filename="t.txt", file_type=".txt", file_size=10, word_count=2, character_count=10
    )


# =============================================================================
# KeywordPreFilter
# =============================================================================

class TestKeywordPreFilter:

    def test_sections_merge_overlaps_and_keep_order(self):
        text = "Intro sentence. Termination may occur with notice. Unrelated. " * 3
        sections = KeywordPreFilter().find_relevant_sections(
            text, ["termination", "notice"], context_chars=10
        )
        assert sections
        # Each keyword hit is inside a returned section, in document order
        positions = [text.find(s) for s in sections]
        assert positions == sorted(positions)
        for s in sections:
            assert "termination" in s.lower() or "notice" in s.lower()

    def test_focused_excerpt_keeps_middle_of_long_document(self):
        filler = "Lorem ipsum dolor sit amet. " * 400  # ~11k chars
        clause = "Either party may terminate this Agreement with thirty days notice period."
        text = filler + clause + filler
        excerpt = KeywordPreFilter().build_focused_excerpt(
            text, ["terminate", "notice period"], max_length=2000
        )
        assert len(excerpt) <= 2000 + 50
        assert "terminate this Agreement" in excerpt

    def test_focused_excerpt_falls_back_to_head_and_tail(self):
        text = "A" * 5000 + "B" * 5000
        excerpt = KeywordPreFilter().build_focused_excerpt(text, ["zzz"], max_length=1000)
        assert excerpt.startswith("A" * 500)
        assert excerpt.endswith("B" * 500)
        assert "truncated" in excerpt

    def test_short_text_returned_unchanged(self):
        text = "short contract"
        assert KeywordPreFilter().build_focused_excerpt(text, ["x"], 100) == text


# =============================================================================
# ClauseExtractor normalisation
# =============================================================================

class TestClauseExtractorNormalisation:

    def test_valid_response(self, rule):
        llm = FakeLLM([{
            "found": True, "status": "present",
            "extracted_text": "Either party may terminate with 30 days notice.",
            "found_elements": ["Notice period"], "missing_elements": ["Effect of termination"],
            "confidence": 0.9, "issues": [], "notes": "ok",
        }])
        result = ClauseExtractor(llm).extract_clause("text", rule)
        assert result["status"] == ClauseStatus.PRESENT
        assert result["confidence"] == 0.9
        assert result["tokens_used"] == 120
        assert result["truncated"] is False

    def test_null_and_wrong_types_do_not_crash(self, rule):
        llm = FakeLLM([{
            "found": "yes", "status": None, "extracted_text": 42,
            "found_elements": None, "missing_elements": "Notice period",
            "confidence": "high", "issues": None, "notes": None,
        }])
        result = ClauseExtractor(llm).extract_clause("text", rule)
        assert result["status"] in (ClauseStatus.PARTIAL, ClauseStatus.PRESENT)
        assert result["extracted_text"] == "42"
        assert result["found_elements"] == []
        assert result["missing_elements"] == ["Notice period"]
        assert result["confidence"] == 0.0
        assert result["issues"] == []

    def test_confidence_is_clamped(self, rule):
        llm = FakeLLM([{"status": "present", "extracted_text": "x", "confidence": 7}])
        assert ClauseExtractor(llm).extract_clause("text", rule)["confidence"] == 1.0

    def test_not_applicable_variants(self, rule):
        for status in ("n/a", "NA", "not applicable", "Not_Applicable"):
            llm = FakeLLM([{"status": status}])
            result = ClauseExtractor(llm).extract_clause("text", rule)
            assert result["status"] == ClauseStatus.NOT_APPLICABLE

    def test_missing_elements_default_excludes_found(self, rule):
        llm = FakeLLM([{"status": "partial", "extracted_text": "x", "found_elements": ["Notice period"]}])
        result = ClauseExtractor(llm).extract_clause("text", rule)
        assert result["missing_elements"] == ["Effect of termination"]

    def test_json_embedded_in_prose(self, rule):
        llm = FakeLLM(['Sure! Here is the result:\n{"status": "missing", "found": false}\nThanks'])
        result = ClauseExtractor(llm).extract_clause("text", rule)
        assert result["status"] == ClauseStatus.MISSING

    def test_json_array_response_is_handled(self, rule):
        llm = FakeLLM(['[1, 2, 3]'])
        result = ClauseExtractor(llm).extract_clause("text", rule)
        assert result["status"] == ClauseStatus.MISSING
        assert "error" in result

    def test_long_contract_uses_prefilter(self, rule):
        llm = FakeLLM([{"status": "present", "extracted_text": "x", "confidence": 1}])
        filler = "Lorem ipsum dolor sit amet. " * 400
        clause = "The Client may terminate this Agreement with a notice period of 30 days."
        text = filler + clause + filler
        result = ClauseExtractor(llm, max_text_length=3000).extract_clause(text, rule)
        assert result["truncated"] is True
        assert clause in llm.prompts[0]


# =============================================================================
# ContractAnalyzer
# =============================================================================

class TestContractAnalyzer:

    def test_tokens_are_summed_from_llm_responses(self, rule, metadata):
        llm = FakeLLM([
            {"status": "present", "extracted_text": "Either party may terminate with notice.", "confidence": 0.8,
             "found_elements": ["Notice period"], "missing_elements": []},
            {"similarity_score": 0.7},
        ])
        result = ContractAnalyzer(llm, [rule]).analyze("text", metadata)
        assert result.tokens_used == 240
        assert result.clause_analyses[0].similarity_score == 0.7

    def test_similarity_is_clamped(self, rule, metadata):
        llm = FakeLLM([
            {"status": "present", "extracted_text": "Either party may terminate with notice.", "confidence": 0.8},
            {"similarity_score": 3.5},
        ])
        result = ContractAnalyzer(llm, [rule]).analyze("text", metadata)
        assert result.clause_analyses[0].similarity_score == 1.0

    def test_missing_clause_has_zero_completeness(self, rule, metadata):
        llm = FakeLLM([{"status": "missing", "found_elements": [], "missing_elements": []}])
        result = ContractAnalyzer(llm, [rule]).analyze("text", metadata)
        analysis = result.clause_analyses[0]
        assert analysis.completeness_score == 0.0
        assert analysis.overall_score == 0.0
        assert analysis.risk_level == RiskLevel.HIGH

    def test_not_applicable_clause_excluded_from_document_score(self, metadata):
        rules = [
            ClauseRule(name="A", id="a", description="d"),
            ClauseRule(name="B", id="b", description="d"),
        ]
        llm = FakeLLM([
            {"status": "missing"},
            {"status": "n/a"},
        ])
        result = ContractAnalyzer(llm, rules).analyze("text", metadata)
        score = result.compliance_score
        assert score.clauses_analyzed == 2
        assert score.clauses_not_applicable == 1
        assert score.clauses_missing == 1
        assert score.overall_score == 0.0
        assert score.coverage_percentage == 0.0
        assert all(r.clause_id != "b" for r in result.recommendations)

    def test_custom_scoring_config_is_used(self, rule, metadata):
        llm = FakeLLM([{"status": "present", "extracted_text": "short", "confidence": 1.0,
                        "found_elements": ["Notice period", "Effect of termination"]}])
        analyzer = ContractAnalyzer(
            llm, [rule],
            component_weights={"presence": 1.0, "similarity": 0.0, "completeness": 0.0},
            risk_thresholds={"low_risk": 0.99, "medium_risk": 0.5},
        )
        result = analyzer.analyze("text", metadata)
        analysis = result.clause_analyses[0]
        assert analysis.overall_score == pytest.approx(1.0)
        assert analysis.risk_level == RiskLevel.LOW


# =============================================================================
# Scorer helpers
# =============================================================================

class TestScorerHelpers:

    def test_contract_type_modifiers_scale_and_clamp(self):
        base = {"confidentiality": 0.8, "payment": 0.7}
        adjusted = apply_contract_type_modifiers(base, {"confidentiality": 1.5, "payment": 0.5, "new": 2})
        assert adjusted["confidentiality"] == 1.0
        assert adjusted["payment"] == pytest.approx(0.35)
        assert adjusted["new"] == 1.0  # default weight 0.5 * 2
        assert base == {"confidentiality": 0.8, "payment": 0.7}  # not mutated

    def test_partial_weight_override_is_merged(self):
        scorer = ComplianceScorer(weights={"presence": 0.5})
        assert set(scorer.weights) == {"presence", "similarity", "completeness"}
        assert sum(scorer.weights.values()) == pytest.approx(1.0)

    def test_zero_weights_rejected(self):
        with pytest.raises(ValueError):
            ComplianceScorer(weights={"presence": 0, "similarity": 0, "completeness": 0})


# =============================================================================
# ComplianceAgent wiring (offline)
# =============================================================================

class TestAgentWiring:

    def test_nda_modifiers_change_weights(self):
        with ComplianceAgent() as agent:
            general = agent._get_analyzer(ContractType.GENERAL, None)
            nda = agent._get_analyzer(ContractType.NDA, None)
            assert nda.scorer.clause_weights["confidentiality"] > general.scorer.clause_weights["confidentiality"]
            assert nda.scorer.clause_weights["payment"] < general.scorer.clause_weights["payment"]

    def test_scoring_section_from_rules_yaml(self):
        with ComplianceAgent() as agent:
            analyzer = agent._get_analyzer(ContractType.GENERAL, None)
            assert analyzer.scorer.thresholds["low_risk"] == pytest.approx(0.7)
            assert analyzer.scorer.weights["similarity"] == pytest.approx(0.4)

    def test_focus_clauses_filter_rules(self):
        with ComplianceAgent() as agent:
            analyzer = agent._get_analyzer(ContractType.GENERAL, ["termination", "payment"])
            assert {r.id for r in analyzer.clause_rules} == {"termination", "payment"}


# =============================================================================
# Model name matching
# =============================================================================

@pytest.mark.parametrize("requested,available,expected", [
    ("llama3.1:8b", ["llama3.1:8b"], True),
    ("llama3.1", ["llama3.1:8b"], True),
    ("llama3.1:8b", ["llama3.1:latest"], True),
    ("llama3.1:8b", ["llama3.1:70b"], False),
    ("qwen2.5:7b", ["llama3.1:8b"], False),
])
def test_model_matches(requested, available, expected):
    assert model_matches(requested, available) is expected
