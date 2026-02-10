"""
Integration tests for Contract Compliance Agent.

These tests require Ollama to be running with a model available.
Run with: pytest tests/test_integration.py -v --integration
"""

import pytest
from pathlib import Path

from agent import ComplianceAgent
from models import RiskLevel, ContractType


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_nda_text():
    """Sample NDA contract text for testing."""
    return """
    NON-DISCLOSURE AGREEMENT
    
    This Non-Disclosure Agreement ("Agreement") is entered into as of January 1, 2025
    by and between Company A ("Disclosing Party") and Company B ("Receiving Party").
    
    1. CONFIDENTIAL INFORMATION
    "Confidential Information" means any information disclosed by the Disclosing Party
    to the Receiving Party, either directly or indirectly, in writing, orally, or by
    inspection of tangible objects, that is designated as "Confidential" or would
    reasonably be understood to be confidential given the nature of the information.
    
    2. OBLIGATIONS
    The Receiving Party agrees to:
    (a) Hold the Confidential Information in strict confidence;
    (b) Not disclose Confidential Information to any third parties;
    (c) Use Confidential Information only for the purpose of evaluating a potential
        business relationship between the parties.
    
    3. TERM
    This Agreement shall remain in effect for a period of two (2) years from the
    date of execution.
    
    4. RETURN OF INFORMATION
    Upon termination of this Agreement, the Receiving Party shall return or destroy
    all Confidential Information and any copies thereof.
    
    5. GOVERNING LAW
    This Agreement shall be governed by and construed in accordance with the laws
    of the State of Delaware.
    
    IN WITNESS WHEREOF, the parties have executed this Agreement as of the date
    first written above.
    """


@pytest.fixture
def incomplete_contract_text():
    """Incomplete contract for testing missing clause detection."""
    return """
    SERVICE AGREEMENT
    
    This Agreement is between Provider and Client.
    
    1. SERVICES
    Provider agrees to provide consulting services to Client.
    
    2. FEES
    Client shall pay Provider $10,000 per month.
    
    [Note: This contract is intentionally incomplete for testing]
    """


# =============================================================================
# Integration Tests (Require Ollama)
# =============================================================================

@pytest.mark.integration
class TestOllamaIntegration:
    """Integration tests requiring Ollama."""
    
    def test_agent_initialization(self):
        """Test agent can be initialized."""
        agent = ComplianceAgent(verbose=False)
        assert agent is not None
        agent.close()
    
    def test_ollama_status_check(self):
        """Test Ollama status checking."""
        agent = ComplianceAgent()
        status = agent.check_status()
        
        assert "available" in status
        assert "models" in status
        agent.close()
    
    @pytest.mark.skipif(
        not Path("/tmp/ollama_available").exists(),
        reason="Ollama not available"
    )
    def test_analyze_nda(self, sample_nda_text):
        """Test analyzing an NDA contract."""
        agent = ComplianceAgent(verbose=False)
        
        result = agent.analyze_text(
            sample_nda_text,
            filename="test_nda.txt",
            contract_type=ContractType.NDA
        )
        
        assert result is not None
        assert result.compliance_score is not None
        assert len(result.clause_analyses) > 0
        
        agent.close()
    
    @pytest.mark.skipif(
        not Path("/tmp/ollama_available").exists(),
        reason="Ollama not available"
    )
    def test_analyze_incomplete_contract(self, incomplete_contract_text):
        """Test analyzing incomplete contract detects missing clauses."""
        agent = ComplianceAgent(verbose=False)
        
        result = agent.analyze_text(
            incomplete_contract_text,
            filename="incomplete.txt"
        )
        
        # Should detect missing clauses
        assert result.compliance_score.clauses_missing > 0
        
        # Risk should not be LOW due to missing clauses
        assert result.compliance_score.risk_level != RiskLevel.LOW
        
        agent.close()


# =============================================================================
# Offline Tests (No Ollama Required)
# =============================================================================

class TestOffline:
    """Tests that don't require Ollama."""
    
    def test_agent_config_loading(self):
        """Test configuration is loaded correctly."""
        agent = ComplianceAgent(verbose=False)
        
        assert agent.config is not None
        assert len(agent.clause_rules) > 0
        
        agent.close()
    
    def test_clause_rules_structure(self):
        """Test clause rules have required fields."""
        agent = ComplianceAgent()
        
        for rule in agent.clause_rules:
            assert rule.id is not None
            assert rule.name is not None
            assert rule.description is not None
            assert isinstance(rule.keywords, list)
            assert isinstance(rule.expected_elements, list)
        
        agent.close()
    
    def test_context_manager(self):
        """Test agent works as context manager."""
        with ComplianceAgent() as agent:
            assert agent is not None
            status = agent.check_status()
            assert "rules_loaded" in status
    
    def test_custom_model_config(self):
        """Test custom model configuration."""
        agent = ComplianceAgent(model="qwen2.5:7b")
        
        assert agent.config.ollama.model == "qwen2.5:7b"
        
        agent.close()


# =============================================================================
# Document Loading Tests
# =============================================================================

class TestDocumentLoading:
    """Tests for document loading functionality."""
    
    def test_loader_supported_formats(self):
        """Test document loader reports supported formats."""
        agent = ComplianceAgent()
        formats = agent.loader.get_supported_formats()
        
        assert formats[".txt"] is True
        assert formats[".md"] is True
        # PDF and DOCX depend on optional dependencies
        
        agent.close()
    
    def test_load_text_file(self, tmp_path, sample_nda_text):
        """Test loading a text file."""
        # Create temp file
        test_file = tmp_path / "test_contract.txt"
        test_file.write_text(sample_nda_text)
        
        agent = ComplianceAgent()
        text = agent.loader.load_text(test_file)
        
        assert "NON-DISCLOSURE AGREEMENT" in text
        assert len(text) > 0
        
        agent.close()


# =============================================================================
# Report Generation Tests
# =============================================================================

class TestReportGeneration:
    """Tests for report generation."""
    
    def test_report_generator_markdown(self):
        """Test Markdown report generation."""
        from agent.reporter import ReportGenerator
        from models import (
            AnalysisResult, DocumentMetadata, ComplianceScore,
            ClauseAnalysis, ClauseStatus, MaturityLevel
        )
        from datetime import datetime
        
        # Create minimal test data
        result = AnalysisResult(
            analysis_id="test-123",
            model_used="test-model",
            document=DocumentMetadata(
                filename="test.txt",
                file_type=".txt",
                file_size=1000,
                word_count=500,
                character_count=3000
            ),
            compliance_score=ComplianceScore(
                overall_score=0.75,
                risk_level=RiskLevel.LOW,
                maturity_level=MaturityLevel.ESTABLISHED,
                critical_clause_score=0.8,
                coverage_percentage=80.0,
                clauses_analyzed=5,
                clauses_present=3,
                clauses_partial=1,
                clauses_missing=1
            ),
            clause_analyses=[
                ClauseAnalysis(
                    clause_id="test",
                    clause_name="Test Clause",
                    status=ClauseStatus.PRESENT,
                    presence_score=0.9,
                    similarity_score=0.8,
                    completeness_score=0.7,
                    overall_score=0.8,
                    risk_level=RiskLevel.LOW
                )
            ],
            recommendations=[]
        )
        
        generator = ReportGenerator()
        markdown = generator.generate_markdown_report(result)
        
        assert "# Contract Compliance Analysis Report" in markdown
        assert "test.txt" in markdown
        assert "0.75" in markdown
    
    def test_report_generator_json(self):
        """Test JSON report generation."""
        from agent.reporter import ReportGenerator
        from models import (
            AnalysisResult, DocumentMetadata, ComplianceScore,
            MaturityLevel
        )
        import json
        
        result = AnalysisResult(
            analysis_id="test-123",
            model_used="test-model",
            document=DocumentMetadata(
                filename="test.txt",
                file_type=".txt",
                file_size=1000,
                word_count=500,
                character_count=3000
            ),
            compliance_score=ComplianceScore(
                overall_score=0.75,
                risk_level=RiskLevel.LOW,
                maturity_level=MaturityLevel.ESTABLISHED,
                critical_clause_score=0.8,
                coverage_percentage=80.0,
                clauses_analyzed=5,
                clauses_present=3,
                clauses_partial=1,
                clauses_missing=1
            ),
            clause_analyses=[],
            recommendations=[]
        )
        
        generator = ReportGenerator()
        json_str = generator.generate_json_report(result)
        
        # Should be valid JSON
        data = json.loads(json_str)
        assert data["analysis_id"] == "test-123"
        assert data["compliance_score"]["overall_score"] == 0.75


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require Ollama)"
    )
