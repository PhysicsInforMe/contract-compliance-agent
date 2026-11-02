"""
Core Compliance Agent for Contract Compliance Review.

This is the main entry point for the compliance analysis system,
providing a high-level interface for contract analysis.
"""

from pathlib import Path
from typing import Optional, Union

from config.settings import AgentConfig, get_config, load_rules, load_weights
from tools.llm_client import OllamaClient, check_ollama_status, model_matches
from tools.document_loader import DocumentLoader
from agent.analyzer import ContractAnalyzer
from agent.scorer import apply_contract_type_modifiers
from agent.reporter import ReportGenerator, print_report, save_report
from models.schemas import (
    AnalysisResult,
    ClauseRule,
    ContractType,
    DocumentMetadata
)


# =============================================================================
# Compliance Agent
# =============================================================================

class ComplianceAgent:
    """
    Main Contract Compliance Review Agent.
    
    Provides a simple interface for analyzing contracts against
    configurable compliance rules and generating detailed reports.
    
    Example:
        >>> agent = ComplianceAgent()
        >>> result = agent.analyze("contract.pdf")
        >>> print(f"Score: {result.compliance_score.overall_score}")
        >>> agent.save_report(result, "report.md")
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        rules_path: Optional[Path] = None,
        config: Optional[AgentConfig] = None,
        verbose: bool = False
    ):
        """
        Initialize the Compliance Agent.
        
        Args:
            model: LLM model to use (e.g., "llama3.1:8b", "qwen2.5:7b")
            rules_path: Path to custom compliance rules YAML
            config: Optional full configuration object
            verbose: Enable verbose output
        """
        # Load configuration (explicit arguments override a supplied config)
        if config is None:
            config = get_config(model=model, rules_path=rules_path, verbose=verbose)
        else:
            if model:
                config.ollama.model = model
            if rules_path:
                config.rules_path = Path(rules_path)
            if verbose:
                config.verbose = True
        self.config = config
        self.verbose = verbose or config.verbose

        # Load compliance rules
        self._rules_data = load_rules(self.config.rules_path)
        self._weights_data = load_weights(self.config.weights_path)

        # Parse clause rules and scoring configuration
        self.clause_rules = self._parse_clause_rules()
        self.clause_weights = self._weights_data.get("weights", {}) or {}
        self.contract_type_modifiers = self._weights_data.get("contract_type_modifiers", {}) or {}
        self.component_weights, self.risk_thresholds = self._parse_scoring_config()

        # Initialize components (lazy loading)
        self._llm_client: Optional[OllamaClient] = None
        self._document_loader: Optional[DocumentLoader] = None
        self._analyzers: dict[tuple, ContractAnalyzer] = {}
        self._reporter: Optional[ReportGenerator] = None
    
    # -------------------------------------------------------------------------
    # Properties (Lazy Initialization)
    # -------------------------------------------------------------------------
    
    @property
    def llm(self) -> OllamaClient:
        """Get or create the LLM client."""
        if self._llm_client is None:
            self._llm_client = OllamaClient(
                base_url=self.config.ollama.base_url,
                model=self.config.ollama.model,
                timeout=self.config.ollama.timeout,
                max_retries=self.config.max_retries
            )
        return self._llm_client
    
    @property
    def loader(self) -> DocumentLoader:
        """Get or create the document loader."""
        if self._document_loader is None:
            self._document_loader = DocumentLoader()
        return self._document_loader
    
    @property
    def analyzer(self) -> ContractAnalyzer:
        """Get or create the default (general contract type) analyzer."""
        return self._get_analyzer(ContractType.GENERAL, None)

    def _get_analyzer(
        self,
        contract_type: ContractType,
        focus_clauses: Optional[list[str]]
    ) -> ContractAnalyzer:
        """
        Get or create an analyzer for a contract type and clause subset.

        Clause weights are adjusted with the ``contract_type_modifiers``
        from the weights configuration (e.g. confidentiality counts more
        in an NDA).
        """
        key = (contract_type, tuple(focus_clauses) if focus_clauses else None)
        if key not in self._analyzers:
            rules = self.clause_rules
            if focus_clauses:
                wanted = set(focus_clauses)
                rules = [r for r in self.clause_rules if r.id in wanted]

            weights = apply_contract_type_modifiers(
                self.clause_weights,
                self.contract_type_modifiers.get(contract_type.value)
            )
            self._analyzers[key] = ContractAnalyzer(
                llm_client=self.llm,
                clause_rules=rules,
                clause_weights=weights,
                component_weights=self.component_weights,
                risk_thresholds=self.risk_thresholds
            )
        return self._analyzers[key]
    
    @property
    def reporter(self) -> ReportGenerator:
        """Get or create the report generator."""
        if self._reporter is None:
            report_config = self._rules_data.get("report", {})
            self._reporter = ReportGenerator(
                include_details=report_config.get("include_details", True),
                include_recommendations=report_config.get("include_recommendations", True),
                max_recommendations=report_config.get("max_recommendations", 10)
            )
        return self._reporter
    
    # -------------------------------------------------------------------------
    # Main Analysis Methods
    # -------------------------------------------------------------------------
    
    def analyze(
        self,
        contract_path: Union[str, Path],
        contract_type: ContractType = ContractType.GENERAL,
        focus_clauses: Optional[list[str]] = None
    ) -> AnalysisResult:
        """
        Analyze a contract for compliance.
        
        Args:
            contract_path: Path to the contract file (PDF, DOCX, or TXT)
            contract_type: Type of contract for specialized analysis
            focus_clauses: Optional list of clause IDs to focus on
            
        Returns:
            AnalysisResult with comprehensive compliance assessment
            
        Raises:
            FileNotFoundError: If contract file doesn't exist
            ValueError: If file type is not supported
            ConnectionError: If Ollama is not available
        """
        # Validate Ollama availability
        if not self.llm.is_available():
            raise ConnectionError(
                "Ollama is not available. Please ensure Ollama is running: "
                "https://ollama.ai/download"
            )
        
        # Check model availability
        if not self.llm.model_exists():
            raise ConnectionError(
                f"Model '{self.config.ollama.model}' not found. "
                f"Pull it with: ollama pull {self.config.ollama.model}"
            )
        
        if self.verbose:
            print(f"Loading document: {contract_path}")
        
        # Load document
        contract_text, metadata = self.loader.load(contract_path)
        
        if self.verbose:
            print(f"Document loaded: {metadata.word_count} words")
            print(f"Analyzing with model: {self.config.ollama.model}")
        
        # Run analysis with the analyzer for this contract type / clause subset
        analyzer = self._get_analyzer(contract_type, focus_clauses)
        result = analyzer.analyze(
            contract_text=contract_text,
            document_metadata=metadata,
            contract_type=contract_type
        )
        
        if self.verbose:
            print(f"Analysis complete in {result.processing_time_seconds:.2f}s")
        
        return result
    
    def analyze_text(
        self,
        contract_text: str,
        filename: str = "inline_contract.txt",
        contract_type: ContractType = ContractType.GENERAL
    ) -> AnalysisResult:
        """
        Analyze contract text directly (without file loading).
        
        Args:
            contract_text: The contract text to analyze
            filename: Name to use in the report
            contract_type: Type of contract
            
        Returns:
            AnalysisResult with compliance assessment
        """
        # Create metadata for inline text
        metadata = DocumentMetadata(
            filename=filename,
            file_type=".txt",
            file_size=len(contract_text.encode()),
            word_count=len(contract_text.split()),
            character_count=len(contract_text)
        )
        
        return self._get_analyzer(contract_type, None).analyze(
            contract_text=contract_text,
            document_metadata=metadata,
            contract_type=contract_type
        )
    
    # -------------------------------------------------------------------------
    # Reporting Methods
    # -------------------------------------------------------------------------
    
    def print_report(self, result: AnalysisResult):
        """
        Print a formatted report to the terminal.
        
        Args:
            result: Analysis result to report
        """
        print_report(result, verbose=self.verbose)
    
    def save_report(
        self,
        result: AnalysisResult,
        filepath: Union[str, Path],
        format: str = "markdown"
    ):
        """
        Save a report to file.
        
        Args:
            result: Analysis result
            filepath: Output file path
            format: Report format ("markdown", "json", "text")
        """
        save_report(result, str(filepath), format)
        
        if self.verbose:
            print(f"Report saved to: {filepath}")
    
    def generate_batch_report(
        self,
        results: list[dict],
        filepath: Union[str, Path]
    ):
        """
        Generate a summary report for multiple contract analyses.
        
        Args:
            results: List of dicts with file, score, risk info
            filepath: Output file path
        """
        lines = [
            "# Batch Contract Analysis Summary",
            "",
            "| File | Score | Risk Level |",
            "|------|-------|------------|",
        ]
        
        for r in results:
            lines.append(f"| {r['file']} | {r['score']:.2f} | {r['risk']} |")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    
    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    
    def _parse_clause_rules(self) -> list[ClauseRule]:
        """Parse clause rules from configuration."""
        rules = []
        
        for clause_data in self._rules_data.get("clauses", []):
            rule = ClauseRule(
                name=clause_data["name"],
                id=clause_data["id"],
                required=clause_data.get("required", True),
                risk_if_missing=clause_data.get("risk_if_missing", "medium"),
                description=clause_data.get("description", ""),
                keywords=clause_data.get("keywords", []),
                expected_elements=clause_data.get("expected_elements", [])
            )
            rules.append(rule)
        
        return rules

    def _parse_scoring_config(self) -> tuple[Optional[dict], Optional[dict]]:
        """
        Read component weights and risk thresholds from the rules file.

        The ``scoring`` section of the rules YAML takes precedence; the
        ``ScoringConfig`` from the agent configuration is used otherwise.
        """
        scoring = self._rules_data.get("scoring") or {}
        component_weights = scoring.get("component_weights") or {
            "presence": self.config.scoring.presence_weight,
            "similarity": self.config.scoring.similarity_weight,
            "completeness": self.config.scoring.completeness_weight,
        }
        thresholds = scoring.get("thresholds") or {
            "low_risk": self.config.scoring.medium_risk_threshold,
            "medium_risk": self.config.scoring.high_risk_threshold,
        }
        return component_weights, thresholds

    def check_status(self) -> dict:
        """
        Check the status of the agent and its dependencies.
        
        Returns:
            Dict with status information
        """
        status = check_ollama_status(self.config.ollama.base_url)
        status["configured_model"] = self.config.ollama.model
        status["model_available"] = model_matches(
            self.config.ollama.model, status.get("models", [])
        )
        status["rules_loaded"] = len(self.clause_rules)
        return status
    
    def close(self):
        """Close connections and release resources."""
        if self._llm_client:
            self._llm_client.close()
            self._llm_client = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_analyze(
    contract_path: Union[str, Path],
    model: str = "llama3.1:8b"
) -> AnalysisResult:
    """
    Quick one-shot contract analysis.
    
    Args:
        contract_path: Path to contract file
        model: LLM model to use
        
    Returns:
        Analysis result
    """
    with ComplianceAgent(model=model) as agent:
        return agent.analyze(contract_path)
