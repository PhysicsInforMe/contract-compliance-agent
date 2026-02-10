"""
Contract Compliance Agent package.

Provides AI-powered contract compliance analysis using local LLMs.
"""

from agent.core import ComplianceAgent, quick_analyze
from agent.analyzer import ContractAnalyzer
from agent.scorer import ComplianceScorer
from agent.extractor import ClauseExtractor
from agent.reporter import ReportGenerator, print_report, save_report

__all__ = [
    "ComplianceAgent",
    "quick_analyze",
    "ContractAnalyzer",
    "ComplianceScorer",
    "ClauseExtractor",
    "ReportGenerator",
    "print_report",
    "save_report",
]

__version__ = "0.1.0"
