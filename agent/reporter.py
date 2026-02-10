"""
Report Generator for Contract Compliance Agent.

Generates formatted compliance reports in multiple formats
including terminal output, Markdown, and JSON.
"""

import json
from datetime import datetime
from typing import Optional

from models.schemas import (
    AnalysisResult,
    ClauseAnalysis,
    RiskLevel,
    ClauseStatus,
    MaturityLevel
)


# =============================================================================
# Report Templates
# =============================================================================

REPORT_HEADER = """
═══════════════════════════════════════════════════════════════════════════════
                    CONTRACT COMPLIANCE ANALYSIS REPORT
═══════════════════════════════════════════════════════════════════════════════
"""

EXECUTIVE_SUMMARY_TEMPLATE = """
───────────────────────────────────────────────────────────────────────────────
                              EXECUTIVE SUMMARY
───────────────────────────────────────────────────────────────────────────────

Document: {filename}
Analysis Date: {date}
Model: {model}

Overall Compliance Score: {score:.2f} / 1.00
Risk Level: {risk_emoji} {risk_level}
Maturity Level: {maturity_level} ({maturity_name})
Clauses Analyzed: {clauses_analyzed}
Issues Identified: {issues_count}
"""

CLAUSE_SECTION_TEMPLATE = """
───────────────────────────────────────────────────────────────────────────────
                              CLAUSE ANALYSIS
───────────────────────────────────────────────────────────────────────────────
"""

RECOMMENDATIONS_TEMPLATE = """
───────────────────────────────────────────────────────────────────────────────
                              RECOMMENDATIONS
───────────────────────────────────────────────────────────────────────────────
"""


# =============================================================================
# Risk Level Formatting
# =============================================================================

RISK_EMOJI = {
    RiskLevel.LOW: "🟢",
    RiskLevel.MEDIUM: "🟡",
    RiskLevel.HIGH: "🔴"
}

RISK_TEXT = {
    RiskLevel.LOW: "LOW RISK",
    RiskLevel.MEDIUM: "MEDIUM RISK",
    RiskLevel.HIGH: "HIGH RISK"
}

STATUS_EMOJI = {
    ClauseStatus.PRESENT: "✅",
    ClauseStatus.PARTIAL: "⚠️",
    ClauseStatus.MISSING: "❌",
    ClauseStatus.NOT_APPLICABLE: "➖"
}

MATURITY_NAMES = {
    MaturityLevel.BASIC: "Basic",
    MaturityLevel.DEVELOPING: "Developing",
    MaturityLevel.ESTABLISHED: "Established",
    MaturityLevel.ADVANCED: "Advanced",
    MaturityLevel.OPTIMIZED: "Optimized"
}


# =============================================================================
# Report Generator
# =============================================================================

class ReportGenerator:
    """
    Generates formatted compliance reports from analysis results.
    """
    
    def __init__(
        self,
        include_details: bool = True,
        include_recommendations: bool = True,
        max_recommendations: int = 10
    ):
        """
        Initialize the report generator.
        
        Args:
            include_details: Whether to include detailed clause analysis
            include_recommendations: Whether to include recommendations
            max_recommendations: Maximum number of recommendations to show
        """
        self.include_details = include_details
        self.include_recommendations = include_recommendations
        self.max_recommendations = max_recommendations
    
    # -------------------------------------------------------------------------
    # Main Report Methods
    # -------------------------------------------------------------------------
    
    def generate_text_report(self, result: AnalysisResult) -> str:
        """
        Generate a formatted text report for terminal display.
        
        Args:
            result: Analysis result
            
        Returns:
            Formatted text report
        """
        parts = [REPORT_HEADER]
        
        # Executive Summary
        parts.append(self._format_executive_summary(result))
        
        # Clause Analysis
        if self.include_details:
            parts.append(self._format_clause_section(result))
        
        # Recommendations
        if self.include_recommendations and result.recommendations:
            parts.append(self._format_recommendations(result))
        
        # Footer
        parts.append("\n" + "═" * 79 + "\n")
        
        return "\n".join(parts)
    
    def generate_markdown_report(self, result: AnalysisResult) -> str:
        """
        Generate a Markdown-formatted report.
        
        Args:
            result: Analysis result
            
        Returns:
            Markdown report
        """
        lines = [
            "# Contract Compliance Analysis Report",
            "",
            f"**Document:** {result.document.filename}",
            f"**Analysis Date:** {result.analysis_timestamp.strftime('%Y-%m-%d %H:%M')}",
            f"**Model:** {result.model_used}",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Overall Score | {result.compliance_score.overall_score:.2f} / 1.00 |",
            f"| Risk Level | {RISK_EMOJI[result.compliance_score.risk_level]} {RISK_TEXT[result.compliance_score.risk_level]} |",
            f"| Maturity Level | Level {result.compliance_score.maturity_level.value} ({MATURITY_NAMES[result.compliance_score.maturity_level]}) |",
            f"| Coverage | {result.compliance_score.coverage_percentage:.1f}% |",
            f"| Critical Score | {result.compliance_score.critical_clause_score:.2f} |",
            "",
            "### Risk Summary",
            "",
            f"- 🔴 High Risk Issues: {result.compliance_score.high_risk_issues}",
            f"- 🟡 Medium Risk Issues: {result.compliance_score.medium_risk_issues}",
            f"- 🟢 Low Risk Issues: {result.compliance_score.low_risk_issues}",
            "",
        ]
        
        # Clause Details
        if self.include_details:
            lines.extend([
                "---",
                "",
                "## Clause Analysis",
                "",
            ])
            
            for analysis in result.clause_analyses:
                lines.extend(self._format_clause_markdown(analysis))
        
        # Recommendations
        if self.include_recommendations and result.recommendations:
            lines.extend([
                "---",
                "",
                "## Recommendations",
                "",
            ])
            
            for i, rec in enumerate(result.recommendations[:self.max_recommendations], 1):
                priority_emoji = RISK_EMOJI.get(rec.priority, "📌")
                lines.extend([
                    f"### {i}. {priority_emoji} {rec.title}",
                    "",
                    f"**Priority:** {rec.priority.value.upper()}",
                    f"**Clause:** {rec.clause_id}",
                    "",
                    rec.description,
                    "",
                ])
                
                if rec.action_items:
                    lines.append("**Action Items:**")
                    for item in rec.action_items:
                        lines.append(f"- {item}")
                    lines.append("")
        
        # Disclaimer
        lines.extend([
            "---",
            "",
            "## Disclaimer",
            "",
            "*This analysis is generated by an AI system for informational purposes only. ",
            "It is not legal advice and should not be relied upon for legal decisions. ",
            "Always consult with qualified legal professionals for contract review.*",
            "",
            f"*Processing time: {result.processing_time_seconds:.2f} seconds*",
        ])
        
        return "\n".join(lines)
    
    def generate_json_report(self, result: AnalysisResult) -> str:
        """
        Generate a JSON-formatted report.
        
        Args:
            result: Analysis result
            
        Returns:
            JSON string
        """
        return result.model_dump_json(indent=2)
    
    # -------------------------------------------------------------------------
    # Formatting Helpers
    # -------------------------------------------------------------------------
    
    def _format_executive_summary(self, result: AnalysisResult) -> str:
        """Format the executive summary section."""
        score = result.compliance_score
        
        return EXECUTIVE_SUMMARY_TEMPLATE.format(
            filename=result.document.filename,
            date=result.analysis_timestamp.strftime("%Y-%m-%d"),
            model=result.model_used,
            score=score.overall_score,
            risk_emoji=RISK_EMOJI[score.risk_level],
            risk_level=RISK_TEXT[score.risk_level],
            maturity_level=f"Level {score.maturity_level.value}",
            maturity_name=MATURITY_NAMES[score.maturity_level],
            clauses_analyzed=score.clauses_analyzed,
            issues_count=score.high_risk_issues + score.medium_risk_issues
        )
    
    def _format_clause_section(self, result: AnalysisResult) -> str:
        """Format the clause analysis section."""
        lines = [CLAUSE_SECTION_TEMPLATE]
        
        for analysis in result.clause_analyses:
            status_emoji = STATUS_EMOJI[analysis.status]
            risk_text = f"[{RISK_TEXT[analysis.risk_level]}]"
            
            # Main clause line
            clause_line = (
                f"{status_emoji} {analysis.clause_name:<35} "
                f"Score: {analysis.overall_score:.2f}  {risk_text}"
            )
            lines.append(clause_line)
            
            # Details for non-present clauses
            if analysis.status != ClauseStatus.PRESENT:
                if analysis.missing_elements:
                    lines.append(f"   Missing: {', '.join(analysis.missing_elements[:2])}")
                if analysis.recommendations:
                    lines.append(f"   Recommendation: {analysis.recommendations[0]}")
            else:
                if analysis.found_elements:
                    lines.append(f"   Found: {', '.join(analysis.found_elements[:2])}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_recommendations(self, result: AnalysisResult) -> str:
        """Format the recommendations section."""
        lines = [RECOMMENDATIONS_TEMPLATE]
        
        for i, rec in enumerate(result.recommendations[:self.max_recommendations], 1):
            priority_marker = {
                RiskLevel.HIGH: "[HIGH]",
                RiskLevel.MEDIUM: "[MEDIUM]",
                RiskLevel.LOW: "[LOW]"
            }.get(rec.priority, "[INFO]")
            
            lines.append(f"{i}. {priority_marker} {rec.title}")
        
        return "\n".join(lines)
    
    def _format_clause_markdown(self, analysis: ClauseAnalysis) -> list[str]:
        """Format a single clause analysis for Markdown."""
        status_emoji = STATUS_EMOJI[analysis.status]
        risk_emoji = RISK_EMOJI[analysis.risk_level]
        
        lines = [
            f"### {status_emoji} {analysis.clause_name}",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Status | {analysis.status.value.title()} |",
            f"| Overall Score | {analysis.overall_score:.2f} |",
            f"| Risk Level | {risk_emoji} {analysis.risk_level.value.upper()} |",
            f"| Confidence | {analysis.confidence:.0%} |",
            "",
        ]
        
        if analysis.extracted_text:
            # Truncate long text
            text = analysis.extracted_text[:300]
            if len(analysis.extracted_text) > 300:
                text += "..."
            lines.extend([
                "**Extracted Text:**",
                f"> {text}",
                "",
            ])
        
        if analysis.found_elements:
            lines.append("**Found Elements:** " + ", ".join(analysis.found_elements))
            lines.append("")
        
        if analysis.missing_elements:
            lines.append("**Missing Elements:** " + ", ".join(analysis.missing_elements))
            lines.append("")
        
        if analysis.issues:
            lines.append("**Issues:**")
            for issue in analysis.issues:
                lines.append(f"- {issue}")
            lines.append("")
        
        return lines


# =============================================================================
# Convenience Functions
# =============================================================================

def print_report(result: AnalysisResult, verbose: bool = False):
    """
    Print a formatted report to the terminal.
    
    Args:
        result: Analysis result
        verbose: Whether to include detailed clause analysis
    """
    generator = ReportGenerator(
        include_details=verbose,
        include_recommendations=True
    )
    print(generator.generate_text_report(result))


def save_report(
    result: AnalysisResult,
    filepath: str,
    format: str = "markdown"
):
    """
    Save a report to file.
    
    Args:
        result: Analysis result
        filepath: Output file path
        format: Report format (markdown, json, text)
    """
    generator = ReportGenerator()
    
    if format == "json":
        content = generator.generate_json_report(result)
    elif format == "text":
        content = generator.generate_text_report(result)
    else:
        content = generator.generate_markdown_report(result)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
