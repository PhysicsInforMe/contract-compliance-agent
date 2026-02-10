"""
Data models and schemas for Contract Compliance Agent.

Uses Pydantic for validation and serialization of all data structures
used throughout the analysis pipeline.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================

class RiskLevel(str, Enum):
    """Risk level classification based on compliance scores."""
    LOW = "low"           # Green - Score >= 0.7
    MEDIUM = "medium"     # Yellow - Score 0.4-0.7
    HIGH = "high"         # Red - Score < 0.4


class ClauseStatus(str, Enum):
    """Status of a clause in the analyzed contract."""
    PRESENT = "present"           # Clause found and complete
    PARTIAL = "partial"           # Clause found but incomplete
    MISSING = "missing"           # Clause not found
    NOT_APPLICABLE = "n/a"        # Clause not relevant for this contract type


class MaturityLevel(int, Enum):
    """ISO 37302-inspired compliance maturity levels."""
    BASIC = 1           # Minimal coverage
    DEVELOPING = 2      # Some elements present
    ESTABLISHED = 3     # Most elements present
    ADVANCED = 4        # Comprehensive coverage
    OPTIMIZED = 5       # Excellent with all elements


# =============================================================================
# Clause Models
# =============================================================================

class ClauseRule(BaseModel):
    """Definition of a compliance rule for a clause category."""
    
    name: str = Field(..., description="Human-readable clause name")
    id: str = Field(..., description="Unique identifier for the clause")
    required: bool = Field(default=True, description="Whether clause is required")
    risk_if_missing: RiskLevel = Field(
        default=RiskLevel.MEDIUM,
        description="Risk level if clause is missing"
    )
    description: str = Field(..., description="What this clause should cover")
    keywords: list[str] = Field(
        default_factory=list,
        description="Keywords indicating clause presence"
    )
    expected_elements: list[str] = Field(
        default_factory=list,
        description="Elements that should be included"
    )


class ClauseAnalysis(BaseModel):
    """Analysis result for a single clause category."""
    
    # Identification
    clause_id: str = Field(..., description="Clause identifier")
    clause_name: str = Field(..., description="Human-readable name")
    
    # Status and scoring
    status: ClauseStatus = Field(..., description="Clause presence status")
    presence_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Score for clause presence (0-1)"
    )
    similarity_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Score for language similarity (0-1)"
    )
    completeness_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Score for element completeness (0-1)"
    )
    overall_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Weighted overall score (0-1)"
    )
    risk_level: RiskLevel = Field(..., description="Risk classification")
    
    # Details
    extracted_text: Optional[str] = Field(
        default=None,
        description="Relevant text extracted from contract"
    )
    found_elements: list[str] = Field(
        default_factory=list,
        description="Expected elements that were found"
    )
    missing_elements: list[str] = Field(
        default_factory=list,
        description="Expected elements that were missing"
    )
    
    # Analysis
    issues: list[str] = Field(
        default_factory=list,
        description="Identified issues with this clause"
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Recommendations for improvement"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in the analysis"
    )


# =============================================================================
# Document Models
# =============================================================================

class DocumentMetadata(BaseModel):
    """Metadata about the analyzed document."""
    
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File extension/type")
    file_size: int = Field(..., description="File size in bytes")
    page_count: Optional[int] = Field(
        default=None,
        description="Number of pages (if applicable)"
    )
    word_count: int = Field(default=0, description="Approximate word count")
    character_count: int = Field(default=0, description="Character count")
    language: str = Field(default="en", description="Detected language")


class ContractType(str, Enum):
    """Common contract types for specialized analysis."""
    NDA = "nda"
    SERVICE_AGREEMENT = "service_agreement"
    EMPLOYMENT = "employment"
    SOFTWARE_LICENSE = "software_license"
    GENERAL = "general"


# =============================================================================
# Analysis Result Models
# =============================================================================

class ComplianceScore(BaseModel):
    """Overall compliance scoring results."""
    
    # Aggregate scores
    overall_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall weighted compliance score"
    )
    risk_level: RiskLevel = Field(..., description="Overall risk classification")
    maturity_level: MaturityLevel = Field(
        ...,
        description="Compliance maturity level"
    )
    
    # Component scores
    critical_clause_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score for critical clauses only"
    )
    coverage_percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of expected clauses present"
    )
    
    # Statistics
    clauses_analyzed: int = Field(..., description="Total clauses analyzed")
    clauses_present: int = Field(..., description="Clauses found present")
    clauses_partial: int = Field(..., description="Clauses partially present")
    clauses_missing: int = Field(..., description="Clauses missing")
    
    # Risk breakdown
    high_risk_issues: int = Field(default=0, description="High risk issue count")
    medium_risk_issues: int = Field(default=0, description="Medium risk issue count")
    low_risk_issues: int = Field(default=0, description="Low risk issue count")


class Recommendation(BaseModel):
    """A specific recommendation for improving compliance."""
    
    priority: RiskLevel = Field(..., description="Priority level")
    clause_id: str = Field(..., description="Related clause")
    title: str = Field(..., description="Short recommendation title")
    description: str = Field(..., description="Detailed recommendation")
    action_items: list[str] = Field(
        default_factory=list,
        description="Specific actions to take"
    )


class AnalysisResult(BaseModel):
    """Complete analysis result for a contract."""
    
    # Metadata
    analysis_id: str = Field(..., description="Unique analysis identifier")
    analysis_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When analysis was performed"
    )
    model_used: str = Field(..., description="LLM model used for analysis")
    
    # Document info
    document: DocumentMetadata = Field(..., description="Document metadata")
    contract_type: ContractType = Field(
        default=ContractType.GENERAL,
        description="Detected or specified contract type"
    )
    
    # Scoring
    compliance_score: ComplianceScore = Field(
        ...,
        description="Overall compliance scoring"
    )
    
    # Detailed analysis
    clause_analyses: list[ClauseAnalysis] = Field(
        default_factory=list,
        description="Per-clause analysis results"
    )
    
    # Recommendations
    recommendations: list[Recommendation] = Field(
        default_factory=list,
        description="Prioritized recommendations"
    )
    
    # Execution metadata
    processing_time_seconds: float = Field(
        default=0.0,
        description="Total processing time"
    )
    tokens_used: int = Field(default=0, description="LLM tokens consumed")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# LLM Interaction Models
# =============================================================================

class LLMRequest(BaseModel):
    """Request to the LLM for analysis."""
    
    prompt: str = Field(..., description="The prompt to send")
    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system prompt"
    )
    temperature: float = Field(default=0.1, description="Generation temperature")
    max_tokens: int = Field(default=4096, description="Maximum tokens")


class LLMResponse(BaseModel):
    """Response from the LLM."""
    
    content: str = Field(..., description="Generated text content")
    model: str = Field(..., description="Model that generated response")
    tokens_used: int = Field(default=0, description="Tokens consumed")
    generation_time: float = Field(default=0.0, description="Time to generate")


# =============================================================================
# Helper Functions
# =============================================================================

def determine_risk_level(score: float, thresholds: dict = None) -> RiskLevel:
    """
    Determine risk level from a score.
    
    Args:
        score: Compliance score (0-1)
        thresholds: Optional custom thresholds
        
    Returns:
        RiskLevel enum value
    """
    if thresholds is None:
        thresholds = {"low_risk": 0.7, "medium_risk": 0.4}
    
    if score >= thresholds["low_risk"]:
        return RiskLevel.LOW
    elif score >= thresholds["medium_risk"]:
        return RiskLevel.MEDIUM
    else:
        return RiskLevel.HIGH


def determine_maturity_level(score: float) -> MaturityLevel:
    """
    Determine maturity level from overall score.
    
    Args:
        score: Overall compliance score (0-1)
        
    Returns:
        MaturityLevel enum value
    """
    if score >= 0.95:
        return MaturityLevel.OPTIMIZED
    elif score >= 0.8:
        return MaturityLevel.ADVANCED
    elif score >= 0.6:
        return MaturityLevel.ESTABLISHED
    elif score >= 0.4:
        return MaturityLevel.DEVELOPING
    else:
        return MaturityLevel.BASIC
