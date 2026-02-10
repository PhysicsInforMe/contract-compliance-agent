"""
Configuration management for Contract Compliance Agent.

Handles loading and validating configuration from environment variables,
config files, and command-line arguments.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
import yaml


# =============================================================================
# Configuration Models
# =============================================================================

class OllamaConfig(BaseModel):
    """Configuration for Ollama LLM connection."""
    
    # Base URL for Ollama API (default: local instance)
    base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL"
    )
    
    # Default model to use for analysis
    model: str = Field(
        default="llama3.1:8b",
        description="LLM model identifier"
    )
    
    # Request timeout in seconds
    timeout: int = Field(
        default=120,
        description="Request timeout in seconds"
    )
    
    # Maximum tokens for generation
    max_tokens: int = Field(
        default=4096,
        description="Maximum tokens to generate"
    )
    
    # Temperature for generation (0.0 = deterministic)
    temperature: float = Field(
        default=0.1,
        description="Generation temperature (lower = more deterministic)"
    )


class ScoringConfig(BaseModel):
    """Configuration for compliance scoring calculations."""
    
    # Weight for clause presence in final score
    presence_weight: float = Field(
        default=0.3,
        description="Weight for clause presence (0-1)"
    )
    
    # Weight for similarity to expected language
    similarity_weight: float = Field(
        default=0.4,
        description="Weight for language similarity (0-1)"
    )
    
    # Weight for clause completeness
    completeness_weight: float = Field(
        default=0.3,
        description="Weight for clause completeness (0-1)"
    )
    
    # Risk level thresholds
    high_risk_threshold: float = Field(
        default=0.4,
        description="Score below this = HIGH RISK (Red)"
    )
    
    medium_risk_threshold: float = Field(
        default=0.7,
        description="Score below this (but above high) = MEDIUM RISK (Yellow)"
    )


class AgentConfig(BaseModel):
    """Main agent configuration."""
    
    # Ollama settings
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    
    # Scoring settings
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    
    # Path to compliance rules file
    rules_path: Path = Field(
        default=Path(__file__).parent / "default_rules.yaml",
        description="Path to compliance rules YAML"
    )
    
    # Path to clause weights file
    weights_path: Path = Field(
        default=Path(__file__).parent / "clause_weights.yaml",
        description="Path to clause weights YAML"
    )
    
    # Enable verbose output
    verbose: bool = Field(
        default=False,
        description="Enable verbose logging"
    )
    
    # Maximum retries for LLM calls
    max_retries: int = Field(
        default=3,
        description="Maximum retries for failed LLM calls"
    )


# =============================================================================
# Configuration Loading Functions
# =============================================================================

def load_yaml_config(path: Path) -> dict:
    """
    Load configuration from a YAML file.
    
    Args:
        path: Path to the YAML file
        
    Returns:
        Dictionary with configuration data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_rules(path: Optional[Path] = None) -> dict:
    """
    Load compliance rules from YAML file.
    
    Args:
        path: Optional path to rules file. Uses default if not provided.
        
    Returns:
        Dictionary containing compliance rules
    """
    if path is None:
        path = Path(__file__).parent / "default_rules.yaml"
    
    return load_yaml_config(path)


def load_weights(path: Optional[Path] = None) -> dict:
    """
    Load clause weights from YAML file.
    
    Args:
        path: Optional path to weights file. Uses default if not provided.
        
    Returns:
        Dictionary containing clause weights
    """
    if path is None:
        path = Path(__file__).parent / "clause_weights.yaml"
    
    return load_yaml_config(path)


def get_config(
    model: Optional[str] = None,
    rules_path: Optional[Path] = None,
    verbose: bool = False
) -> AgentConfig:
    """
    Get agent configuration with optional overrides.
    
    Args:
        model: Override default model
        rules_path: Override default rules path
        verbose: Enable verbose mode
        
    Returns:
        Configured AgentConfig instance
    """
    config = AgentConfig(verbose=verbose)
    
    # Apply environment variable overrides
    if env_url := os.getenv("OLLAMA_BASE_URL"):
        config.ollama.base_url = env_url
    
    if env_model := os.getenv("OLLAMA_MODEL"):
        config.ollama.model = env_model
    
    # Apply explicit overrides
    if model:
        config.ollama.model = model
    
    if rules_path:
        config.rules_path = rules_path
    
    return config


# =============================================================================
# Project Paths
# =============================================================================

# Root directory of the project
PROJECT_ROOT = Path(__file__).parent.parent

# Directory for sample contracts
SAMPLES_DIR = PROJECT_ROOT / "samples"

# Directory for test expected results
EXPECTED_RESULTS_DIR = SAMPLES_DIR / "expected_results"


# =============================================================================
# Default Configuration Instance
# =============================================================================

# Global default configuration (can be overridden)
default_config = AgentConfig()
