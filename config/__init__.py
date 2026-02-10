"""
Configuration package for Contract Compliance Agent.
"""

from config.settings import (
    AgentConfig,
    OllamaConfig,
    ScoringConfig,
    get_config,
    load_rules,
    load_weights,
    load_yaml_config,
    PROJECT_ROOT,
    SAMPLES_DIR,
)

__all__ = [
    "AgentConfig",
    "OllamaConfig",
    "ScoringConfig",
    "get_config",
    "load_rules",
    "load_weights",
    "load_yaml_config",
    "PROJECT_ROOT",
    "SAMPLES_DIR",
]
