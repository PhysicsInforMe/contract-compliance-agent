"""
Clause Extractor for Contract Compliance Agent.

Uses LLM-based analysis to identify and extract specific clause
categories from contract text.
"""

import json
import re
from typing import Optional

from tools.llm_client import OllamaClient
from models.schemas import ClauseRule, ClauseStatus


# =============================================================================
# System Prompts
# =============================================================================

EXTRACTION_SYSTEM_PROMPT = """You are a legal contract analyst AI assistant. Your task is to analyze contract text and extract specific clause information.

When analyzing contracts:
1. Be precise and extract only relevant text
2. Identify clause boundaries accurately
3. Note when clauses are missing or incomplete
4. Provide confidence scores based on clarity of the language

Always respond in valid JSON format as specified in the user prompt."""


CLAUSE_EXTRACTION_PROMPT = """Analyze the following contract text and extract information about the "{clause_name}" clause.

CLAUSE DEFINITION:
{clause_description}

EXPECTED ELEMENTS:
{expected_elements}

KEYWORDS TO LOOK FOR:
{keywords}

CONTRACT TEXT:
---
{contract_text}
---

Respond with a JSON object containing:
{{
    "found": true/false,
    "status": "present" | "partial" | "missing",
    "extracted_text": "The relevant text from the contract (or null if not found)",
    "found_elements": ["list of expected elements that were found"],
    "missing_elements": ["list of expected elements that were NOT found"],
    "confidence": 0.0-1.0,
    "issues": ["any issues or concerns with this clause"],
    "notes": "any additional observations"
}}

Be precise and only extract text that directly relates to this clause category."""


# =============================================================================
# Clause Extractor
# =============================================================================

class ClauseExtractor:
    """
    Extracts clause information from contract text using LLM analysis.
    
    Works with the Ollama client to analyze contracts and identify
    the presence and quality of specific clause categories.
    """
    
    def __init__(
        self,
        llm_client: OllamaClient,
        temperature: float = 0.1
    ):
        """
        Initialize the clause extractor.
        
        Args:
            llm_client: Configured Ollama client
            temperature: LLM temperature for extraction (lower = more deterministic)
        """
        self.llm = llm_client
        self.temperature = temperature
    
    def extract_clause(
        self,
        contract_text: str,
        clause_rule: ClauseRule,
        max_text_length: int = 8000
    ) -> dict:
        """
        Extract information about a specific clause from contract text.
        
        Args:
            contract_text: Full contract text or relevant section
            clause_rule: The clause rule definition to search for
            max_text_length: Maximum text length to send to LLM
            
        Returns:
            Dict with extraction results
        """
        # Truncate text if needed (keep beginning and end for context)
        if len(contract_text) > max_text_length:
            half = max_text_length // 2
            contract_text = (
                contract_text[:half] + 
                "\n\n[... middle section truncated ...]\n\n" + 
                contract_text[-half:]
            )
        
        # Build the prompt
        prompt = CLAUSE_EXTRACTION_PROMPT.format(
            clause_name=clause_rule.name,
            clause_description=clause_rule.description,
            expected_elements="\n".join(f"- {e}" for e in clause_rule.expected_elements),
            keywords=", ".join(clause_rule.keywords),
            contract_text=contract_text
        )
        
        # Get LLM response
        response = self.llm.generate(
            prompt=prompt,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            temperature=self.temperature,
            format_json=True
        )
        
        # Parse response
        try:
            result = json.loads(response.content)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            result = self._extract_json_from_text(response.content)
        
        # Normalize the result
        return self._normalize_extraction_result(result, clause_rule)
    
    def extract_all_clauses(
        self,
        contract_text: str,
        clause_rules: list[ClauseRule]
    ) -> list[dict]:
        """
        Extract all defined clauses from a contract.
        
        Args:
            contract_text: Full contract text
            clause_rules: List of clause rules to extract
            
        Returns:
            List of extraction results for each clause
        """
        results = []
        
        for rule in clause_rules:
            result = self.extract_clause(contract_text, rule)
            result["clause_id"] = rule.id
            result["clause_name"] = rule.name
            result["required"] = rule.required
            result["risk_if_missing"] = rule.risk_if_missing
            results.append(result)
        
        return results
    
    def _extract_json_from_text(self, text: str) -> dict:
        """
        Attempt to extract JSON object from text that may contain other content.
        
        Args:
            text: Text potentially containing JSON
            
        Returns:
            Parsed JSON dict or default structure
        """
        # Try to find JSON block
        json_patterns = [
            r'\{[^{}]*\}',  # Simple object
            r'\{(?:[^{}]|\{[^{}]*\})*\}',  # Nested object
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        # Return default structure if no valid JSON found
        return {
            "found": False,
            "status": "missing",
            "extracted_text": None,
            "confidence": 0.0,
            "error": "Could not parse LLM response as JSON"
        }
    
    def _normalize_extraction_result(
        self,
        result: dict,
        clause_rule: ClauseRule
    ) -> dict:
        """
        Normalize extraction result to ensure consistent structure.
        
        Args:
            result: Raw extraction result
            clause_rule: The clause rule for context
            
        Returns:
            Normalized result dict
        """
        # Ensure all required fields exist with defaults
        normalized = {
            "found": result.get("found", False),
            "status": result.get("status", "missing"),
            "extracted_text": result.get("extracted_text"),
            "found_elements": result.get("found_elements", []),
            "missing_elements": result.get("missing_elements", clause_rule.expected_elements.copy()),
            "confidence": result.get("confidence", 0.0),
            "issues": result.get("issues", []),
            "notes": result.get("notes", "")
        }
        
        # Convert status string to enum-compatible value
        status_map = {
            "present": ClauseStatus.PRESENT,
            "partial": ClauseStatus.PARTIAL,
            "missing": ClauseStatus.MISSING,
            "n/a": ClauseStatus.NOT_APPLICABLE
        }
        normalized["status"] = status_map.get(
            normalized["status"].lower(), 
            ClauseStatus.MISSING
        )
        
        # Ensure confidence is in valid range
        normalized["confidence"] = max(0.0, min(1.0, float(normalized["confidence"])))
        
        return normalized


# =============================================================================
# Keyword-Based Pre-Filtering
# =============================================================================

class KeywordPreFilter:
    """
    Fast keyword-based pre-filter for clause detection.
    
    Used to quickly identify which sections of a contract
    are likely to contain specific clause types before
    sending to the LLM for detailed analysis.
    """
    
    def __init__(self):
        """Initialize the pre-filter."""
        pass
    
    def find_relevant_sections(
        self,
        text: str,
        keywords: list[str],
        context_chars: int = 500
    ) -> list[str]:
        """
        Find text sections containing any of the given keywords.
        
        Args:
            text: Full document text
            keywords: Keywords to search for
            context_chars: Characters of context to include around matches
            
        Returns:
            List of relevant text sections
        """
        sections = []
        text_lower = text.lower()
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Find all occurrences
            start = 0
            while True:
                pos = text_lower.find(keyword_lower, start)
                if pos == -1:
                    break
                
                # Extract context around the match
                section_start = max(0, pos - context_chars)
                section_end = min(len(text), pos + len(keyword) + context_chars)
                
                # Extend to sentence boundaries if possible
                section_start = self._find_sentence_start(text, section_start)
                section_end = self._find_sentence_end(text, section_end)
                
                section = text[section_start:section_end]
                if section not in sections:
                    sections.append(section)
                
                start = pos + 1
        
        return sections
    
    @staticmethod
    def _find_sentence_start(text: str, pos: int) -> int:
        """Find the start of the sentence containing position."""
        sentence_ends = ".!?\n"
        while pos > 0 and text[pos - 1] not in sentence_ends:
            pos -= 1
        return pos
    
    @staticmethod
    def _find_sentence_end(text: str, pos: int) -> int:
        """Find the end of the sentence containing position."""
        sentence_ends = ".!?\n"
        while pos < len(text) and text[pos] not in sentence_ends:
            pos += 1
        return min(pos + 1, len(text))
    
    def keyword_score(self, text: str, keywords: list[str]) -> float:
        """
        Calculate a quick keyword presence score.
        
        Args:
            text: Text to analyze
            keywords: Keywords to search for
            
        Returns:
            Score from 0-1 based on keyword presence
        """
        if not keywords:
            return 0.0
        
        text_lower = text.lower()
        found = sum(1 for kw in keywords if kw.lower() in text_lower)
        
        return found / len(keywords)
