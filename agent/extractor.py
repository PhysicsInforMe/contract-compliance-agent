"""
Clause Extractor for Contract Compliance Agent.

Uses LLM-based analysis to identify and extract specific clause
categories from contract text.
"""

import json
import re
from typing import Any, Optional

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
# Keyword-Based Pre-Filtering
# =============================================================================

class KeywordPreFilter:
    """
    Fast keyword-based pre-filter for clause detection.

    Used to quickly identify which sections of a contract
    are likely to contain specific clause types before
    sending to the LLM for detailed analysis.
    """

    def find_relevant_sections(
        self,
        text: str,
        keywords: list[str],
        context_chars: int = 500
    ) -> list[str]:
        """
        Find text sections containing any of the given keywords.

        Sections are returned in document order and overlapping
        matches are merged into a single section.

        Args:
            text: Full document text
            keywords: Keywords to search for
            context_chars: Characters of context to include around matches

        Returns:
            List of relevant text sections
        """
        if not text or not keywords:
            return []

        text_lower = text.lower()
        spans: list[tuple[int, int]] = []

        for keyword in keywords:
            keyword_lower = keyword.lower().strip()
            if not keyword_lower:
                continue

            start = 0
            while True:
                pos = text_lower.find(keyword_lower, start)
                if pos == -1:
                    break

                section_start = max(0, pos - context_chars)
                section_end = min(len(text), pos + len(keyword_lower) + context_chars)

                # Extend to sentence boundaries if possible
                section_start = self._find_sentence_start(text, section_start)
                section_end = self._find_sentence_end(text, section_end)
                spans.append((section_start, section_end))

                start = pos + 1

        if not spans:
            return []

        # Merge overlapping / adjacent spans, preserving document order
        spans.sort()
        merged: list[list[int]] = [list(spans[0])]
        for s, e in spans[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])

        return [text[s:e] for s, e in merged]

    def build_focused_excerpt(
        self,
        text: str,
        keywords: list[str],
        max_length: int,
        context_chars: int = 500
    ) -> str:
        """
        Build an excerpt of at most ``max_length`` characters that keeps the
        parts of the document most likely to contain the clause.

        Keyword-matching sections are concatenated in document order. If no
        keyword matches, the beginning and end of the document are kept.

        Args:
            text: Full document text
            keywords: Keywords indicating the clause
            max_length: Maximum excerpt length in characters
            context_chars: Context to keep around each keyword hit

        Returns:
            Excerpt text, never longer than ``max_length`` (plus separators)
        """
        if len(text) <= max_length:
            return text

        separator = "\n\n[...]\n\n"
        sections = self.find_relevant_sections(text, keywords, context_chars)

        if sections:
            parts: list[str] = []
            used = 0
            for section in sections:
                if used + len(section) > max_length:
                    remaining = max_length - used
                    if remaining > 200:
                        parts.append(section[:remaining])
                    break
                parts.append(section)
                used += len(section)
            if parts:
                return separator.join(parts)

        # Fallback: keep beginning and end for context
        half = max_length // 2
        return text[:half] + "\n\n[... middle section truncated ...]\n\n" + text[-half:]

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


# =============================================================================
# Clause Extractor
# =============================================================================

class ClauseExtractor:
    """
    Extracts clause information from contract text using LLM analysis.

    Works with the Ollama client to analyze contracts and identify
    the presence and quality of specific clause categories. Long
    documents are narrowed down with a keyword pre-filter so that the
    relevant sections (not just the beginning and end) reach the LLM.
    """

    def __init__(
        self,
        llm_client: OllamaClient,
        temperature: float = 0.1,
        max_text_length: int = 8000
    ):
        """
        Initialize the clause extractor.

        Args:
            llm_client: Configured Ollama client
            temperature: LLM temperature for extraction (lower = more deterministic)
            max_text_length: Maximum characters of contract text sent per clause
        """
        self.llm = llm_client
        self.temperature = temperature
        self.max_text_length = max_text_length
        self.prefilter = KeywordPreFilter()

    def extract_clause(
        self,
        contract_text: str,
        clause_rule: ClauseRule,
        max_text_length: Optional[int] = None
    ) -> dict:
        """
        Extract information about a specific clause from contract text.

        Args:
            contract_text: Full contract text or relevant section
            clause_rule: The clause rule definition to search for
            max_text_length: Override for the maximum text length sent to the LLM

        Returns:
            Dict with extraction results. Includes ``tokens_used`` (actual
            tokens reported by Ollama) and ``truncated`` (whether the text
            was narrowed before being sent).
        """
        limit = max_text_length or self.max_text_length
        truncated = len(contract_text) > limit
        if truncated:
            contract_text = self.prefilter.build_focused_excerpt(
                contract_text, clause_rule.keywords, limit
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
        result: Any
        try:
            result = json.loads(response.content)
        except (json.JSONDecodeError, TypeError):
            result = self._extract_json_from_text(response.content or "")

        if not isinstance(result, dict):
            result = self._default_result("LLM response was not a JSON object")

        # Normalize the result
        normalized = self._normalize_extraction_result(result, clause_rule)
        normalized["tokens_used"] = response.tokens_used
        normalized["truncated"] = truncated
        return normalized

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

    @staticmethod
    def _default_result(error: str) -> dict:
        """Default structure used when the LLM output cannot be parsed."""
        return {
            "found": False,
            "status": "missing",
            "extracted_text": None,
            "confidence": 0.0,
            "error": error
        }

    def _extract_json_from_text(self, text: str) -> dict:
        """
        Attempt to extract JSON object from text that may contain other content.

        Args:
            text: Text potentially containing JSON

        Returns:
            Parsed JSON dict or default structure
        """
        json_patterns = [
            r'\{(?:[^{}]|\{[^{}]*\})*\}',  # Object with one level of nesting
            r'\{[^{}]*\}',  # Flat object
        ]

        for pattern in json_patterns:
            for match in re.findall(pattern, text, re.DOTALL):
                try:
                    parsed = json.loads(match)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    return parsed

        return self._default_result("Could not parse LLM response as JSON")

    @staticmethod
    def _as_str_list(value: Any) -> list[str]:
        """Coerce an LLM-provided value into a list of non-empty strings."""
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple)):
            return []
        return [str(v).strip() for v in value if v is not None and str(v).strip()]

    def _normalize_extraction_result(
        self,
        result: dict,
        clause_rule: ClauseRule
    ) -> dict:
        """
        Normalize extraction result to ensure consistent structure.

        Tolerates missing, null or wrongly-typed fields coming from the LLM.

        Args:
            result: Raw extraction result
            clause_rule: The clause rule for context

        Returns:
            Normalized result dict
        """
        extracted_text = result.get("extracted_text")
        if extracted_text is not None and not isinstance(extracted_text, str):
            extracted_text = str(extracted_text)
        if isinstance(extracted_text, str) and not extracted_text.strip():
            extracted_text = None

        found_elements = self._as_str_list(result.get("found_elements"))
        if "missing_elements" in result:
            missing_elements = self._as_str_list(result.get("missing_elements"))
        else:
            # Default: everything not reported as found is missing
            missing_elements = [
                e for e in clause_rule.expected_elements if e not in found_elements
            ]

        notes = result.get("notes")
        normalized = {
            "found": bool(result.get("found", False)),
            "status": result.get("status"),
            "extracted_text": extracted_text,
            "found_elements": found_elements,
            "missing_elements": missing_elements,
            "confidence": result.get("confidence"),
            "issues": self._as_str_list(result.get("issues")),
            "notes": str(notes) if notes is not None else ""
        }
        if "error" in result:
            normalized["error"] = result["error"]

        # Convert status string to enum value
        status_map = {
            "present": ClauseStatus.PRESENT,
            "partial": ClauseStatus.PARTIAL,
            "missing": ClauseStatus.MISSING,
            "n/a": ClauseStatus.NOT_APPLICABLE,
            "na": ClauseStatus.NOT_APPLICABLE,
            "not applicable": ClauseStatus.NOT_APPLICABLE,
            "not_applicable": ClauseStatus.NOT_APPLICABLE,
        }
        raw_status = normalized["status"]
        status_key = raw_status.strip().lower() if isinstance(raw_status, str) else ""
        status = status_map.get(status_key)
        if status is None:
            # Fall back on the "found" flag and the extracted text
            if normalized["found"] or extracted_text:
                status = ClauseStatus.PARTIAL if missing_elements else ClauseStatus.PRESENT
            else:
                status = ClauseStatus.MISSING
        normalized["status"] = status
        normalized["found"] = status in (ClauseStatus.PRESENT, ClauseStatus.PARTIAL)

        # Ensure confidence is a float in [0, 1]
        try:
            confidence = float(normalized["confidence"])
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence != confidence:  # NaN guard
            confidence = 0.0
        normalized["confidence"] = max(0.0, min(1.0, confidence))

        return normalized
