"""
LLM Client for Ollama integration.

Provides a clean interface for interacting with local LLMs through Ollama,
handling connection management, retries, and response parsing.
"""

import json
import time
from typing import Optional, Generator

import httpx

from models.schemas import LLMResponse


# =============================================================================
# Ollama Client
# =============================================================================

class OllamaClient:
    """
    Client for interacting with Ollama API.
    
    Handles all communication with the local Ollama instance,
    including health checks, model management, and generation requests.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        timeout: int = 120,
        max_retries: int = 3,
        retry_base_delay: float = 1.0
    ):
        """
        Initialize the Ollama client.

        Args:
            base_url: Ollama API base URL
            model: Default model to use
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failed requests
            retry_base_delay: Base delay in seconds for exponential backoff
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.retry_base_delay = retry_base_delay
        
        # HTTP client with connection pooling
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
        )
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.close()
    
    def close(self):
        """Close the HTTP client and release resources."""
        self._client.close()
    
    # -------------------------------------------------------------------------
    # Health and Status Methods
    # -------------------------------------------------------------------------
    
    def is_available(self) -> bool:
        """
        Check if Ollama is available and responding.
        
        Returns:
            True if Ollama is available, False otherwise
        """
        try:
            response = self._client.get("/api/tags")
            return response.status_code == 200
        except httpx.RequestError:
            return False
    
    def list_models(self) -> list[str]:
        """
        List available models in Ollama.
        
        Returns:
            List of model names
        """
        try:
            response = self._client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except httpx.RequestError as e:
            raise ConnectionError(f"Failed to connect to Ollama: {e}")
    
    def model_exists(self, model: Optional[str] = None) -> bool:
        """
        Check if a specific model is available.

        Args:
            model: Model name to check (uses default if not specified)

        Returns:
            True if model exists, False otherwise
        """
        model = model or self.model
        return model_matches(model, self.list_models())
    
    # -------------------------------------------------------------------------
    # Generation Methods
    # -------------------------------------------------------------------------
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        format_json: bool = False
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt for context
            model: Model to use (uses default if not specified)
            temperature: Generation temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            format_json: Whether to request JSON output format
            
        Returns:
            LLMResponse with generated content
            
        Raises:
            ConnectionError: If Ollama is not available
            RuntimeError: If generation fails after retries
        """
        model = model or self.model
        start_time = time.time()
        
        # Build request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        # Add system prompt if provided
        if system_prompt:
            payload["system"] = system_prompt
        
        # Request JSON format if needed
        if format_json:
            payload["format"] = "json"
        
        # Attempt generation with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.post(
                    "/api/generate",
                    json=payload
                )
                response.raise_for_status()
                
                data = response.json()
                return self._build_response(data.get("response", ""), data, model, start_time)

            except httpx.TimeoutException:
                last_error = f"Request timed out (attempt {attempt + 1}/{self.max_retries})"
                self._backoff(attempt)

            except httpx.HTTPStatusError as e:
                last_error = f"HTTP error {e.response.status_code}: {e.response.text}"
                if e.response.status_code >= 500:
                    self._backoff(attempt)  # Server error, retry
                else:
                    break  # Client error, don't retry

            except httpx.RequestError as e:
                last_error = f"Connection error: {e}"
                self._backoff(attempt)

        raise RuntimeError(f"Generation failed after {self.max_retries} attempts: {last_error}")

    def _backoff(self, attempt: int) -> None:
        """Sleep with exponential backoff (1s, 2s, 4s, ...) before the next retry."""
        if attempt + 1 < self.max_retries:
            time.sleep(self.retry_base_delay * (2 ** attempt))

    @staticmethod
    def _build_response(content: str, data: dict, model: str, start_time: float) -> LLMResponse:
        """Build an LLMResponse from an Ollama JSON payload with full token accounting."""
        prompt_tokens = int(data.get("prompt_eval_count") or 0)
        completion_tokens = int(data.get("eval_count") or 0)
        return LLMResponse(
            content=content,
            model=model,
            tokens_used=prompt_tokens + completion_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            generation_time=time.time() - start_time
        )
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1
    ) -> Generator[str, None, None]:
        """
        Generate a streaming response from the LLM.
        
        Yields tokens as they are generated for real-time output.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            model: Model to use
            temperature: Generation temperature
            
        Yields:
            Generated text tokens
        """
        model = model or self.model
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        with self._client.stream("POST", "/api/generate", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if token := data.get("response"):
                        yield token
    
    def chat(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096
    ) -> LLMResponse:
        """
        Send a chat completion request.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use
            temperature: Generation temperature
            max_tokens: Maximum tokens
            
        Returns:
            LLMResponse with assistant's reply
        """
        model = model or self.model
        start_time = time.time()
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        response = self._client.post("/api/chat", json=payload)
        response.raise_for_status()

        data = response.json()
        return self._build_response(
            data.get("message", {}).get("content", ""), data, model, start_time
        )


# =============================================================================
# Helper Functions
# =============================================================================

def model_matches(model: str, available: list[str]) -> bool:
    """
    Check whether a requested model name matches one of the installed models.

    An exact match is accepted. A request without an explicit tag
    ("llama3.1") matches any tag of that base name ("llama3.1:8b").
    A request with an explicit tag ("llama3.1:8b") also matches the
    ":latest" alias of the same base name, but not a different tag.

    Args:
        model: Requested model name
        available: Installed model names as reported by Ollama

    Returns:
        True if a compatible model is installed
    """
    if model in available:
        return True
    base, _, tag = model.partition(":")
    for name in available:
        a_base, _, a_tag = name.partition(":")
        if a_base != base:
            continue
        if not tag or a_tag in ("", "latest"):
            return True
    return False


def create_client(
    base_url: str = "http://localhost:11434",
    model: str = "llama3.1:8b",
    **kwargs
) -> OllamaClient:
    """
    Create an Ollama client with default settings.
    
    Args:
        base_url: Ollama API URL
        model: Default model
        **kwargs: Additional client options
        
    Returns:
        Configured OllamaClient instance
    """
    return OllamaClient(base_url=base_url, model=model, **kwargs)


def check_ollama_status(base_url: str = "http://localhost:11434") -> dict:
    """
    Get detailed Ollama status information.
    
    Args:
        base_url: Ollama API URL
        
    Returns:
        Dict with status information
    """
    client = OllamaClient(base_url=base_url)
    
    status = {
        "available": client.is_available(),
        "models": [],
        "error": None
    }
    
    if status["available"]:
        try:
            status["models"] = client.list_models()
        except Exception as e:
            status["error"] = str(e)
    
    client.close()
    return status
