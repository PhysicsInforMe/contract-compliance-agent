"""
LLM Client for Ollama integration.

Provides a clean interface for interacting with local LLMs through Ollama,
handling connection management, retries, and response parsing.
"""

import json
import time
from typing import Optional, Generator

import httpx

from models.schemas import LLMRequest, LLMResponse


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
        max_retries: int = 3
    ):
        """
        Initialize the Ollama client.
        
        Args:
            base_url: Ollama API base URL
            model: Default model to use
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failed requests
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        
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
    
    def model_exists(self, model: str = None) -> bool:
        """
        Check if a specific model is available.
        
        Args:
            model: Model name to check (uses default if not specified)
            
        Returns:
            True if model exists, False otherwise
        """
        model = model or self.model
        available = self.list_models()
        return model in available or model.split(":")[0] in [m.split(":")[0] for m in available]
    
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
                generation_time = time.time() - start_time
                
                return LLMResponse(
                    content=data.get("response", ""),
                    model=model,
                    tokens_used=data.get("eval_count", 0),
                    generation_time=generation_time
                )
                
            except httpx.TimeoutException:
                last_error = f"Request timed out (attempt {attempt + 1}/{self.max_retries})"
                time.sleep(1)  # Brief pause before retry
                
            except httpx.HTTPStatusError as e:
                last_error = f"HTTP error {e.response.status_code}: {e.response.text}"
                if e.response.status_code >= 500:
                    time.sleep(1)  # Server error, retry
                else:
                    break  # Client error, don't retry
                    
            except httpx.RequestError as e:
                last_error = f"Connection error: {e}"
                time.sleep(1)
        
        raise RuntimeError(f"Generation failed after {self.max_retries} attempts: {last_error}")
    
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
        generation_time = time.time() - start_time
        
        return LLMResponse(
            content=data.get("message", {}).get("content", ""),
            model=model,
            tokens_used=data.get("eval_count", 0),
            generation_time=generation_time
        )


# =============================================================================
# Helper Functions
# =============================================================================

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
