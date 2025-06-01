# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""
Ollama LLM provider implementation.

This module provides an implementation of the LLMProvider interface for Ollama.
"""

import json
import aiohttp
from typing import Dict, Any, Optional
from .llm_provider import LLMProvider

class OllamaProvider(LLMProvider):
    """
    Implementation of LLMProvider for Ollama.
    
    This provider connects to a local Ollama instance running on http://localhost:11434.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama provider.
        
        Args:
            base_url: The base URL of the Ollama API (default: http://localhost:11434)
        """
        self.base_url = base_url
        self._client = None
    
    @classmethod
    def get_client(cls):
        """
        Get or initialize the Ollama client.
        
        Returns:
            An instance of OllamaProvider
        """
        if not hasattr(cls, '_instance'):
            cls._instance = cls()
        return cls._instance
    
    async def get_completion(
        self,
        prompt: str,
        schema: Dict[str, Any],
        model: Optional[str] = "llama2",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: float = 30.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a completion request to Ollama and return the parsed response.
        
        Args:
            prompt: The text prompt to send to Ollama
            schema: JSON schema that the response should conform to
            model: The specific model to use (default: llama2)
            temperature: Controls randomness of the output (0-1)
            max_tokens: Maximum tokens in the generated response
            timeout: Request timeout in seconds
            **kwargs: Additional provider-specific arguments
            
        Returns:
            Parsed JSON response from Ollama
            
        Raises:
            TimeoutError: If the request times out
            ValueError: If the response cannot be parsed or request fails
        """
        if not self._client:
            self._client = aiohttp.ClientSession()
            
        try:
            async with self._client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False
                },
                timeout=timeout
            ) as response:
                if response.status != 200:
                    raise ValueError(f"Ollama API request failed with status {response.status}")
                
                result = await response.json()
                return self.clean_response(result["response"])
                
        except aiohttp.ClientError as e:
            raise ValueError(f"Failed to connect to Ollama: {str(e)}")
    
    @classmethod
    def clean_response(cls, content: str) -> Dict[str, Any]:
        """
        Clean and parse the raw response content into a structured dict.
        
        Args:
            content: Raw response content from Ollama
            
        Returns:
            Parsed JSON as a Python dictionary
            
        Raises:
            ValueError: If the content doesn't contain valid JSON
        """
        try:
            # Try to parse the response as JSON
            return json.loads(content)
        except json.JSONDecodeError:
            # If it's not valid JSON, wrap it in a response object
            return {"response": content} 