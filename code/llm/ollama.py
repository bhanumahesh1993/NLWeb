# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""
Ollama LLM provider implementation.
"""

import os
import json
import aiohttp
import asyncio
import threading
from typing import Dict, Any, Optional

from llm.llm_provider import LLMProvider
from config.config import CONFIG
from utils.logging_config_helper import get_configured_logger

logger = get_configured_logger("ollama_llm")

class OllamaProvider(LLMProvider):
    """
    Implementation of LLMProvider for Ollama.
    """
    
    _client_lock = threading.Lock()
    _client = None
    
    @classmethod
    def get_ollama_endpoint(cls) -> str:
        """Get Ollama endpoint from config or environment."""
        provider_config = CONFIG.llm_endpoints.get("ollama")
        if provider_config and provider_config.endpoint:
            endpoint = provider_config.endpoint
            if endpoint:
                return endpoint.strip('"')
        
        # Fallback to environment variable
        return os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
    
    @classmethod
    def get_client(cls):
        """Get or create Ollama HTTP client."""
        with cls._client_lock:
            if cls._client is None:
                timeout = aiohttp.ClientTimeout(total=60)
                cls._client = aiohttp.ClientSession(timeout=timeout)
                logger.debug("Ollama client initialized successfully")
        return cls._client
    
    @classmethod
    def clean_response(cls, content: str) -> Dict[str, Any]:
        """Clean and parse the response content."""
        if not content:
            return {"response": ""}
            
        # Remove any markdown formatting
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            # If all else fails, wrap in a response object
            return {"response": content}
    
    async def get_completion(
        self,
        prompt: str,
        schema: Dict[str, Any],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: float = 30.0,
        **kwargs
    ) -> Dict[str, Any]:
        """Send a completion request to Ollama."""
        
        # Get model from config if not provided
        if model is None:
            provider_config = CONFIG.llm_endpoints.get("ollama")
            if provider_config and provider_config.models:
                model = provider_config.models.high
            else:
                model = "llama3.2:latest"  # Default model
        
        logger.debug(f"Getting Ollama completion with model: {model}")
        
        client = self.get_client()
        endpoint = self.get_ollama_endpoint()
        
        # Build the prompt with schema instructions
        full_prompt = f"""You are a helpful assistant that responds with valid JSON.

Your response must match this exact JSON schema:
{json.dumps(schema, indent=2)}

Important: Only return the JSON object, no explanation or markdown formatting.

User request: {prompt}"""
        
        try:
            async with client.post(
                f"{endpoint}/api/generate",
                json={
                    "model": model,
                    "prompt": full_prompt,
                    "temperature": temperature,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature
                    }
                },
                timeout=timeout
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Ollama API request failed with status {response.status}: {error_text}")
                    raise ValueError(f"Ollama API request failed with status {response.status}")
                
                result = await response.json()
                
                if "response" not in result:
                    logger.error(f"Invalid response from Ollama: {result}")
                    raise ValueError(f"Invalid response from Ollama: {result}")
                
                response_content = result["response"]
                logger.debug(f"Ollama response received, length: {len(response_content)}")
                
                return self.clean_response(response_content)
                
        except asyncio.TimeoutError:
            logger.error(f"Ollama completion timed out after {timeout}s")
            raise
        except aiohttp.ClientError as e:
            logger.error(f"Failed to connect to Ollama: {str(e)}")
            raise ValueError(f"Failed to connect to Ollama: {str(e)}")
        except Exception as e:
            logger.error(f"Ollama completion failed: {type(e).__name__}: {str(e)}")
            raise


# Create a singleton instance
provider = OllamaProvider()