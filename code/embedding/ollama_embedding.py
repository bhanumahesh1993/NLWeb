# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""
Ollama embedding implementation using Ollama's embedding API.
"""

import os
import asyncio
import threading
import aiohttp
from typing import List, Optional

from config.config import CONFIG
from utils.logging_config_helper import get_configured_logger, LogLevel

logger = get_configured_logger("ollama_embedding")

# Add lock for thread-safe client access
_client_lock = threading.Lock()
ollama_client = None

def get_ollama_endpoint() -> str:
    """
    Retrieve the Ollama endpoint from configuration.
    """
    provider_config = CONFIG.get_embedding_provider("ollama")
    if provider_config and provider_config.endpoint:
        endpoint = provider_config.endpoint
        if endpoint:
            return endpoint.strip('"')
    
    # Fallback to environment variable
    endpoint = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
    return endpoint

async def get_async_client() -> aiohttp.ClientSession:
    """
    Configure and return an asynchronous HTTP client for Ollama.
    """
    global ollama_client
    with _client_lock:
        if ollama_client is None:
            timeout = aiohttp.ClientTimeout(total=60)
            ollama_client = aiohttp.ClientSession(timeout=timeout)
            logger.debug("Ollama client initialized successfully")
    
    return ollama_client

async def get_ollama_embeddings(
    text: str,
    model: Optional[str] = None,
    timeout: float = 30.0
) -> List[float]:
    """
    Generate an embedding for a single text using Ollama API.
    
    Args:
        text: The text to embed
        model: Optional model name (defaults to provider's configured model)
        timeout: Maximum time to wait for the embedding response in seconds
        
    Returns:
        List of floats representing the embedding vector
    """
    if model is None:
        provider_config = CONFIG.get_embedding_provider("ollama")
        if provider_config and provider_config.model:
            model = provider_config.model
        else:
            model = "nomic-embed-text:v1.5"  # Default model
    
    logger.debug(f"Generating Ollama embedding with model: {model}")
    logger.debug(f"Text length: {len(text)} chars")
    
    client = await get_async_client()
    endpoint = get_ollama_endpoint()
    
    try:
        # Clean input text (replace newlines with spaces)
        text = text.replace("\n", " ").strip()
        
        if not text:
            logger.warning("Empty text provided for embedding")
            # Return a zero vector of expected dimension (768 for nomic-embed-text)
            return [0.0] * 768
        
        async with client.post(
            f"{endpoint}/api/embeddings",
            json={
                "model": model,
                "prompt": text
            },
            timeout=timeout
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"Ollama API request failed with status {response.status}: {error_text}")
                raise ValueError(f"Ollama API request failed with status {response.status}: {error_text}")
            
            result = await response.json()
            
            if "embedding" not in result:
                logger.error(f"Invalid response from Ollama: {result}")
                raise ValueError(f"Invalid response from Ollama: missing 'embedding' field")
            
            embedding = result["embedding"]
            
            if not embedding or len(embedding) == 0:
                logger.error(f"Empty embedding received from Ollama for text: '{text[:50]}...'")
                # Return a zero vector of expected dimension
                return [0.0] * 768
            
            logger.debug(f"Ollama embedding generated, dimension: {len(embedding)}")
            return embedding
            
    except asyncio.TimeoutError:
        logger.error(f"Ollama embedding request timed out after {timeout}s")
        # Return a zero vector instead of failing
        logger.warning("Returning zero vector due to timeout")
        return [0.0] * 768
    except Exception as e:
        logger.exception("Error generating Ollama embedding")
        logger.log_with_context(
            LogLevel.ERROR,
            "Ollama embedding generation failed",
            {
                "model": model,
                "text_length": len(text),
                "error_type": type(e).__name__,
                "error_message": str(e),
                "endpoint": endpoint
            }
        )
        # Return a zero vector instead of failing completely
        logger.warning("Returning zero vector due to error")
        return [0.0] * 768

async def get_ollama_batch_embeddings(
    texts: List[str],
    model: Optional[str] = None,
    timeout: float = 60.0
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts using Ollama API.
    Note: Ollama doesn't have native batch support, so we process one by one.
    
    Args:
        texts: List of texts to embed
        model: Optional model name (defaults to provider's configured model)
        timeout: Maximum time to wait for batch embedding response in seconds
        
    Returns:
        List of embedding vectors, each a list of floats
    """
    if model is None:
        provider_config = CONFIG.get_embedding_provider("ollama")
        if provider_config and provider_config.model:
            model = provider_config.model
        else:
            model = "nomic-embed-text:v1.5"  # Default model
    
    logger.debug(f"Generating Ollama batch embeddings with model: {model}")
    logger.debug(f"Batch size: {len(texts)} texts")
    
    try:
        # Process texts one by one since Ollama doesn't support batch embeddings
        results = []
        for i, text in enumerate(texts):
            if i % 10 == 0:  # Log progress every 10 items
                logger.debug(f"Processing embedding {i+1}/{len(texts)}")
            
            embedding = await get_ollama_embeddings(text, model, timeout=30.0)
            results.append(embedding)
            
            # Small delay to avoid overwhelming the server
            if i < len(texts) - 1:  # Don't delay after the last item
                await asyncio.sleep(0.1)
        
        logger.debug(f"Ollama batch embeddings generated, count: {len(results)}")
        return results
        
    except Exception as e:
        logger.exception("Error generating Ollama batch embeddings")
        logger.log_with_context(
            LogLevel.ERROR,
            "Ollama batch embedding generation failed",
            {
                "model": model,
                "batch_size": len(texts),
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
        )
        raise

async def close_client():
    """Close the Ollama client session."""
    global ollama_client
    if ollama_client:
        await ollama_client.close()
        ollama_client = None