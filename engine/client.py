from asyncio.log import logger
import os
from abc import ABC
from threading import Lock
from typing import Optional, Dict, Any
from openai import OpenAI
import json


class Client(ABC):
    """
    Abstract base class for AI API clients with structured output support.
    """

    def __init__(self, api_key: str, base_url: Optional[str] = None, **kwargs):
        """
        Initialize the client with API key and optional base URL.

        Args:
            api_key (str): API key for authentication
            base_url (Optional[str]): Base URL for the API
            **kwargs: Extra optional arguments (e.g., proxies) accepted for compatibility
        """
        if not api_key:
            raise ValueError("API key is required")

        # Filter out any unsupported arguments that might be passed
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        # Optionally support proxies via httpx client if provided
        # We don't fail if unsupported; we just ignore unknown extras gracefully
        proxies = kwargs.get("proxies")
        if proxies:
            try:
                import httpx

                client_kwargs["http_client"] = httpx.Client(proxies=proxies)
            except Exception:
                # Silently ignore if httpx or proxies fail; fall back to default client
                pass

        self.client = OpenAI(**client_kwargs)

    def chat(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[Dict[str, Any]] = None,
        **config,
    ):
        """
        Returns a chat completion for the given model, system prompt, and user prompt.

        Args:
            model (str): The model to use for the chat completion.
            system_prompt (str): The system prompt for the chat completion.
            user_prompt (str): The user prompt for the chat completion.
            response_format (Optional[Dict[str, Any]]): JSON schema for structured output.
            **config: Additional configuration for the chat completion.

        Returns:
            Union[str, Dict]: The chat completion content or parsed JSON if response_format is provided.
        """
        if not system_prompt or not user_prompt:
            raise ValueError("system_prompt and user_prompt are required")
        
        # Prepare the request parameters
        request_params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **config,
        }

        # Add response format if provided
        if response_format:
            request_params["response_format"] = response_format

        response = self.client.chat.completions.create(**request_params)

        # If structured output was requested, try to parse as JSON
        if response_format and response_format.get("type") == "json_object":
            try:
                return json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                # Fallback to raw content if JSON parsing fails
                return response.choices[0].message.content

        return response.choices[0].message.content

    def chat_structured(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema: Dict[str, Any],
        **config,
    ):
        """
        Returns a structured chat completion using JSON schema.

        Args:
            model (str): The model to use for the chat completion.
            system_prompt (str): The system prompt for the chat completion.
            user_prompt (str): The user prompt for the chat completion.
            schema (Dict[str, Any]): JSON schema defining the expected response structure.
            **config: Additional configuration for the chat completion.

        Returns:
            Dict: The parsed JSON response matching the provided schema.
        """
        response_format = {"type": "json_object", "schema": schema}

        # Add instruction to follow the schema in the system prompt
        enhanced_system_prompt = f"{system_prompt}\n\nIMPORTANT: Respond with valid JSON that matches this schema: {schema}"

        return self.chat(
            model=model,
            system_prompt=enhanced_system_prompt,
            user_prompt=user_prompt,
            response_format=response_format,
            **config,
        )


class OpenAIClient(Client):
    """
    Client for OpenAI API.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI client.

        Args:
            api_key (Optional[str]): OpenAI API key. If not provided, will use OPENAI_API_KEY env var.
        """
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set in environment variables")

        super().__init__(api_key=api_key)


class OpenRouterClient(Client):
    """
    Singleton client for OpenRouter API.
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        """
        Returns a singleton instance of the OpenRouterClient.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(OpenRouterClient, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, **kwargs):
        """
        Initializes the OpenRouterClient.

        Accepts extra kwargs (e.g., proxies) and forwards to base Client.
        """
        if self._initialized:
            return

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set in environment variables")

        # Initialize the parent Client class
        super().__init__(
            api_key=api_key, base_url="https://openrouter.ai/api/v1", **kwargs
        )
        self._initialized = True


class OllamaClient(Client):
    """
    Singleton client for Ollama local models.

    Ollama exposes an OpenAI-compatible API at /v1, so we reuse the
    base Client which wraps the openai SDK.
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        """
        Returns a singleton instance of the OllamaClient.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(OllamaClient, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, **kwargs):
        """
        Initializes the OllamaClient.

        Uses OLLAMA_HOST env var (default: http://ollama:11434) to connect
        to the Ollama server's OpenAI-compatible endpoint.
        """
        if self._initialized:
            return

        ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        # Ensure no trailing slash before appending /v1
        base_url = f"{ollama_host.rstrip('/')}/v1"

        # Ollama doesn't require a real API key, but the openai SDK demands one
        super().__init__(
            api_key="ollama",  # dummy key — Ollama ignores it
            base_url=base_url,
            **kwargs,
        )
        self._initialized = True


def get_client(provider: str = None) -> Client:
    """
    Factory function that returns the appropriate Client singleton based
    on the configured (or explicitly requested) provider.

    Args:
        provider: One of 'ollama', 'openrouter', 'openai'.
                  When None, reads from config / env.

    Returns:
        A Client instance ready to call .chat() / .chat_structured().
    """
    if provider is None:
        # Try to infer from environment — prefer Ollama when OLLAMA_HOST is set
        if os.getenv("OLLAMA_HOST"):
            provider = "ollama"
        elif os.getenv("OPENROUTER_API_KEY"):
            provider = "openrouter"
        else:
            provider = "ollama"  # default for local / Power9 setups

    provider = provider.lower()

    if provider == "ollama":
        return OllamaClient()
    elif provider == "openrouter":
        return OpenRouterClient()
    elif provider == "openai":
        return OpenAIClient()
    else:
        # Fallback: treat unknown providers as Ollama (local-first philosophy)
        return OllamaClient()
