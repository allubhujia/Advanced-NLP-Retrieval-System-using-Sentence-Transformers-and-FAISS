import os
import json
import logging
import time
from dataclasses import dataclass
from typing import Generator, Optional

import requests

logger = logging.getLogger(__name__)

class LLMConfig:
    def __init__(self, provider: str = "openai", model: str = "gpt-3.5-turbo", api_key: str ="", base_url: str = "https://api.openai.com/v1",
    temperature: float = 0.7, max_tokens: int = 512, top_p: float = 1.0, timeout: int = 30, 
    system_prompt: str = (
        "You are a helpful assistant. Answer the question using ONLY the provided context. "
        "If the context does not contain enough information, say: "
        "'I don't know based on the provided context. '"
        )
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.timeout = timeout
        self.system_prompt = system_prompt
    
    def to_dict(self) -> dict:
        return {k : v for k, v in self.__dict__.items() if k != "api_key"}
    
    def from_dict(self, d: dict):
        valid = {k: v for k, v in d.items() if k in self.__dict__}
        return LLMConfig(**valid)
    
class LLMProviders:

    def openai(self, model: str = "gpt-3.5-turbo", api_key: str = "") -> LLMConfig:
        return LLMConfig(
            provider="openai",
            model = model,
            api_key = api_key or os.getenv("OPENAI_API_KEY", ""),
            base_url = "https://api.openai.com/v1",
        )
    
    def ollama(self, model: str = "llama3", host: str = "http://localhost:11434") -> LLMConfig:
        return LLMConfig(
            provider="ollama",
            model = model,
            api_key = "ollama",
            base_url = f"{host}/v1"
        )

import openai

class LLMClient:
    """
    Sends a structured prompt to an OpenAI-compatible LLM API manually or via openai package.
    """
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = openai.OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url
        )

    def generate(self, prompt: str) -> str:
        if self.config.provider in ['openai', 'ollama']:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "system", "content": self.config.system_prompt}, {"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
            )
            return response.choices[0].message.content.strip()
        return ""

    