"""
utils.py
========
LLM adapters — same interface as in promptbreeder-mini.
"""

from __future__ import annotations
import os
import time
import random
from typing import Callable

LLMFn = Callable[[str], str]


def groq_llm(
    model: str = "llama3-8b-8192",
    temperature: float = 0.3,   # lower than promptbreeder — we want consistent compression
    max_tokens: int = 1024,
    api_key: str | None = None,
) -> LLMFn:
    """Groq API adapter. Requires: pip install groq + GROQ_API_KEY env var."""
    try:
        from groq import Groq
    except ImportError as e:
        raise ImportError("pip install groq") from e

    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise EnvironmentError("Set GROQ_API_KEY or pass api_key=...")

    client = Groq(api_key=key)

    def call(prompt: str) -> str:
        delay = 1.0
        for _ in range(5):
            try:
                r = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return r.choices[0].message.content or ""
            except Exception as exc:
                if "429" in str(exc):
                    time.sleep(delay); delay *= 2
                else:
                    raise
        return ""

    return call


def openai_compatible_llm(
    base_url: str,
    model: str,
    api_key: str = "none",
    temperature: float = 0.3,
) -> LLMFn:
    """OpenAI-compatible endpoint (Ollama, LM Studio, etc.)."""
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError("pip install openai") from e
    client = OpenAI(base_url=base_url, api_key=api_key)

    def call(prompt: str) -> str:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1024,
        )
        return r.choices[0].message.content or ""

    return call


# ------------------------------------------------------------------ #
# Mock LLM for offline testing                                        #
# ------------------------------------------------------------------ #

_COMPRESSED = [
    "Classify sentiment: positive, negative, or neutral.",
    "Label sentiment.",
    "Sentiment?",
    "Answer the question briefly.",
    "Answer concisely.",
    "Short answer only.",
    "Summarise in one sentence.",
    "One-sentence summary.",
]

_ANSWERS = {
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "paris": "Paris",
    "france": "France",
    "42": "42",
}


def mock_llm(seed: int | None = 42) -> LLMFn:
    """
    Deterministic mock LLM for offline testing.
    Simulates compression + classification/QA responses.
    """
    rng = random.Random(seed)
    call_count = {"n": 0}

    def call(prompt: str) -> str:
        time.sleep(0.005)
        call_count["n"] += 1
        p = prompt.lower()

        # Compression call
        if "compress" in p or "rewrite" in p or "concise" in p or "shorten" in p:
            return rng.choice(_COMPRESSED)

        # Judge call
        if "rate" in p and "scale" in p:
            return str(rng.randint(6, 9))

        # Classification call
        if "label:" in p or "sentiment" in p:
            if "great" in p or "love" in p or "good" in p or "best" in p:
                return "positive"
            if "terrible" in p or "bad" in p or "disappoint" in p:
                return "negative"
            return "neutral"

        # QA call
        for key, val in _ANSWERS.items():
            if key in p:
                return val

        return rng.choice(["positive", "negative", "neutral", "Paris", "42"])

    return call
