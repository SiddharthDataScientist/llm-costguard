"""
Wraps OpenAI chat completion calls.

Responsibilities:
- Make the actual (or mocked) call to OpenAI
- Track token usage and compute cost per request
- Support MOCK_MODE so we can test the whole pipeline without spending money
"""

import os
import time
import random
from dataclasses import dataclass
from openai import OpenAI

MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"

# Pricing per 1M tokens, in USD. Update these if OpenAI changes pricing.
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}

_client = None


def get_client() -> OpenAI:
    """Lazily create the OpenAI client so importing this module doesn't require an API key to be set yet."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


@dataclass
class CompletionResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    mocked: bool


def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = PRICING[model]
    return (input_tokens / 1_000_000) * rates["input"] + (output_tokens / 1_000_000) * rates["output"]


def _mock_completion(prompt: str, model: str) -> CompletionResult:
    """
    Fakes a completion so you can test the whole pipeline (cache, router, logging)
    without spending real money. Token counts are rough estimates (word count * 1.3)
    just to make the cost numbers look plausible during dev.
    """
    time.sleep(random.uniform(0.05, 0.2))  # pretend there's network latency
    input_tokens = int(len(prompt.split()) * 1.3)
    output_tokens = random.randint(30, 150)
    cost = _compute_cost(model, input_tokens, output_tokens)
    return CompletionResult(
        text=f"[MOCKED RESPONSE from {model}] This is a placeholder answer to: '{prompt[:50]}...'",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
        latency_ms=round(random.uniform(50, 200), 1),
        mocked=True,
    )


def get_completion(prompt: str, model: str = "gpt-4o-mini") -> CompletionResult:
    """
    Main entrypoint. Returns a CompletionResult regardless of whether we're
    mocking or hitting the real API, so callers never need to know the difference.
    """
    if MOCK_MODE:
        return _mock_completion(prompt, model)

    start = time.perf_counter()
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    latency_ms = (time.perf_counter() - start) * 1000

    usage = response.usage
    cost = _compute_cost(model, usage.prompt_tokens, usage.completion_tokens)

    return CompletionResult(
        text=response.choices[0].message.content,
        model=model,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        cost_usd=cost,
        latency_ms=round(latency_ms, 1),
        mocked=False,
    )