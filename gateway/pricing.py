"""Model pricing + savings accounting (ADR-0008).

Prices are USD per 1M tokens (input, output). Mock models carry simulated
prices mirroring real tiers so load tests produce meaningful savings numbers
at $0 actual spend. `counterfactual_cost` prices every request at the flagship
rate — savings = 1 - actual/counterfactual, attributed separately to cache
hits vs down-routing. Never blend the two.
"""

from __future__ import annotations

# (input $/MTok, output $/MTok)
PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "anthropic/claude-sonnet": (3.00, 15.00),
    "anthropic/claude-haiku": (0.80, 4.00),
    "ollama/llama3.1-8b": (0.00, 0.00),
    # Simulated prices for the chaos mock (cheap-a ~ mini, mid-b ~ haiku, top-c ~ 4o).
    "mock/cheap-a": (0.15, 0.60),
    "mock/mid-b": (0.80, 4.00),
    "mock/top-c": (2.50, 10.00),
}

FLAGSHIP_MODEL = "openai/gpt-4o"

# Only these providers spend real dollars; the MAX_DAILY_SPEND kill-switch
# applies to them and never blocks free/simulated traffic.
REAL_SPEND_PROVIDERS = frozenset({"openai", "anthropic"})


def cost_usd(model_key: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Simulated/actual cost of a call at this model's price."""
    if model_key not in PRICING_PER_MTOK:
        raise KeyError(f"no pricing for model {model_key!r}")
    in_price, out_price = PRICING_PER_MTOK[model_key]
    return (prompt_tokens * in_price + completion_tokens * out_price) / 1_000_000


def counterfactual_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    """What the same tokens would have cost at the flagship model (ADR-0008)."""
    return cost_usd(FLAGSHIP_MODEL, prompt_tokens, completion_tokens)


def is_real_spend(model_key: str) -> bool:
    return model_key.split("/", 1)[0] in REAL_SPEND_PROVIDERS
