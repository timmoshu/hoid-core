"""LLM-based execution mode router — replaces keyword heuristics with a cheap LLM call.

Classifies user messages into "single", "rewoo", or "react" execution modes.
Uses ROUTING_MODEL (fast/cheap, e.g. groq/qwen/qwen3-32b via OpenRouter).

Falls back to keyword-based classify_execution_mode() on parse failure or timeout.
"""
from __future__ import annotations

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

# Feature flag — on by default when ROUTING_MODEL is set.
# Set LLM_ROUTING=FALSE in .env to force keyword-only routing.
ENABLED = os.getenv("LLM_ROUTING", "").upper() != "FALSE"

_ROUTER_SYSTEM = (
    "You are a request classifier for a personal AI assistant. "
    "Classify the user's message into an execution mode. "
    "Reply with ONLY strict JSON — no markdown, no explanation."
)

_ROUTER_PROMPT = """\
Classify this user message into an execution mode.

User message: {message}

Return ONLY strict JSON:
{{
  "mode": "single" | "rewoo" | "react",
  "reasoning": "One sentence why"
}}

Definitions:
- "single": Simple request needing zero or one tool call. Conversational questions, \
single lookups, simple commands like "add a todo", "remember that", "what's the weather", \
"search for X". Most messages are single.
- "rewoo": Explicit sequential multi-step pipeline where step order is known upfront. \
The user specifies a chain: "research X and save a note", "look up Y then add a todo", \
"first check Z then send an email". Keywords: "and then", "and save", "first...then".
- "react": Iterative execution where the next step depends on seeing the previous result. \
Bulk operations over unknown-sized sets ("review all my todos", "go through each"), \
conditional logic ("investigate", "if that fails"), scan-then-select ("anything interesting \
on r/tech", "compare ETH and BTC this week"), or multi-source analysis requiring evaluation \
between steps. The agent can't plan all steps upfront because it needs to see data first.

Rules:
- Default to "single" when unsure — it handles most requests correctly
- "rewoo" requires the user to explicitly chain actions (verb1 ... then/and verb2)
- "react" requires iteration, conditionals, or evaluation of fetched data before deciding next steps
- Time-period analysis ("this week", "past month", "YTD") that requires fetching + computing → "react"
- Simple "compare X and Y" without time analysis → "single" (one tool call + reasoning)"""


async def classify(
    message: str,
    chat_fn=None,
    model: str | None = None,
) -> str | None:
    """Run LLM-based mode classification.

    Returns "single", "rewoo", "react", or None on failure (caller should fall back).
    """
    if not ENABLED:
        return None

    if chat_fn is None:
        from openrouter import chat as chat_fn_default, ROUTING_MODEL
        chat_fn = chat_fn_default
        if model is None:
            model = ROUTING_MODEL
    elif model is None:
        from openrouter import ROUTING_MODEL
        model = ROUTING_MODEL

    if not model:
        # ROUTING_MODEL not configured — skip LLM routing
        return None

    prompt = _ROUTER_PROMPT.format(message=message)

    try:
        result = await chat_fn(
            prompt,
            system=_ROUTER_SYSTEM,
            use_tools=False,
            model=model,
        )
    except Exception:
        logger.warning("LLM router call failed", exc_info=True)
        return None

    raw = result.get("content") or ""
    return _parse_response(raw)


def _parse_response(raw: str) -> str | None:
    """Extract and validate the mode from the JSON response."""
    try:
        data = json.loads(raw.strip())
    except (json.JSONDecodeError, TypeError):
        m = re.search(r'\{[^}]+\}', raw or "")
        if not m:
            logger.warning("LLM router: no JSON found in response: %s", raw[:200])
            return None
        try:
            data = json.loads(m.group())
        except (json.JSONDecodeError, TypeError):
            logger.warning("LLM router: JSON parse failed: %s", raw[:200])
            return None

    mode = data.get("mode", "").lower().strip()
    if mode not in ("single", "rewoo", "react"):
        logger.warning("LLM router: invalid mode=%r", mode)
        return None

    reasoning = data.get("reasoning", "")
    logger.info("LLM router: mode=%s reason=%s", mode, reasoning)
    return mode
