# ~/core/ — Shared Engine

This directory is the shared engine used by **both** [Hoid](~/hoid/) (personal assistant Telegram bot) and [Vespyn](~/vespyn/) (multi-tenant SaaS for scheduled agents). Changes here affect both systems — test accordingly.

## Modules

| File | Purpose | Dependencies |
|---|---|---|
| `openrouter.py` | LLM client (OpenRouter API), execution modes (single/ReWOO/ReAct), model routing, tool-result synthesis | Injected: registry module + context loader via `init()` |
| `registry.py` | Tool framework: `register()`, `get_schemas()`, `dispatch()`, `execute_plan()`. ZERO tool registrations — each consumer registers its own tools. | None |
| `search.py` | Web search (Tavily) and URL fetching (direct + Jina fallback) | `TAVILY_API_KEY` env var |
| `resend.py` | Email via Resend API | `RESEND_API_KEY`, `FROM_EMAIL` env vars |

### `openrouter.py` — full API surface

**Constants** (all read from env, fall back to each other):
- `DEFAULT_MODEL`, `NOTES_MODEL`, `SUMMARIZATION_MODEL`, `COMPLEX_MODEL`
- `MAX_AGENT_TURNS = 6` — ReAct loop ceiling

**`init(*, registry=None, context_loader=None, synthesis_shortcircuit=None)`** — inject dependencies once at startup.
- `registry`: module with `get_schemas()`, `dispatch()`, optionally `is_dispatch_error()`, `claims_tool_action()`
- `context_loader`: `() -> str` — returns default system prompt
- `synthesis_shortcircuit`: `callable(tool_results: list) -> str | None` — if it returns a string, `synthesize_tool_result()` returns it immediately instead of making a synthesis API call. Used by Hoid for `run_code_task` async placeholder. Default: `None` (no shortcircuit).

**`route_model(message: str) -> str`** — returns `COMPLEX_MODEL` or `DEFAULT_MODEL` based on message length and complexity signals.

**`classify_execution_mode(message: str) -> str`** — returns `"rewoo"`, `"react"`, or `"single"`. ReWOO first (explicit sequential), ReAct second (iterative/bulk), then single.

**`chat(message=None, *, model, history=None, tools=None, system=None, messages=None) -> dict`** — single LLM call. Returns `{"type": "text"|"tool_calls", "content": str, "calls": [...]}`.

**`plan_chat(message, *, history, model, tools_override=None) -> list[dict]`** — ReWOO planner. Returns list of `{"var": "#E1", "tool": "name", "args": {...}}` steps.

**`run_single_pass(message, model, history=None, tools_override=None) -> dict`** — single-pass execution with tool dispatch and synthesis. Returns `{"type": "reply"|"disambig", "content": str, "tools": [tool_names], "tokens_used": int}`. Token count accumulates across all API calls (initial chat, escalation retry, synthesis).

**`run_react_loop(message, model, history=None, tools_override=None) -> dict`** — iterative ReAct loop up to `MAX_AGENT_TURNS`. Returns same shape as `run_single_pass()` including `tokens_used`.

**`synthesize_tool_result(message, tool_results, history) -> str`** — narrates tool results using `SUMMARIZATION_MODEL`.

**`synthesize_plan_result(message, plan_results, history) -> str`** — narrates ReWOO plan results using `DEFAULT_MODEL`.

**`prune_tool_results(messages, keep_last=3) -> list`** — trims old tool-result content to reduce token waste. Soft-trims at 4+ assistant turns old; hard-clears at 5+.

### `registry.py` — full API surface

**Constants:**
- `TOOL_TIMEOUT = 60` — default per-tool dispatch timeout in seconds
- `AGENT_BLOCKED_TOOLS` — frozenset of tool names agents must never call (agent management tools)

**`register(schema, handler, timeout=None)`** — register a tool by OpenAI tool schema + callable handler.

**`get_schemas(scope=None, exclude=None) -> list`** — return tool schemas, optionally filtered to a scope list and/or excluding names.

**`get_handler(name) -> callable|None`** — retrieve a registered handler by name.

**`dispatch(name, args) -> str`** — async; calls handler with timeout; returns result string or error string.

**`execute_plan(plan: list[dict]) -> list[dict]`** — async; ReWOO sequential executor, resolves `#E1`/`#E2` placeholders from prior step results. Returns list of `{"var", "tool", "content"}`.

**`claims_tool_action(text) -> bool`** — Default is a no-op (`return False`). Each consumer should define their own version in their `tools/registry.py` with patterns specific to their tool set (e.g. Hoid checks for hallucinated vault/todo/memory actions). `init()` in `openrouter.py` picks it up via `getattr(registry, 'claims_tool_action', ...)`.
**`is_dispatch_error(result) -> bool`** — True if dispatch result indicates a model-level failure worth escalating (unknown tool, tool exception).

### `search.py` — full API surface

**`web_search(query: str) -> str`** — async; Tavily search, returns formatted results string.

**`fetch_url(url: str) -> str`** — async; direct httpx → Jina.ai fallback. Validates content via `_is_usable_content()` (rejects <200 chars; junk pattern check for 200–499 chars). Known limitations: client-side-rendered pages (claude.ai/share/, JS SPAs), paywalled articles.

### `resend.py` — full API surface

**`send_email(to, subject, body, html=None, reply_to=None, extra_headers=None) -> None`** — async; sends via Resend API. Raises on HTTP error.

## How consumers use this

### Hoid (`~/hoid/`)
- `tools/registry.py`: imports framework from `~/core/registry.py`, registers ~20 hoid-specific tools (memory, todos, vault, notes, sandbox, digest, agents), then calls `openrouter.init()` at the bottom
- `services/openrouter.py`: thin wrapper that re-exports everything from `~/core/openrouter.py`
- Entry points: `bot.py`, `agent_runner.py`, standalone scripts (morning.py, nightly.py, etc.)

### Vespyn (`~/vespyn/backend/`)
- `tools/registry.py`: imports framework from `~/core/registry.py`, registers 4 V1 readonly tools (web_search, fetch_url, gmail_read, todoist_read), then calls `openrouter.init()` at the bottom
- Imports `openrouter` directly via `sys.path` (set in `main.py` and `agent_runner.py`)
- Agent management tools are registered per-request in `routers/chat.py` (not in the global registry)

## Dependency injection pattern

`openrouter.py` uses **no hardcoded imports** for consumer-specific code. Instead:

```python
import openrouter
openrouter.init(
    registry=my_registry_module,    # must have get_schemas(), dispatch(), etc.
    context_loader=my_load_context,  # callable() -> str (default system prompt)
)
```

Each consumer calls `init()` once at startup (typically at the bottom of its `tools/registry.py` after all tools are registered).

## Rules for modifying ~/core/

1. **Never add consumer-specific imports** — no `from tools.X`, `from config`, `from services.X`
2. **Never register tools here** — tool registrations belong in each consumer's `tools/registry.py`
3. **Test both consumers** after any change:
   - Hoid: `cd ~/hoid && python -c "from tools.registry import get_schemas; print(len(get_schemas()))"`
   - Vespyn: `cd ~/vespyn/backend && python -c "from tools.registry import get_schemas; print(len(get_schemas()))"`
4. **Keep `init()` backward-compatible** — add new optional params, don't remove existing ones
5. **The error messages in synthesis functions are generic** — no "Tim", no "user" assumptions
