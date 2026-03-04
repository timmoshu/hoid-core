# ~/core/ — Shared Engine

This directory is the shared engine used by **both** [Hoid](~/hoid/) (personal assistant Telegram bot) and [Vespyn](~/vespyn/) (multi-tenant SaaS for scheduled agents). Changes here affect both systems — test accordingly.

## Modules

| File | Purpose | Dependencies |
|---|---|---|
| `openrouter.py` | LLM client (OpenRouter API), execution modes (single/ReWOO/ReAct), model routing, tool-result synthesis | Injected: registry module + context loader via `init()` |
| `registry.py` | Tool framework: `register()`, `get_schemas()`, `dispatch()`, `execute_plan()`. ZERO tool registrations — each consumer registers its own tools. | None |
| `search.py` | Web search (Tavily) and URL fetching (direct + Jina fallback) | `TAVILY_API_KEY` env var |
| `resend.py` | Email via Resend API | `RESEND_API_KEY`, `FROM_EMAIL` env vars |

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
