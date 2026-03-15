"""Tool registry framework — shared between hoid and vespyn.

Provides: register(), get_schemas(), dispatch(), execute_plan().
Each consumer registers its own tools at startup. This file has ZERO
tool registrations — it's a framework only.

Consumers should define their own claims_tool_action() in their
tools/registry.py to detect hallucinated actions specific to their
tool set. The default here is a no-op.
"""
import asyncio
import inspect

_registry: dict[str, dict] = {}

TOOL_TIMEOUT = 60  # seconds; per-tool ceiling, leaves headroom for synthesis call

# Tools that agents must never call (prevents recursive agent creation/modification).
# Enforced in get_schemas() — these are stripped from any agent tool scope.
AGENT_BLOCKED_TOOLS = frozenset({
    "create_agent", "update_agent", "delete_agent",
    "disable_agent", "enable_agent", "list_agents",
})


def register(schema: dict, handler, timeout: int | None = None):
    """Register a tool.

    schema:  full OpenAI tool object {"type": "function", "function": {...}}
    handler: callable(args: dict) → str, sync or async
    timeout: optional per-tool dispatch timeout in seconds (default: TOOL_TIMEOUT)
    """
    name = schema["function"]["name"]
    _registry[name] = {"schema": schema, "handler": handler, "timeout": timeout}


def get_schemas(scope: list[str] | None = None,
                exclude: frozenset[str] | None = None) -> list:
    """Return tool schemas. If scope is provided, only include named tools.
    If exclude is provided, remove those tools regardless of scope."""
    if scope is None:
        schemas = [entry["schema"] for name, entry in _registry.items()
                   if not exclude or name not in exclude]
    else:
        schemas = [entry["schema"] for name, entry in _registry.items()
                   if name in scope and (not exclude or name not in exclude)]
    return schemas


def get_handler(name: str):
    entry = _registry.get(name)
    return entry["handler"] if entry else None


def claims_tool_action(text: str) -> bool:
    """True if a text-only response claims to have performed a tool action.

    Default is a no-op. Each consumer should define their own version in
    their tools/registry.py with patterns specific to their tool set.
    init() in openrouter.py picks it up via getattr().
    """
    return False


def is_dispatch_error(result: str) -> bool:
    """True if dispatch result indicates a model-level failure worth escalating."""
    if result.startswith("Unknown tool:"):
        return True
    if result.startswith("Tool '") and "failed:" in result:
        return True
    return False


async def dispatch(name: str, args: dict) -> str:
    handler = get_handler(name)
    if not handler:
        return f"Unknown tool: {name}"
    try:
        entry = _registry[name]
        timeout = entry.get("timeout") or TOOL_TIMEOUT
        result = handler(args)
        if inspect.isawaitable(result):
            result = await asyncio.wait_for(result, timeout=timeout)
    except asyncio.TimeoutError:
        return f"Tool '{name}' timed out after {timeout}s."
    except Exception as e:
        return f"Tool '{name}' failed: {e}"
    return result


async def execute_plan(plan: list[dict]) -> list[dict]:
    """Execute a ReWOO plan sequentially, substituting #E placeholders with prior results."""
    evidences: dict[str, str] = {}
    failed_vars: set[str] = set()
    results = []
    for step in plan:
        resolved_args = {}
        for key, val in step.get("args", {}).items():
            if isinstance(val, str):
                for var, evidence in evidences.items():
                    # Skip substitution for failed steps — prevents error strings
                    # from corrupting downstream args
                    if var not in failed_vars:
                        val = val.replace(var, evidence)
            resolved_args[key] = val
        result = await dispatch(step["tool"], resolved_args)
        evidences[step["var"]] = result
        if result.startswith("Unknown tool:") or (
            result.startswith("Tool '") and ("failed:" in result or "timed out" in result)
        ):
            failed_vars.add(step["var"])
        results.append({"var": step["var"], "tool": step["tool"], "content": result})
    return results
