"""OpenRouter LLM client — shared engine for hoid and vespyn.

Provides: chat(), plan_chat(), run_single_pass(), run_react_loop(),
          synthesize_tool_result(), synthesize_plan_result(),
          route_model(), classify_execution_mode(), prune_tool_results()

Dependencies are injected via init() at startup — each consumer provides
its own registry and context loader. Call init() before first use.
"""
import asyncio
import os
import json
import re
import random
import string
import httpx
from datetime import datetime
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")
_ID_CHARS = string.ascii_letters + string.digits  # for Mistral tool call ID remapping
from dotenv import load_dotenv

load_dotenv()

_API_KEY = os.getenv("OPENROUTER_API_KEY")
_API_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "google/gemini-2.5-flash-lite")
NOTES_MODEL = os.getenv("NOTES_MODEL", DEFAULT_MODEL)
SUMMARIZATION_MODEL = os.getenv("SUMMARIZATION_MODEL", DEFAULT_MODEL)
COMPLEX_MODEL = os.getenv("COMPLEX_MODEL", SUMMARIZATION_MODEL)


# ---------------------------------------------------------------------------
# Dependency injection — set via init() at startup
# ---------------------------------------------------------------------------

_get_schemas = None          # () -> list[dict]
_dispatch = None             # (name: str, args: dict) -> str
_is_dispatch_error = None    # (result: str) -> bool
_claims_tool_action = None   # (text: str) -> bool
_load_context = None         # () -> str


def init(*, registry=None, context_loader=None):
    """Inject dependencies. Call once at startup before any chat/dispatch calls.

    registry:       module with get_schemas(), dispatch(), is_dispatch_error(),
                    claims_tool_action() — typically tools.registry or equivalent
    context_loader: callable() -> str that returns the default system prompt
    """
    global _get_schemas, _dispatch, _is_dispatch_error, _claims_tool_action, _load_context
    if registry is not None:
        _get_schemas = registry.get_schemas
        _dispatch = registry.dispatch
        _is_dispatch_error = getattr(registry, 'is_dispatch_error', lambda r: False)
        _claims_tool_action = getattr(registry, 'claims_tool_action', lambda t: False)
    if context_loader is not None:
        _load_context = context_loader


# ---------------------------------------------------------------------------
# Model routing
# ---------------------------------------------------------------------------

# Signals that indicate a reasoning-heavy request deserving COMPLEX_MODEL.
# Score ONLY the user message — never the system prompt.
_COMPLEXITY_SIGNALS = (
    "analyze", "analysis", "compare", "comparison", "synthesize",
    "think through", "help me think", "help me understand", "explain why",
    "what should i", "should i", "recommend", "recommendation",
    "strategy", "pros and cons", "tradeoff", "trade-off",
    "decide", "decision", "weigh", "consider my options",
    "this week", "last week", "what have i", "what did i accomplish",
    "review my", "walk me through", "step by step",
)
_COMPLEX_WORD_THRESHOLD = 80  # messages longer than this get a complexity bump


def route_model(message: str) -> str:
    """Return the appropriate model tier for a user message.

    Scores only the user message (not the system prompt) to avoid
    inflating every request to the expensive tier.
    """
    lower = message.lower()
    if len(message.split()) > _COMPLEX_WORD_THRESHOLD:
        return COMPLEX_MODEL
    if any(s in lower for s in _COMPLEXITY_SIGNALS):
        return COMPLEX_MODEL
    return DEFAULT_MODEL


# Signals that indicate a multi-step request requiring sequential execution.
_REWOO_SUBSTRINGS = (
    "and then",
    "then save", "then add", "then update", "then send", "then note", "then create",
    "and save", "and note", "and capture",
    "and add a todo", "and add to my", "and create a todo", "and put it",
)
_REWOO_PATTERNS = (
    r"first\b.{1,60}\bthen\b",
    r"research.{1,60}and (save|add|note|capture)",
    r"look up.{1,60}and (save|add|note)",
    r"find.{1,60}and (add|note|save)",
)


# Signals that indicate iterative/conditional execution — operates over unknown data at runtime.
_REACT_SUBSTRINGS = (
    # Bulk operations on an unknown-sized set
    "review all", "go through all", "go through my",
    "check all", "enrich all", "update all my", "fill in all",
    "for each",
    # Iteration over search/vault/web results
    "each result", "each article",
    "go through each",  # "go through each search result", "go through each note"
    "look up each",     # "look up each competitor"
    "fetch each",       # "fetch each of these URLs"
    "read each",        # "read each article and pull the key points"
    # Conditional/investigative — step 2 depends on evaluating step 1
    "investigate", "figure out",
    "if that fails", "if that doesn't work", "if there's nothing",
    "try different", "depends on", "depending on",
    # Market period queries — need web_search or run_code_task, not fetch_market_data
    "ytd", "year to date", "year-to-date",
    "mtd", "month to date", "month-to-date",
    "since january", "since the start of the year",
    "past month", "past week", "past year",
    "this month", "this year",
    "over the past", "over the last",
)
_REACT_PATTERNS = (
    # Todo-specific bulk patterns
    r"review (all|each|my) (task|todo)",
    r"(all|each) (task|todo).{0,30}(enrich|fill|update|add)",
    r"enrich (all|my|each).{0,10}(task|todo)",        # "enrich my todos" — primary use case
    r"fill in.{0,25}(all|my|each).{0,10}(task|todo)", # "fill in descriptions for my tasks"
    # Research/vault iteration
    r"find all.{0,25}(note|article|result)",           # "find all my notes on X"
    r"(search|look|research).{1,40}(multiple|several)", # "search multiple angles", "look through several sources"
    r"try .{1,40}(if|when).{1,40}(fail|doesn.t work|nothing)",
)


def classify_execution_mode(message: str) -> str:
    """Return 'rewoo', 'react', or 'single' based on the message's execution pattern.

    Check order: ReWOO first (explicit sequential), ReAct second (iterative), then single.
    """
    msg = message.lower()
    if any(s in msg for s in _REWOO_SUBSTRINGS):
        return "rewoo"
    if any(re.search(p, msg) for p in _REWOO_PATTERNS):
        return "rewoo"
    if any(s in msg for s in _REACT_SUBSTRINGS):
        return "react"
    if any(re.search(p, msg) for p in _REACT_PATTERNS):
        return "react"
    return "single"



def prune_tool_results(messages: list, keep_last: int = 3) -> list:
    """Trim old tool-result content to reduce token waste.

    Walks messages backwards counting assistant turns. Tool results beyond
    the keep_last window get progressively trimmed:
      - 4+ assistant turns old AND >4000 chars → soft-trim (first/last 200 chars)
      - 5+ assistant turns old → hard-clear (placeholder only)
    Returns a new list; does not mutate the input.
    """
    SOFT_TRIM_CHARS = 4000
    SOFT_TRIM_KEEP = 200
    SOFT_TRIM_AFTER = keep_last + 1   # assistant turns before soft-trim
    HARD_CLEAR_AFTER = keep_last + 2  # assistant turns before hard-clear

    # Build a map of message index → how many assistant turns ago it is.
    # Walk backwards so the most recent assistant turn = 1.
    assistant_age = {}
    turn_count = 0
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            turn_count += 1
        assistant_age[i] = turn_count

    pruned = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "tool":
            pruned.append(msg)
            continue

        age = assistant_age.get(i, 0)
        content = str(msg.get("content", ""))
        orig_len = len(content)

        if age >= HARD_CLEAR_AFTER:
            pruned.append({**msg, "content": f"[Old tool result cleared — {orig_len} chars]"})
        elif age >= SOFT_TRIM_AFTER and orig_len > SOFT_TRIM_CHARS:
            trimmed = (
                content[:SOFT_TRIM_KEEP]
                + f"\n[...trimmed {orig_len} chars...]\n"
                + content[-SOFT_TRIM_KEEP:]
            )
            pruned.append({**msg, "content": trimmed})
        else:
            pruned.append(msg)

    return pruned


# ---------------------------------------------------------------------------
# Core chat
# ---------------------------------------------------------------------------

async def chat(message: str = None, system: str = None, use_tools: bool = True, history: list = None, model: str = None, messages: list = None, tools_override: list = None) -> dict:
    """Call OpenRouter and return either a text response or a tool call.

    Returns:
        {"type": "text", "content": "..."}
        {"type": "tool_call", "name": "...", "args": {...}}

    Pass messages= to continue an accumulated conversation (ReAct loop). When
    messages= is provided, message/system/history are ignored.
    Pass tools_override= to use a scoped tool list instead of the full registry.
    """
    if messages is None:
        if system is None:
            system = _load_context() if _load_context else ""

        now_et = datetime.now(_ET)
        timestamp = now_et.strftime("%A, %Y-%m-%d %I:%M %p ET (UTC%z)")
        system = f"Current date and time: {timestamp}\n\n{system}"

        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": message})

    messages = prune_tool_results(messages)

    payload = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
    }
    if use_tools:
        payload["tools"] = tools_override if tools_override is not None else (_get_schemas() if _get_schemas else [])
        payload["tool_choice"] = "auto"

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            _API_URL,
            headers={
                "Authorization": f"Bearer {_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        if response.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"OpenRouter {response.status_code}: {response.text}",
                request=response.request,
                response=response,
            )

    msg = response.json()["choices"][0]["message"]

    if msg.get("tool_calls"):
        return {
            "type": "tool_call",
            "calls": [
                {
                    "name": tc["function"]["name"],
                    "args": json.loads(tc["function"]["arguments"]),
                    "id": tc["id"],
                }
                for tc in msg["tool_calls"]
            ],
            "_assistant_msg": msg,
            "_messages": messages,
        }

    return {"type": "text", "content": msg["content"]}


def _is_tool_error(content: str) -> bool:
    """True if a dispatch() result string indicates a tool-level failure."""
    s = str(content)
    return (
        (s.startswith("Tool '") and ("failed:" in s or "timed out" in s))
        or s.startswith("Unknown tool:")
    )


async def synthesize_tool_result(
    messages: list,
    assistant_msg: dict,
    tool_results: list,
) -> str:
    """Make a follow-up API call so the model can narrate all tool results in natural language.

    tool_results: list of {"id": str, "name": str, "content": str}
    """
    # run_code_task returns a "Working on it" placeholder — pass through verbatim so the
    # synthesis model doesn't hallucinate actual results before the sandbox finishes.
    if any(
        r.get("name") == "run_code_task"
        and str(r.get("content", "")).startswith("Working on it")
        for r in tool_results
    ):
        return "Working on it — I'll send you the result when it's done."
    # Mistral requires tool call IDs to be exactly 9 alphanumeric chars.
    # Remap whatever format the initial model generated to a safe short ID.
    id_map = {r["id"]: "".join(random.choices(_ID_CHARS, k=9)) for r in tool_results}

    patched_assistant = dict(assistant_msg)
    if patched_assistant.get("tool_calls"):
        patched_assistant["tool_calls"] = [
            {**tc, "id": id_map.get(tc["id"], tc["id"])}
            for tc in patched_assistant["tool_calls"]
        ]

    follow_up = messages + [patched_assistant] + [
        {
            "role": "tool",
            "tool_call_id": id_map[r["id"]],
            "name": r["name"],
            "content": str(r["content"]),
        }
        for r in tool_results
    ]
    # If any tool returned an error, inject an explicit instruction so the synthesis
    # model doesn't narrate over the failure and claim success.
    error_names = [r["name"] for r in tool_results if _is_tool_error(str(r.get("content", "")))]
    if error_names:
        follow_up.append({
            "role": "user",
            "content": (
                f"[SYSTEM NOTE: The following tool(s) returned errors: {', '.join(error_names)}. "
                "Report these failures accurately — do NOT claim an action succeeded "
                "if its result contains an error message.]"
            ),
        })
    # Synthesis uses SUMMARIZATION_MODEL for better prose than DEFAULT_MODEL.
    payload = {"model": SUMMARIZATION_MODEL, "messages": follow_up}
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            _API_URL,
            headers={
                "Authorization": f"Bearer {_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        if resp.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"OpenRouter {resp.status_code}: {resp.text}",
                request=resp.request,
                response=resp,
            )
    content = resp.json()["choices"][0]["message"]["content"]
    if not content:
        content = "\n".join(r["content"] for r in tool_results)
    return content


async def plan_chat(message: str, history: list, model: str, tools_override: list = None) -> list[dict]:
    """Generate a ReWOO execution plan: list of {var, tool, args} steps.

    Uses COMPLEX_MODEL with a custom planning system prompt (no persona context needed).
    Raises json.JSONDecodeError if the model returns unparseable output — caller falls back.
    """
    schemas = tools_override if tools_override is not None else (_get_schemas() if _get_schemas else [])
    schemas_json = json.dumps(schemas, indent=2)
    planning_system = f"""You are a task planner. Given the user's request, output a JSON array of steps.

Available tools:
{schemas_json}

Rules:
- Each step: {{"var": "#E1", "tool": "tool_name", "args": {{...}}}}
- Number steps #E1, #E2, etc.
- Use placeholder values in args when a step needs a prior step's output (e.g. "content": "#E1")
- Output ONLY a valid JSON array. No explanation, no markdown."""

    result = await chat(message, system=planning_system, use_tools=False,
                        history=history, model=model)
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", result["content"].strip(), flags=re.DOTALL)
    return json.loads(raw)


async def run_single_pass(message: str, model: str, history: list = None, tools_override: list = None) -> dict:
    """Run single-pass chat: dispatch, escalation, synthesis.

    Returns:
        {"type": "reply", "content": str}
        {"type": "disambig", "matches": list}
    """
    async def _run_call(call: dict) -> dict:
        content = await _dispatch(call["name"], call["args"])
        return {"id": call["id"], "name": call["name"], "content": content}

    _is_err = _is_dispatch_error or (lambda r: False)
    _claims = _claims_tool_action or (lambda t: False)

    result = await chat(message, history=history or [], model=model, tools_override=tools_override)

    if result["type"] == "tool_call":
        tool_results = list(await asyncio.gather(*[_run_call(c) for c in result["calls"]]))

        # Check for disambiguation sentinel before escalation/synthesis
        for tr in tool_results:
            if tr["content"].startswith("__DISAMBIG__|"):
                payload = json.loads(tr["content"].split("|", 1)[1])
                return {
                    "type": "disambig",
                    "matches": payload["matches"],
                    "action": payload.get("action", "done"),
                    "params": payload.get("params", {}),
                    "tools": [c["name"] for c in result["calls"]],
                }

        # Escalate: if DEFAULT_MODEL produced a tool error, retry on COMPLEX_MODEL
        if model == DEFAULT_MODEL and any(_is_err(tr["content"]) for tr in tool_results):
            result = await chat(message, history=history or [], model=COMPLEX_MODEL, tools_override=tools_override)
            if result["type"] == "tool_call":
                tool_results = list(await asyncio.gather(*[_run_call(c) for c in result["calls"]]))
                content = await synthesize_tool_result(result["_messages"], result["_assistant_msg"], tool_results)
            else:
                content = result["content"]
            return {"type": "reply", "content": content, "tools": [c["name"] for c in result.get("calls", [])]}

        content = await synthesize_tool_result(result["_messages"], result["_assistant_msg"], tool_results)
        return {"type": "reply", "content": content, "tools": [c["name"] for c in result["calls"]]}

    # Text response — escalate if DEFAULT_MODEL claimed a tool action without calling one
    if model == DEFAULT_MODEL and _claims(result["content"]):
        result = await chat(message, history=history or [], model=COMPLEX_MODEL, tools_override=tools_override)
        if result["type"] == "tool_call":
            tool_results = list(await asyncio.gather(*[_run_call(c) for c in result["calls"]]))
            content = await synthesize_tool_result(result["_messages"], result["_assistant_msg"], tool_results)
        else:
            content = result["content"]
        return {"type": "reply", "content": content, "tools": [c["name"] for c in result.get("calls", [])]}

    return {"type": "reply", "content": result["content"], "tools": []}


MAX_AGENT_TURNS = 6


async def run_react_loop(message: str, model: str, history: list = None, tools_override: list = None) -> dict:
    """ReAct agent loop: iterative tool use until task complete or MAX_AGENT_TURNS.

    Always uses the provided model (caller should pass COMPLEX_MODEL).
    Returns same shape as run_single_pass():
        {"type": "reply", "content": str, "tools": list}
        {"type": "disambig", ...}
    """
    async def _run_call(call):
        content = await _dispatch(call["name"], call["args"])
        return {"id": call["id"], "name": call["name"], "content": content}

    all_tools = []
    accumulated = None
    result = await chat(message, history=history or [], model=model, tools_override=tools_override)

    for _ in range(MAX_AGENT_TURNS):
        if result["type"] == "text":
            return {"type": "reply", "content": result["content"], "tools": all_tools}

        # Dispatch all tool calls for this turn concurrently
        tool_results = list(await asyncio.gather(*[_run_call(c) for c in result["calls"]]))
        all_tools.extend(c["name"] for c in result["calls"])

        # Surface disambiguation sentinel immediately — can't show buttons from inside a loop
        for tr in tool_results:
            if tr["content"].startswith("__DISAMBIG__|"):
                payload = json.loads(tr["content"].split("|", 1)[1])
                return {
                    "type": "disambig",
                    "matches": payload["matches"],
                    "action": payload.get("action", "done"),
                    "params": payload.get("params", {}),
                    "tools": all_tools,
                }

        # Remap tool call IDs for Mistral compatibility
        id_map = {r["id"]: "".join(random.choices(_ID_CHARS, k=9)) for r in tool_results}
        patched_assistant = dict(result["_assistant_msg"])
        if patched_assistant.get("tool_calls"):
            patched_assistant["tool_calls"] = [
                {**tc, "id": id_map.get(tc["id"], tc["id"])}
                for tc in patched_assistant["tool_calls"]
            ]

        # Accumulate: prior messages + this turn's assistant + tool results
        accumulated = result["_messages"] + [patched_assistant] + [
            {
                "role": "tool",
                "tool_call_id": id_map[r["id"]],
                "name": r["name"],
                "content": str(r["content"]),
            }
            for r in tool_results
        ]

        # Continue: model sees full history including all prior observations
        result = await chat(messages=accumulated, model=model, tools_override=tools_override)

    # MAX_AGENT_TURNS reached — synthesize what was accomplished
    step_results = []
    if accumulated:
        for m in accumulated:
            if m.get("role") == "tool":
                step_results.append({"tool": m["name"], "content": m["content"]})
    content = await synthesize_plan_result(message, step_results, history or [])
    return {"type": "reply", "content": content, "tools": all_tools}


async def synthesize_plan_result(user_message: str, plan_results: list, history: list) -> str:
    """Synthesize ReWOO plan results into a natural-language reply.

    plan_results: list of {var, tool, content} dicts from execute_plan().
    Uses load_context() as system prompt (system=None → falls through to default).
    """
    results_text = "\n\n".join(f"[{r['tool']}]: {r['content']}" for r in plan_results)
    error_tools = [r["tool"] for r in plan_results if _is_tool_error(str(r.get("content", "")))]
    error_note = ""
    if error_tools:
        error_note = (
            f"\n\nIMPORTANT: The following tool(s) returned errors: {', '.join(error_tools)}. "
            "Report these failures accurately — do NOT say an action succeeded "
            "if the tool result contains an error message."
        )
    synthesis_msg = (
        f"Complete the following request using these tool results. "
        f"Respond naturally — do not mention tools or the plan.{error_note}\n\n"
        f"Request: {user_message}\n\nTool results:\n{results_text}"
    )
    result = await chat(synthesis_msg, use_tools=False, history=history, model=DEFAULT_MODEL)
    return result["content"]
