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
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "")

# ReAct loop addenda — appended to system prompt for interactive vs. agent contexts.
# Callers pass these via system_addendum= to run_react_loop().
REACT_ADDENDUM_AGENT = (
    "- Always make at least one tool call before producing a final answer. Never return an empty first-turn response.\n"
    "- If tools return no useful information, state exactly what you searched for and confirm the information could not be found.\n"
    "- Stop as soon as you have a complete answer — do not make additional tool calls once the task is done.\n"
    "- After 5 tool calls without a complete answer, stop and report what you found so far.\n"
    "- Your final answer must directly address the request and nothing else. No unsolicited context, caveats, or padding.\n"
    "- Never expose your internal process in the final answer. Do not include tool call parameters, search queries, "
    "chain-of-thought reasoning, or \"I searched for X\" narration. Present only polished results."
)

REACT_ADDENDUM_CHAT = (
    "- For conversational questions or topics answerable from existing knowledge, respond directly without making unnecessary tool calls.\n"
    "- For questions requiring current data, use the appropriate tool before answering. This includes topics such as prices, weather, news, scores, and market odds.\n"
    "- Before any significant tool call, state in one brief line the purpose of the call and the minimal inputs being used.\n"
    "- Use only the provided tools. If a needed tool is unavailable or returns no information, briefly say so and then offer the best available knowledge instead.\n"
    "- After each tool call, briefly validate whether the result answered the question; if not, make at most one additional targeted tool call when necessary.\n"
    "- Keep responses concise; this is an interactive chat session, not a report.\n"
    "- Stop once the question is answered. Do not make additional tool calls after you have a satisfactory answer.\n"
    "- Reason internally as needed, but do not reveal private chain-of-thought unless explicitly requested."
)

# Token usage accumulator — reset by run_single_pass()/run_react_loop() at
# entry, incremented by chat() and synthesize_tool_result() on each API call.
# Safe in asyncio single-threaded context (no concurrent top-level calls).
_request_tokens = 0


# ---------------------------------------------------------------------------
# LLM retry with backoff
# ---------------------------------------------------------------------------

_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
_MAX_LLM_RETRIES = 3
_LLM_RETRY_BASE = 1.0  # seconds


async def _post_with_retry(
    client: httpx.AsyncClient, url: str, headers: dict, payload: dict,
) -> httpx.Response:
    """POST to OpenRouter with exponential backoff on transient errors."""
    for attempt in range(_MAX_LLM_RETRIES):
        response = await client.post(url, headers=headers, json=payload)
        if response.status_code not in _RETRYABLE_STATUSES:
            return response
        if attempt < _MAX_LLM_RETRIES - 1:
            wait = _LLM_RETRY_BASE * (2 ** attempt)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = min(float(retry_after), 10.0)
                    except (ValueError, TypeError):
                        pass
            await asyncio.sleep(wait)
    return response


# ---------------------------------------------------------------------------
# Dependency injection — set via init() at startup
# ---------------------------------------------------------------------------

_get_schemas = None          # () -> list[dict]
_dispatch = None             # (name: str, args: dict) -> str
_is_dispatch_error = None    # (result: str) -> bool
_claims_tool_action = None   # (text: str) -> bool
_load_context = None         # () -> str
_synthesis_shortcircuit = None  # (tool_results: list) -> str | None


def init(*, registry=None, context_loader=None, synthesis_shortcircuit=None):
    """Inject dependencies. Call once at startup before any chat/dispatch calls.

    registry:       module with get_schemas(), dispatch(), is_dispatch_error(),
                    claims_tool_action() — typically tools.registry or equivalent
    context_loader: callable() -> str that returns the default system prompt
    synthesis_shortcircuit: callable(tool_results) -> str | None — if it returns
                    a string, synthesize_tool_result() returns it immediately
                    instead of making a synthesis API call
    """
    global _get_schemas, _dispatch, _is_dispatch_error, _claims_tool_action, _load_context, _synthesis_shortcircuit
    if registry is not None:
        _get_schemas = registry.get_schemas
        _dispatch = registry.dispatch
        _is_dispatch_error = getattr(registry, 'is_dispatch_error', lambda r: False)
        _claims_tool_action = getattr(registry, 'claims_tool_action', lambda t: False)
    if context_loader is not None:
        _load_context = context_loader
    if synthesis_shortcircuit is not None:
        _synthesis_shortcircuit = synthesis_shortcircuit


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
# Strong ReWOO: explicit sequential save/write/create — always beats react.
_REWOO_SUBSTRINGS_STRONG = (
    "and then",
    "then save", "then add", "then update", "then send", "then note", "then create",
    "and save", "and note", "and capture",
    "and add a todo", "and add to my", "and create a todo", "and put it",
)
_REWOO_PATTERNS_STRONG = (
    r"first\b.{1,60}\bthen\b",
    r"research.{1,60}and (save|add|note|capture)",
    r"look up.{1,60}and (save|add|note)",
    r"find.{1,60}and (add|note|save)",
)

# Pipeline ReWOO: multi-source sequential fetch — beats single but yields to react ("for each").
_REWOO_SUBSTRINGS_PIPELINE = (
    "then search", "then fetch", "then look", "then check", "then email",
)
_REWOO_PATTERNS_PIPELINE = (
    r"(read|fetch|get|search|check).{1,80},\s*then\b",
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
    # Scan-then-select: fetch a list/feed, evaluate what came back, drill in
    # These imply the agent can't know which items to act on until it sees the data.
    "most interesting", "most relevant", "most important",
    "worth reading", "worth noting", "worth knowing",
    "stand out", "stands out",
    "anything interesting", "anything notable", "anything worth",
    "anything relevant", "anything important",
    "relevant posts", "relevant articles", "relevant results",
    "notable posts", "notable articles",
    "if there are any", "if there's anything", "if anything is",
    "scan for", "monitor for", "watch for", "look for any", "filter for",
    "pick the best", "pick the top", "pick the most",
    "select the best", "select the most relevant",
    "if the odds", "if the score", "if the price", "if the result",
    "compare sources", "cross-reference",
    "follow up", "follow-up",
    "based on the results", "based on what",
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
    r"(search|look|research).{1,40}(multiple|several)", # "search multiple angles"
    r"try .{1,40}(if|when).{1,40}(fail|doesn.t work|nothing)",
    # Scan-then-select patterns (new tools: reddit, predictions, sports, crypto)
    r"(most|top).{1,20}(interesting|relevant|important|notable|noteworthy)",
    r"anything (interesting|notable|important|relevant|worth)",
    r"(monitor|watch|scan|look).{1,60}(for any|for relevant|for mentions|for posts|for threads)",
    r"(filter|select|pick).{1,40}(relevant|interesting|best|top|most)",
    r"if (there are|there.?s|any).{1,40}(relevant|interesting|notable|worth)",
    r"(summarize|highlight|pull out).{1,40}(best|top|most|relevant|notable|interesting)",
    r"(check|see).{1,60}(if (there|any|it)).{1,40}(interest|notabl|worth|relevant)",
)


def classify_execution_mode(message: str) -> str:
    """Return 'rewoo', 'react', or 'single' based on the message's execution pattern.

    Priority order:
      1. Strong ReWOO — explicit sequential save/create/update (beats everything)
      2. ReAct — iterative over unknown data ("for each", scan-then-select, bulk ops)
      3. Pipeline ReWOO — multi-source sequential (only when no react signal)
      4. Single — everything else
    """
    msg = message.lower()
    if any(s in msg for s in _REWOO_SUBSTRINGS_STRONG):
        return "rewoo"
    if any(re.search(p, msg) for p in _REWOO_PATTERNS_STRONG):
        return "rewoo"
    if any(s in msg for s in _REACT_SUBSTRINGS):
        return "react"
    if any(re.search(p, msg) for p in _REACT_PATTERNS):
        return "react"
    if any(s in msg for s in _REWOO_SUBSTRINGS_PIPELINE):
        return "rewoo"
    if any(re.search(p, msg) for p in _REWOO_PATTERNS_PIPELINE):
        return "rewoo"
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

async def chat(message: str = None, system: str = None, use_tools: bool = True, history: list = None, model: str = None, messages: list = None, tools_override: list = None, system_addendum: str = None) -> dict:
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

        if system_addendum:
            system = system + "\n\n" + system_addendum

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

    _headers = {"Authorization": f"Bearer {_API_KEY}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=60) as client:
        response = await _post_with_retry(client, _API_URL, _headers, payload)
        if response.status_code >= 400:
            used_model = model or DEFAULT_MODEL
            if FALLBACK_MODEL and used_model != FALLBACK_MODEL:
                payload["model"] = FALLBACK_MODEL
                response = await _post_with_retry(client, _API_URL, _headers, payload)
            if response.status_code >= 400:
                raise httpx.HTTPStatusError(
                    f"OpenRouter {response.status_code}: {response.text}",
                    request=response.request,
                    response=response,
                )

    msg = response.json()["choices"][0]["message"]

    global _request_tokens
    _request_tokens += response.json().get("usage", {}).get("total_tokens", 0)

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


_REACT_RESULT_CAP = 12_000

def _cap_tool_result(content: str) -> str:
    """Cap a fresh ReAct tool result to prevent large outputs from corrupting next-turn JSON."""
    if len(content) <= _REACT_RESULT_CAP:
        return content
    return content[:_REACT_RESULT_CAP] + f"\n[...truncated — {len(content)} total chars]"


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
    # Consumer-defined shortcircuit (e.g. async tools that return a placeholder)
    if _synthesis_shortcircuit is not None:
        shortcircuit = _synthesis_shortcircuit(tool_results)
        if shortcircuit is not None:
            return shortcircuit
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
    _headers = {"Authorization": f"Bearer {_API_KEY}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await _post_with_retry(client, _API_URL, _headers, payload)
        if resp.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"OpenRouter {resp.status_code}: {resp.text}",
                request=resp.request,
                response=resp,
            )
    content = resp.json()["choices"][0]["message"]["content"]

    global _request_tokens
    _request_tokens += resp.json().get("usage", {}).get("total_tokens", 0)

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
        {"type": "reply", "content": str, "tools": list, "tokens_used": int}
        {"type": "disambig", "matches": list, "tokens_used": int}
    """
    global _request_tokens
    _request_tokens = 0

    async def _run_call(call: dict) -> dict:
        content = await _dispatch(call["name"], call["args"])
        return {"id": call["id"], "name": call["name"], "content": content}

    async def _safe_synthesize(messages, assistant_msg, tool_results):
        try:
            return await synthesize_tool_result(messages, assistant_msg, tool_results)
        except Exception:
            return "\n".join(r["content"] for r in tool_results)

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
                    "tokens_used": _request_tokens,
                }

        # Escalate: if DEFAULT_MODEL produced a tool error, retry on COMPLEX_MODEL
        if model == DEFAULT_MODEL and any(_is_err(tr["content"]) for tr in tool_results):
            result = await chat(message, history=history or [], model=COMPLEX_MODEL, tools_override=tools_override)
            if result["type"] == "tool_call":
                tool_results = list(await asyncio.gather(*[_run_call(c) for c in result["calls"]]))
                content = await _safe_synthesize(result["_messages"], result["_assistant_msg"], tool_results)
            else:
                content = result["content"]
            return {"type": "reply", "content": content, "tools": [c["name"] for c in result.get("calls", [])], "tokens_used": _request_tokens}

        content = await _safe_synthesize(result["_messages"], result["_assistant_msg"], tool_results)
        return {"type": "reply", "content": content, "tools": [c["name"] for c in result["calls"]], "tokens_used": _request_tokens}

    # Text response — escalate if DEFAULT_MODEL claimed a tool action without calling one
    if model == DEFAULT_MODEL and _claims(result["content"]):
        result = await chat(message, history=history or [], model=COMPLEX_MODEL, tools_override=tools_override)
        if result["type"] == "tool_call":
            tool_results = list(await asyncio.gather(*[_run_call(c) for c in result["calls"]]))
            content = await _safe_synthesize(result["_messages"], result["_assistant_msg"], tool_results)
        else:
            content = result["content"]
        return {"type": "reply", "content": content, "tools": [c["name"] for c in result.get("calls", [])], "tokens_used": _request_tokens}

    return {"type": "reply", "content": result["content"], "tools": [], "tokens_used": _request_tokens}


MAX_AGENT_TURNS = 6


async def run_react_loop(message: str, model: str, history: list = None, tools_override: list = None, system_addendum: str = None) -> dict:
    """ReAct agent loop: iterative tool use until task complete or MAX_AGENT_TURNS.

    Always uses the provided model (caller should pass COMPLEX_MODEL).
    Returns same shape as run_single_pass():
        {"type": "reply", "content": str, "tools": list, "tokens_used": int}
        {"type": "disambig", ..., "tokens_used": int}
    """
    global _request_tokens
    _request_tokens = 0

    async def _run_call(call):
        content = await _dispatch(call["name"], call["args"])
        return {"id": call["id"], "name": call["name"], "content": _cap_tool_result(content)}

    all_tools = []
    accumulated = None
    _nudged = False
    _empty_retried = False
    result = await chat(message, history=history or [], model=model, tools_override=tools_override, system_addendum=system_addendum)

    for _ in range(MAX_AGENT_TURNS):
        if result["type"] == "text":
            content = result["content"]

            # Recovery 1: model returned empty text without calling any real tool — nudge it
            if not content.strip() and not all_tools and not _nudged:
                _nudged = True
                nudge_messages = result.get("_messages", []) + [
                    {"role": "assistant", "content": ""},
                    {"role": "user", "content": "Please use the available tools to complete the request."},
                ]
                result = await chat(messages=nudge_messages, model=model, tools_override=tools_override)
                continue

            # Recovery 2: model called tools but returned empty final text — retry once
            if not content.strip() and accumulated and not _empty_retried:
                _empty_retried = True
                retry_messages = accumulated + [
                    {"role": "user", "content": "You have tool results above. Please summarize them into a complete response."},
                ]
                result = await chat(messages=retry_messages, model=model, tools_override=tools_override)
                continue

            return {"type": "reply", "content": content, "tools": all_tools, "tokens_used": _request_tokens}

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
                    "tokens_used": _request_tokens,
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
    try:
        content = await synthesize_plan_result(message, step_results, history or [])
    except Exception:
        content = "\n".join(r["content"] for r in step_results)
    return {"type": "reply", "content": content, "tools": all_tools, "tokens_used": _request_tokens}


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
