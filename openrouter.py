"""OpenRouter LLM client — shared engine for hoid and vespyn.

Provides: chat(), run_react_loop(),
          synthesize_tool_result(), synthesize_plan_result(),
          prune_tool_results()

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

# Alternative backends — models with these prefixes route directly, bypassing OpenRouter.
_ALT_BACKENDS: dict[str, tuple[str, str, dict]] = {}  # prefix → (url, api_key, extra_payload)
_groq_key = os.getenv("GROQ_API_KEY", "")
if _groq_key:
    _ALT_BACKENDS["groq/"] = (
        "https://api.groq.com/openai/v1/chat/completions",
        _groq_key,
        {"reasoning_format": "hidden"},
    )

_gemini_key = os.getenv("GEMINI_API_KEY", "")
if _gemini_key:
    _ALT_BACKENDS["google/"] = (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        _gemini_key,
        {},
    )

_anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
if _anthropic_key:
    # "ant/" prefix routes directly to Anthropic's OpenAI-compatible endpoint.
    # Saves OpenRouter's 5% markup + reduces latency (one fewer hop).
    # Docs: https://platform.claude.com/docs/en/api/openai-sdk
    _ALT_BACKENDS["ant/"] = (
        "https://api.anthropic.com/v1/chat/completions",
        _anthropic_key,
        {},
    )

_openai_key = os.getenv("OPENAI_API_KEY", "")
if _openai_key:
    # "oai/" prefix routes directly to OpenAI API, bypassing OpenRouter.
    # Use "openai/" prefix (no alt backend) for OpenRouter-hosted OpenAI models
    # like openai/gpt-oss-120b that don't exist on OpenAI's own API.
    _ALT_BACKENDS["oai/"] = (
        "https://api.openai.com/v1/chat/completions",
        _openai_key,
        {},
    )



def _resolve_endpoint(model: str) -> tuple[str, dict, str, dict]:
    """Return (api_url, headers, model_id, extra_payload) for the given model.

    For alt backends, strips the routing prefix so the downstream API receives
    the native model name (e.g. 'groq/qwen/qwen3-32b' → 'qwen/qwen3-32b').
    """
    for prefix, (url, key, extra) in _ALT_BACKENDS.items():
        if model and model.startswith(prefix):
            native_id = model[len(prefix):]
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            return url, headers, native_id, extra
    # Default: OpenRouter
    headers = {"Authorization": f"Bearer {_API_KEY}", "Content-Type": "application/json"}
    return _API_URL, headers, model, {}

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "google/gemini-2.5-flash-lite")
NOTES_MODEL = os.getenv("NOTES_MODEL", DEFAULT_MODEL)
SUMMARIZATION_MODEL = os.getenv("SUMMARIZATION_MODEL", DEFAULT_MODEL)
COMPLEX_MODEL = os.getenv("COMPLEX_MODEL", SUMMARIZATION_MODEL)
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", COMPLEX_MODEL)
WORKER_MODEL = os.getenv("WORKER_MODEL", "google/gemini-3.1-flash-lite")
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "anthropic/claude-sonnet-4-6")
SYNTHESIS_MODEL = os.getenv("SYNTHESIS_MODEL", SUMMARIZATION_MODEL)

# ReAct loop addenda — appended to system prompt for interactive vs. agent contexts.
# Callers pass these via system_addendum= to run_react_loop().
REACT_ADDENDUM_AGENT = (
    "- Always make at least one tool call before producing a final answer. Never return an empty first-turn response.\n"
    "- If tools return no useful information, state exactly what you searched for and confirm the information could not be found.\n"
    "- Stop as soon as you have a complete answer — do not make additional tool calls once the task is done.\n"
    "- After 5 tool calls without a complete answer, stop and report what you found so far.\n"
    "- Your final answer must directly address the request and nothing else. No unsolicited context, caveats, or padding.\n"
    "- Never expose your internal process in the final answer. Do not include tool call parameters, search queries, "
    "chain-of-thought reasoning, or \"I searched for X\" narration. Present only polished results.\n"
    "- NEVER output tool calls as JSON text. Always use the tool calling mechanism provided. Your text responses must be natural language only."
)

REACT_ADDENDUM_CHAT = (
    "- For conversational questions or topics answerable from existing knowledge, respond directly without making unnecessary tool calls.\n"
    "- For questions requiring current data, use the appropriate tool before answering. This includes topics such as prices, weather, news, scores, and market odds.\n"
    "- Before any significant tool call, state in one brief line the purpose of the call and the minimal inputs being used.\n"
    "- Use only the provided tools. If a needed tool is unavailable or returns no information, briefly say so and then offer the best available knowledge instead.\n"
    "- After each tool call, briefly validate whether the result answered the question; if not, make at most one additional targeted tool call when necessary.\n"
    "- Keep responses concise; this is an interactive chat session, not a report.\n"
    "- Stop once the question is answered. Do not make additional tool calls after you have a satisfactory answer.\n"
    "- Reason internally as needed, but do not reveal private chain-of-thought unless explicitly requested.\n"
    "- NEVER output tool calls as JSON text. Always use the tool calling mechanism provided. Your text responses must be natural language only."
)

# Token usage accumulator — reset by run_react_loop() at entry,
# incremented by chat() and synthesize_tool_result() on each API call.
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
            # Some providers return 200 with an error body and no choices
            try:
                body = response.json()
                if "error" in body and "choices" not in body:
                    if attempt < _MAX_LLM_RETRIES - 1:
                        await asyncio.sleep(_LLM_RETRY_BASE * (2 ** attempt))
                        continue
            except (ValueError, KeyError):
                pass
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
_run_worker = None           # (tool_name: str, task: str, *, user_intent: str) -> WorkerResult


def init(*, registry=None, context_loader=None, synthesis_shortcircuit=None, worker=None):
    """Inject dependencies. Call once at startup before any chat/dispatch calls.

    registry:       module with get_schemas(), dispatch(), is_dispatch_error(),
                    claims_tool_action() — typically tools.registry or equivalent
    context_loader: callable() -> str that returns the default system prompt
    synthesis_shortcircuit: callable(tool_results) -> str | None — if it returns
                    a string, synthesize_tool_result() returns it immediately
                    instead of making a synthesis API call
    """
    global _get_schemas, _dispatch, _is_dispatch_error, _claims_tool_action, _load_context, _synthesis_shortcircuit, _run_worker
    if registry is not None:
        _get_schemas = registry.get_schemas
        _dispatch = registry.dispatch
        _is_dispatch_error = getattr(registry, 'is_dispatch_error', lambda r: False)
        _claims_tool_action = getattr(registry, 'claims_tool_action', lambda t: False)
    if context_loader is not None:
        _load_context = context_loader
    if synthesis_shortcircuit is not None:
        _synthesis_shortcircuit = synthesis_shortcircuit
    if worker is not None:
        _run_worker = worker


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

async def chat(message: str = None, system: str = None, use_tools: bool = True, history: list = None, model: str = None, messages: list = None, tools_override: list = None, system_addendum: str = None, json_mode: bool = False) -> dict:
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

    used_model = model or DEFAULT_MODEL
    api_url, api_headers, resolved_model, extra_payload = _resolve_endpoint(used_model)

    payload = {
        "model": resolved_model,
        "messages": messages,
        **extra_payload,
    }
    if json_mode:
        # Anthropic's OpenAI-compatible endpoint doesn't support json_object mode —
        # only json_schema with a full schema. Skip response_format for ant/ models;
        # the system prompt already instructs JSON output.
        if not (used_model and used_model.startswith("ant/")):
            payload["response_format"] = {"type": "json_object"}
    if use_tools:
        payload["tools"] = tools_override if tools_override is not None else (_get_schemas() if _get_schemas else [])
        payload["tool_choice"] = "auto"

    async with httpx.AsyncClient(timeout=90) as client:
        response = await _post_with_retry(client, api_url, api_headers, payload)
        if response.status_code >= 400:
            # If a direct backend failed, retry the same model via OpenRouter
            _OR_PREFIX_MAP = {"oai/": "openai/", "ant/": "anthropic/", "groq/": ""}
            matched_prefix = next((p for p in _ALT_BACKENDS if used_model and used_model.startswith(p)), None)
            if matched_prefix:
                or_model = _OR_PREFIX_MAP.get(matched_prefix, matched_prefix) + used_model[len(matched_prefix):]
                or_headers = {"Authorization": f"Bearer {_API_KEY}", "Content-Type": "application/json"}
                payload["model"] = or_model
                payload.pop("reasoning_format", None)
                response = await _post_with_retry(client, _API_URL, or_headers, payload)
            if response.status_code >= 400 and FALLBACK_MODEL and used_model != FALLBACK_MODEL:
                fb_url, fb_headers, fb_model, fb_extra = _resolve_endpoint(FALLBACK_MODEL)
                payload["model"] = fb_model
                payload.update(fb_extra)
                response = await _post_with_retry(client, fb_url, fb_headers, payload)
            if response.status_code >= 400:
                raise httpx.HTTPStatusError(
                    f"LLM {response.status_code}: {response.text}",
                    request=response.request,
                    response=response,
                )

    body = response.json()
    if "choices" not in body or not body["choices"]:
        raise httpx.HTTPStatusError(
            f"LLM returned no choices: {response.text[:500]}",
            request=response.request,
            response=response,
        )
    msg = body["choices"][0]["message"]

    global _request_tokens
    _request_tokens += body.get("usage", {}).get("total_tokens", 0)

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
    async with httpx.AsyncClient(timeout=90) as client:
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


MAX_AGENT_TURNS = 6


def _try_parse_text_tool_call(text: str) -> tuple[str, dict] | None:
    """Detect when a model outputs a JSON tool call as text instead of using tool_calls.

    Returns (tool_name, args_dict) if detected, else None.
    Looks for patterns like: {"name": "capture_note", "args": {...}}
    """
    stripped = text.strip()
    # Must look like JSON
    if not stripped.startswith("{"):
        return None
    try:
        data = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        # Try extracting JSON from surrounding text
        m = re.search(r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"args"\s*:\s*\{[^}]*\}[^}]*\}', stripped)
        if not m:
            return None
        try:
            data = json.loads(m.group())
        except (json.JSONDecodeError, TypeError):
            return None

    name = data.get("name")
    args = data.get("args")
    if isinstance(name, str) and isinstance(args, dict):
        return name, args
    return None


async def run_react_loop(message: str, model: str, history: list = None, tools_override: list = None, system_addendum: str = None, user_intent: str = None) -> dict:
    """ReAct agent loop: iterative tool use until task complete or MAX_AGENT_TURNS.

    When user_intent is provided and a worker is configured, tool calls are
    dispatched through scoped workers instead of direct dispatch.

    Returns:
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

            # Recovery 0: model output a JSON tool call as text instead of using
            # the tool calling mechanism (common with some models in ReAct loops).
            # Parse it and dispatch if it looks like {"name": "...", "args": {...}}.
            parsed_call = _try_parse_text_tool_call(content)
            if parsed_call and _dispatch:
                call_name, call_args = parsed_call
                tr_content = await _dispatch(call_name, call_args)
                tr_content = _cap_tool_result(tr_content)
                all_tools.append(call_name)
                # Build accumulated context and continue the loop
                synth_messages = (accumulated or result.get("_messages", [])) + [
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": f"Tool {call_name} returned:\n{tr_content}\n\nPlease provide a natural language response based on this result."},
                ]
                result = await chat(messages=synth_messages, model=model, tools_override=tools_override)
                continue

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

        all_tools.extend(c["name"] for c in result["calls"])

        if user_intent and _run_worker:
            # Extract prior tool results for context-hungry tools (e.g. run_code_task)
            _prior_context = []
            if accumulated:
                for m in accumulated:
                    if m.get("role") == "tool" and m.get("content"):
                        _prior_context.append({"tool": m.get("name", ""), "content": m["content"][:2000]})

            # Worker dispatch path: each tool call → scoped worker
            async def _dispatch_worker(i, call):
                return await _run_worker(
                    call["name"],
                    call["args"].get("task", json.dumps(call["args"])),
                    user_intent=user_intent,
                    prior_context=_prior_context or None,
                )

            worker_results = list(
                await asyncio.gather(*[_dispatch_worker(i, c) for i, c in enumerate(result["calls"])])
            )

            # Build tool results from worker outputs
            tool_results = [
                {
                    "id": result["calls"][i]["id"],
                    "name": result["calls"][i]["name"],
                    "content": wr.output if wr.success else f"[Worker error: {wr.error}]",
                }
                for i, wr in enumerate(worker_results)
            ]
        else:
            # Direct dispatch path (no workers)
            tool_results = list(await asyncio.gather(*[_run_call(c) for c in result["calls"]]))

        # Surface disambiguation sentinel immediately
        for tr in tool_results:
            if tr["content"].startswith("__DISAMBIG__|"):
                payload = json.loads(tr["content"].split("|", 1)[1])
                result = {
                    "type": "disambig",
                    "matches": payload["matches"],
                    "action": payload.get("action", "done"),
                    "params": payload.get("params", {}),
                    "tools": all_tools,
                    "tokens_used": _request_tokens,
                }
                # Forward extra keys (e.g. contact_files) for downstream handlers
                for k in payload:
                    if k not in ("matches", "action", "params"):
                        result[k] = payload[k]
                return result

        # Surface sandbox shortcircuit
        for tr in tool_results:
            if tr["content"].startswith("Working on it"):
                return {
                    "type": "reply",
                    "content": tr["content"],
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
                "content": _cap_tool_result(str(r["content"])),
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
