"""
mcp_agent.py — ReAct agent that communicates with the MCP server.

How this differs from rag_agent.py (direct vs MCP-mediated tool calls)
-----------------------------------------------------------------------
rag_agent.py calls tool logic DIRECTLY inside the agent process:

    Agent → LangChain tool function → rag.retriever.retrieve() → Pinecone

mcp_agent.py is different — tool logic lives in a SEPARATE MCP server process.
The agent never imports or calls retrieval/summarisation code itself.  Instead,
it communicates with the MCP server over the network using the Model Context
Protocol (JSON-RPC over SSE):

    Agent → LangChain tool stub → MCP server (port 8001) → tool function → result

Advantages of MCP mediation:
  • Tools can be updated, scaled, or replaced without touching the agent code.
  • Multiple agents can share the same MCP server (single source of truth).
  • The agent only needs to know the MCP server address — it discovers which
    tools exist at runtime via `tools/list`, no hard-coded tool names.
  • Tool implementations are isolated from the LLM process, so a crash in a
    tool doesn't kill the agent (the MCP server absorbs it).

How the agent decides which tool to use (ReAct reasoning)
----------------------------------------------------------
LangChain's ReAct agent follows this loop for every query:

  1. Thought — the LLM reads the user query plus the list of available tools
               (their names and descriptions) and produces a reasoning trace
               explaining what information it needs and which tool to call.

  2. Action  — the LLM emits:
                 Action: <tool_name>
                 Action Input: <arguments>
               LangChain parses this and calls the corresponding tool stub.

  3. Observe — the tool stub sends an MCP `tools/call` request to the server,
               receives the result, and returns it as a string to LangChain,
               which appends it to the scratchpad as "Observation: ...".

  4. Repeat  — the LLM sees the Observation and either issues another
               Action (if it needs more information) or emits "Final Answer".

Tool selection is driven purely by the LLM reading each tool's description.
A well-written description is the most important signal — the LLM will
prefer the tool whose description best matches the current goal.

What happens when no tool is needed (direct LLM response)
----------------------------------------------------------
If the LLM's first Thought concludes it already knows the answer (e.g. a
general-knowledge question with no domain-specific data needed), it emits
"Final Answer" without any Action.  The AgentExecutor detects this and
returns the answer immediately without contacting the MCP server at all.
This is identical behaviour to rag_agent.py — both are ReAct agents; the
difference is only in where tool *execution* happens when a tool IS called.

The MCP round-trip: agent → server → tool → server → agent
-----------------------------------------------------------
  ┌───────────────────────────────────────────────────────────────────────┐
  │ 1. LangChain tool stub receives "Action Input" string from the LLM.   │
  │ 2. Stub parses the string → arguments dict (JSON or {"query": text}). │
  │ 3. Stub opens an SSE connection to http://{host}:{port}/sse.          │
  │ 4. Stub calls session.initialize() → MCP handshake.                   │
  │ 5. Stub calls session.call_tool(name, arguments) → JSON-RPC request.  │
  │    MCP server receives the request, validates arguments against the    │
  │    tool's JSON Schema, calls the Python handler function.              │
  │ 6. Handler returns a dict (e.g. { context, sources }).                 │
  │    MCP server serialises it to JSON and sends it back over SSE.        │
  │ 7. ClientSession receives CallToolResult and returns it to the stub.   │
  │ 8. Stub serialises the result to a string → LangChain Observation.     │
  │ 9. LLM reads Observation and continues the ReAct loop.                 │
  └───────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import site as _site
import sys
from typing import Any

# ---------------------------------------------------------------------------
# MCP SDK import — resolving the `mcp` namespace collision
#
# Our local backend/mcp/ package and the installed Anthropic MCP SDK both use
# the top-level name `mcp`.  We need to import the SDK's ClientSession and
# sse_client before our local package shadows them in sys.modules.
#
# The same two-phase sys.modules + sys.path trick used in mcp/server.py:
#   Phase 1 — evict local mcp.* from sys.modules, prepend site-packages.
#   Phase 2 — import SDK client, then restore local mcp.* entries.
# ---------------------------------------------------------------------------
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_saved_modules = {
    k: sys.modules.pop(k)
    for k in list(sys.modules)
    if k == "mcp" or k.startswith("mcp.")
}

_sdk_paths = _site.getsitepackages()
for _p in reversed(_sdk_paths):
    sys.path.insert(0, _p)

from mcp import ClientSession          # Anthropic MCP SDK  # noqa: E402
from mcp.client.sse import sse_client  # SSE transport client  # noqa: E402

sys.modules.update(_saved_modules)
for _p in _sdk_paths:
    try:
        sys.path.remove(_p)
    except ValueError:
        pass

if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# ---------------------------------------------------------------------------
# Standard imports (after sys.path is set up for backend/)
# ---------------------------------------------------------------------------
from langchain.agents import AgentExecutor, create_react_agent  # noqa: E402
from langchain_core.prompts import PromptTemplate  # noqa: E402
from langchain_core.tools import Tool  # noqa: E402
from langchain_groq import ChatGroq  # noqa: E402

from config import settings  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MCP server connection details
#
# The SSE endpoint follows FastMCP's default URL scheme:
#   GET  http://{host}:{port}/sse      — long-lived SSE stream (server→agent)
#   POST http://{host}:{port}/messages/ — JSON-RPC messages (agent→server)
#
# ClientSession manages the session ID and routes messages automatically;
# we only need to point sse_client at the /sse URL.
# ---------------------------------------------------------------------------
_MCP_SSE_URL = (
    f"http://{settings.MCP_SERVER_HOST}:{settings.MCP_SERVER_PORT}/sse"
)

# Module-level state: track the most recently used tool name and input so
# run_mcp_agent() can include them in its structured return value.
_last_tool_call: dict[str, Any] = {"name": None, "input": None}


# ---------------------------------------------------------------------------
# Async MCP client helpers
#
# Every interaction with the MCP server is async (SSE is inherently async).
# We open a fresh ClientSession per call to keep sessions short-lived and
# stateless — no persistent connection to manage or reconnect.
# ---------------------------------------------------------------------------

async def _async_list_tools() -> list[dict[str, Any]]:
    """
    Query the MCP server for its registered tools via `tools/list`.

    Opens an SSE session, performs the MCP handshake (initialize), then
    calls list_tools() which sends a JSON-RPC `tools/list` request.  The
    server responds with a ListToolsResult containing Tool objects that
    include the name, description, and JSON-Schema input spec for each tool.

    Returns
    -------
    list[dict]
        One dict per tool: { name, description, inputSchema }
    """
    async with sse_client(_MCP_SSE_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "inputSchema": t.inputSchema or {},
                }
                for t in result.tools
            ]


async def _async_call_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    """
    Send a `tools/call` JSON-RPC request to the MCP server.

    The MCP round-trip:
      1. Open SSE connection → server sends session ID + capabilities.
      2. session.initialize() completes the MCP protocol handshake.
      3. session.call_tool() sends:
           { "method": "tools/call",
             "params": { "name": tool_name, "arguments": arguments } }
      4. Server validates arguments against the tool's JSON Schema.
      5. Server executes the Python handler and wraps the return value in a
         CallToolResult with a list of ContentBlock objects.
      6. We extract the text/structured content and return it as a string
         so LangChain can append it to the ReAct scratchpad as an Observation.

    Parameters
    ----------
    tool_name : str
        The registered MCP tool name (e.g. "rag_retriever").
    arguments : dict
        Validated arguments matching the tool's input schema.

    Returns
    -------
    str
        Serialised tool result ready for use as a ReAct Observation.
    """
    async with sse_client(_MCP_SSE_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)

            # isError is True when the MCP server caught an exception inside
            # the tool handler — surface it as a string so the agent can
            # reason about the failure and either retry or answer differently.
            if result.isError:
                error_text = " ".join(
                    getattr(block, "text", str(block))
                    for block in (result.content or [])
                )
                return f"[MCP tool error] {error_text}"

            if not result.content:
                return "[MCP tool returned no content]"

            # CallToolResult.content is a list of ContentBlock objects.
            # For our tools (which return dicts), the SDK serialises the dict
            # to JSON and wraps it in a TextContent block.  We join all text
            # blocks so the agent receives the full result as one string.
            parts: list[str] = []
            for block in result.content:
                text = getattr(block, "text", None)
                if text is not None:
                    parts.append(text)
                else:
                    parts.append(str(block))

            return "\n".join(parts)


# ---------------------------------------------------------------------------
# Sync bridge
#
# LangChain tool functions must be synchronous; MCP client calls are async.
# _run_async() bridges this gap:
#   • If no event loop is running (standalone script), asyncio.run() works.
#   • If an event loop is already running (FastAPI/uvicorn), asyncio.run()
#     raises RuntimeError("This event loop is already running").  In that
#     case we spin up a ThreadPoolExecutor and call asyncio.run() in a fresh
#     thread which has no event loop attached — safe and deadlock-free.
# ---------------------------------------------------------------------------

def _run_async(coro) -> Any:
    """Run an async coroutine from a synchronous context."""
    try:
        asyncio.get_running_loop()
        # Already inside an event loop — run in a new thread instead.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        # No running loop — safe to call asyncio.run() directly.
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# LangChain tool factory
#
# For each tool advertised by the MCP server, we create a LangChain Tool
# object whose `func` is a closure that:
#   1. Parses the agent's "Action Input" string into an arguments dict.
#   2. Records the tool call in _last_tool_call for the return payload.
#   3. Calls _run_async(_async_call_tool(...)) to reach the MCP server.
#
# Why Tool (not @tool decorator)?
# langchain_core.tools.Tool accepts a `name` parameter at construction time,
# making it the right choice for dynamically-named tools.  The @tool decorator
# derives the name from the Python function name — unusable for tools whose
# names we only know at runtime.
#
# How the agent passes arguments:
# In the ReAct text format, "Action Input" is a plain string.  Our tools
# primarily accept a `query` field, so we try JSON parsing first (in case
# the LLM emits a JSON object), then fall back to wrapping the string as
# {"query": input}.  This handles both cases gracefully.
# ---------------------------------------------------------------------------

def _make_mcp_tool(name: str, description: str) -> Tool:
    """
    Create a LangChain Tool that proxies calls to the named MCP tool.

    Parameters
    ----------
    name : str
        MCP tool name (e.g. "rag_retriever", "doc_summarizer").
    description : str
        Tool description shown to the LLM in the ReAct prompt.

    Returns
    -------
    langchain_core.tools.Tool
    """
    def _invoke(action_input: str) -> str:
        """
        Parse Action Input → arguments dict → call MCP server → return result.

        The ReAct agent emits "Action Input: <string>".  We accept two forms:
          • JSON object string: {"query": "...", "top_k": 5}
          • Plain query string: "What is machine learning?"

        The plain-string fallback covers the common case where the LLM emits
        a simple query phrase rather than a structured JSON object.
        """
        global _last_tool_call

        raw = action_input.strip()

        # Attempt to parse as JSON; fall back to wrapping in {"query": ...}.
        try:
            arguments = json.loads(raw)
            if not isinstance(arguments, dict):
                arguments = {"query": raw}
        except (json.JSONDecodeError, ValueError):
            arguments = {"query": raw}

        # Record this call so run_mcp_agent() can surface it in the return dict.
        _last_tool_call = {"name": name, "input": arguments}

        logger.debug("MCP tool call → %s(%s)", name, arguments)

        return _run_async(_async_call_tool(name, arguments))

    return Tool(name=name, description=description, func=_invoke)


# ---------------------------------------------------------------------------
# ReAct prompt
#
# Mirrors the style of rag_agent.py's RAG_REACT_PROMPT.  The key difference:
# tool descriptions come from the MCP server's `tools/list` response, so
# the agent's tool knowledge is always in sync with what the server offers —
# no stale hardcoded descriptions.
# ---------------------------------------------------------------------------

_MCP_REACT_PROMPT = PromptTemplate.from_template(
    """You are a helpful assistant with access to a set of tools provided by an MCP server.

You have access to the following tools:
{tools}

Guidelines:
1. For questions requiring knowledge retrieval, use rag_retriever or knowledge_search.
2. For summarising long text, use doc_summarizer.
3. Use knowledge_search when the query relates to a specific category or domain.
4. If no tool is relevant, answer directly from your own knowledge.
5. Base your final answer on tool results when tools were called.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
)

# ---------------------------------------------------------------------------
# Agent builder (lazy, cached)
#
# Building the agent requires fetching the tool list from the MCP server.
# We cache the executor so repeated run_mcp_agent() calls reuse the same
# LangChain objects (avoids the MCP list_tools round-trip each time).
# The cache is invalidated by calling _reset_agent_cache() — useful in tests
# or when the MCP server restarts with a changed tool set.
# ---------------------------------------------------------------------------
_agent_executor_cache: AgentExecutor | None = None


def _reset_agent_cache() -> None:
    """Invalidate the cached agent executor (forces a rebuild on next call)."""
    global _agent_executor_cache
    _agent_executor_cache = None


def _build_mcp_agent_executor() -> AgentExecutor:
    """
    Build and cache a LangChain AgentExecutor backed by MCP tools.

    Steps:
      1. Fetch the tool list from the MCP server via `tools/list`.
      2. Create a LangChain Tool wrapper for each MCP tool.
      3. Initialise the Groq LLM (llama-3.3-70b-versatile).
      4. Wire LLM + tools + prompt into a create_react_agent chain.
      5. Wrap in AgentExecutor and cache.

    Raises
    ------
    RuntimeError
        If the MCP server is unreachable or returns no tools.

    Returns
    -------
    AgentExecutor
    """
    global _agent_executor_cache
    if _agent_executor_cache is not None:
        return _agent_executor_cache

    # Step 1 — discover tools from the live MCP server.
    try:
        mcp_tools_meta = _run_async(_async_list_tools())
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach MCP server at {_MCP_SSE_URL}.  "
            f"Ensure 'python -m mcp.server' is running.  Error: {exc}"
        ) from exc

    if not mcp_tools_meta:
        raise RuntimeError(
            f"MCP server at {_MCP_SSE_URL} returned an empty tool list."
        )

    logger.info(
        "MCP agent discovered %d tool(s): %s",
        len(mcp_tools_meta),
        [t["name"] for t in mcp_tools_meta],
    )

    # Step 2 — wrap each MCP tool in a LangChain Tool stub.
    langchain_tools = [
        _make_mcp_tool(name=t["name"], description=t["description"])
        for t in mcp_tools_meta
    ]

    # Step 3 — Groq LLM (same model as rag_agent.py for consistency).
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=settings.GROQ_API_KEY,
    )

    # Step 4 — ReAct agent chain.
    react_agent = create_react_agent(
        llm=llm,
        tools=langchain_tools,
        prompt=_MCP_REACT_PROMPT,
    )

    # Step 5 — AgentExecutor manages the Thought→Action→Observe loop.
    _agent_executor_cache = AgentExecutor(
        agent=react_agent,
        tools=langchain_tools,
        verbose=True,          # print full ReAct trace to stdout
        handle_parsing_errors=True,  # let LLM self-correct on bad output
        max_iterations=6,      # cap loop depth to prevent runaway chains
    )
    return _agent_executor_cache


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_available_tools() -> list[dict[str, Any]]:
    """
    Ask the MCP server which tools are currently registered.

    This function is useful for:
      • Admin / health endpoints that want to display available capabilities.
      • Dynamic UI that shows users what the agent can do.
      • Testing that the MCP server is reachable and correctly configured.

    Returns
    -------
    list[dict]
        Each dict has keys: name (str), description (str), inputSchema (dict).
        Returns an empty list if the MCP server is unreachable.

    Examples
    --------
    >>> tools = list_available_tools()
    >>> [t["name"] for t in tools]
    ['rag_retriever', 'doc_summarizer', 'knowledge_search']
    """
    try:
        return _run_async(_async_list_tools())
    except Exception as exc:
        logger.error("list_available_tools failed: %s", exc)
        return []


def run_mcp_agent(query: str) -> dict[str, Any]:
    """
    Execute the MCP-backed ReAct agent on a user query.

    The agent:
      1. Fetches available tools from the MCP server (cached after first call).
      2. Runs the ReAct loop — the LLM reasons and optionally calls MCP tools.
      3. Each tool call sends a JSON-RPC request to the MCP server, which
         executes the real tool function (retrieval, summarisation, search)
         and streams the result back.
      4. The LLM composes a Final Answer from the accumulated Observations.

    When no tool is needed (direct LLM response)
    ---------------------------------------------
    If the LLM's first Thought decides it already has enough knowledge to
    answer, it emits "Final Answer" without any Action.  No MCP request is
    sent.  tool_used and tool_input will be None in the return dict.

    Parameters
    ----------
    query : str
        The user's natural-language question or instruction.

    Returns
    -------
    dict with keys:
      query      : str  — the original query (echoed for traceability)
      response   : str  — the agent's final answer
      tool_used  : str | None — name of the last MCP tool called (or None)
      tool_input : dict | None — arguments passed to that tool (or None)
      status     : "success" | "error"
    """
    global _last_tool_call
    _last_tool_call = {"name": None, "input": None}

    if not query or not query.strip():
        return {
            "query": query,
            "response": "Query is empty.",
            "tool_used": None,
            "tool_input": None,
            "status": "error",
        }

    try:
        executor = _build_mcp_agent_executor()
        result = executor.invoke({"input": query})

        return {
            "query": query,
            "response": result.get("output", ""),
            "tool_used": _last_tool_call["name"],
            "tool_input": _last_tool_call["input"],
            "status": "success",
        }

    except RuntimeError as exc:
        # RuntimeError from _build_mcp_agent_executor means the MCP server
        # is not reachable.  Surface a clear message rather than a traceback.
        logger.error("MCP agent setup failed: %s", exc)
        return {
            "query": query,
            "response": str(exc),
            "tool_used": None,
            "tool_input": None,
            "status": "error",
        }

    except Exception as exc:
        logger.error("MCP agent execution failed: %s", exc)
        return {
            "query": query,
            "response": f"Agent execution failed: {exc}",
            "tool_used": _last_tool_call["name"],
            "tool_input": _last_tool_call["input"],
            "status": "error",
        }
