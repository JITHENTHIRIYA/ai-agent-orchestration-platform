"""
server.py — Standalone MCP server for the AI Agent Orchestration Platform.

What is MCP?
------------
The Model Context Protocol (MCP) is an open standard by Anthropic that lets
LLM agents discover and invoke "tools" hosted on a separate server over
JSON-RPC.  Instead of baking tool logic into the agent process, the agent:

  1. Connects to this server on startup.
  2. Calls `tools/list` to get a JSON Schema description of every available
     tool — name, description, parameter types, defaults.
  3. When the LLM decides to use a tool, the agent calls `tools/call` with
     the tool name and a dict of arguments.
  4. This server validates the arguments, executes the tool, and streams the
     result back to the agent.

MCP request / response lifecycle
----------------------------------
  Agent                          MCP Server (this file)
  ──────                         ──────────────────────
  GET /sse  ──────────────────►  Opens SSE stream, sends server-info + caps
  tools/list ─────────────────►  Iterates _registry, returns JSON Schema list
  tools/call { name, args } ──►  Routes to handler via FastMCP dispatch
                                    handler(query=..., top_k=...) executes
  ◄─────────────── result JSON      Returns { context, sources } (or error)

Why a separate port from FastAPI (8000)?
-----------------------------------------
FastAPI (8000) is the public-facing REST API for the frontend and external
clients.  It owns HTTP/REST conventions, CORS, auth middleware, etc.

The MCP server (8001) is an internal service consumed only by the agent layer.
Keeping it separate means:
  • Different auth and rate-limiting policies per service.
  • Each service can be restarted/scaled independently.
  • MCP JSON-RPC traffic never flows through REST middleware designed for
    different semantics (CORS headers on a JSON-RPC stream would be noise).

How tool routing works
-----------------------
FastMCP maintains an internal dispatch table keyed by tool name.  When a
`tools/call` request arrives, FastMCP:
  1. Looks up the handler by name in its internal table.
  2. Validates the incoming arguments against the JSON Schema derived from
     the handler's type annotations.
  3. Calls the handler and serialises the return value to JSON.

We feed FastMCP's dispatch table via `mcp.add_tool(handler, name, description)`
during `_register_all_tools()`.  The source of truth for which tools exist is
our own `tool_registry` module — FastMCP only sees what we push into it.

Starting the server
-------------------
As a standalone process:
    python -m mcp.server          # from backend/
    python backend/mcp/server.py  # from project root

The server listens on MCP_SERVER_HOST:MCP_SERVER_PORT (default localhost:8001)
using Server-Sent Events (SSE) transport, which is the recommended transport
for networked MCP servers.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any

# ---------------------------------------------------------------------------
# SDK import — resolving the `mcp` namespace collision
#
# Problem
# -------
# Our local package lives at  backend/mcp/  and is also named `mcp`.
# The Anthropic MCP SDK is installed as the `mcp` package in site-packages.
# Both share the same top-level name, so whichever appears first in sys.path
# wins.  Worse, by the time this file's code runs, Python has already:
#   1. Found backend/mcp/__init__.py and registered it as sys.modules['mcp'].
#   2. Started importing this file as sys.modules['mcp.server'].
# So even manipulating sys.path alone isn't enough — sys.modules must also be
# temporarily cleared of our local entries.
#
# Fix (two-phase)
# ---------------
# Phase 1 — before importing FastMCP:
#   a. Save all our local mcp.* entries from sys.modules and remove them so
#      the SDK's mcp package can be freshly imported without interference.
#   b. Prepend site-packages to sys.path so the SDK's mcp/ wins the race
#      against any remaining backend/ entries.
#
# Phase 2 — after importing FastMCP:
#   a. Restore our local mcp.* entries to sys.modules so subsequent imports
#      of mcp.tool_registry, mcp.tools.* resolve to our files.
#   b. Remove the temporarily prepended site-packages paths.
#   c. Ensure backend/ is in sys.path for config, rag, etc.
#
# Why this is safe at runtime
# ---------------------------
# FastMCP loads all of its internal dependencies (mcp.types, mcp.server.*,
# etc.) during module execution in Phase 1.  Those SDK sub-modules stay in
# sys.modules under their own keys.  Restoring our `mcp` / `mcp.server`
# entries in Phase 2 only affects the top-level and server keys — it does not
# evict the SDK's already-loaded sub-modules, so FastMCP's internals are
# untouched at runtime.
# ---------------------------------------------------------------------------
import site as _site

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Phase 1a — evict our local mcp.* from sys.modules.
_saved_modules = {
    k: sys.modules.pop(k)
    for k in list(sys.modules)
    if k == "mcp" or k.startswith("mcp.")
}

# Phase 1b — prepend site-packages so the SDK's mcp/ is found first.
_sdk_paths = _site.getsitepackages()
for _p in reversed(_sdk_paths):
    sys.path.insert(0, _p)

from mcp.server.fastmcp import FastMCP  # now resolves to installed SDK  # noqa: E402

# Phase 2a — restore our local mcp.* entries.
sys.modules.update(_saved_modules)

# Phase 2b — remove the temporarily prepended site-packages entries.
for _p in _sdk_paths:
    try:
        sys.path.remove(_p)
    except ValueError:
        pass

# Phase 2c — ensure backend/ is on sys.path for config, rag, tools, etc.
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from config import settings  # noqa: E402
from mcp.tool_registry import register_tool, list_tools  # noqa: E402

# Tool handler functions and their descriptions
from mcp.tools.rag_tool import rag_retriever, DESCRIPTION as RAG_DESC  # noqa: E402
from mcp.tools.summarizer_tool import doc_summarizer, DESCRIPTION as SUM_DESC  # noqa: E402
from mcp.tools.search_tool import knowledge_search, DESCRIPTION as SEARCH_DESC  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastMCP instance
#
# host / port are read from config so they can be overridden via .env without
# touching source code — useful when deploying behind a reverse proxy or in
# Docker where the MCP server may bind to 0.0.0.0 instead of localhost.
# ---------------------------------------------------------------------------
mcp = FastMCP(
    name=settings.MCP_SERVER_NAME,   # "ai-agent-platform-mcp"
    host=settings.MCP_SERVER_HOST,   # default: "localhost"
    port=settings.MCP_SERVER_PORT,   # default: 8001
    log_level="INFO",
)


# ---------------------------------------------------------------------------
# Tool registration
#
# Why not hardcode tools in this file?
# Every tool added here would mean editing the server — increasing the blast
# radius of changes and coupling unrelated tools together.  Instead:
#   1. register_tool() adds the entry to our application-level registry.
#   2. _bind_tools_to_mcp() iterates the registry and calls mcp.add_tool()
#      to push each entry into FastMCP's dispatch table.
#
# Adding a new tool in future weeks = one register_tool() call here (or in a
# dedicated setup module), zero changes to the dispatch / routing logic below.
# ---------------------------------------------------------------------------

def _register_all_tools() -> None:
    """
    Populate the tool registry and bind every entry to the FastMCP instance.

    This is intentionally a two-phase process:
      Phase 1 (register_tool) — record handler + description in our own dict.
      Phase 2 (mcp.add_tool)  — push handler into FastMCP's dispatch table.

    Separating the phases means we can introspect the registry (health checks,
    admin endpoints) independently of the FastMCP instance.
    """
    # Phase 1 — register in our application registry.
    # register_tool() raises ValueError on duplicate names, catching bugs where
    # the same tool is accidentally registered twice during startup.
    register_tool(
        name="rag_retriever",
        description=RAG_DESC,
        handler=rag_retriever,
    )
    register_tool(
        name="doc_summarizer",
        description=SUM_DESC,
        handler=doc_summarizer,
    )
    register_tool(
        name="knowledge_search",
        description=SEARCH_DESC,
        handler=knowledge_search,
    )

    # Phase 2 — bind to FastMCP.
    # mcp.add_tool() derives the JSON Schema from the handler's type annotations
    # and adds it to FastMCP's internal dispatch table.  From this point on,
    # `tools/list` and `tools/call` will include these tools.
    for entry in list_tools():
        mcp.add_tool(
            fn=entry.handler,
            name=entry.name,
            description=entry.description,
        )
        logger.info("Registered MCP tool: %s", entry.name)


# ---------------------------------------------------------------------------
# Health check
#
# Called by monitoring systems (Docker HEALTHCHECK, Kubernetes liveness probe)
# or by the FastAPI backend to confirm the MCP server is reachable before
# routing agent requests to it.
# ---------------------------------------------------------------------------

def health_check() -> dict[str, Any]:
    """
    Return a structured health status dict for this MCP server.

    The check verifies two things:
      1. The server process is running (trivially true if this function runs).
      2. All expected tools are registered (guards against partial startup
         failures where one tool's import raised an exception).

    Returns
    -------
    dict
        {
          "status":      "ok" | "degraded",
          "server_name": str,
          "host":        str,
          "port":        int,
          "tools":       list[str],   # names of registered tools
          "tool_count":  int,
        }

    A status of "degraded" means fewer tools than expected are registered;
    the caller should alert and investigate before routing agent traffic here.
    """
    registered = list_tools()
    tool_names = [t.name for t in registered]

    expected = {"rag_retriever", "doc_summarizer", "knowledge_search"}
    all_present = expected.issubset(set(tool_names))

    return {
        "status": "ok" if all_present else "degraded",
        "server_name": settings.MCP_SERVER_NAME,
        "host": settings.MCP_SERVER_HOST,
        "port": settings.MCP_SERVER_PORT,
        "tools": tool_names,
        "tool_count": len(tool_names),
    }


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def create_server() -> FastMCP:
    """
    Initialise and return the configured FastMCP server instance.

    Calling this function:
      1. Registers all tools in the application registry.
      2. Binds each tool to the FastMCP dispatch table.
      3. Returns the server instance ready to be started with .run().

    Separating creation from startup (mcp.run()) makes it possible to import
    the server object in tests or other processes without immediately binding
    a network port.

    Returns
    -------
    FastMCP
        Configured, tool-loaded server instance.
    """
    _register_all_tools()
    status = health_check()
    logger.info(
        "MCP server '%s' initialised — %d tool(s) registered: %s",
        status["server_name"],
        status["tool_count"],
        ", ".join(status["tools"]),
    )
    if status["status"] == "degraded":
        logger.warning(
            "Health check: DEGRADED — some expected tools are missing: %s",
            status,
        )
    return mcp


# ---------------------------------------------------------------------------
# Entry point
#
# Allows the server to be started directly:
#   python -m mcp.server       (from backend/)
#   python backend/mcp/server.py
#
# Transport: SSE (Server-Sent Events) is the recommended transport for
# networked MCP servers.  It uses a long-lived HTTP GET /sse connection for
# server→agent messages and POST /messages/ for agent→server messages.
# Use "stdio" only for local subprocess-based MCP (e.g. Claude Desktop).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    server = create_server()

    logger.info(
        "Starting MCP server on %s:%s (transport=sse)",
        settings.MCP_SERVER_HOST,
        settings.MCP_SERVER_PORT,
    )

    # mcp.run() is synchronous and blocks until the server is stopped.
    # It internally runs an asyncio event loop using uvicorn / anyio.
    server.run(transport="sse")
