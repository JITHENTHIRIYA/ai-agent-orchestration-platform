"""
tool_registry.py — Central registry for all MCP tools.

Why register tools instead of hardcoding them in the server?
-------------------------------------------------------------
Hardcoding tool wiring directly in server.py creates a tight coupling:
every new tool requires editing the server file, risking accidental breakage
of existing tools and making the file grow unbounded over time.

A registry pattern solves this by acting as the single source of truth for
tool metadata.  The server iterates the registry at startup and registers
each entry with FastMCP — it never needs to know tool names or descriptions
individually.  Adding a new tool in future weeks means:

  1. Write the handler function in backend/mcp/tools/my_tool.py.
  2. Call register_tool() once (e.g. in a tools/setup.py or directly here).
  3. Done — the server picks it up automatically on next restart.

Registry vs FastMCP's internal tool list
-----------------------------------------
FastMCP maintains its own internal tool list used for JSON-RPC dispatch.
This registry is our application-level bookkeeping layer that sits above it:

  • It stores our handler callables and descriptions in one place.
  • It makes it easy to introspect registered tools without going through the
    FastMCP instance (useful for health checks, admin endpoints, tests).
  • It decouples tool authoring from the FastMCP API — if we ever swap the
    MCP SDK version, only the server.py binding code changes, not the tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ToolEntry:
    """
    Metadata record for a single registered MCP tool.

    Attributes
    ----------
    name : str
        Unique snake_case identifier; must match the name used in FastMCP
        registration so that `tools/call` dispatches correctly.
    description : str
        Human-readable description shown to the LLM in `tools/list`.
        Good descriptions are the single most important signal the LLM uses
        to decide *which* tool to call — write them as agent-facing docs.
    handler : Callable
        The Python function that implements the tool logic.  It will be passed
        to FastMCP's add_tool() so its type annotations form the JSON Schema.
    """
    name: str
    description: str
    handler: Callable[..., Any]


# ---------------------------------------------------------------------------
# Registry store
#
# A module-level dict keyed by tool name.  Using a plain dict (not a class)
# keeps the interface simple: the registry is a singleton by virtue of Python's
# module import system — importing this module twice returns the same object.
# ---------------------------------------------------------------------------
_registry: dict[str, ToolEntry] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_tool(name: str, description: str, handler: Callable[..., Any]) -> None:
    """
    Add a tool to the registry.

    This function should be called once per tool, typically in the server
    startup sequence before the FastMCP instance is started.

    Parameters
    ----------
    name : str
        Unique tool identifier.  Raises ValueError if already registered to
        prevent silent overwrites (a duplicate name usually means a bug).
    description : str
        Description surfaced to the LLM via `tools/list`.
    handler : Callable
        The tool implementation function.  Its parameter names and type
        annotations are used by FastMCP to generate the JSON Schema for
        `tools/list`.

    Raises
    ------
    ValueError
        If a tool with the same name is already registered.

    Examples
    --------
    >>> from mcp.tools.rag_tool import rag_retriever, DESCRIPTION
    >>> register_tool("rag_retriever", DESCRIPTION, rag_retriever)
    """
    if name in _registry:
        raise ValueError(
            f"Tool '{name}' is already registered.  "
            "Use a unique name or call unregister_tool() first."
        )
    _registry[name] = ToolEntry(name=name, description=description, handler=handler)


def get_tool(name: str) -> ToolEntry | None:
    """
    Look up a registered tool by name.

    Returns None (rather than raising) so callers can gracefully handle
    unknown tool names — e.g. the server's dispatch layer can return a
    structured error instead of crashing.

    Parameters
    ----------
    name : str
        Tool identifier to look up.

    Returns
    -------
    ToolEntry or None
    """
    return _registry.get(name)


def list_tools() -> list[ToolEntry]:
    """
    Return all registered tools as an ordered list.

    The server calls this at startup to iterate entries and bind each one to
    the FastMCP instance.  The health check endpoint also uses this to report
    which tools are available.

    Returns
    -------
    list[ToolEntry]
        Snapshot of all currently registered tools.  Mutations to the returned
        list do not affect the registry.
    """
    return list(_registry.values())


def unregister_tool(name: str) -> bool:
    """
    Remove a tool from the registry.

    Primarily useful in tests where you need a clean registry state between
    test cases.  In production, tools are registered once at startup and never
    removed.

    Parameters
    ----------
    name : str
        Tool name to remove.

    Returns
    -------
    bool
        True if the tool was found and removed, False if it was not registered.
    """
    if name in _registry:
        del _registry[name]
        return True
    return False
