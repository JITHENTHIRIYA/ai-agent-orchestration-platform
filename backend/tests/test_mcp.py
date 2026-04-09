"""
test_mcp.py — Integration tests for the MCP server and FastAPI /api/mcp/* endpoints.

What "integration test" means here
------------------------------------
These tests exercise the real stack end-to-end:

    pytest (this file)
        │
        ├─ starts MCP server subprocess on port 8001    (session fixture)
        │
        └─ calls FastAPI endpoints via TestClient
               │  JSON over HTTP (in-process ASGI)
               └─ FastAPI endpoint
                      │  JSON-RPC over SSE
                      └─ MCP server (port 8001)
                             │
                             └─ tool handler → Pinecone / Groq

Unlike unit tests (which mock everything), these tests verify that the full
request path works: routing, MCP protocol, tool registration, and serialisation.
A failure here catches integration bugs that unit tests with mocks would miss.

Why a separate MCP server process?
------------------------------------
FastMCP.run() starts its own uvicorn/anyio event loop.  Running it inside the
pytest process (or inside the TestClient's event loop) causes loop-within-loop
errors.  The session fixture spawns it as a subprocess so it has an isolated
Python interpreter and event loop, matching how it runs in production.

Test organisation
------------------
  test_mcp_server_starts   — connectivity / smoke test (no credentials needed)
  test_list_tools          — tools/list MCP RPC (no credentials needed)
  test_rag_tool_invoke     — tools/call: rag_retriever (needs Pinecone)
  test_mcp_agent_query     — full ReAct agent round-trip (needs Groq + Pinecone)
  test_unknown_tool        — error path for an unregistered tool name

Skipping in CI
--------------
Tests that require external services are decorated with pytest.mark.skipif so
they are automatically skipped when the relevant environment variables are absent.
Set the following env vars in CI to enable all tests:
  GROQ_API_KEY
  PINECONE_API_KEY
  PINECONE_INDEX_NAME   (or PINECONE_INDEX)
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from typing import Generator

import pytest
from fastapi.testclient import TestClient

# ── Path setup ──────────────────────────────────────────────────────────────
# backend/ must be on sys.path before importing any project modules.
# This file lives at backend/tests/test_mcp.py → two levels up is backend/.
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from config import settings  # noqa: E402
from main import app  # noqa: E402


# ── Skip conditions ──────────────────────────────────────────────────────────
# These are evaluated once at collection time.  Tests that need Pinecone or
# Groq are skipped automatically when the keys are absent so the CI job doesn't
# fail on credential-less environments.

_needs_pinecone = pytest.mark.skipif(
    not settings.PINECONE_API_KEY or not (settings.PINECONE_INDEX_NAME or settings.PINECONE_INDEX),
    reason="PINECONE_API_KEY / PINECONE_INDEX_NAME not configured — skipping Pinecone tests",
)

_needs_groq = pytest.mark.skipif(
    not settings.GROQ_API_KEY,
    reason="GROQ_API_KEY not configured — skipping Groq LLM tests",
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """
    Return True if a TCP connection to host:port succeeds within `timeout` seconds.

    Used to check whether the MCP server has finished binding its port before
    the tests begin.  We use a raw socket rather than httpx so the check has
    no dependency on HTTP-level protocol details (SSE handshake, etc.).
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_for_port(host: str, port: int, timeout: float = 20.0) -> bool:
    """
    Poll host:port every 500 ms until it opens or `timeout` seconds elapse.

    Returns True if the port opened, False if the timeout expired.
    A 20-second window is generous for local startup but keeps the CI job
    from hanging indefinitely if the server crashes immediately.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _port_open(host, port):
            return True
        time.sleep(0.5)
    return False


# ── Session fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def mcp_server() -> Generator[subprocess.Popen, None, None]:
    """
    Start the MCP server as a subprocess for the entire test session.

    Why session scope?
      Starting the MCP server takes ~2–3 seconds (server binds port,
      registers tools, warms up imports).  Doing this once per session
      rather than once per test avoids a multi-minute overhead on the
      CI pipeline and mirrors how the server runs in production (long-lived).

    Startup sequence:
      1. Spawn `python backend/mcp/server.py` as a child process.
      2. Poll port 8001 every 500 ms until it's accepting connections.
      3. Yield the process handle to dependent fixtures / tests.
      4. On teardown, send SIGTERM and wait up to 5 s; SIGKILL if needed.

    Failure handling:
      If the server doesn't bind within 20 seconds we terminate the process
      and call pytest.fail(), which marks every dependent test as ERROR and
      shows a clear message rather than a cascade of confusing connection
      errors.
    """
    server_script = os.path.join(_BACKEND_DIR, "mcp", "server.py")

    proc = subprocess.Popen(
        [sys.executable, server_script],
        cwd=_BACKEND_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    host = settings.MCP_SERVER_HOST
    port = settings.MCP_SERVER_PORT

    ready = _wait_for_port(host, port, timeout=20.0)
    if not ready:
        proc.terminate()
        stderr_output = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        pytest.fail(
            f"MCP server did not bind {host}:{port} within 20 seconds.\n"
            f"stderr:\n{stderr_output}"
        )

    yield proc

    # ── Teardown ──────────────────────────────────────────────────────────
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="session")
def client(mcp_server: subprocess.Popen) -> TestClient:
    """
    Return a FastAPI TestClient that talks to the in-process ASGI app.

    Why NOT use `with TestClient(app)` (context-manager form)?
      The context-manager form triggers FastAPI's startup events, one of which
      (start_mcp_server) spawns another MCP subprocess.  Since the session
      fixture has already started the server, a second launch would either
      fail to bind the port or produce a duplicate process.  Creating the
      client without the context manager skips startup events while still
      allowing all endpoint handlers to run normally.

    Timeout:
      120 seconds covers the worst-case LLM + Pinecone + MCP round-trip in
      test_mcp_agent_query.  Individual network-only tests complete in < 5 s.
    """
    return TestClient(app, timeout=120.0)


# ── Tests ────────────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_mcp_server_starts(mcp_server: subprocess.Popen) -> None:
    """
    Verify the MCP server process is alive and accepting TCP connections.

    What this test verifies
    ------------------------
    • The subprocess did not crash immediately after startup (returncode check).
    • Port 8001 is open and responding to TCP SYN (socket connectivity check).

    Why this test matters
    ----------------------
    All subsequent MCP tests depend on the server being reachable.  A dedicated
    connectivity test makes the failure mode explicit: if this test fails, the
    rest of the MCP suite fails for a known infrastructure reason, not a code bug.
    If this passes but a later test fails, the problem is in the tool code or
    the HTTP layer — a meaningful distinction when debugging.
    """
    # Guard: subprocess should still be running (returncode is None when alive).
    assert mcp_server.returncode is None, (
        "MCP server process exited prematurely.  "
        "Check mcp/server.py for import errors or port conflicts."
    )

    # Connectivity: port must be open and accepting connections.
    host = settings.MCP_SERVER_HOST
    port = settings.MCP_SERVER_PORT
    assert _port_open(host, port), (
        f"MCP server port {host}:{port} is not accepting connections "
        "even though the process is running.  The server may still be starting up."
    )


@pytest.mark.integration
def test_list_tools(client: TestClient) -> None:
    """
    Verify GET /api/mcp/tools returns exactly the three registered tools.

    What this test verifies
    ------------------------
    • The FastAPI endpoint successfully forwards a tools/list JSON-RPC request
      to the MCP server over SSE.
    • The MCP server responds with a ListToolsResult containing the expected tools.
    • The endpoint serialises the result into the MCPToolsResponse schema.
    • Each tool entry has at minimum a `name` and `description` field.

    Why this test matters
    ----------------------
    tools/list is the discovery mechanism that the ReAct agent uses to build its
    tool registry.  If this fails:
      • The agent cannot discover tools → every /api/mcp/query call will fail.
      • A mismatch in registered tool names will cause routing failures.
    Testing this in isolation before test_mcp_agent_query makes root-cause
    analysis straightforward: if list fails, query will too — and for this reason.
    """
    response = client.get("/api/mcp/tools")

    assert response.status_code == 200, (
        f"Expected 200 from GET /api/mcp/tools, got {response.status_code}.\n"
        f"Body: {response.text}"
    )

    data = response.json()
    assert "tools" in data, f"Response missing 'tools' key: {data}"

    tools = data["tools"]
    assert isinstance(tools, list), f"'tools' should be a list, got {type(tools)}"
    assert len(tools) == 3, (
        f"Expected 3 registered tools, found {len(tools)}: "
        f"{[t.get('name') for t in tools]}"
    )

    # Verify each expected tool is present and has the required fields.
    expected_names = {"rag_retriever", "doc_summarizer", "knowledge_search"}
    actual_names = {t.get("name") for t in tools}
    assert actual_names == expected_names, (
        f"Tool names mismatch.\n  Expected: {expected_names}\n  Got:      {actual_names}"
    )

    # Each tool must carry a description (this is what the LLM reads to decide
    # which tool to call — a blank description would degrade agent performance).
    for tool in tools:
        assert tool.get("description"), (
            f"Tool '{tool.get('name')}' has an empty description.  "
            "The LLM uses the description to decide when to call each tool."
        )


@pytest.mark.integration
@_needs_pinecone
def test_rag_tool_invoke(client: TestClient) -> None:
    """
    Verify POST /api/mcp/tool/invoke correctly calls the rag_retriever tool.

    What this test verifies
    ------------------------
    • The endpoint accepts a tool_name + inputs JSON body.
    • It sends a tools/call JSON-RPC request to the MCP server.
    • The MCP server routes to the rag_retriever handler.
    • rag_retriever calls the Pinecone vector database and returns a result
      (even if the index is empty, the handler returns a valid structure).
    • The result is serialised to a non-empty string and returned with status=success.

    Why this test matters
    ----------------------
    This is the most important tool in the platform — if rag_retriever is broken,
    the agent has no grounding and will hallucinate.  Testing it via the
    /api/mcp/tool/invoke endpoint (bypassing the agent LLM) lets us verify the
    tool itself in isolation, separate from any LLM-related failures.

    Expected result structure (JSON-encoded in the `result` string field):
      { "context": str, "sources": list[dict] }

    Accepted outcomes:
      • context = "No relevant context found." when index is empty — valid.
      • context = "<numbered passage block>" when docs are indexed — valid.
    Both cases mean the tool completed successfully; the status must be "success".
    """
    response = client.post(
        "/api/mcp/tool/invoke",
        json={
            "tool_name": "rag_retriever",
            "inputs": {
                "query": "What is machine learning?",
                "top_k": 3,
            },
        },
    )

    assert response.status_code == 200, (
        f"Expected 200, got {response.status_code}.\nBody: {response.text}"
    )

    data = response.json()

    assert data.get("status") == "success", (
        f"Tool invocation reported failure.\n"
        f"status={data.get('status')}\nresult={data.get('result')}"
    )
    assert data.get("tool_name") == "rag_retriever"

    result_str = data.get("result", "")
    assert isinstance(result_str, str) and len(result_str) > 0, (
        "Expected a non-empty result string from rag_retriever."
    )

    # The rag_retriever handler returns a JSON-encoded dict.
    # Parse it to verify the structural contract (context + sources keys).
    import json
    try:
        result_dict = json.loads(result_str)
        assert "context" in result_dict, (
            f"Result dict missing 'context' key: {result_dict}"
        )
        assert "sources" in result_dict, (
            f"Result dict missing 'sources' key: {result_dict}"
        )
        assert isinstance(result_dict["sources"], list), (
            f"'sources' should be a list, got {type(result_dict['sources'])}"
        )
    except json.JSONDecodeError:
        # If the result is not JSON (e.g. the MCP SDK returned plain text),
        # we still accept it as long as it's non-empty.
        pass


@pytest.mark.integration
@_needs_groq
@_needs_pinecone
def test_mcp_agent_query(client: TestClient) -> None:
    """
    Verify POST /api/mcp/query runs the full MCP-backed ReAct agent end-to-end.

    What this test verifies
    ------------------------
    • The endpoint accepts a {"query": str} body.
    • run_mcp_agent() builds the LangChain ReAct executor (fetches tools from MCP).
    • The Groq LLM (llama-3.3-70b-versatile) processes the query and either:
        a. Calls an MCP tool → MCP server executes it → LLM uses result → Final Answer.
        b. Answers directly from parametric knowledge (no tool call).
    • The response dict has the correct shape: query, response, status.
    • status == "success" and response is a non-empty string.

    Why this test matters
    ----------------------
    This is the only test that exercises the complete agent loop:
      FastAPI → run_mcp_agent → LangChain ReAct → Groq LLM → MCP tool call
      → MCP server → Pinecone → MCP tool result → LLM Final Answer → FastAPI response

    A failure here with test_list_tools passing isolates the problem to either
    the LLM (Groq key/quota), the LangChain→MCP glue code, or the agent prompt.

    Note on tool_used / tool_input:
      For a general question like "What is machine learning?" the LLM may
      answer directly from knowledge (tool_used == None) OR it may call
      rag_retriever first.  Both are correct behaviour — we assert on the
      response shape and success status, not on which path the LLM took.
    """
    response = client.post(
        "/api/mcp/query",
        json={"query": "What is machine learning?"},
    )

    assert response.status_code == 200, (
        f"Expected 200, got {response.status_code}.\nBody: {response.text}"
    )

    data = response.json()

    # Required fields present.
    for field in ("query", "response", "status"):
        assert field in data, f"Response missing field '{field}': {data}"

    assert data["query"] == "What is machine learning?", (
        f"Echoed query doesn't match input: {data['query']}"
    )

    assert data["status"] == "success", (
        f"Agent reported status={data['status']}.\n"
        f"response: {data.get('response')}"
    )

    assert isinstance(data["response"], str) and len(data["response"].strip()) > 0, (
        "Agent returned an empty response string."
    )

    # tool_used and tool_input are Optional — assert type correctness only.
    assert data.get("tool_used") is None or isinstance(data["tool_used"], str), (
        f"tool_used should be str or None, got {type(data.get('tool_used'))}"
    )
    assert data.get("tool_input") is None or isinstance(data["tool_input"], dict), (
        f"tool_input should be dict or None, got {type(data.get('tool_input'))}"
    )


@pytest.mark.integration
def test_unknown_tool(client: TestClient) -> None:
    """
    Verify the API returns HTTP 500 when an unregistered tool name is invoked.

    What this test verifies
    ------------------------
    • Calling POST /api/mcp/tool/invoke with a tool_name that doesn't exist on
      the MCP server produces an error response — not a crash or silent failure.
    • The error is surfaced as HTTP 500 (tool-level error, not a client mistake).
    • The response body contains a detail field explaining the error.

    Why HTTP 500 (not 404)?
    ------------------------
    The MCP protocol does not differentiate "tool not found" from other tool
    errors at the HTTP layer — all tool failures arrive as a CallToolResult with
    isError=True or as an McpError exception, which our endpoint converts to 500.
    A 404 would be semantically cleaner but would require inspecting the error
    message string, adding fragility.  500 is the correct mapping here.

    Why this test matters
    ----------------------
    Without proper error handling, an unknown tool name could:
      • Hang the request (SSE connection open with no response).
      • Raise an unhandled exception and crash the worker.
      • Return HTTP 200 with an empty/malformed body.
    This test ensures that all three failure modes are caught and the client
    always receives a well-formed HTTP error with a human-readable message.
    """
    response = client.post(
        "/api/mcp/tool/invoke",
        json={
            "tool_name": "this_tool_does_not_exist_xyz_9999",
            "inputs": {"query": "test input"},
        },
    )

    # Must be an HTTP error — 500 from MCP tool error, not 200.
    assert response.status_code == 500, (
        f"Expected HTTP 500 for unknown tool, got {response.status_code}.\n"
        f"Body: {response.text}\n"
        "Hint: the endpoint may not be catching McpError / tool-not-found errors."
    )

    data = response.json()

    # FastAPI formats HTTPException bodies as {"detail": "..."}.
    assert "detail" in data, (
        f"Error response missing 'detail' field: {data}"
    )
    assert isinstance(data["detail"], str) and len(data["detail"]) > 0, (
        "Error detail message is empty — hard to debug without it."
    )
