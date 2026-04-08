"""
config.py — Application configuration module.

Loads environment variables from a .env file using python-dotenv and exposes
them as a Pydantic settings object for type-safe access throughout the app.
"""

from pydantic import BaseModel
from dotenv import load_dotenv
import os

# Load variables from .env file into the process environment
load_dotenv()


class Settings(BaseModel):
    """
    Central configuration container.

    Reads required secrets and connection details from environment variables
    so that sensitive values are never hard-coded in source control.
    """

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX: str = os.getenv("PINECONE_INDEX", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "ai-agent-platform")
    PINECONE_DIMENSION: int = int(os.getenv("PINECONE_DIMENSION", "384"))
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # -------------------------------------------------------------------------
    # MCP Server — Model Context Protocol (https://modelcontextprotocol.io)
    #
    # MCP is an open protocol developed by Anthropic that standardises how LLM
    # agents discover, invoke, and receive results from "tools" (arbitrary
    # server-side functions).  Instead of hard-coding tool logic inside the
    # agent, the agent queries an MCP server over JSON-RPC and the server
    # executes the function and returns a structured response.
    #
    # Why a separate port from FastAPI (8000)?
    #   • FastAPI (port 8000) is the public-facing REST API consumed by the
    #     frontend and external clients.  It handles HTTP/1.1 + WebSocket
    #     traffic and owns the request-response contract with users.
    #   • The MCP server (port 8001) is an internal service consumed only by
    #     the agent layer.  Keeping it on a dedicated port lets us:
    #       – Apply different auth / rate-limiting policies per service.
    #       – Scale or restart each service independently.
    #       – Avoid routing MCP JSON-RPC traffic through FastAPI middleware
    #         (CORS, auth guards, etc.) that is designed for REST, not RPC.
    #
    # How the agent communicates with this server:
    #   1. On startup, the agent connects to
    #      http://{MCP_SERVER_HOST}:{MCP_SERVER_PORT} via the `mcp` SDK.
    #   2. It calls `tools/list` to discover all registered tools and their
    #      JSON-Schema parameter definitions.
    #   3. When the LLM decides to use a tool, the agent calls `tools/call`
    #      with the tool name and arguments; the MCP server executes it and
    #      streams the result back to the agent.
    # -------------------------------------------------------------------------
    MCP_SERVER_HOST: str = os.getenv("MCP_SERVER_HOST", "localhost")
    MCP_SERVER_PORT: int = int(os.getenv("MCP_SERVER_PORT", "8001"))
    # Human-readable identifier for this MCP server instance; surfaced in
    # server-info responses so agents can verify they connected to the right
    # service.
    MCP_SERVER_NAME: str = os.getenv("MCP_SERVER_NAME", "ai-agent-platform-mcp")


# Singleton settings instance used by the rest of the application
settings = Settings()
