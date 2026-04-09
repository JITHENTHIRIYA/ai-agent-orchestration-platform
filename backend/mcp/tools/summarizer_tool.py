"""
summarizer_tool.py — MCP tool: "doc_summarizer"

What this file does
-------------------
Exposes a Groq-backed text summarisation capability as an MCP tool.
The agent calls this tool when it needs to condense a long passage to a
controlled length before presenting it to the user or passing it to another
tool in a multi-step chain.

When would an agent choose this tool?
--------------------------------------
An LLM agent operating under the Model Context Protocol chooses tools based
on the descriptions surfaced by `tools/list`.  The agent will favour
"doc_summarizer" in scenarios like:

  1. **Post-retrieval compression** — the "rag_retriever" tool may return
     several long passages.  The agent can pipe each passage through
     "doc_summarizer" to shrink the context before composing its final answer,
     preventing prompt-length limits from being exceeded.

  2. **User-facing summaries** — when the user explicitly asks for a summary
     ("Summarise this document in 50 words"), the agent maps that intent to
     this tool directly rather than attempting to summarise inline.

  3. **Multi-step pipelines** — in an agentic chain such as
       retrieve → summarise → compare → respond,
     the summariser sits between retrieval and downstream reasoning steps,
     acting as a lossy-but-focused information filter.

  4. **Token budget management** — when the agent needs to fit content into a
     constrained downstream context (e.g. a notification, a tweet, a card),
     it uses max_words to enforce the budget programmatically.

MCP tool contract
-----------------
- Tool name   : "doc_summarizer"
- Input model : DocSummarizerInput  →  { text: str, max_words: int = 100 }
- Output      : { summary: str, word_count: int }
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP
from groq import Groq  # noqa: E402

from config import settings  # noqa: E402


# ---------------------------------------------------------------------------
# Input schema
#
# Typed inputs let the MCP SDK:
#   • Generate a JSON Schema for `tools/list` — the LLM reads this to know
#     what arguments to pass without any hand-holding.
#   • Validate agent arguments before the handler runs — prevents the Groq
#     API call from being made with malformed inputs.
# ---------------------------------------------------------------------------
class DocSummarizerInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description=(
            "The raw text to summarise.  Can be a document excerpt, a retrieved "
            "RAG passage, or any freeform content the agent wants condensed."
        ),
    )
    max_words: int = Field(
        default=100,
        ge=10,
        le=500,
        description=(
            "Target maximum word count for the summary.  The LLM is instructed "
            "to stay within this limit; actual count may vary slightly.  "
            "Must be between 10 and 500."
        ),
    )


# ---------------------------------------------------------------------------
# Internal helper — Groq client
# ---------------------------------------------------------------------------

def _groq_client() -> Groq:
    """Return a Groq client using the API key from app settings."""
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not configured.  Add it to your .env file.")
    return Groq(api_key=settings.GROQ_API_KEY)


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def doc_summarizer(text: str, max_words: int = 100) -> dict:
    """
    MCP tool handler for "doc_summarizer".

    Sends the provided text to the Groq LLM (llama3-8b-8192 by default) with
    a system prompt that instructs it to produce a concise summary within the
    requested word budget.

    MCP tool contract
    -----------------
    - Tool name   : "doc_summarizer"
    - Input       : { text: str, max_words: int = 100 }
    - Output      : { summary: str, word_count: int }

    The ``word_count`` field in the output reflects the actual number of words
    in the summary, which the calling agent can use to:
      - Verify the LLM honoured the budget.
      - Log/audit token usage across a pipeline.
      - Decide whether to truncate further before presenting to the user.

    Parameters
    ----------
    text : str
        The content to summarise.  No pre-processing is applied; the full
        string is embedded in the prompt as-is.
    max_words : int
        Approximate upper bound on the summary length in words.

    Returns
    -------
    dict
        ``{ "summary": str, "word_count": int }``

    Raises
    ------
    ValueError
        If GROQ_API_KEY is absent from the environment.
    groq.APIError
        If the Groq API returns a non-2xx response.
    """
    client = _groq_client()

    system_prompt = (
        "You are a precise summarisation assistant.  "
        "When given a block of text your only job is to produce a clear, "
        "faithful summary.  Do not add information not present in the source."
    )

    user_prompt = (
        f"Summarise the following text in no more than {max_words} words.\n\n"
        f"Text:\n{text}\n\n"
        f"Summary (≤{max_words} words):"
    )

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # Cap generation tokens; 4 tokens ≈ 3 words on average, so
        # max_words * 2 gives comfortable headroom without over-generating.
        max_tokens=max_words * 2,
        temperature=0.3,  # Low temperature keeps summaries factual and stable.
    )

    summary: str = response.choices[0].message.content.strip()
    word_count: int = len(summary.split())

    return {"summary": summary, "word_count": word_count}


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

def register(mcp: FastMCP) -> None:
    """
    Register the doc_summarizer tool on the given FastMCP server instance.

    Call this once during MCP server startup:

        from mcp.tools.summarizer_tool import register
        register(mcp_server)

    After registration, ``tools/list`` will include "doc_summarizer" and
    ``tools/call`` with name="doc_summarizer" will invoke the handler above.
    The LLM agent will see the description below in `tools/list` and use it
    to decide when this tool is the right choice.
    """

    @mcp.tool(name="doc_summarizer", description=(
        "Summarise a block of text to a target word count using an LLM.  "
        "Use this when retrieved passages are too long to fit in a prompt, "
        "when the user asks for a summary, or when condensing content for a "
        "downstream step in a multi-tool pipeline.  "
        "Returns the summary text and its actual word count."
    ))
    def _handler(text: str, max_words: int = 100) -> dict:
        return doc_summarizer(text=text, max_words=max_words)
