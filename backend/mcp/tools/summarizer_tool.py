"""
summarizer_tool.py — MCP tool handler: "doc_summarizer"

MCP tool contract
-----------------
Name        : doc_summarizer
Input       : { text: str, max_words: int = 100 }
Output      : { summary: str, word_count: int }

When would an agent choose this tool?
--------------------------------------
The LLM agent selects tools based on the description returned in `tools/list`.
"doc_summarizer" is the right choice when:

  1. Post-retrieval compression — "rag_retriever" or "knowledge_search" may
     return long passages.  The agent pipes them through here to shrink the
     context before composing its final answer, staying within prompt limits.

  2. Explicit user request — "summarise this in 50 words" maps directly to
     this tool with max_words=50; the agent doesn't need to summarise inline.

  3. Multi-step pipelines — in a retrieve → summarise → compare → respond
     chain, this tool sits between retrieval and downstream reasoning steps,
     acting as a focused information filter.

  4. Token budget management — max_words lets the agent enforce an exact budget
     before injecting content into a constrained context (notification, card).
"""

from __future__ import annotations

import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from groq import Groq  # noqa: E402
from config import settings  # noqa: E402


DESCRIPTION = (
    "Summarise a block of text to a target word count using a Groq LLM. "
    "Use this when retrieved passages are too long to fit in a prompt, "
    "when the user explicitly asks for a summary, or when condensing content "
    "for a downstream pipeline step. "
    "Returns the summary text and its actual word count."
)


def _client() -> Groq:
    """Return a Groq client; raises if API key is absent."""
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set.  Add it to your .env file.")
    return Groq(api_key=settings.GROQ_API_KEY)


def doc_summarizer(text: str, max_words: int = 100) -> dict[str, Any]:
    """
    Summarise the given text to approximately max_words words using Groq.

    MCP contract
    ------------
    Input  : text      (str)       — the content to summarise.
             max_words (int, 100)  — target upper word-count bound (10–500).
    Output : {
               "summary":    str,  # the generated summary
               "word_count": int   # actual word count of the summary
             }

    The word_count field lets the calling agent:
      • Verify the LLM honoured the budget.
      • Log/audit generation across a multi-step pipeline.
      • Decide whether to truncate further before presenting to the user.

    Implementation notes
    --------------------
    • Model  : llama3-8b-8192 — fast, accurate, well-suited for summarisation.
    • Temp   : 0.3 — low temperature keeps summaries factual and stable across
                     repeated calls with the same input.
    • Tokens : max_words * 2 — ~4 tokens per 3 words; gives comfortable
                               headroom without over-generating.

    Parameters
    ----------
    text : str
        Raw content to summarise.  No pre-processing is applied.
    max_words : int
        Approximate upper bound on summary length in words.

    Returns
    -------
    dict
    """
    client = _client()

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise summarisation assistant.  "
                    "Produce a clear, faithful summary.  "
                    "Do not add information not present in the source text."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Summarise the following text in no more than {max_words} words.\n\n"
                    f"Text:\n{text}\n\n"
                    f"Summary (≤{max_words} words):"
                ),
            },
        ],
        max_tokens=max_words * 2,
        temperature=0.3,
    )

    summary: str = response.choices[0].message.content.strip()
    return {"summary": summary, "word_count": len(summary.split())}
