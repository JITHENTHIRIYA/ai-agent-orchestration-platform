"""
rag_agent.py - ReAct agent with Retrieval-Augmented Generation (RAG).

This module is similar in structure to base_agent.py, but differs in one key way:
it gives the agent a retrieval tool backed by Pinecone so answers can be grounded
in indexed project knowledge, not only the model's built-in parametric memory.
"""

from __future__ import annotations

from typing import Dict, List
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # noqa: E402

from langchain.agents import AgentExecutor, create_react_agent  # noqa: E402
from langchain_core.prompts import PromptTemplate  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langchain_groq import ChatGroq  # noqa: E402
from config import settings  # noqa: E402
from rag.retriever import retrieve, retrieve_as_context  # noqa: E402

# Stores source file names from the most recent retrieval call.
# "sources" in API responses means the document names (usually file names) that
# provided context chunks used to ground the generated answer.
_last_sources: List[str] = []


@tool
def rag_lookup_tool(input: str) -> str:
    """
    Retrieve relevant context snippets from Pinecone for the given query.

    How tool usage is decided:
    The ReAct agent is prompted with explicit instructions to call this tool
    before drafting a final answer. During the ReAct loop, the model decides
    when to issue an Action call, and that Action invokes this function.
    """
    global _last_sources

    query = input.strip()
    if not query:
        _last_sources = []
        return "No query provided for retrieval."

    retrieved = retrieve(query=query, top_k=5)
    _last_sources = sorted(
        {
            item.get("metadata", {}).get("source", "unknown")
            for item in retrieved
            if item.get("metadata", {})
        }
    )
    return retrieve_as_context(query)


tools = [rag_lookup_tool]

RAG_REACT_PROMPT = PromptTemplate.from_template(
    """You are a helpful assistant with access to a retrieval tool.

You have access to the following tools:
{tools}

IMPORTANT behavior rules:
1) For factual or domain-specific questions, you MUST call rag_lookup_tool first.
2) Use the retrieved context as your primary evidence.
3) If retrieval returns no relevant context, clearly say so and then provide your best effort answer.
4) Do not claim unsupported facts that are absent from retrieved context.

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

_rag_agent_executor = None


def _build_rag_agent_executor() -> AgentExecutor:
    """Lazily construct and cache the Groq-backed ReAct RAG agent executor."""
    global _rag_agent_executor
    if _rag_agent_executor is not None:
        return _rag_agent_executor

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=settings.GROQ_API_KEY,
    )

    react_agent = create_react_agent(llm=llm, tools=tools, prompt=RAG_REACT_PROMPT)
    _rag_agent_executor = AgentExecutor(
        agent=react_agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=6,
    )
    return _rag_agent_executor


def run_rag_agent(query: str) -> Dict[str, object]:
    """
    Execute a RAG-enabled ReAct agent and return structured output.

    Returns dictionary with:
    - query: original user query
    - response: final agent answer text
    - sources: list of retrieved source document names
    - status: "success" or "error"

    How this differs from base_agent.py:
    - base_agent.py uses a demo echo tool and can answer without retrieval.
    - this module provides a real retrieval tool and explicitly prioritizes
      retrieval-grounded answering before model-only knowledge.
    """
    global _last_sources
    _last_sources = []

    if not query or not query.strip():
        return {
            "query": query,
            "response": "Query is empty.",
            "sources": [],
            "status": "error",
        }

    try:
        executor = _build_rag_agent_executor()
        result = executor.invoke({"input": query})

        return {
            "query": query,
            "response": result.get("output", ""),
            "sources": _last_sources,
            "status": "success",
        }
    except Exception as exc:
        return {
            "query": query,
            "response": f"RAG agent execution failed: {exc}",
            "sources": _last_sources,
            "status": "error",
        }
