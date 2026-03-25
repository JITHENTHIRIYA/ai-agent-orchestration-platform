"""
base_agent.py — A basic ReAct agent powered by LangChain and Groq (LLaMA3).

What is a ReAct Agent?
──────────────────────
ReAct stands for "Reasoning + Acting". It is an agent paradigm where a large
language model iteratively:

  1. **Thinks**  – Produces an internal reasoning trace (the "Thought" step).
     The model analyses the user query, decides whether it already has enough
     information to answer, and if not, determines which tool to call next.

  2. **Acts**   – Invokes an external tool with the arguments it chose during
     the Thought step.  A "tool" can be anything: a calculator, a database
     lookup, an API call, a web search, etc.

  3. **Observes** – Receives the output (Observation) returned by the tool and
     feeds it back into the model context so it can reason about the result.

The loop repeats (Thought → Action → Observation → Thought → …) until the
model decides it has a final answer, at which point it emits a "Final Answer"
instead of another Action.

This file creates a minimal ReAct agent using:
  • Groq's hosted LLaMA3-8b-8192 as the LLM backbone
  • A single demo tool ("echo_tool") to illustrate tool invocation
  • LangChain's create_react_agent helper + AgentExecutor for the loop
"""

import sys
import os

# ---------------------------------------------------------------------------
# Ensure the parent directory (backend/) is on sys.path so we can import
# config.py regardless of where this script is executed from.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from config import settings

# ═══════════════════════════════════════════════════════════════════════════
# 1. DEFINE TOOLS
# ═══════════════════════════════════════════════════════════════════════════
# Tools are Python functions decorated with @tool.  LangChain inspects the
# function name, docstring, and type hints to describe the tool to the LLM
# so it knows *when* and *how* to call it.


@tool
def echo_tool(input: str) -> str:
    """
    A simple echo tool for demonstration purposes.
    It receives an input string and returns a confirmation message.
    Use this tool whenever you need to echo or repeat information.
    """
    return f"Tool called with: {input}"


# Collect all tools into a list.  The agent will be told about every tool
# in this list and can choose to call any of them during the ReAct loop.
tools = [echo_tool]

# ═══════════════════════════════════════════════════════════════════════════
# 2. REACT PROMPT TEMPLATE
# ═══════════════════════════════════════════════════════════════════════════
# The prompt template tells the LLM *how* to behave inside the ReAct loop.
# It must contain the following placeholders that LangChain fills at runtime:
#   {tools}            – formatted descriptions of all available tools
#   {tool_names}       – comma-separated list of tool names
#   {input}            – the user's original query
#   {agent_scratchpad} – running log of Thought/Action/Observation steps
#
# The explicit "Thought → Action → Action Input → Observation" format is
# what drives the ReAct loop; the LLM learns to follow this pattern.

REACT_PROMPT = PromptTemplate.from_template(
    """Answer the following questions as best you can. You have access to the following tools:

{tools}

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


# ═══════════════════════════════════════════════════════════════════════════
# 3. LAZY BUILDER — _build_agent_executor()
# ═══════════════════════════════════════════════════════════════════════════
# The LLM and AgentExecutor are created lazily (on first call) so that
# importing this module never fails — the GROQ_API_KEY is only required
# when the agent is actually invoked.

_agent_executor = None


def _build_agent_executor() -> AgentExecutor:
    """
    Construct the LLM, ReAct agent, and executor on first use.

    Why lazy?
    ---------
    ChatGroq validates the API key at construction time.  By deferring
    construction we allow the module to be imported (e.g. for testing or
    IDE introspection) even when the key is not yet set.
    """
    global _agent_executor
    if _agent_executor is not None:
        return _agent_executor

    # ── 3a. Initialise the LLM ──────────────────────────────────────────
    # We use Groq's blazing-fast inference API with the LLaMA3-8b-8192
    # model.  The API key is pulled from our centralised Settings object
    # (config.py), which in turn reads it from the .env file.
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,          # deterministic output for reproducibility
        groq_api_key=settings.GROQ_API_KEY,
    )

    # ── 3b. Create the agent & executor ─────────────────────────────────
    # create_react_agent  – wires the LLM + tools + prompt into a runnable
    #                       that can parse the LLM's text output into Actions.
    # AgentExecutor       – manages the iterative loop:
    #   while not final_answer:
    #       thought, action = agent.plan(input, scratchpad)
    #       observation      = tool.run(action.input)
    #       scratchpad      += thought + action + observation
    #   return final_answer
    #
    # handle_parsing_errors – if the LLM produces malformed output that can't
    #   be parsed into an Action, the executor feeds the error back to the LLM
    #   so it can self-correct rather than crashing.
    # verbose – prints each Thought/Action/Observation to stdout, which is
    #   invaluable for understanding how the ReAct loop unfolds.
    react_agent = create_react_agent(llm=llm, tools=tools, prompt=REACT_PROMPT)

    _agent_executor = AgentExecutor(
        agent=react_agent,
        tools=tools,
        verbose=True,              # show the full ReAct trace in the console
        handle_parsing_errors=True,  # let the LLM retry on bad output format
        max_iterations=5,          # safety cap to prevent infinite loops
    )
    return _agent_executor


# ═══════════════════════════════════════════════════════════════════════════
# 4. PUBLIC HELPER — run_agent()
# ═══════════════════════════════════════════════════════════════════════════
def run_agent(query: str) -> str:
    """
    Execute the ReAct agent on the given user query and return the response.

    Parameters
    ----------
    query : str
        The natural-language question or instruction from the user.

    Returns
    -------
    str
        The agent's final answer after completing the ReAct loop.
    """
    executor = _build_agent_executor()
    result = executor.invoke({"input": query})
    return result["output"]


# ═══════════════════════════════════════════════════════════════════════════
# 5. TEST FUNCTION
# ═══════════════════════════════════════════════════════════════════════════
def test_agent():
    """
    Quick smoke test — runs the agent with a simple arithmetic question.

    Expected behaviour:
      • The LLM reasons about the query ("What is 2+2?").
      • It may choose to call echo_tool or answer directly.
      • A final answer is printed to stdout.
    """
    query = "What is 2+2?"
    print(f"\n{'='*60}")
    print(f"  TEST QUERY: {query}")
    print(f"{'='*60}\n")

    response = run_agent(query)

    print(f"\n{'='*60}")
    print(f"  FINAL RESPONSE: {response}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    test_agent()
