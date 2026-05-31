"""
LangGraph-based Finlo AI agent.

`create_react_agent` wraps a Claude model in an agent loop with structured
tools. The frontend sends a structured financial snapshot per request; tools
defined in services.tools read it from a contextvar and let the model query
specific slices (accounts, spending by period, bills, etc.) instead of
reading a flattened blob.
"""
from datetime import datetime, timezone
from typing import AsyncIterator

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.prebuilt import create_react_agent

from config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MAX_TOKENS,
    CLAUDE_MODEL,
    CLAUDE_TIMEOUT_S,
)
from services import tools as agent_tools

HISTORY_LIMIT = 20

_llm = ChatAnthropic(
    api_key=ANTHROPIC_API_KEY,
    model=CLAUDE_MODEL,
    max_tokens=CLAUDE_MAX_TOKENS,
    streaming=True,
    timeout=CLAUDE_TIMEOUT_S,
    max_retries=1,
)

_agent = create_react_agent(_llm, tools=agent_tools.all_tools())


def _system_prompt(default_currency: str) -> str:
    today = datetime.now(timezone.utc).strftime("%A, %d %B %Y")
    return (
        "You are Finlo AI, a personal financial advisor inside the Finlo app. "
        "Help the user understand their finances and make smart money decisions. "
        "Be friendly, concise, specific. Always reference the user's actual "
        "numbers. Use bullet points for lists. Keep simple answers to 2-4 "
        "sentences.\n\n"
        f"Today: {today}. Default currency: {default_currency}.\n\n"
        "You have tools to look up the user's data. PREFER calling tools over "
        "guessing. Examples:\n"
        "- For balance / net worth questions → get_accounts\n"
        "- For 'how much did I spend' → summarize_spending\n"
        "- For 'where does my money go' → top_spending_categories\n"
        "- For bills due → list_bills\n"
        "- For loans / debts → list_debts (owed by user) or list_lendings "
        "(owed to user)\n"
        "- For 'am I on budget' → get_budgets\n\n"
        "Chain multiple tool calls if useful (e.g. compare this month vs last "
        "month). Don't speculate about data that tools could answer for you."
    )


def _to_lc_messages(
    system_prompt: str,
    history: list[dict],
    user_message: str,
) -> list:
    msgs = [SystemMessage(content=system_prompt)]
    for m in history[-HISTORY_LIMIT:]:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
    msgs.append(HumanMessage(content=user_message))
    return msgs


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


async def stream_chat(
    snapshot: dict,
    history: list[dict],
    user_message: str,
) -> AsyncIterator[str]:
    """Yields plain-text deltas. `snapshot` holds the user's accounts,
    transactions, etc. — bound into a contextvar for the tools to read."""
    default_currency = snapshot.get("default_currency", "USD")
    messages = _to_lc_messages(
        _system_prompt(default_currency),
        history,
        user_message,
    )

    token = agent_tools.set_snapshot(snapshot)
    try:
        async for msg_chunk, _meta in _agent.astream(
            {"messages": messages},
            stream_mode="messages",
        ):
            text = _extract_text(getattr(msg_chunk, "content", ""))
            if text:
                yield text
    finally:
        agent_tools.reset_snapshot(token)
