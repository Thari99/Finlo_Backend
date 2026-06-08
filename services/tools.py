"""
LangGraph tools that read the per-request financial snapshot.

The Flutter app sends a structured snapshot of the user's finances on every
chat call. We stash that snapshot in a contextvar so the @tool functions can
look at it. Tools return short, readable text — the agent decides which to
call and how to combine their answers.

Adding a new tool: define an @tool-decorated function with a clear docstring
+ typed args, then add it to `all_tools()`. The docstring is the description
the model uses to decide whether/how to call it — make it concrete.
"""
from __future__ import annotations

import contextvars
from datetime import datetime, timedelta, timezone
from typing import Optional

from langchain_core.tools import tool

# Per-request snapshot bound by services.agent.stream_chat.
#
# Note: we deliberately don't use reset(token). LangGraph's async streaming
# runs the agent in a child asyncio context, and reset(token) must be called
# in the same context where set() ran — otherwise it raises ValueError
# ("token was created in a different context"). Each FastAPI request runs in
# its own contextvars Context, so the value is naturally scoped per-request
# and cleans up when the task ends.
_snapshot: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "finlo_snapshot", default={}
)


def set_snapshot(data: dict) -> None:
    _snapshot.set(data)


def reset_snapshot(_token=None) -> None:
    """No-op kept for backwards compatibility with existing callers."""
    pass


# ── Internal helpers ─────────────────────────────────────────────────────────

def _snap() -> dict:
    try:
        return _snapshot.get()
    except LookupError:
        return {}


def _accounts() -> list[dict]:
    return _snap().get("accounts", []) or []


def _transactions() -> list[dict]:
    return _snap().get("transactions", []) or []


def _bills() -> list[dict]:
    return _snap().get("bills", []) or []


def _debts() -> list[dict]:
    return _snap().get("debts", []) or []


def _lendings() -> list[dict]:
    return _snap().get("lendings", []) or []


def _budgets() -> list[dict]:
    return _snap().get("budgets", []) or []


def _default_currency() -> str:
    return _snap().get("default_currency", "USD") or "USD"


def _today() -> datetime:
    return datetime.now(timezone.utc)


def _parse_date(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _period_range(period: str) -> tuple[datetime, datetime, str]:
    """Returns (start, end, label) for a period keyword."""
    now = _today()
    p = period.lower().strip()
    if p == "this_month":
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return start, now, start.strftime("%B %Y")
    if p == "last_month":
        first_this = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        last_end = first_this - timedelta(microseconds=1)
        last_start = last_end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return last_start, last_end, last_start.strftime("%B %Y")
    if p == "last_7_days":
        return now - timedelta(days=7), now, "last 7 days"
    if p == "last_30_days":
        return now - timedelta(days=30), now, "last 30 days"
    if p == "year_to_date":
        start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        return start, now, f"year-to-date ({now.year})"
    # Fallback: this month
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return start, now, start.strftime("%B %Y")


def _fmt_amount(amount: float, currency: str) -> str:
    return f"{amount:,.2f} {currency}"


# ── Tools ────────────────────────────────────────────────────────────────────

@tool
def get_accounts() -> str:
    """List the user's accounts with balances and currency, plus net worth if
    all accounts share a currency. Use this when the user asks about their
    accounts, balance, net worth, or how much money they have."""
    accs = _accounts()
    if not accs:
        return "The user has no accounts set up yet."

    lines = []
    for a in accs:
        name = a.get("name", "?")
        atype = a.get("type", "")
        bal = float(a.get("balance", 0))
        cur = a.get("currency") or _default_currency()
        lines.append(f"- {name} [{atype}]: {_fmt_amount(bal, cur)}")

    same_currency = len({a.get("currency") for a in accs}) == 1
    if same_currency and len(accs) > 1:
        total = sum(float(a.get("balance", 0)) for a in accs)
        cur = accs[0].get("currency") or _default_currency()
        lines.append(f"\nNet worth: {_fmt_amount(total, cur)}")

    return "\n".join(lines)


@tool
def summarize_spending(
    period: str = "this_month",
    category: Optional[str] = None,
) -> str:
    """Summarize income, expenses, and net cash flow over a period, optionally
    filtered by category. Use for questions like "how much did I spend this
    month" or "how much have I spent on food".

    Args:
        period: One of "this_month", "last_month", "last_7_days",
            "last_30_days", "year_to_date". Defaults to "this_month".
        category: Optional category name (e.g. "Food", "Transport"). When set,
            income is ignored and only expenses in that category are returned.
    """
    start, end, label = _period_range(period)
    cur = _default_currency()
    txns = _transactions()

    income = 0.0
    expense = 0.0
    cat_filtered = 0.0
    cat_matched = False

    for t in txns:
        d = _parse_date(t.get("date", ""))
        if d is None or d < start or d > end:
            continue
        amt = float(t.get("amount", 0))
        ttype = t.get("type")
        if ttype == "income":
            income += amt
        elif ttype == "expense":
            expense += amt
            if category and t.get("category", "").lower() == category.lower():
                cat_filtered += amt
                cat_matched = True

    if category:
        if not cat_matched:
            return (
                f"No '{category}' expenses found in {label}."
            )
        return (
            f"{label}: spent {_fmt_amount(cat_filtered, cur)} on "
            f"'{category}'."
        )

    net = income - expense
    return (
        f"{label}:\n"
        f"- Income: {_fmt_amount(income, cur)}\n"
        f"- Expenses: {_fmt_amount(expense, cur)}\n"
        f"- Net: {_fmt_amount(net, cur)}"
    )


@tool
def top_spending_categories(period: str = "this_month", n: int = 5) -> str:
    """List the user's top N spending categories for a period. Use for
    questions like "what am I spending the most on" or "where does my money go".

    Args:
        period: See `summarize_spending`. Defaults to "this_month".
        n: How many categories to return. Defaults to 5.
    """
    start, end, label = _period_range(period)
    cur = _default_currency()
    totals: dict[str, float] = {}
    for t in _transactions():
        if t.get("type") != "expense":
            continue
        d = _parse_date(t.get("date", ""))
        if d is None or d < start or d > end:
            continue
        cat = t.get("category", "Uncategorized")
        totals[cat] = totals.get(cat, 0) + float(t.get("amount", 0))

    if not totals:
        return f"No expenses recorded in {label}."

    sorted_cats = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:n]
    lines = [f"Top {len(sorted_cats)} spending categories for {label}:"]
    for cat, amt in sorted_cats:
        lines.append(f"- {cat}: {_fmt_amount(amt, cur)}")
    return "\n".join(lines)


@tool
def list_bills(status: str = "upcoming", within_days: int = 30) -> str:
    """List the user's bills. Use for questions about what's due, what's
    overdue, or what's coming up.

    Args:
        status: "upcoming" (due within `within_days`), "overdue" (past due),
            or "all".
        within_days: For upcoming, how far ahead to look. Defaults to 30.
    """
    bills = _bills()
    if not bills:
        return "The user has no bills tracked."

    cur = _default_currency()
    # Bills are date-shaped, not instant-shaped. Compare at day granularity so
    # a bill due "today" doesn't flip to overdue as soon as the clock crosses
    # midnight UTC (or get mis-labelled for users in +offset timezones whose
    # local midnight stores as previous-day UTC).
    today = _today().replace(hour=0, minute=0, second=0, microsecond=0)
    cutoff = today + timedelta(days=within_days)
    lines: list[str] = []

    for b in bills:
        due_dt = _parse_date(b.get("due_date", ""))
        if due_dt is None:
            continue
        due = due_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        days_until = (due - today).days
        item = f"- {b.get('name','?')}: {_fmt_amount(float(b.get('amount',0)), cur)}"
        if status == "overdue":
            if due < today:
                item += f" (overdue {abs(days_until)}d)"
                lines.append(item)
        elif status == "upcoming":
            if today <= due <= cutoff:
                item += " (due today)" if days_until == 0 else f" (due in {days_until}d)"
                lines.append(item)
        else:  # all
            if due < today:
                item += f" (overdue {abs(days_until)}d)"
            elif days_until == 0:
                item += " (due today)"
            else:
                item += f" (due in {days_until}d)"
            lines.append(item)

    if not lines:
        return f"No {status} bills."

    header = {
        "upcoming": f"Bills due in the next {within_days} days:",
        "overdue": "Overdue bills:",
        "all": "All bills:",
    }.get(status, "Bills:")
    return header + "\n" + "\n".join(lines)


@tool
def list_debts() -> str:
    """List active debts the user owes. Use for questions about loans,
    money owed, or paying off balances."""
    debts = [d for d in _debts() if d.get("status") == "active"]
    if not debts:
        return "The user has no active debts."

    cur = _default_currency()
    lines = ["Active debts:"]
    for d in debts:
        total = float(d.get("total_amount", 0))
        paid = float(d.get("paid_amount", 0))
        remaining = total - paid
        lines.append(
            f"- {d.get('lender_name','?')}: "
            f"{_fmt_amount(remaining, cur)} remaining "
            f"(total {_fmt_amount(total, cur)}, paid {_fmt_amount(paid, cur)})"
        )
    return "\n".join(lines)


@tool
def list_lendings() -> str:
    """List money the user has lent out (money owed TO the user). Use for
    questions about who owes them money."""
    lends = [l for l in _lendings() if l.get("status") == "active"]
    if not lends:
        return "Nobody owes the user money right now."

    cur = _default_currency()
    lines = ["Active lendings (owed to user):"]
    for l in lends:
        amt = float(l.get("amount", 0))
        ret = float(l.get("returned_amount", 0))
        remaining = amt - ret
        lines.append(
            f"- {l.get('person_name','?')}: "
            f"{_fmt_amount(remaining, cur)} remaining "
            f"(lent {_fmt_amount(amt, cur)})"
        )
    return "\n".join(lines)


@tool
def get_budgets() -> str:
    """Show the user's budgets and how much of each they've used this month.
    Use for questions about whether they're on track, over budget, etc."""
    budgets = _budgets()
    if not budgets:
        return "The user has no budgets set up."

    start, _end, label = _period_range("this_month")
    cur = _default_currency()
    spent_by_cat: dict[str, float] = {}
    for t in _transactions():
        if t.get("type") != "expense":
            continue
        d = _parse_date(t.get("date", ""))
        if d is None or d < start:
            continue
        cat = t.get("category", "")
        spent_by_cat[cat] = spent_by_cat.get(cat, 0) + float(t.get("amount", 0))

    lines = [f"Budgets ({label}):"]
    for b in budgets:
        cat = b.get("category", "?")
        limit = float(b.get("limit_amount", 0))
        spent = spent_by_cat.get(cat, 0)
        pct = (spent / limit * 100) if limit else 0
        flag = "  ⚠ OVER" if spent > limit else ""
        lines.append(
            f"- {cat}: {_fmt_amount(spent, cur)} / "
            f"{_fmt_amount(limit, cur)} ({pct:.0f}%){flag}"
        )
    return "\n".join(lines)


def all_tools() -> list:
    return [
        get_accounts,
        summarize_spending,
        top_spending_categories,
        list_bills,
        list_debts,
        list_lendings,
        get_budgets,
    ]
