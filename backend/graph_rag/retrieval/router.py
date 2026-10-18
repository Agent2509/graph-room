"""Query routing: cheap keyword hints first, one LLM classification as a fallback."""

SYSTEM = """Classify the user question for a retrieval system.
Return JSON: {"mode": "local" or "global"}.
- local: about specific entities, people, companies, dates, relationships.
- global: broad themes, summaries, comparisons across the whole corpus."""

GLOBAL_HINTS = (
    "summary", "summarize", "overview", "main themes", "key themes", "themes",
    "key findings", "big picture",
    "trends", "executive summary", "all topics", "high level", "high-level",
)


def decide(llm, question: str, has_communities: bool) -> str:
    lowered = question.lower()
    if any(hint in lowered for hint in GLOBAL_HINTS):
        return "global"
    if llm is not None:
        try:
            data = llm.generate_json(f"Classify this question: {question}", SYSTEM)
            mode = data.get("mode")
            if mode in {"local", "global"}:
                return mode if (mode != "global" or has_communities) else "local"
        except Exception:  # noqa: BLE001 - routing must never fail a query
            pass
    return "local"
