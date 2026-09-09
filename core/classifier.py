"""
Phase 1 — Intent Classifier.

Classifies an incoming message into one of:
  complaint | query | guidance | appointment
plus a priority level, using a single structured LLM call.
"""

import json
from core.llm_client import chat

SYSTEM_PROMPT = """You are an intake classifier for a public-sector helpdesk (university, HEC, or hospital).
Classify the user's message into exactly one intent:
- "complaint": something went wrong / is delayed / not working and the user wants it fixed
- "query": the user wants a factual answer (hours, requirements, status, contact info)
- "guidance": the user doesn't know a process and needs step-by-step direction
- "appointment": the user wants to book, reschedule, or cancel an appointment/visit

Also assign a priority: "low", "medium", or "high" (high = urgent/health-critical/deadline-critical).

Respond ONLY with JSON in this exact shape:
{"intent": "<complaint|query|guidance|appointment>", "priority": "<low|medium|high>", "reasoning": "<one short sentence>"}
"""


def classify_intent(message: str) -> dict:
    raw = chat(prompt=f"Classify this message:\n\n\"{message}\"", system=SYSTEM_PROMPT, json_mode=True)
    try:
        data = json.loads(raw)
        intent = data.get("intent", "query")
        priority = data.get("priority", "medium")
        reasoning = data.get("reasoning", "")
    except (json.JSONDecodeError, AttributeError):
        # Defensive fallback if the model returns non-JSON text
        intent, priority, reasoning = "query", "medium", "fallback parse (model returned non-JSON)"

    if intent not in {"complaint", "query", "guidance", "appointment"}:
        intent = "query"
    if priority not in {"low", "medium", "high"}:
        priority = "medium"

    return {"intent": intent, "priority": priority, "reasoning": reasoning}
