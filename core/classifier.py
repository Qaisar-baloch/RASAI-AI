"""
Phase 1 — Intent Classifier (hybrid: trained ML model + LLM fallback).

Fast path: a trained TF-IDF + Logistic Regression model (core/ml_classifier.py)
classifies the message. If it's confident (>= CONFIDENCE_THRESHOLD), we trust it
and skip the API call entirely — faster, free, and demonstrates real model training
rather than only prompting an LLM.

Fallback: for low-confidence/ambiguous messages, we fall back to a single
structured LLM call (also used to determine priority in all cases).
"""

import json
from core.llm_client import chat
from core.ml_classifier import ml_classify

CONFIDENCE_THRESHOLD = 0.55

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

URGENT_KEYWORDS = ["emergency", "urgent", "asap", "immediately", "severe", "critical", "right now"]


def _quick_priority(message: str) -> str:
    text = message.lower()
    if any(k in text for k in URGENT_KEYWORDS):
        return "high"
    return "medium"


def classify_intent(message: str) -> dict:
    ml_result = ml_classify(message)

    if ml_result["confidence"] >= CONFIDENCE_THRESHOLD:
        # Fast path: trust the trained model, skip the API call.
        return {
            "intent": ml_result["intent"],
            "priority": _quick_priority(message),
            "reasoning": f"ML classifier (confidence {ml_result['confidence']:.2f}) — no LLM call needed",
            "method": "ml",
        }

    # Fallback path: low-confidence message, ask the LLM to decide (and set priority).
    raw = chat(prompt=f"Classify this message:\n\n\"{message}\"", system=SYSTEM_PROMPT, json_mode=True)
    try:
        data = json.loads(raw)
        intent = data.get("intent", ml_result["intent"])
        priority = data.get("priority", "medium")
        reasoning = data.get("reasoning", "")
    except (json.JSONDecodeError, AttributeError):
        intent, priority, reasoning = ml_result["intent"], "medium", "fallback parse (model returned non-JSON)"

    if intent not in {"complaint", "query", "guidance", "appointment"}:
        intent = ml_result["intent"]
    if priority not in {"low", "medium", "high"}:
        priority = "medium"

    return {"intent": intent, "priority": priority,
            "reasoning": f"LLM fallback (ML confidence was only {ml_result['confidence']:.2f}) — {reasoning}",
            "method": "llm"}
