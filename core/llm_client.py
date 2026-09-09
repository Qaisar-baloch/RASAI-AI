"""
Unified LLM client for Rasai AI.

Tries Groq first (fast + generous free tier), falls back to Gemini,
and falls back further to a small offline "demo mode" so the app
still runs end-to-end even with no API key configured yet
(useful for testing the UI/flow before you add real keys).
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

_groq_client = None
_gemini_model = None

if GROQ_API_KEY:
    try:
        from groq import Groq
        _groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception:
        _groq_client = None

if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    except Exception:
        _gemini_model = None


def is_live() -> bool:
    """True if at least one real LLM backend is configured."""
    return bool(_groq_client or _gemini_model)


def chat(prompt: str, system: str = "", json_mode: bool = False) -> str:
    """
    Send a prompt to the best available backend.
    Returns raw text (caller parses JSON if json_mode=True and the model supports it).
    Falls back to a deterministic offline stub if no backend is configured,
    so the rest of the pipeline (classifier/RAG/tickets) can still be tested.
    """
    full_prompt = f"{system}\n\n{prompt}" if system else prompt

    if _groq_client:
        try:
            kwargs = dict(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system or "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = _groq_client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[llm_client] Groq call failed, trying Gemini: {e}")

    if _gemini_model:
        try:
            resp = _gemini_model.generate_content(full_prompt)
            return resp.text
        except Exception as e:
            print(f"[llm_client] Gemini call failed, using offline stub: {e}")

    # ---- Offline demo-mode stub (no API key configured) ----
    return _offline_stub(prompt, json_mode)


def _offline_stub(prompt: str, json_mode: bool) -> str:
    """Very simple keyword-based stand-in so the app is testable with zero API keys."""
    text = prompt.lower()
    if json_mode:
        if "classify" in text or "intent" in text:
            if any(k in text for k in ["appointment", "book", "slot", "schedule a visit"]):
                intent = "appointment"
            elif any(k in text for k in ["complain", "complaint", "delay", "not working", "issue", "problem"]):
                intent = "complaint"
            elif any(k in text for k in ["how do i", "how to", "process", "procedure", "guide", "steps"]):
                intent = "guidance"
            else:
                intent = "query"
            return json.dumps({
                "intent": intent,
                "priority": "medium",
                "reasoning": "offline demo-mode keyword match (no LLM API key configured)"
            })
        return json.dumps({"note": "offline demo-mode stub response"})
    return ("[Demo mode — no GROQ_API_KEY or GEMINI_API_KEY set] "
            "This is a placeholder answer. Add an API key to .env to get real generated responses.")
