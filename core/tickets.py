"""
Phase 2 — Ticket generation + storage, with SLA auto-escalation.

Complaints (and guidance requests that need follow-up) become a structured
ticket: category, priority, routing department, and a plain-language summary.
Stored in SQLite so the dashboard can list/filter/update status live.

SLA auto-escalation: each ticket gets a deadline (in days) inferred from the
category/summary text using rules pulled from the actual policy documents
(e.g. "result remarking processed within 21 working days"). If a ticket is
still pending past its deadline, it's automatically flagged overdue —
this is what turns the RAG knowledge base into something *actionable*,
not just a Q&A answer.
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from core.llm_client import chat

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tickets.db")

TICKET_SYSTEM_PROMPT = """You convert a user's raw complaint/guidance message into a structured helpdesk ticket.
Pick the routing department from context clues (e.g. "Registrar", "HEC Scholarships", "IT Support",
"OPD", "Cardiology", "Billing", "General Administration" — infer a sensible one if unclear).
Respond ONLY with JSON in this exact shape:
{"category": "<short category label>", "department": "<routing department>", "summary": "<one plain-language sentence summarizing the issue>"}
"""

# SLA rules pulled from the sample policy docs (data/university_docs, data/hospital_docs).
# Matched by keyword against the ticket's category+summary text, in order — first match wins.
SLA_RULES = [
    (["transcript"], 7),
    (["attestation", "hec"], 15),
    (["scholarship", "financial aid"], 10),
    (["fee", "refund", "finance", "payment"], 10),
    (["exam", "result", "remark"], 21),
    (["billing", "invoice", "overcharge"], 7),
    (["patient relations", "conduct", "staff"], 3),
    (["appointment", "booking"], 2),
    (["lab", "report", "diagnostic"], 2),
]
# Fallback SLA (days) by priority, used when no keyword rule matches.
DEFAULT_SLA_BY_PRIORITY = {"high": 2, "medium": 7, "low": 14}


def _infer_sla_days(category: str, summary: str, priority: str) -> int:
    text = f"{category} {summary}".lower()
    for keywords, days in SLA_RULES:
        if any(k in text for k in keywords):
            return days
    return DEFAULT_SLA_BY_PRIORITY.get(priority, 7)


def _get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vertical TEXT,
            category TEXT,
            department TEXT,
            summary TEXT,
            priority TEXT,
            status TEXT DEFAULT 'pending',
            sla_days INTEGER DEFAULT 7,
            created_at TEXT
        )
    """)
    # Backward-compatible migration: add sla_days if an older DB file is reused.
    cols = [r[1] for r in conn.execute("PRAGMA table_info(tickets)").fetchall()]
    if "sla_days" not in cols:
        conn.execute("ALTER TABLE tickets ADD COLUMN sla_days INTEGER DEFAULT 7")
    return conn


def create_ticket(message: str, priority: str, vertical: str) -> dict:
    raw = chat(prompt=f"Message: \"{message}\"", system=TICKET_SYSTEM_PROMPT, json_mode=True)
    try:
        data = json.loads(raw)
        category = data.get("category", "General")
        department = data.get("department", "General Administration")
        summary = data.get("summary", message[:120])
    except (json.JSONDecodeError, AttributeError):
        category, department, summary = "General", "General Administration", message[:120]

    sla_days = _infer_sla_days(category, summary, priority)

    conn = _get_conn()
    cur = conn.execute(
        "INSERT INTO tickets (vertical, category, department, summary, priority, status, sla_days, created_at) "
        "VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)",
        (vertical, category, department, summary, priority, sla_days,
         datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    ticket_id = cur.lastrowid
    conn.close()

    return {
        "id": ticket_id, "vertical": vertical, "category": category,
        "department": department, "summary": summary, "priority": priority,
        "status": "pending", "sla_days": sla_days,
    }


def _with_sla_status(rows: list[tuple]) -> list[tuple]:
    """Append a computed SLA status ('⚠️ Overdue' / '✅ On track' / '—' if resolved) to each row."""
    result = []
    now = datetime.now()
    for row in rows:
        (tid, vertical, category, department, summary, priority, status, sla_days, created_at) = row
        if status == "resolved":
            sla_label = "—"
        else:
            deadline = datetime.fromisoformat(created_at) + timedelta(days=sla_days)
            sla_label = "⚠️ Overdue" if now > deadline else f"✅ On track ({(deadline - now).days}d left)"
        result.append((tid, vertical, category, department, summary, priority, status, f"{sla_days}d", sla_label, created_at))
    return result


def list_tickets(vertical: str | None = None) -> list[tuple]:
    conn = _get_conn()
    if vertical:
        rows = conn.execute(
            "SELECT id, vertical, category, department, summary, priority, status, sla_days, created_at "
            "FROM tickets WHERE vertical = ? ORDER BY id DESC", (vertical,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, vertical, category, department, summary, priority, status, sla_days, created_at "
            "FROM tickets ORDER BY id DESC"
        ).fetchall()
    conn.close()
    return _with_sla_status(rows)


def update_status(ticket_id: int, new_status: str) -> None:
    conn = _get_conn()
    conn.execute("UPDATE tickets SET status = ? WHERE id = ?", (new_status, ticket_id))
    conn.commit()
    conn.close()
