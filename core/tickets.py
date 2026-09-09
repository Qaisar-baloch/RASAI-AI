"""
Phase 2 — Ticket generation + storage.

Complaints (and guidance requests that need follow-up) become a structured
ticket: category, priority, routing department, and a plain-language summary.
Stored in SQLite so the dashboard can list/filter/update status live.
"""

import sqlite3
import json
import os
from datetime import datetime
from core.llm_client import chat

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tickets.db")

TICKET_SYSTEM_PROMPT = """You convert a user's raw complaint/guidance message into a structured helpdesk ticket.
Pick the routing department from context clues (e.g. "Registrar", "HEC Scholarships", "IT Support",
"OPD", "Cardiology", "Billing", "General Administration" — infer a sensible one if unclear).
Respond ONLY with JSON in this exact shape:
{"category": "<short category label>", "department": "<routing department>", "summary": "<one plain-language sentence summarizing the issue>"}
"""


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
            created_at TEXT
        )
    """)
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

    conn = _get_conn()
    cur = conn.execute(
        "INSERT INTO tickets (vertical, category, department, summary, priority, status, created_at) "
        "VALUES (?, ?, ?, ?, ?, 'pending', ?)",
        (vertical, category, department, summary, priority, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    ticket_id = cur.lastrowid
    conn.close()

    return {
        "id": ticket_id, "vertical": vertical, "category": category,
        "department": department, "summary": summary, "priority": priority, "status": "pending",
    }


def list_tickets(vertical: str | None = None) -> list[tuple]:
    conn = _get_conn()
    if vertical:
        rows = conn.execute(
            "SELECT id, vertical, category, department, summary, priority, status, created_at "
            "FROM tickets WHERE vertical = ? ORDER BY id DESC", (vertical,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, vertical, category, department, summary, priority, status, created_at "
            "FROM tickets ORDER BY id DESC"
        ).fetchall()
    conn.close()
    return rows


def update_status(ticket_id: int, new_status: str) -> None:
    conn = _get_conn()
    conn.execute("UPDATE tickets SET status = ? WHERE id = ?", (new_status, ticket_id))
    conn.commit()
    conn.close()
