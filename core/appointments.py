"""
Phase 3 — Appointment booking (hospital vertical demo).

Fixed sample slots per department, kept in SQLite so bookings persist
for the live dashboard/demo. Simple and reliable beats a real calendar
integration for a 2-day build.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tickets.db")

DEPARTMENTS = ["OPD (General)", "Cardiology", "Dermatology", "Pediatrics", "Orthopedics"]
SAMPLE_SLOTS = [
    "Mon 10:00 AM", "Mon 2:00 PM", "Tue 11:00 AM",
    "Wed 9:30 AM", "Wed 3:00 PM", "Thu 1:00 PM", "Fri 10:30 AM",
]


def _get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT,
            department TEXT,
            slot TEXT,
            status TEXT DEFAULT 'confirmed',
            created_at TEXT
        )
    """)
    return conn


def book_appointment(patient_name: str, department: str, slot: str) -> dict:
    conn = _get_conn()
    # prevent double-booking the same slot+department
    existing = conn.execute(
        "SELECT id FROM appointments WHERE department = ? AND slot = ? AND status = 'confirmed'",
        (department, slot),
    ).fetchone()
    if existing:
        conn.close()
        return {"success": False, "message": f"That slot is already booked. Please choose another."}

    cur = conn.execute(
        "INSERT INTO appointments (patient_name, department, slot, status, created_at) "
        "VALUES (?, ?, ?, 'confirmed', ?)",
        (patient_name, department, slot, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    appt_id = cur.lastrowid
    conn.close()
    return {"success": True, "id": appt_id, "message": f"Appointment #{appt_id} confirmed: {department} on {slot}."}


def list_appointments() -> list[tuple]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, patient_name, department, slot, status, created_at FROM appointments ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return rows


def available_slots(department: str) -> list[str]:
    conn = _get_conn()
    booked = {
        row[0] for row in conn.execute(
            "SELECT slot FROM appointments WHERE department = ? AND status = 'confirmed'", (department,)
        ).fetchall()
    }
    conn.close()
    return [s for s in SAMPLE_SLOTS if s not in booked]
