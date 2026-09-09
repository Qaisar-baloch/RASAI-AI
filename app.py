"""
Rasai AI — Universal Complaint, Query & Guidance Agent
Main Gradio application. One engine, two demo verticals (University/HEC, Hospital).

Run locally:  python app.py
"""

import gradio as gr
import pandas as pd

from core.classifier import classify_intent
from core.rag import VerticalKnowledgeBase
from core.tickets import create_ticket, list_tickets, update_status
from core.appointments import book_appointment, list_appointments, available_slots, DEPARTMENTS
from core.llm_client import is_live

# ---- Phase 1: load a knowledge base per vertical ----
KB = {
    "University / HEC": VerticalKnowledgeBase("data/university_docs"),
    "Hospital": VerticalKnowledgeBase("data/hospital_docs"),
}

VERTICAL_TICKET_KEY = {"University / HEC": "university", "Hospital": "hospital"}


def handle_message(message: str, vertical: str, history: list):
    """Core pipeline: classify -> RAG answer OR ticket generation."""
    if not message or not message.strip():
        return history, "", gr.update()

    result = classify_intent(message)
    intent, priority = result["intent"], result["priority"]

    if intent == "appointment":
        reply = (f"**Intent detected: appointment** (priority: {priority})\n\n"
                 "Head over to the **Appointments** tab to pick a department and time slot.")
    elif intent in ("query", "guidance"):
        kb = KB[vertical]
        rag_result = kb.answer(message)
        sources = ", ".join(rag_result["sources"]) if rag_result["sources"] else "no matching document"
        reply = (f"**Intent: {intent}** (priority: {priority})\n\n"
                 f"{rag_result['answer']}\n\n"
                 f"*Source: {sources}*")
    else:  # complaint
        ticket = create_ticket(message, priority, VERTICAL_TICKET_KEY[vertical])
        reply = (f"**Intent: complaint** (priority: {priority})\n\n"
                 f"A ticket has been created:\n"
                 f"- **Ticket #{ticket['id']}**\n"
                 f"- Category: {ticket['category']}\n"
                 f"- Routed to: {ticket['department']}\n"
                 f"- Summary: {ticket['summary']}\n"
                 f"- Status: pending\n\n"
                 f"You can track this in the **Ticket Dashboard** tab.")

    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": reply}]
    return history, "", gr.update()


def refresh_tickets(vertical_filter: str):
    vkey = None if vertical_filter == "All" else VERTICAL_TICKET_KEY.get(vertical_filter)
    rows = list_tickets(vkey)
    df = pd.DataFrame(rows, columns=["ID", "Vertical", "Category", "Department", "Summary", "Priority", "Status", "Created"])
    return df


def mark_resolved(ticket_id: int):
    if ticket_id is None:
        return refresh_tickets("All")
    update_status(int(ticket_id), "resolved")
    return refresh_tickets("All")


def refresh_slots(department: str):
    slots = available_slots(department)
    return gr.update(choices=slots, value=slots[0] if slots else None)


def do_booking(patient_name: str, department: str, slot: str):
    if not patient_name or not slot:
        return "Please enter your name and pick a slot.", refresh_appts(), gr.update()
    result = book_appointment(patient_name, department, slot)
    return result["message"], refresh_appts(), refresh_slots(department)


def refresh_appts():
    rows = list_appointments()
    return pd.DataFrame(rows, columns=["ID", "Patient", "Department", "Slot", "Status", "Booked At"])


with gr.Blocks(title="Rasai AI") as demo:
    gr.Markdown("# 🇵🇰 Rasai AI — Universal Complaint, Query & Guidance Agent")
    live_note = "🟢 Live LLM connected" if is_live() else "🟡 Demo mode (no GROQ_API_KEY/GEMINI_API_KEY set — add one to .env for real answers)"
    gr.Markdown(f"*One AI engine, reused across sectors by swapping the knowledge base.* — {live_note}")

    with gr.Tab("💬 Chat"):
        vertical_select = gr.Radio(["University / HEC", "Hospital"], value="University / HEC", label="Vertical (swap to prove one engine works for both)")
        chatbot = gr.Chatbot(height=420)
        msg_box = gr.Textbox(label="Type your complaint, query, guidance request, or appointment request", placeholder="e.g. 'my transcript has been delayed for 3 weeks'")
        send_btn = gr.Button("Send", variant="primary")
        send_btn.click(handle_message, [msg_box, vertical_select, chatbot], [chatbot, msg_box, msg_box])
        msg_box.submit(handle_message, [msg_box, vertical_select, chatbot], [chatbot, msg_box, msg_box])

    with gr.Tab("🎫 Ticket Dashboard"):
        gr.Markdown("Every complaint auto-generates a ticket here — category, routing department, priority, status.")
        with gr.Row():
            ticket_filter = gr.Radio(["All", "University / HEC", "Hospital"], value="All", label="Filter by vertical")
            refresh_btn = gr.Button("Refresh")
        ticket_table = gr.Dataframe(headers=["ID", "Vertical", "Category", "Department", "Summary", "Priority", "Status", "Created"])
        with gr.Row():
            resolve_id = gr.Number(label="Ticket ID to mark resolved", precision=0)
            resolve_btn = gr.Button("Mark Resolved")
        refresh_btn.click(refresh_tickets, [ticket_filter], [ticket_table])
        ticket_filter.change(refresh_tickets, [ticket_filter], [ticket_table])
        resolve_btn.click(mark_resolved, [resolve_id], [ticket_table])
        demo.load(refresh_tickets, [ticket_filter], [ticket_table])

    with gr.Tab("📅 Appointments (Hospital demo)"):
        gr.Markdown("AI-assisted appointment booking — pick a department, see live-available slots, confirm.")
        dept_select = gr.Dropdown(DEPARTMENTS, value=DEPARTMENTS[0], label="Department")
        slot_select = gr.Dropdown(available_slots(DEPARTMENTS[0]), label="Available slot")
        patient_name = gr.Textbox(label="Patient name")
        book_btn = gr.Button("Book Appointment", variant="primary")
        booking_status = gr.Markdown()
        appt_table = gr.Dataframe(headers=["ID", "Patient", "Department", "Slot", "Status", "Booked At"])
        dept_select.change(refresh_slots, [dept_select], [slot_select])
        book_btn.click(do_booking, [patient_name, dept_select, slot_select], [booking_status, appt_table, slot_select])
        demo.load(refresh_appts, None, [appt_table])

if __name__ == "__main__":
    demo.launch()
