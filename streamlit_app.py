"""
Rasai AI — Universal Complaint, Query & Guidance Agent (Streamlit Cloud version)
Run locally:  streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd

from core.classifier import classify_intent
from core.rag import VerticalKnowledgeBase
from core.tickets import create_ticket, list_tickets, update_status
from core.appointments import book_appointment, list_appointments, available_slots, DEPARTMENTS
from core.llm_client import is_live

st.set_page_config(page_title="Rasai AI", page_icon="🇵🇰", layout="wide")


@st.cache_resource
def load_knowledge_bases():
    return {
        "University / HEC": VerticalKnowledgeBase("data/university_docs"),
        "Hospital": VerticalKnowledgeBase("data/hospital_docs"),
    }


KB = load_knowledge_bases()
VERTICAL_TICKET_KEY = {"University / HEC": "university", "Hospital": "hospital"}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("🇵🇰 Rasai AI — Universal Complaint, Query & Guidance Agent")
live_note = ("🟢 Live LLM connected" if is_live()
             else "🟡 Demo mode (no GROQ_API_KEY/GEMINI_API_KEY set — add one in Settings → Secrets for real answers)")
st.caption(f"One AI engine, reused across sectors by swapping the knowledge base. — {live_note}")

vertical = st.sidebar.radio("Vertical (swap to prove one engine works for both)",
                             ["University / HEC", "Hospital"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Rasai AI** — built for the hackathon in 4 phases:\n"
                     "1. Intent classification\n2. RAG-grounded answers\n"
                     "3. Auto ticket generation\n4. Appointment booking")

tab_chat, tab_tickets, tab_appts = st.tabs(["💬 Chat", "🎫 Ticket Dashboard", "📅 Appointments"])

# ---------------- Chat tab ----------------
with tab_chat:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Type your complaint, query, guidance request, or appointment request")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Thinking..."):
            result = classify_intent(user_input)
            intent, priority = result["intent"], result["priority"]

            if intent == "appointment":
                reply = (f"**Intent detected: appointment** (priority: {priority})\n\n"
                         "Head over to the **Appointments** tab to pick a department and time slot.")
            elif intent in ("query", "guidance"):
                kb = KB[vertical]
                rag_result = kb.answer(user_input)
                sources = ", ".join(rag_result["sources"]) if rag_result["sources"] else "no matching document"
                reply = (f"**Intent: {intent}** (priority: {priority})\n\n"
                         f"{rag_result['answer']}\n\n*Source: {sources}*")
            else:
                ticket = create_ticket(user_input, priority, VERTICAL_TICKET_KEY[vertical])
                reply = (f"**Intent: complaint** (priority: {priority})\n\n"
                         f"A ticket has been created:\n"
                         f"- **Ticket #{ticket['id']}**\n"
                         f"- Category: {ticket['category']}\n"
                         f"- Routed to: {ticket['department']}\n"
                         f"- Summary: {ticket['summary']}\n"
                         f"- Status: pending\n\n"
                         f"You can track this in the **Ticket Dashboard** tab.")

        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

# ---------------- Ticket Dashboard tab ----------------
with tab_tickets:
    st.markdown("Every complaint auto-generates a ticket here — category, routing department, priority, status.")
    filter_choice = st.radio("Filter by vertical", ["All", "University / HEC", "Hospital"], horizontal=True)
    vkey = None if filter_choice == "All" else VERTICAL_TICKET_KEY.get(filter_choice)
    rows = list_tickets(vkey)
    df = pd.DataFrame(rows, columns=["ID", "Vertical", "Category", "Department", "Summary", "Priority", "Status", "Created"])
    st.dataframe(df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        resolve_id = st.number_input("Ticket ID to mark resolved", min_value=0, step=1)
    with col2:
        st.write("")  # spacing to align button with input
        if st.button("Mark Resolved"):
            if resolve_id:
                update_status(int(resolve_id), "resolved")
                st.rerun()

# ---------------- Appointments tab ----------------
with tab_appts:
    st.markdown("AI-assisted appointment booking — pick a department, see live-available slots, confirm.")
    dept = st.selectbox("Department", DEPARTMENTS)
    slots = available_slots(dept)
    slot = st.selectbox("Available slot", slots) if slots else None
    patient_name = st.text_input("Patient name")

    if st.button("Book Appointment", type="primary"):
        if not patient_name or not slot:
            st.warning("Please enter your name and pick a slot.")
        else:
            result = book_appointment(patient_name, dept, slot)
            if result["success"]:
                st.success(result["message"])
            else:
                st.error(result["message"])
            st.rerun()

    appt_rows = list_appointments()
    appt_df = pd.DataFrame(appt_rows, columns=["ID", "Patient", "Department", "Slot", "Status", "Booked At"])
    st.dataframe(appt_df, use_container_width=True, hide_index=True)
