---
title: Rasai AI
emoji: 🇵🇰
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.31.0
app_file: app.py
pinned: false
---

# Rasai AI — Universal Complaint, Query & Guidance Agent

One AI engine — classify → RAG-answer or auto-ticket → track — reused across two demo
verticals (University/HEC and Hospital) just by swapping the knowledge base.

## What's already built (all phases delivered)

| Phase | What it does | File(s) |
|---|---|---|
| 1 | Intent classification (complaint / query / guidance / appointment) | `core/classifier.py` |
| 1 | RAG pipeline — TF-IDF retrieval over your policy docs + grounded LLM answer | `core/rag.py`, `data/university_docs/`, `data/hospital_docs/` |
| 2 | Auto ticket generation + SQLite storage + status tracking | `core/tickets.py` |
| 3 | Appointment slot booking (hospital vertical) | `core/appointments.py` |
| 4 | Gradio UI tying it all together — Chat / Ticket Dashboard / Appointments, vertical toggle | `app.py` |

Everything runs **without any API key** in "offline demo mode" (keyword-based classifier,
placeholder RAG answers) — that's intentional, so you can test the full flow first, then
flip on real generation once you add a key.

## 1. Setup (5 minutes)

```bash
cd rasai-ai
pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and paste in **one** of these (both are free to get):
- `GROQ_API_KEY` — from https://console.groq.com (recommended: fast, generous free tier)
- `GEMINI_API_KEY` — from https://aistudio.google.com/apikey (fallback / multimodal option)

## 2. Run locally

```bash
python app.py
```

Opens at `http://127.0.0.1:7860`. The banner at the top of the app tells you whether
it's running live (API key detected) or in demo mode.

## 3. Try it (suggested test script for your demo video)

**University/HEC vertical:**
- Type: `my transcript has been delayed for 3 weeks` → should create a **complaint ticket**
- Type: `how long does HEC attestation take` → should give a **RAG-grounded answer** citing the policy doc
- Switch vertical to **Hospital**

**Hospital vertical:**
- Type: `what are OPD timings` → RAG answer from the hospital FAQ
- Type: `i want to book an appointment with cardiology` → routes you to the Appointments tab
- Go to **Appointments** tab → pick Cardiology → pick a slot → book → see it appear in the table

**Ticket Dashboard tab:** shows every complaint generated above, filterable by vertical, with a "mark resolved" action.

## 4. Deploy (free)

**Option A — Hugging Face Spaces (recommended for Gradio):**
1. Create a new Space → SDK: Gradio
2. Push this folder's contents to the Space repo (`app.py`, `core/`, `data/`, `requirements.txt`)
3. Add `GROQ_API_KEY` / `GEMINI_API_KEY` under Space Settings → Repository secrets
4. Space auto-builds and gives you a public URL

**Option B — Streamlit Cloud:** works too, but you'd need to port `app.py`'s UI layer to
Streamlit widgets since the core logic (`core/`) is framework-agnostic and reusable as-is.

## 5. Customize for a real institution
Swap in real policy documents:
```
data/university_docs/*.txt   # replace with real registrar/HEC/finance policy text
data/hospital_docs/*.txt     # replace with real hospital services/FAQ text
```
The RAG pipeline re-indexes automatically on next run — no code changes needed.
This is the "one engine, any sector" story for your presentation: the same
`core/classifier.py`, `core/rag.py`, and `core/tickets.py` work unchanged; only the
`.txt` files and the `DEPARTMENTS` list in `core/appointments.py` change per institution.

## 6. What's intentionally out of scope (v1 / say this in your pitch)
- No real integration with an institution's actual backend/SIS/HMIS
- No live SMS/WhatsApp/telephony channel (chat UI simulates the channel)
- No human-agent handoff
- Urdu voice input was scoped as a stretch goal — not included in this build; the
  architecture supports adding open-source Whisper for it without changing `core/`

## Project structure
```
rasai-ai/
├── app.py                          # Gradio UI (Phase 4 — ties everything together)
├── core/
│   ├── llm_client.py                # Unified Groq/Gemini client + offline demo-mode fallback
│   ├── classifier.py                 # Phase 1 — intent classification
│   ├── rag.py                        # Phase 1 — TF-IDF retrieval + grounded answers
│   ├── tickets.py                    # Phase 2 — ticket generation + SQLite
│   └── appointments.py               # Phase 3 — slot booking + SQLite
├── data/
│   ├── university_docs/university_policy.txt
│   └── hospital_docs/hospital_services.txt
├── requirements.txt
└── .env.example
```
