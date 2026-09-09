# Rasai AI — Universal Complaint, Query & Guidance Agent

One AI engine — classify → RAG-answer or auto-ticket → track — reused across two demo
verticals (University/HEC and Hospital) just by swapping the knowledge base.

Built for **Streamlit Cloud** deployment (`streamlit_app.py` is the main app).
A Gradio version also ships in `gradio_app.py` in case you want it later.

## What's already built (all phases delivered)

| Phase | What it does | File(s) |
|---|---|---|
| 1 | Intent classification (complaint / query / guidance / appointment) | `core/classifier.py` |
| 1 | RAG pipeline — TF-IDF retrieval over your policy docs + grounded LLM answer | `core/rag.py`, `data/university_docs/`, `data/hospital_docs/` |
| 2 | Auto ticket generation + SQLite storage + status tracking | `core/tickets.py` |
| 3 | Appointment slot booking (hospital vertical) | `core/appointments.py` |
| 4 | Streamlit UI tying it all together — Chat / Ticket Dashboard / Appointments, vertical toggle | `streamlit_app.py` |

Everything runs **without any API key** in "offline demo mode" (keyword-based classifier,
placeholder RAG answers) — that's intentional, so you can test the full flow first, then
flip on real generation once you add a key.

## 1. Files to upload to GitHub

Upload the whole folder, keeping this exact structure:
```
rasai-ai/
├── streamlit_app.py           <- main app (this is what Streamlit Cloud runs)
├── gradio_app.py               <- optional alternate UI, not needed for Streamlit Cloud
├── requirements.txt
├── README.md
├── core/
│   ├── __init__.py
│   ├── llm_client.py
│   ├── classifier.py
│   ├── rag.py
│   ├── tickets.py
│   └── appointments.py
└── data/
    ├── university_docs/university_policy.txt
    └── hospital_docs/hospital_services.txt
```
**Do NOT upload:** `.env` (your local secret file) or `data/tickets.db` (created automatically at runtime — it's your local test data, not needed on GitHub).

## 2. Test it locally first (optional but recommended)

```bash
cd rasai-ai
pip install -r requirements.txt
cp .env.example .env        # paste in a free Groq or Gemini key
streamlit run streamlit_app.py
```
Opens at `http://localhost:8501`. The caption under the title tells you whether it's
running live (API key detected) or in demo mode.

## 3. Try it (suggested test script for your demo video)

**University/HEC vertical (pick it in the left sidebar):**
- Type: `my transcript has been delayed for 3 weeks` → creates a **complaint ticket**
- Type: `how long does HEC attestation take` → gives a **RAG-grounded answer** citing the policy doc

**Hospital vertical:**
- Type: `what are OPD timings` → RAG answer from the hospital FAQ
- Type: `i want to book an appointment with cardiology` → points you to the Appointments tab
- Go to **Appointments** tab → pick Cardiology → pick a slot → book → see it appear in the table

**Ticket Dashboard tab:** shows every complaint generated above, filterable by vertical, with a "mark resolved" action.

## 4. Deploy on Streamlit Cloud

1. Push/upload this folder to a GitHub repo (see file list in step 1).
2. Go to **share.streamlit.io** → sign in with GitHub → **New app**.
3. Pick your repo, branch `main`, and set **Main file path** to `streamlit_app.py` (important — it defaults to `app.py`, which doesn't exist in this project, so you must change it).
4. Click **Advanced settings → Secrets** before deploying (or Settings → Secrets after) and paste:
   ```
   GROQ_API_KEY = "your-actual-key-here"
   ```
   (Streamlit's secrets format is TOML — key = "value", with quotes.)
5. Click **Deploy**. Build takes 1-3 minutes. You'll get a public `*.streamlit.app` URL — this is what you submit for the hackathon.

## 5. Common errors and fixes

| Error | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'core'` | The `core/` folder (with `__init__.py` inside) wasn't uploaded, or the repo structure doesn't match step 1 |
| App builds but shows the 🟡 yellow "Demo mode" banner | Your secret key name doesn't exactly match `GROQ_API_KEY`, or wasn't saved — recheck Settings → Secrets |
| `FileNotFoundError` mentioning `.txt` files | The `data/` folder (with both subfolders) wasn't uploaded |
| App won't start / wrong file error | Main file path in Streamlit Cloud settings must be `streamlit_app.py`, not `app.py` |
| `ModuleNotFoundError: No module named 'groq'` or `'google.generativeai'` | `requirements.txt` wasn't uploaded, or the build is still in progress — check the build logs |
| Tickets/appointments disappear after redeploying | Expected — the SQLite file lives in ephemeral storage on Streamlit Cloud and resets on redeploy/restart. Fine for a hackathon demo; a production version would use a persistent database |

## 6. Customize for a real institution
Swap in real policy documents:
```
data/university_docs/*.txt   # replace with real registrar/HEC/finance policy text
data/hospital_docs/*.txt     # replace with real hospital services/FAQ text
```
The RAG pipeline re-indexes automatically on next run — no code changes needed.
This is the "one engine, any sector" story for your presentation: the same
`core/classifier.py`, `core/rag.py`, and `core/tickets.py` work unchanged; only the
`.txt` files and the `DEPARTMENTS` list in `core/appointments.py` change per institution.

## 7. What's intentionally out of scope (v1 / say this in your pitch)
- No real integration with an institution's actual backend/SIS/HMIS
- No live SMS/WhatsApp/telephony channel (chat UI simulates the channel)
- No human-agent handoff
- Urdu voice input was scoped as a stretch goal — not included in this build; the
  architecture supports adding open-source Whisper for it without changing `core/`
