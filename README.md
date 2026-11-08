# RASAI AI — Multi-Vertical Helpdesk Prototype

RASAI AI is an experimental Streamlit helpdesk assistant prototype. It demonstrates how a shared workflow can support two sample knowledge domains by switching between separate document collections:

- **University / HEC**
- **Hospital**

The prototype combines intent classification, document retrieval, LLM-backed answers, complaint-ticket creation and a sample appointment-booking flow.

**Live prototype:** https://rasai-ai-ikwwym8y9hdyrbkbywuewk.streamlit.app/

> **Scope:** educational / hackathon prototype. It is not connected to a real university, HEC, hospital, appointment provider or institutional ticketing system. Sample documents and fixed appointment slots are used. Do not enter real patient, student or other sensitive personal data.

## Features

### 1. Intent classification

Messages are classified into four intents:

- Complaint
- Query
- Guidance
- Appointment

The first pass uses a small TF-IDF + Logistic Regression classifier trained from labelled example messages in core/ml_classifier.py. When confidence is below the configured threshold, the application can use the LLM client as a fallback.

A separate small evaluation set and script are included. Its results should be interpreted as a project-level check, not as a broad benchmark of real-world intent classification.

### 2. Knowledge-grounded answers

Each vertical has its own local text-document folder:

- data/university_docs/
- data/hospital_docs/

The RAG implementation in core/rag.py uses TF-IDF vectorization and cosine similarity to retrieve relevant text chunks. Retrieved passages are passed to the configured LLM client, which is instructed to answer from the supplied context and identify missing information rather than inventing policy facts.

### 3. Complaint tickets

Complaint messages can create structured SQLite tickets containing fields such as vertical, category, department, summary, priority, status and created time.

The ticket module also applies keyword-based SLA-day rules and derives an overdue / on-track status from the ticket's creation time and configured rule. These are demo rules, not official institutional service commitments.

### 4. Sample appointment workflow

The Hospital vertical includes a department selector and a list of predefined sample slots. A booking is stored in SQLite and the app prevents two confirmed bookings for the same department and slot.

These are **static demonstration slots**, not live clinic availability. There is no calendar, provider or hospital-system integration.

## Architecture

~~~text
User message
    |
    v
Intent classifier
(TF-IDF + Logistic Regression)
    |
    +---- low confidence ----> LLM fallback
    |
    v
Intent
    |
    +---- query / guidance ----> Vertical knowledge base
    |                              |
    |                              v
    |                         TF-IDF retrieval
    |                              |
    |                              v
    |                         LLM-grounded answer
    |
    +---- complaint ----------> Ticket generation
    |                              |
    |                              v
    |                         SQLite ticket store
    |
    +---- appointment --------> Appointment tab
                                   |
                                   v
                              Static sample slots
                                   |
                                   v
                              SQLite booking store
~~~

## Technology stack

| Component | Implementation |
|---|---|
| UI | Streamlit |
| Intent classifier | scikit-learn TF-IDF + Logistic Regression |
| Retrieval | scikit-learn TF-IDF + cosine similarity |
| LLM fallback / response generation | Groq, Gemini, or offline demo stub through core/llm_client.py |
| Data storage | SQLite |
| UI tables | Pandas |
| Configuration | environment variables, python-dotenv, Streamlit Secrets |

## Repository structure

~~~text
RASAI-AI/
├── streamlit_app.py
├── core/
│   ├── __init__.py
│   ├── classifier.py
│   ├── ml_classifier.py
│   ├── llm_client.py
│   ├── rag.py
│   ├── tickets.py
│   └── appointments.py
├── data/
│   ├── university_docs/
│   │   └── university_policy.txt
│   └── hospital_docs/
│       └── hospital_services.txt
├── scripts/
│   └── evaluate_classifier.py
├── requirements.txt
└── README.md
~~~

SQLite database files are created at runtime under the data/ directory.

## Run locally

### 1. Clone and install

~~~bash
git clone https://github.com/Qaisar-baloch/RASAI-AI.git
cd RASAI-AI
python -m venv .venv
~~~

Activate the environment:

**Windows**

~~~powershell
.venv\Scripts\activate
~~~

**macOS / Linux**

~~~bash
source .venv/bin/activate
~~~

Install requirements:

~~~bash
pip install -r requirements.txt
~~~

### 2. Configure an LLM provider (optional for offline demo mode)

Create a local .env file:

~~~env
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
~~~

The client attempts configured providers and falls back to a deterministic offline stub if no working API backend is available. Offline mode is useful for checking the application flow, but the stub is not a real LLM and does not provide actual knowledge-grounded answers.

For Streamlit hosting, add provider keys through the app's Secrets settings instead of committing them.

### 3. Launch

~~~bash
streamlit run streamlit_app.py
~~~

Open the local URL printed by Streamlit, normally http://localhost:8501.

## Try the demo

Example messages for the University / HEC vertical:

- My transcript has been delayed for three weeks.
- How long does HEC attestation take?
- How do I request a transcript?

Example messages for the Hospital vertical:

- What are the OPD timings?
- What departments are available?
- I want to book a cardiology appointment.

Use the Ticket Dashboard and Appointments tabs to inspect the resulting sample records.

## Evaluation

The classifier evaluation script uses a small held-out set defined in core/ml_classifier.py. Run:

~~~bash
python scripts/evaluate_classifier.py
~~~

The dataset is small and hand-curated. Treat reported accuracy as an internal project check, not a representative measure of performance across real users, accents, languages or institutions.

## Limitations and responsible use

- The knowledge files are sample documents, not authoritative live policies.
- TF-IDF retrieval is a lightweight lexical/statistical approach; it is not a dense embedding-based vector database.
- Classification examples are small and hand-labelled.
- LLM responses depend on the configured provider, model availability and credentials.
- Offline demo mode returns placeholder responses and must not be mistaken for live AI reasoning.
- Tickets and appointments are local SQLite records. Streamlit-hosted local storage may be reset when the app restarts or redeploys.
- Appointment slots are hard-coded samples, not live availability.
- No institutional authentication, access control, backend integration, human escalation or production data retention is implemented.
- Do not use this prototype to make real medical, legal, scholarship or institutional decisions.

## Acknowledgement

This project was developed as a training and prototype exercise. It explores how one application workflow can be reused across distinct document collections and service scenarios.
