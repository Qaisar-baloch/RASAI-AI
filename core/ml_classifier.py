"""
Feature: Trained ML Intent Classifier (fast first-pass).

A small TF-IDF + Logistic Regression model trained on labeled example messages.
Used as a fast, free, offline-capable first pass before falling back to the LLM
for low-confidence cases. This is the "model training" component of the project —
not just prompting an API — and ships with its own evaluation set/report.

Trained once at import time (dataset is tiny, trains in well under a second).
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ---- Training data: labeled example messages per intent ----
# 20 examples per class, covering university/HEC and hospital phrasing so the
# same classifier generalizes across verticals.
TRAINING_DATA = [
    # complaint
    ("my transcript has been delayed for 3 weeks", "complaint"),
    ("i paid my fee twice and no one has refunded me", "complaint"),
    ("my result was marked wrong and no one is responding", "complaint"),
    ("the registrar office is not answering my emails", "complaint"),
    ("i was overcharged on my hospital bill", "complaint"),
    ("the doctor was rude to me during my visit", "complaint"),
    ("my scholarship payment never arrived this semester", "complaint"),
    ("i have been waiting 2 hours past my appointment time", "complaint"),
    ("the lab report i was promised in 24 hours still isn't ready", "complaint"),
    ("my degree attestation request has been stuck for a month", "complaint"),
    ("nobody responded to my complaint from last week", "complaint"),
    ("this is the third time my fee receipt is wrong", "complaint"),
    ("my remarking request was ignored past the deadline", "complaint"),
    ("i was billed for a test i never took", "complaint"),
    ("the finance office keeps giving me different answers", "complaint"),
    ("my appointment got cancelled with no notice", "complaint"),
    ("i submitted documents twice and they still say missing", "complaint"),
    ("the pharmacy gave me the wrong medicine", "complaint"),
    ("my HEC verification has been pending for two months", "complaint"),
    ("i want to formally complain about staff conduct", "complaint"),
    # query
    ("what are the OPD timings", "query"),
    ("how long does HEC attestation take", "query"),
    ("what documents do i need for new patient registration", "query"),
    ("is the cardiology department open on saturday", "query"),
    ("what is the fee for result remarking", "query"),
    ("when will my lab report be ready", "query"),
    ("what are the finance office working hours", "query"),
    ("how much does a transcript cost", "query"),
    ("is there a scholarship for first year students", "query"),
    ("what is the emergency department's contact number", "query"),
    ("how many days does a refund take", "query"),
    ("what is the dress code for hospital visits", "query"),
    ("where is the billing help desk located", "query"),
    ("what are the visa requirements mentioned in university policy", "query"),
    ("how long is a routine blood test report valid", "query"),
    ("what departments are available at this hospital", "query"),
    ("what is the deadline for scholarship verification", "query"),
    ("how do i contact the patient relations office", "query"),
    ("what is the university ombudsperson's role", "query"),
    ("does the hospital have a pediatrics department", "query"),
    # guidance
    ("how do i apply for HEC scholarship verification", "guidance"),
    ("what is the process to request a transcript", "guidance"),
    ("how do i file a fee complaint", "guidance"),
    ("what steps do i need to get my degree attested", "guidance"),
    ("how can i request a lab report copy", "guidance"),
    ("guide me through the result remarking process", "guidance"),
    ("how do i register as a new patient", "guidance"),
    ("what is the procedure to escalate an unresolved complaint", "guidance"),
    ("how do i apply for a need-based scholarship", "guidance"),
    ("what is the process for a billing dispute", "guidance"),
    ("how do i get my intermediate certificate attested first", "guidance"),
    ("walk me through requesting a refund", "guidance"),
    ("how do i file a complaint about staff conduct", "guidance"),
    ("what's the process to switch my registered department", "guidance"),
    ("how do i request my medical records", "guidance"),
    ("steps to appeal a rejected scholarship application", "guidance"),
    ("how do i update my contact details with the registrar", "guidance"),
    ("what is the process for emergency admission", "guidance"),
    ("how can i verify my enrollment status", "guidance"),
    ("guide me on requesting an insurance claim form", "guidance"),
    # appointment
    ("i want to book an appointment with cardiology", "appointment"),
    ("can i schedule a visit for next monday", "appointment"),
    ("i need to reschedule my dermatology appointment", "appointment"),
    ("book me a slot with pediatrics this week", "appointment"),
    ("i want to cancel my appointment tomorrow", "appointment"),
    ("please schedule a follow-up visit for me", "appointment"),
    ("i'd like to see an orthopedic doctor, any slots open", "appointment"),
    ("can you book the earliest available cardiology slot", "appointment"),
    ("i need an appointment for a check-up", "appointment"),
    ("schedule me for OPD on wednesday", "appointment"),
    ("can i get a same-day appointment", "appointment"),
    ("book a consultation for my child with pediatrics", "appointment"),
    ("i'd like to change my appointment time", "appointment"),
    ("please confirm a slot for dermatology next week", "appointment"),
    ("i want to make a booking with the orthopedic department", "appointment"),
    ("reserve me a time with cardiology on friday", "appointment"),
    ("i need to set up a visit for a checkup", "appointment"),
    ("can you find me an open slot this week", "appointment"),
    ("book an appointment for a follow-up scan", "appointment"),
    ("schedule my next visit please", "appointment"),
]

# Small held-out evaluation set — NOT used in training — for measuring real accuracy.
EVAL_DATA = [
    ("my fee refund still hasn't come through after a month", "complaint"),
    ("the front desk was unhelpful and dismissive", "complaint"),
    ("my second lab test result is still missing", "complaint"),
    ("what time does the emergency room open", "query"),
    ("how much is a follow-up consultation", "query"),
    ("is HEC attestation free of cost", "query"),
    ("how do i apply for a transcript reissue", "guidance"),
    ("what is the process to dispute a hospital bill", "guidance"),
    ("steps to request an official degree verification letter", "guidance"),
    ("i want a cardiology appointment for next tuesday", "appointment"),
    ("can you book me a pediatrics slot", "appointment"),
    ("reschedule my orthopedic visit please", "appointment"),
]

INTENTS = ["complaint", "query", "guidance", "appointment"]

_texts = [t for t, _ in TRAINING_DATA]
_labels = [l for _, l in TRAINING_DATA]

_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
    ("clf", LogisticRegression(max_iter=1000)),
])
_pipeline.fit(_texts, _labels)


def ml_classify(message: str) -> dict:
    """Fast first-pass classification. Returns intent + confidence (0-1)."""
    probs = _pipeline.predict_proba([message])[0]
    classes = _pipeline.classes_
    best_idx = probs.argmax()
    return {"intent": classes[best_idx], "confidence": float(probs[best_idx])}


def evaluate() -> dict:
    """Run the held-out eval set and return an accuracy report (used by scripts/evaluate_classifier.py)."""
    correct = 0
    per_class = {intent: {"correct": 0, "total": 0} for intent in INTENTS}
    rows = []
    for text, true_label in EVAL_DATA:
        pred = ml_classify(text)
        is_correct = pred["intent"] == true_label
        correct += int(is_correct)
        per_class[true_label]["total"] += 1
        per_class[true_label]["correct"] += int(is_correct)
        rows.append({"text": text, "true": true_label, "predicted": pred["intent"],
                      "confidence": round(pred["confidence"], 3), "correct": is_correct})
    accuracy = correct / len(EVAL_DATA)
    return {"accuracy": accuracy, "n": len(EVAL_DATA), "per_class": per_class, "rows": rows}
