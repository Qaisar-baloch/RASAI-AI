"""
Phase 1 — RAG pipeline.

Uses TF-IDF + cosine similarity for retrieval (open-source, lightweight,
no large model download needed — reliable to set up inside a 2-day hackathon).
Retrieved chunks are then passed to the LLM so answers stay grounded in the
institution's actual documents instead of the model guessing.
"""

import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from core.llm_client import chat

ANSWER_SYSTEM_PROMPT = """You are a helpdesk assistant. Answer the user's question using ONLY the
CONTEXT provided below. If the context does not contain the answer, say clearly
that you don't have that information on file and suggest they contact the relevant
department directly — do not invent procedures or facts.
Keep the answer short, plain-language, and cite which document snippet you used.
"""


def _chunk_text(text: str, chunk_size: int = 400) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)] or [text]


class VerticalKnowledgeBase:
    """One knowledge base per vertical (e.g. 'university' or 'hospital')."""

    def __init__(self, doc_folder: str):
        self.chunks: list[str] = []
        self.sources: list[str] = []
        for path in sorted(glob.glob(os.path.join(doc_folder, "*.txt"))):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            for chunk in _chunk_text(text):
                self.chunks.append(chunk)
                self.sources.append(os.path.basename(path))

        self.vectorizer = TfidfVectorizer(stop_words="english")
        if self.chunks:
            self.matrix = self.vectorizer.fit_transform(self.chunks)
        else:
            self.matrix = None

    def retrieve(self, query: str, top_k: int = 2) -> list[dict]:
        if not self.chunks or self.matrix is None:
            return []
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        top_idx = sims.argsort()[::-1][:top_k]
        return [
            {"text": self.chunks[i], "source": self.sources[i], "score": float(sims[i])}
            for i in top_idx if sims[i] > 0
        ]

    def answer(self, query: str, top_k: int = 2) -> dict:
        retrieved = self.retrieve(query, top_k=top_k)
        context = "\n\n".join(f"[{r['source']}] {r['text']}" for r in retrieved) or "(no relevant document found)"
        prompt = f"CONTEXT:\n{context}\n\nQUESTION: {query}"
        answer_text = chat(prompt=prompt, system=ANSWER_SYSTEM_PROMPT)
        return {
            "answer": answer_text,
            "sources": list({r["source"] for r in retrieved}),
            "retrieved_chunks": retrieved,
        }
