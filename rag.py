"""RAG — policy-grounded answers.

Task: answer a customer question using ONLY the retrieved help-center text. If the
answer isn't in the docs, the model must say it doesn't know (don't hallucinate).

Fill in YOUR_SYSTEM_PROMPT and YOUR_RETRIEVER.
Run:  python rag.py
"""
import os
import re

from ollama import chat

MODEL = "llama3.1:8b"
DATA = os.path.join(os.path.dirname(__file__), "data", "help_center.txt")


def load_chunks() -> list[str]:
    with open(DATA, encoding="utf-8") as f:
        # one chunk per non-empty line
        return [line.strip() for line in f if line.strip()]


_STOPWORDS = {
    "the", "and", "for", "you", "your", "are", "can", "does", "did", "how", "what",
    "when", "where", "who", "why", "which", "have", "has", "was", "were", "will",
    "with", "that", "this", "there", "our", "any", "but", "not", "may", "get",
    "its", "his", "her", "from", "into", "about", "long", "much", "many", "most",
}


def _stem(word: str) -> str:
    """Very small suffix stripper so 'shipping'/'ship' and 'international(ly)' match."""
    for suffix in ("ally", "ing", "ed", "es", "al", "ly", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if len(word) > 3 and word[-1] == word[-2]:  # shipp -> ship
        word = word[:-1]
    return word


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {_stem(w) for w in words if len(w) >= 3 and w not in _STOPWORDS}


def YOUR_RETRIEVER(question: str, chunks: list[str], top_k: int = 3) -> list[str]:
    """Keyword-overlap retriever: rank chunks by shared stemmed keywords, keep the
    best few, and return nothing when there is no overlap at all."""
    q = _keywords(question)
    scored = [(len(q & _keywords(c)), i, c) for i, c in enumerate(chunks)]
    scored = [t for t in scored if t[0] > 0]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [c for _, _, c in scored[:top_k]]


YOUR_SYSTEM_PROMPT = """You are a customer-support assistant for a small online store.
Answer the customer's question using ONLY the text under "Context". Do not use outside knowledge and do not guess.

Rules:
1. If the Context contains the answer, reply in one or two short sentences, reusing the exact facts and numbers from the Context (for example "30 days", "international shipping").
2. If the Context is empty or does not contain the answer, reply with exactly this sentence and nothing else:
I don't know based on our help center, but I can connect you with a human agent who can help.
3. Never invent policies, prices, addresses, names, or phone numbers.
4. Be polite and concise."""

CASES = [
    ("How long do I have to return an item?", ["30 days"]),
    ("Do you ship internationally?", ["international"]),
    ("What's your CEO's home address?", ["don't know", "do not know", "not sure", "cannot"]),
]


def answer(question: str) -> str:
    context = "\n".join(YOUR_RETRIEVER(question, load_chunks()))
    resp = chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": YOUR_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        options={"temperature": 0.0},
    )
    return resp.message.content.strip().lower()


def run() -> bool:
    passed = 0
    for question, needles in CASES:
        got = answer(question)
        ok = any(n.lower() in got for n in needles)
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {question!r} -> {got[:80]!r}")
    print(f"{passed}/{len(CASES)} correct")
    return passed == len(CASES)


if __name__ == "__main__":
    run()
