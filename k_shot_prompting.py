"""K-shot prompting — ticket classification.

Task: classify a customer message into one of: billing, shipping, returns,
technical, other. Use in-context EXAMPLES (k-shot) so the model outputs exactly the
label, lowercase, with no extra text.

Only edit YOUR_SYSTEM_PROMPT.
Run:  python k_shot_prompting.py
"""
from ollama import chat

MODEL = "llama3.1:8b"
NUM_RUNS = 5

YOUR_SYSTEM_PROMPT = """You are a ticket classifier for an online store's support desk.
Classify each customer message into exactly ONE of these labels:

billing   - charges, payments, invoices, double charges, payment methods, refunds of money already charged to a card
shipping  - delivery status, lost or late packages, tracking, delivery address, shipping cost
returns   - sending an item back, return/exchange requests, return labels, refund for a returned product
technical - website or app bugs, crashes, login problems, errors, things not loading
other     - anything that fits none of the above (greetings, feedback, general questions)

Rules:
- Reply with ONLY the label: one lowercase word, no punctuation, no explanation.
- Pick the single best label even if the message touches several topics.

Examples:
Message: I was billed $59 but my order total said $49.
Label: billing

Message: Why was I charged twice for my subscription this month?
Label: billing

Message: My parcel says delivered but nothing is at my door.
Label: shipping

Message: Can I change the delivery address? The order hasn't arrived yet.
Label: shipping

Message: The jacket doesn't fit, how do I exchange it for a larger size?
Label: returns

Message: Please send me a return label for the damaged blender.
Label: returns

Message: I can't log in, the site keeps saying "something went wrong".
Label: technical

Message: The checkout page freezes when I press Pay.
Label: technical

Message: Thanks, you guys are great!
Label: other
"""

CASES = [
    ("My card was charged twice for order 4412.", "billing"),
    ("Where is my package? It's been two weeks.", "shipping"),
    ("I want to send the shoes back for a refund.", "returns"),
    ("The app crashes every time I open my cart.", "technical"),
]


def classify(system_prompt: str, message: str) -> str:
    resp = chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        options={"temperature": 0.0},
    )
    return resp.message.content.strip().lower()


def run() -> bool:
    passed = 0
    for message, expected in CASES:
        got = classify(YOUR_SYSTEM_PROMPT, message)
        ok = got == expected
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {message!r} -> got {got!r}, expected {expected!r}")
    print(f"{passed}/{len(CASES)} correct")
    return passed == len(CASES)


if __name__ == "__main__":
    run()
