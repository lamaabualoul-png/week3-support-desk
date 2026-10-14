"""Chain-of-thought — refund eligibility.

Task: given the refund policy and an order, decide YES or NO. Use chain-of-thought
so the model reasons step by step, but make the FINAL line exactly `DECISION: YES`
or `DECISION: NO` so it can be graded.

Only edit YOUR_SYSTEM_PROMPT.
Run:  python chain_of_thought.py
"""
from ollama import chat

MODEL = "llama3.1:8b"

POLICY = """Refund policy:
- Refunds allowed within 30 days of purchase.
- Item must be unused / in original condition.
- Final-sale items are never refundable.
"""

YOUR_SYSTEM_PROMPT = """You are a careful refund-eligibility agent. Decide whether the order below qualifies for a refund under the policy that follows.

Think step by step before deciding. Work through every rule one at a time:
Step 1: Time window - how many days since purchase? Is that 30 days or fewer?
Step 2: Condition - is the item unused / in original condition?
Step 3: Final sale - is it a final-sale item? (If yes, it is never refundable.)
Step 4: Combine - a refund is allowed ONLY IF every rule passes. If any single rule fails, the answer is NO.

Be concise: one short line per step, quoting the facts from the order. Do not invent facts that are not in the order.

Formatting rules (very important):
- The LAST line of your reply must be exactly `DECISION: YES` or exactly `DECISION: NO`.
- Nothing may come after that line: no punctuation, no bold/markdown, no explanation.

Example
Order: Bought 20 days ago, opened and used, not final sale.
Step 1: 20 days is within 30 days - pass.
Step 2: The item was used - fail.
Step 3: Not final sale - pass.
Step 4: One rule failed, so no refund.
DECISION: NO
"""

CASES = [
    ("Bought 10 days ago, unused, not final sale.", "YES"),
    ("Bought 45 days ago, unused.", "NO"),
    ("Bought 5 days ago but it was a final-sale item.", "NO"),
]


def decide(system_prompt: str, order: str) -> str:
    resp = chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt + "\n" + POLICY},
            {"role": "user", "content": order},
        ],
        options={"temperature": 0.0},
    )
    text = resp.message.content.strip()
    last = text.splitlines()[-1].upper() if text else ""
    return "YES" if "DECISION: YES" in last else "NO" if "DECISION: NO" in last else "?"


def run() -> bool:
    passed = 0
    for order, expected in CASES:
        got = decide(YOUR_SYSTEM_PROMPT, order)
        ok = got == expected
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {order!r} -> {got}, expected {expected}")
    print(f"{passed}/{len(CASES)} correct")
    return passed == len(CASES)


if __name__ == "__main__":
    run()
