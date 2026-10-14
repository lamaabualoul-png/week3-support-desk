
# Week 3 — Prompting Techniques: The Support-Desk Assistant

Model: `llama3.1:8b` via Ollama, `temperature=0.0`. Only the `TODO` parts (prompts, tool schema, retriever) were edited.

---

## 1. K-shot prompting — ticket classification (`k_shot_prompting.py`)

**Final prompt**

```text
You are a ticket classifier for an online store's support desk.
Classify each customer message into exactly ONE of these labels:

billing   - charges, payments, invoices, double charges, payment methods, refunds of money already charged to a card
shipping  - delivery status, lost or late packages, tracking, delivery address, shipping cost
returns   - sending an item back, return/exchange requests, return labels, refund for a returned product
technical - website or app bugs, crashes, login problems, errors, things not loading
other     - anything that fits none of the above (greetings, feedback, general questions)

Rules:
- Reply with ONLY the label: one lowercase word, no punctuation, no explanation.
- Pick the single best label even if the message touches several topics.

Examples:  (9 labeled "Message: ... / Label: ..." pairs: 2 billing, 2 shipping,
2 returns, 2 technical, 1 other — see k_shot_prompting.py for the full list)
```

**Sample input / output**

```text
Input : My card was charged twice for order 4412.
Output: got 'billing', expected 'billing'
```

**What I learned:** Labeled examples teach the output *format*  as much as the task itself without them the model tends to add explanations that break an exact-match check.

---

## 2. Chain-of-thought — refund eligibility (`chain_of_thought.py`)

**Final prompt**

```text
You are a careful refund-eligibility agent. Decide whether the order below qualifies for a refund under the policy that follows.

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
```

**Sample input / output**

```text
Input : Bought 45 days ago, unused.
Output: Step 1: 45 days is more than 30 days - fail.
        Step 2: The item is unused - pass.
        Step 3: Not stated as final sale - pass.
        Step 4: One rule failed, so no refund.
        DECISION: NO, expected NO
```

**What I learned:** Forcing the model to check each policy rule separately before answering stops it from "pattern-matching" to a quick yes, and a strict last-line format makes the free-form reasoning machine-gradable.

---

## 3. Tool calling — order lookup (`tool_calling.py`)

**Final prompt + tool description**

```text
System prompt:
You are a support-desk assistant for a small online store.

When a customer asks about an order, you MUST call the get_order_status tool with the order id from their message BEFORE answering. Never guess or make up an order status.
After the tool returns, reply in one or two friendly sentences that state the status you were given. If the tool says "not found", tell the customer you could not find that order and ask them to double-check the number.
If the customer does not give an order id, ask for it instead of calling the tool.

Tool description:
Look up the current status of a customer's order (for example processing, shipped, arriving date, or not found) using its order id. Call this whenever the customer asks about, checks on, or tracks an order.

Parameter `order_id`:
The order id exactly as the customer wrote it, digits only, e.g. '4412'. Do not include '#' or any other words.
```

**Sample input / output**

```text
Input : Hi, can you check on order 4412 for me?
Output: [PASS] tool called with order_id='4412'
        tool result: shipped, arriving Friday
        (final reply, once the result is fed back: "Good news - order 4412 has shipped and is arriving Friday.")
```

**What I learned:** The model only calls a tool when the *description* tells it when and how a precise tool/parameter description plus a "look it up before answering" instruction turns a guess into a grounded, verifiable action.

---

## 4. RAG — grounded answers (`rag.py`)

**Final prompt + retriever**

```text
System prompt:
You are a customer-support assistant for a small online store.
Answer the customer's question using ONLY the text under "Context". Do not use outside knowledge and do not guess.

Rules:
1. If the Context contains the answer, reply in one or two short sentences, reusing the exact facts and numbers from the Context (for example "30 days", "international shipping").
2. If the Context is empty or does not contain the answer, reply with exactly this sentence and nothing else:
I don't know based on our help center, but I can connect you with a human agent who can help.
3. Never invent policies, prices, addresses, names, or phone numbers.
4. Be polite and concise.
```

Retriever: lower-case the question, drop stop-words and short tokens, apply a tiny suffix stemmer (so `ship`/`shipping`, `international`/`internationally` match), score each help-center line by shared keywords, return the top 3 lines with a score > 0 — and **nothing** if there is no overlap.

**Sample input / output**

```text
Input : Do you ship internationally?
Output: Yes, we offer international shipping to most countries; delivery takes 7-21 business days.

Input : What's your CEO's home address?
Output: I don't know based on our help center, but I can connect you with a human agent.
```

**What I learned:** Retrieval decides what the model can know, and the prompt decides what it is allowed to say an empty context plus an explicit "say you don't know" rule is what turns a hallucination into a polite refusal.

---

## Closing — what I learned across all four

LLMs are good at structured tasks because they are strong pattern matchers over language: show them a few labeled examples, a clear schema, or a fixed output format and they generalize the pattern reliably, which is why k-shot classification and tool-call argument extraction worked with very short prompts. But a small local model is not a reliable reasoner or fact store on its own. It needs scaffolding: chain-of-thought to make it check each rule instead of jumping to an answer, a tool so it fetches live data rather than inventing it, and retrieval plus a refusal rule so it stays inside known facts. In every case the biggest gains came from constraining the output (exact label, final `DECISION:` line, exact "I don't know" sentence) and giving the model the right information, rather than from adding more clever wording.
