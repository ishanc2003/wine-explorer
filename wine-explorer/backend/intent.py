import re
import httpx

OLLAMA_BASE = "http://localhost:11434"
INTENT_MODEL = "gemma2:9b"
_PROMPT = """\
Extract wine search filters from this query. Reply with ONLY a single line of key=value pairs. Omit any filter not mentioned.

Allowed keys:
price_max=<number>
price_min=<number>
color=<red|white|rose|sparkling|fortified>
region=<string>
country=<string>
sort=<cheapest|most_expensive|best_rated>

Rules:
- "under X" / "below X" / "less than X" → price_max=X
- "over X" / "above X" / "at least X" → price_min=X
- "broke" / "budget" / "cheap" → sort=cheapest
- "splurge" / "no budget" / "most expensive" → sort=most_expensive
- "best rated" / "top rated" → sort=best_rated
- "not red" → do NOT set color=red

Examples:
Query: cheap white wine from France under 30
Answer: price_max=30 color=white country=France sort=cheapest

Query: best rated Burgundy reds
Answer: color=red region=Burgundy sort=best_rated

Query: something nice to drink
Answer:

Query: {query}
Answer:"""

def extract(query: str) -> dict:
    try:
        response = httpx.post(
            f"{OLLAMA_BASE}/api/chat",
            json={
                "model": INTENT_MODEL,
                "stream": False,
                "messages": [{"role": "user", "content": _PROMPT.format(query=query)}],
            },
            timeout=20.0,
        )
        response.raise_for_status()
        raw = response.json()["message"]["content"].strip()
        print(f"[INTENT RAW] {repr(raw)}")
        result: dict = {}
        m = re.search(r"price_max=(\d+(?:\.\d+)?)", raw)
        if m:
            result["price_max"] = float(m.group(1))

        m = re.search(r"price_min=(\d+(?:\.\d+)?)", raw)
        if m:
            result["price_min"] = float(m.group(1))

        m = re.search(r"color=(red|white|rose|sparkling|fortified)", raw)
        if m:
            result["color"] = m.group(1)

        m = re.search(r"region=([^\s]+(?:\s[^\s=]+)*?)(?=\s+\w+=|$)", raw)
        if m:
            result["region"] = m.group(1).strip()

        m = re.search(r"country=([^\s]+(?:\s[^\s=]+)*?)(?=\s+\w+=|$)", raw)
        if m:
            result["country"] = m.group(1).strip()

        m = re.search(r"sort=(cheapest|most_expensive|best_rated)", raw)
        if m:
            result["sort"] = m.group(1)

        print(f"[INTENT PARSED] {result}")
        return result
    except Exception as e:
        print(f"Intent extraction failed ({e!r}), using regex fallback.")
        return {}
