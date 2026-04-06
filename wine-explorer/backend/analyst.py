import re
import httpx
import numpy as np
import pandas as pd

OLLAMA_BASE = "http://localhost:11434"
ANALYST_MODEL = "qwen2.5:7b"
_ANALYTICAL_KEYWORDS = [
    "most expensive", "least expensive", "cheapest", "priciest", "costliest",
    "lowest price", "highest price", "highest priced", "lowest priced",
    "average price", "median price", "mean price",
    "how many", "how much", "count", "total number",
    "price of", "cost of", "what does", "what is the price", "what does it cost",
    "best rated", "worst rated", "highest rated", "lowest rated",
    "average rating", "average score", "highest score", "lowest score",
]

def is_analytical(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in _ANALYTICAL_KEYWORDS)

_CODE_PROMPT = """\
You have a pandas DataFrame called `df` with these columns:
  Name        - wine name (string)
  color       - "red", "white", "rose", "sparkling", or "fortified"
  Varietal    - grape variety (string)
  Region      - wine region (string)
  Appellation - sub-region / appellation (string)
  Country     - country of origin (string)
  Vintage     - year (string or number)
  Retail      - price in USD (float)
  Volume      - bottle size in ml (string)
  max_score   - highest professional score 0-100 (float, 0 if unrated)

Write Python code that assigns the answer to a variable called `result`.
You may use multiple lines. Do not use imports (pd and np are already available).
`result` must be a string, number, DataFrame, or Series.

Rules:
- For "most/least expensive": sort by Retail descending/ascending
- For "best/worst/highest/lowest rated": sort by max_score descending/ascending
- For filtered queries: apply boolean filters first, then sort
- To combine two results: result = pd.concat([df.nlargest(1,'Retail'), df.nsmallest(1,'Retail')])
- NEVER use df.append() — use pd.concat() instead
- "rating" and "score" both refer to max_score
- Exclude max_score==0 when looking for lowest rated
- For multiple scalar values, assign a formatted string:
    usa = df[df["Country"]=="United States"]["Retail"].mean()
    fra = df[df["Country"]=="France"]["Retail"].mean()
    result = f"United States avg: ${{usa:.2f}}, France avg: ${{fra:.2f}}"
- Always use full country names as they appear in the Country column
    (e.g., "United States" not "usa", "France" not "fra")
- You MUST always assign your final answer to a variable called `result`
- Do NOT put backslashes (\\n) or complex expressions inside f-string braces.
    Instead, build strings with .to_string() or concatenation outside the f-string.
- When returning rows, always return full rows (do not drop columns), limited to 25

Query: {query}

Code:"""

_MAX_RETRIES = 2

def _generate_code(messages: list[dict], timeout: float = 30.0) -> str:
    response = httpx.post(
        f"{OLLAMA_BASE}/api/chat",
        json={"model": ANALYST_MODEL, "stream": False, "messages": messages},
        timeout=timeout,
    )
    response.raise_for_status()
    raw = response.json()["message"]["content"].strip()
    raw = re.sub(r"^```[\w]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw).strip()
    return raw

def _extract_result(ns: dict, query: str) -> str | None:
    result = ns.get("result")
    if result is None:
        _skip = {"__builtins__", "df", "np", "pd", "result"}
        user_vars = {
            k: v for k, v in ns.items()
            if k not in _skip and not k.startswith("_")
        }
        if user_vars:
            parts = []
            for k, v in user_vars.items():
                if isinstance(v, float):
                    parts.append(f"{k} = {v:.2f}")
                else:
                    parts.append(f"{k} = {v}")
            result = "; ".join(parts)
            print(f"Analyst: recovered from user vars: {result}")
        else:
            return None

    if isinstance(result, tuple):
        formatted = ", ".join(
            f"{v:.2f}" if isinstance(v, float) else str(v) for v in result
        )
        return f"Query: {query}\nValues: {formatted}"

    if isinstance(result, pd.DataFrame):
        cols = ["Name", "Retail", "color", "Region", "Country", "max_score"]
        cols = [c for c in cols if c in result.columns]
        return result[cols].head(25).to_string(index=False)

    if isinstance(result, pd.Series):
        return result.head(25).to_string()

    return str(result)

def run(query: str, df: pd.DataFrame) -> str | None:
    messages = [{"role": "user", "content": _CODE_PROMPT.format(query=query)}]
    for attempt in range(_MAX_RETRIES):
        try:
            raw = _generate_code(messages, timeout=30.0 if attempt == 0 else 60.0)
            print(f"[ANALYST CODE attempt={attempt}]\n{raw}")

            ns: dict = {"__builtins__": __builtins__, "df": df, "np": np, "pd": pd}
            exec(raw, ns, ns) 

            result_str = _extract_result(ns, query)
            if result_str is None:
                print("Analyst: code ran but 'result' was not set.")
                return None

            print(f"[ANALYST RESULT] {result_str[:300]}")
            return result_str

        except Exception as e:
            print(f"Analyst attempt {attempt} failed: {e!r}")
            if attempt + 1 < _MAX_RETRIES:
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": (
                        f"That code raised: {e!r}\n"
                        f"Remember: filter with df[df[\"Country\"]==\"United States\"], "
                        f"not df[\"usa\"]. Column names are: {list(df.columns)}\n"
                        f"Fix the code and assign the answer to `result`."
                    ),
                })
            else:
                print("Analyst: all retries exhausted, falling back to RAG.")
                return None
