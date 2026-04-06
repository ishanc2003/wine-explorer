import hashlib
import json
import os
import re
import httpx
import intent as intent_mod
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz import process as fz_process

OLLAMA_BASE = "http://localhost:11434"
EMBED_MODEL = "qwen3-embedding:8b"
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "wines.csv")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "embeddings_cache.npz")

_df: pd.DataFrame | None = None      
_wines: list[str] = []               
_names: list[str] = []               
_matrix: np.ndarray | None = None    

def _csv_hash() -> str:
    with open(DATA_PATH, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def _wine_to_text(row: pd.Series) -> str:
    parts = []
    for col, val in row.items():
        if pd.notna(val) and str(val).strip():
            parts.append(f"{col}: {val}")
    return ", ".join(parts)

def _max_score(ratings_json) -> float:
    try:
        ratings = json.loads(ratings_json) if isinstance(ratings_json, str) else []
        scores = [r["score"] for r in ratings if "score" in r]
        return float(max(scores)) if scores else 0.0
    except Exception:
        return 0.0

def _embed(texts: list[str]) -> np.ndarray:
    response = httpx.post(
        f"{OLLAMA_BASE}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=120.0,
    )
    response.raise_for_status()
    return np.array(response.json()["embeddings"], dtype=np.float32)

def _load_cache(csv_hash: str) -> bool:
    global _wines, _names, _matrix
    if not os.path.exists(CACHE_PATH):
        return False
    try:
        cache = np.load(CACHE_PATH, allow_pickle=True)
        if cache["csv_hash"].item() != csv_hash:
            print("CSV changed — rebuilding embeddings cache.")
            return False
        _matrix = cache["matrix"]
        _wines = json.loads(cache["wines"].item())
        _names = json.loads(cache["names"].item())
        if len(_names) != len(_wines):
            print(f"Cache names/wines length mismatch ({len(_names)} vs {len(_wines)}) — rebuilding.")
            return False
        print(f"Loaded embeddings from cache ({len(_wines)} wines).")
        return True
    except Exception as e:
        print(f"Cache load failed ({e}) — rebuilding.")
        return False

def _save_cache(csv_hash: str) -> None:
    np.savez_compressed(
        CACHE_PATH,
        matrix=_matrix,
        wines=np.array(json.dumps(_wines)),
        names=np.array(json.dumps(_names)),
        csv_hash=np.array(csv_hash),
    )
    print(f"Embeddings cached to {CACHE_PATH}")

def build_index() -> None:
    global _df, _wines, _names, _matrix
    df = pd.read_csv(DATA_PATH)
    df["max_score"] = df["professional_ratings"].apply(_max_score)
    _df = df
    print(f"Loaded {len(df)} wines from CSV.")
    csv_hash = _csv_hash()
    if _load_cache(csv_hash):
        return

    _wines = [_wine_to_text(row) for _, row in df.iterrows()]
    _names = df["Name"].fillna("").astype(str).tolist()
    print(f"Embedding {len(_wines)} wines via Ollama ({EMBED_MODEL})...")
    batch_size = 32
    embeddings = []
    for i in range(0, len(_wines), batch_size):
        batch = _wines[i : i + batch_size]
        embeddings.append(_embed(batch))
        print(f"  Embedded {min(i + batch_size, len(_wines))}/{len(_wines)}")

    _matrix = np.vstack(embeddings)
    norms = np.linalg.norm(_matrix, axis=1, keepdims=True)
    _matrix = _matrix / np.where(norms == 0, 1, norms)
    _save_cache(csv_hash)
    print(f"RAG index built ({len(_wines)} wines).")

_REGIONS = [
    "burgundy", "bordeaux", "champagne", "tuscany", "piedmont", "alsace",
    "rhône", "rhone", "provence", "loire", "rioja", "napa", "sonoma",
    "barossa", "marlborough", "douro", "priorat",
]
_COUNTRIES = [
    "france", "italy", "spain", "argentina", "chile", "australia",
    "germany", "portugal", "new zealand",
]
_COLOR_MAP = {
    "red":       ["red"],
    "white":     ["white"],
    "rosé":      ["rose", "rosé"],
    "rose":      ["rose", "rosé"],
    "sparkling": ["sparkling"],
    "champagne": ["sparkling"],
    "prosecco":  ["sparkling"],
    "fortified": ["fortified"],
}

def _filters_from_intent(parsed: dict) -> tuple[np.ndarray | None, bool]:
    """Apply structured filter dict (from intent extraction) to _df."""
    if _df is None:
        return None, False

    mask = pd.Series([True] * len(_df), index=_df.index)
    color = (parsed.get("color") or "").lower()
    if color and color in _COLOR_MAP:
        mask &= _df["color"].str.lower().isin(_COLOR_MAP[color])

    price_max = parsed.get("price_max")
    if price_max is not None:
        mask &= _df["Retail"] <= float(price_max)

    price_min = parsed.get("price_min")
    if price_min is not None:
        mask &= _df["Retail"] >= float(price_min)

    region = (parsed.get("region") or "").lower()
    if region:
        mask &= (
            _df["Region"].str.lower().str.contains(region, na=False)
            | _df["Appellation"].str.lower().str.contains(region, na=False)
        )

    country = (parsed.get("country") or "").lower()
    if country:
        mask &= _df["Country"].str.lower().str.contains(country, na=False)

    positions = np.where(mask.values)[0]
    sort = parsed.get("sort")

    if sort == "most_expensive":
        order = np.argsort(-_df["Retail"].values[positions])
        return positions[order[:20]], True

    if sort == "cheapest":
        order = np.argsort(_df["Retail"].values[positions])
        return positions[order[:20]], True

    if sort == "best_rated":
        order = np.argsort(-_df["max_score"].values[positions])
        return positions[order[:20]], True

    if len(positions) < len(_df):
        return positions, False

    return None, False

def _filters_from_regex(query: str) -> tuple[np.ndarray | None, bool]:
    if _df is None:
        return None, False

    q = query.lower()
    mask = pd.Series([True] * len(_df), index=_df.index)
    _NEGATIONS = ["no ", "not ", "avoid ", "without ", "don't want ", "hate ", "detest "]
    for keyword, colors in _COLOR_MAP.items():
        if keyword in q:
            negated = any(f"{neg}{keyword}" in q for neg in _NEGATIONS)
            if not negated:
                mask &= _df["color"].str.lower().isin(colors)
            break

    _WORD_NUMS = {
        "a hundred": "100", "one hundred": "100", "two hundred": "200",
        "three hundred": "300", "four hundred": "400", "five hundred": "500",
        "six hundred": "600", "seven hundred": "700", "eight hundred": "800",
        "nine hundred": "900", "a thousand": "1000", "one thousand": "1000",
        "two thousand": "2000", "fifty": "50", "forty": "40", "thirty": "30",
        "twenty": "20", "ten": "10", "hundred": "100", "thousand": "1000",
    }
    q_p = q
    for word, num in _WORD_NUMS.items():
        q_p = q_p.replace(word, num)

    m = re.search(
        r"(?:under|below|less than|cheaper than|on a|afford|budget(?:\s+of)?)\s*\$?(\d+(?:\.\d+)?)",
        q_p,
    )
    if m:
        mask &= _df["Retail"] < float(m.group(1))

    m = re.search(r"(?:over|above|more than|at least)\s*\$?(\d+(?:\.\d+)?)", q_p)
    if m:
        mask &= _df["Retail"] > float(m.group(1))

    for region in _REGIONS:
        if region in q:
            mask &= (
                _df["Region"].str.lower().str.contains(region, na=False)
                | _df["Appellation"].str.lower().str.contains(region, na=False)
            )
            break

    for country in _COUNTRIES:
        if country in q:
            mask &= _df["Country"].str.lower().str.contains(country, na=False)
            break
    if any(kw in q for kw in ["american", "united states", "united states of america",
                               "california", "napa", "sonoma"]):
        mask &= _df["Country"].str.lower().str.contains("united states", na=False)

    positions = np.where(mask.values)[0]
    if any(p in q for p in ["most expensive", "priciest", "costliest", "highest price"]):
        order = np.argsort(-_df["Retail"].values[positions])
        return positions[order[:8]], True

    if any(p in q for p in ["cheapest", "least expensive", "lowest price", "most affordable",
                              "broke", "budget", "inexpensive", "cheap"]):
        order = np.argsort(_df["Retail"].values[positions])
        return positions[order[:8]], True

    if any(p in q for p in ["best rated", "best-rated", "highest rated", "top rated",
                              "highest score", "most acclaimed"]):
        order = np.argsort(-_df["max_score"].values[positions])
        return positions[order[:8]], True

    if len(positions) < len(_df):
        return positions, False

    return None, False

def _apply_filters(query: str) -> tuple[np.ndarray | None, bool]:
    parsed = intent_mod.extract(query)
    if parsed:
        return _filters_from_intent(parsed)
    return _filters_from_regex(query)

def retrieve(query: str, top_k: int = 8) -> list[str]:
    if _matrix is None or not _wines:
        raise RuntimeError("RAG index not built. Call build_index() first.")

    candidate_indices, already_sorted = _apply_filters(query)
    if already_sorted and candidate_indices is not None:
        candidate_indices = candidate_indices[candidate_indices < len(_wines)]
        return [_wines[i] for i in candidate_indices[:top_k]]

    if candidate_indices is not None and len(candidate_indices) > 0:
        candidate_indices = candidate_indices[candidate_indices < len(_names)]
        top_k = max(top_k, min(len(candidate_indices), 20))

    if candidate_indices is not None and len(candidate_indices) > 0:
        search_matrix = _matrix[candidate_indices]
        search_names  = [_names[i] for i in candidate_indices]
    else:
        candidate_indices = np.arange(len(_wines))
        search_matrix = _matrix
        search_names  = _names

    seen: set[int] = set()
    results: list[str] = []
    fuzzy_hits = fz_process.extract(
        query,
        search_names,
        scorer=fuzz.token_set_ratio,
        limit=top_k,
        score_cutoff=60,
    )
    for _name, _score, local_idx in sorted(fuzzy_hits, key=lambda x: x[1], reverse=True):
        global_idx = candidate_indices[local_idx]

        if global_idx not in seen:
            seen.add(global_idx)
            results.append(_wines[global_idx])

    remaining = top_k - len(results)
    if remaining > 0:
        q_vec = _embed([query])[0]
        norm = np.linalg.norm(q_vec)

        if norm > 0:
            q_vec = q_vec / norm

        scores = search_matrix @ q_vec
        for local_idx in np.argsort(scores)[::-1]:
            global_idx = candidate_indices[local_idx]
            
            if global_idx not in seen:
                seen.add(global_idx)
                results.append(_wines[global_idx])
                remaining -= 1
            
                if remaining == 0:
                    break

    return results
