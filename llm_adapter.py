# llm_adapter.py
"""
LLM adapter for MCP server (robust, modern OpenAI client usage).

Exports:
  - call_llm_to_structured(nl, default_table=None, api_key=None, model=None, temperature=None)
  - call_llm_for_payload(prompt, api_key=None, model=None, temperature=None)

Notes:
 - Expects openai>=1.0.0 style client: from openai import OpenAI
 - Reads OPENAI_API_KEY from env if api_key not passed
 - Defensive extraction of assistant text from response object
"""

import os
import json
import textwrap
import re
from typing import Dict, Any, Optional, List
from openai import OpenAI
from schema_cache import get_table_columns
from nl_parser import TABLE_CANONICAL_MAP  # reuse canonical map for validation
import logging

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

# model / sampling defaults
LLM_MODEL = os.getenv("MCP_LLM_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("MCP_LLM_TEMPERATURE", "0.0"))

# canonical region mapping (lowercase keys)
REGION_CANONICAL_MAP = {
    "london": "uk-england-london",
    "uk london": "uk-england-london",
    "uk-england-london": "uk-england-london",
    "glasgow": "uk-scotland-glasgow",
    "scotland glasgow": "uk-scotland-glasgow",
    "uk-scotland-glasgow": "uk-scotland-glasgow",
    "dublin": "ie-dublin",
    "ie dublin": "ie-dublin",
    "ie-dublin": "ie-dublin",
    "berlin": "de-berlin",
    "de berlin": "de-berlin",
    "de-berlin": "de-berlin",
    "bengaluru": "in-bengaluru",
    "bangalore": "in-bengaluru",
    "in bengaluru": "in-bengaluru",
    "in-bengaluru": "in-bengaluru",
    "us-remote": "us-remote",
    "us remote": "us-remote"
    # add more mappings as you discover them
}

def normalize_region_token(token: str) -> str:
    """Normalize free-text region token to canonical region used in tables.
       Returns original token if no mapping found (preserves user input)."""
    if not token:
        return token
    k = token.strip().lower()
    # remove punctuation and common stopwords
    k = re.sub(r"[^\w\s\-]", " ", k)
    k = re.sub(r"\s+", " ", k).strip()
    return REGION_CANONICAL_MAP.get(k, token)

# instructions for JSON-only output
JSON_RESPONSE_INSTRUCTIONS = textwrap.dedent("""
You must respond ONLY with a single valid JSON object (no surrounding text, no code fences).
The JSON object must have keys:
 - payload: object with keys {table, columns, filters, group_by, agg, order_by, limit}
 - intent: short string (e.g. "trend", "topk", "select", "count", "cohort")
 - score: numeric between 0 and 1 (confidence score)

Rules:
 - Use only the column names provided in the prompt. Do not invent columns.
 - If you cannot produce a valid payload, return payload=null and set intent='fail', score=0.0 and include an "error" string.
""").strip()

# few-shot examples (keeps LLM steer stable)
FEWSHOT = [
    {
        "nl": "Show MRR by region for 2025",
        "out": {
            "payload": {
                "table": "gold.mrr",
                "columns": None,
                "filters": [{"col":"year","op":"=","val":2025}],
                "group_by": ["region"],
                "agg": {"total_revenue_sum": "total_revenue"},
                "order_by": [{"col":"total_revenue_sum","dir":"desc"}],
                "limit": 50
            },
            "intent": "trend",
            "score": 0.85
        }
    },
    {
        "nl": "Top 10 features by unique users",
        "out": {
            "payload": {
                "table": "gold.feature_adoption",
                "columns": None,
                "filters": [],
                "group_by": ["feature_name"],
                "agg": {"feature_users_sum": "feature_users"},
                "order_by": [{"col":"feature_users_sum","dir":"desc"}],
                "limit": 10
            },
            "intent": "topk",
            "score": 0.9
        }
    }
]

def _build_system_prompt(table: Optional[str]) -> str:
    base = [
        "You are a strict JSON-outputting assistant that maps user natural-language analytics requests to a structured payload.",
        JSON_RESPONSE_INSTRUCTIONS
    ]
    if table:
        cols = get_table_columns(table) or []
        base.append(f"Table: {table}")
        base.append("Columns: " + ", ".join(cols))
        agg_candidates = [c for c in cols if any(k in c.lower() for k in ("revenue","amount","mrr","total","avg","count","users"))]
        if agg_candidates:
            base.append("Typical aggregatable columns: " + ", ".join(agg_candidates))
    else:
        base.append("No table specified. Prefer gold.mrr/gold.churn_rate/gold.feature_adoption/gold.ltv/gold.ltv_predicted/gold.user_retention_cohort.")

    base.append("Examples (NL -> JSON):")
    for ex in FEWSHOT:
        base.append(f"NL: {ex['nl']}\nJSON: {json.dumps(ex['out'], ensure_ascii=False)}")
    return "\n\n".join(base)

def _extract_text_from_response(resp) -> str:
    """
    Safely extract the assistant text content from modern OpenAI responses.
    Tries several access patterns and falls back to str(resp).
    """
    try:
        # choices likely present
        choices = getattr(resp, "choices", None) or resp.get("choices") if isinstance(resp, dict) else None
        if not choices:
            return str(resp)
        choice0 = choices[0]
        # try attribute path: choice0.message.content
        msg = getattr(choice0, "message", None)
        if msg is None and isinstance(choice0, dict):
            msg = choice0.get("message")
        if msg is None:
            # maybe older shape: choice0.text or choice0.get("text")
            text_val = getattr(choice0, "text", None) or (choice0.get("text") if isinstance(choice0, dict) else None)
            if text_val:
                return text_val
            return str(choice0)
        # msg could be dict-like or object with .content
        content = None
        if isinstance(msg, dict):
            content = msg.get("content") or msg.get("text") or None
        else:
            content = getattr(msg, "content", None)
        if content is None:
            return str(msg)
        return content
    except Exception as e:
        LOG.exception("Failed to extract text from LLM response: %s", e)
        return str(resp)

def _parse_assistant_json(text: str) -> Dict[str, Any]:
    """
    Extract the first {...} JSON object in text and parse it.
    Raises ValueError if none found or invalid JSON.
    """
    if not text or not isinstance(text, str):
        raise ValueError("empty LLM response")
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM did not return an obvious JSON object")
    snippet = text[start:end+1]
    try:
        obj = json.loads(snippet)
    except Exception as e:
        # try a relaxed cleanup: remove surrounding backticks / triple ticks
        cleaned = re.sub(r"^```(?:json)?\s*", "", snippet)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            obj = json.loads(cleaned)
        except Exception as e2:
            raise ValueError(f"Failed to parse JSON from LLM response: {e}; cleaned attempt: {e2}. Excerpt: {snippet[:400]}")
    return obj

def _resolve_col(table: str, cand: str) -> str:
    """
    Resolve a candidate column to an actual column name in the table schema.
    Raises ValueError if no match.
    """
    cols = get_table_columns(table) or []
    if cand in cols:
        return cand
    # canonical mapping
    cm = TABLE_CANONICAL_MAP.get(table, {})
    mapped = cm.get(cand.lower())
    if mapped and mapped in cols:
        return mapped
    # case-insensitive match
    for c in cols:
        if cand.lower() == c.lower():
            return c
    # substring match
    for c in cols:
        if cand.lower() in c.lower() or c.lower() in cand.lower():
            return c
    raise ValueError(f"unknown column '{cand}' for table {table}")

def _validate_payload_table_schema(payload: Dict[str, Any]) -> None:
    if not payload or "table" not in payload:
        raise ValueError("payload missing table")
    table = payload["table"]
    cols = get_table_columns(table) or []
    if not cols:
        raise ValueError(f"unknown table or empty schema: {table}")

    # columns
    columns = payload.get("columns")
    if columns is not None:
        if not isinstance(columns, list):
            raise ValueError("columns must be array or null")
        for c in columns:
            _resolve_col(table, c)

    # filters
    for f in payload.get("filters", []) or []:
        if "col" not in f or "op" not in f:
            raise ValueError("filter objects must have col and op")
        _resolve_col(table, f["col"])
        if f["op"].lower() == "in" and not isinstance(f.get("val", []), list):
            raise ValueError("IN filter requires list value")

    # group_by
    for g in payload.get("group_by", []) or []:
        _resolve_col(table, g)

    # agg
    agg = payload.get("agg") or {}
    if not isinstance(agg, dict):
        raise ValueError("agg must be an object mapping alias->source_col")
    for alias, src in agg.items():
        _resolve_col(table, src)

    # order_by
    for o in payload.get("order_by", []) or []:
        oc = o.get("col")
        if not oc:
            raise ValueError("order_by entries must contain col")
        # allow order by alias (if present in agg) OR real col
        if oc in agg:
            continue
        _resolve_col(table, oc)

    # limit
    lim = payload.get("limit")
    if lim is not None and not isinstance(lim, int):
        raise ValueError("limit must be integer")

def call_llm_for_payload(prompt: str, api_key: Optional[str] = None,
                         model: Optional[str] = None, temperature: Optional[float] = None) -> str:
    """
    Low-level call: return assistant text. Raises ValueError on failure.
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not provided (OPENAI_API_KEY)")

    model = model or LLM_MODEL
    temperature = TEMPERATURE if temperature is None else temperature

    client = OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JSON_RESPONSE_INSTRUCTIONS},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=800
        )
    except Exception as e:
        raise ValueError(f"LLM call failed: {e}")

    text = _extract_text_from_response(resp)
    return text

def call_llm_to_structured(nl: str, default_table: Optional[str] = None,
                           api_key: Optional[str] = None,
                           model: Optional[str] = None,
                           temperature: Optional[float] = None) -> Dict[str, Any]:
    """
    High-level: returns {"payload":..., "intent":..., "score":...}
    Performs:
     - builds system prompt (with table columns if known)
     - calls LLM
     - extracts and parses JSON
     - validates payload against schema (raises ValueError if invalid)
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY or pass api_key")

    model = model or LLM_MODEL
    temperature = TEMPERATURE if temperature is None else temperature

    # choose table heuristically if not provided
    table = default_table
    if not table:
        t = (nl or "").lower()
        if "churn" in t:
            table = "gold.churn_rate"
        elif "feature" in t or "adoption" in t:
            table = "gold.feature_adoption"
        elif "mrr" in t or "revenue" in t:
            table = "gold.mrr"
        elif "ltv predicted" in t or "predicted" in t:
            table = "gold.ltv_predicted"
        elif "ltv" in t or "lifetime value" in t:
            table = "gold.ltv"
        elif "cohort" in t or "retention" in t:
            table = "gold.user_retention_cohort"
        else:
            table = "gold.mrr"

    system_prompt = _build_system_prompt(table)
    user_prompt = f"User NL: {nl}\n\nReturn the JSON payload exactly as instructed."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    client = OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=800
        )
    except Exception as e:
        raise ValueError(f"LLM call failed: {e}")

    assistant_text = _extract_text_from_response(resp)
    parsed = _parse_assistant_json(assistant_text)

    # Normalize/validate top-level shape
    intent = parsed.get("intent", "llm")
    score = float(parsed.get("score", 0.0)) if parsed.get("score") is not None else 0.0
    payload = parsed.get("payload")

     # --- Normalize region values in filters (best-effort) ---
    for f in payload.get("filters", []) or []:
        try:
            # if the filter column references region, normalize the incoming value
            if isinstance(f.get("col"), str) and f["col"].lower().endswith("region"):
                v = f.get("val")
                if isinstance(v, str):
                    f["val"] = normalize_region_token(v)
                elif isinstance(v, list):
                    f["val"] = [normalize_region_token(x) if isinstance(x, str) else x for x in v]
        except Exception:
            # don't fail LLM parsing just because normalization had trouble — continue
            pass

    if payload is None:
        # LLM signalled failure: surface LLM-provided error if available
        err = parsed.get("error", "LLM returned payload=null")
        raise ValueError(f"LLM returned payload=null; reason: {err}")

    # ensure table present
    if "table" not in payload or not payload["table"]:
        payload["table"] = table

    # Validate payload against table schema (raises ValueError when invalid)
    try:
        _validate_payload_table_schema(payload)
    except ValueError as e:
        raise ValueError(f"LLM returned invalid payload: {e}")

    return {"payload": payload, "intent": intent, "score": score}