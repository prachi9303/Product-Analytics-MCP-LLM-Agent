# nl_parser.py
# Rule-based NL -> structured payload mapper tuned to the six gold tables and your schema.
# Relies on schema_cache.get_table_columns to validate column existence.

import re
from typing import Dict, Any, List, Tuple
from schema_cache import get_table_columns
import config
import dateparser

# Table-specific canonical mapping derived from your schema output
# Keys: table FQN -> mapping of canonical alias -> actual column
TABLE_CANONICAL_MAP = {
    "gold.churn_rate": {
        "users": "churned_users",
        "churned_users": "churned_users",
        "active_users": "active_users",
        "region": "region",
        "year": "year",
        "month": "month",
        "churn_rate": "churn_rate"
    },
    "gold.feature_adoption": {
        "feature": "feature_name",
        "users": "feature_users",
        "total_active_users": "total_active_users",
        "adoption_rate": "adoption_rate",
        "region": "region",
        "year": "year",
        "month": "month"
    },
    "gold.ltv": {
        "region": "region",
        "plan": "subscription_plan",
        "subscription_plan": "subscription_plan",
        "avg_ltv": "avg_ltv",
        "users": "user_count"
    },
    "gold.ltv_predicted": {
        "year": "year",
        "month": "month",
        "region": "region",
        "plan": "subscription_plan",
        "amount": "total_revenue",
        "mrr": "mrr",
        "predicted_ltv": "predicted_ltv",
        "avg_revenue_per_user": "avg_revenue_per_user",
        "churned_users": "churned_users"
    },
    "gold.mrr": {
        "year": "year",
        "month": "month",
        "region": "region",
        "plan": "subscription_plan",
        "amount": "total_revenue",
        "mrr": "mrr",
        "avg_revenue_per_user": "avg_revenue_per_user"
    },
    "gold.user_retention_cohort": {
        "region": "region",
        "signup_month": "signup_month",
        "activity_month": "activity_month",
        "active_users": "active_users",
        "cohort_size": "cohort_size",
        "retention_rate": "retention_rate",
        "months_since_signup": "months_since_signup",
        "month": "signup_month"
    }
}
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

# Light intent detector
def detect_intent(text: str) -> Tuple[str, float]:
    t = (text or "").lower()
    if any(k in t for k in ["trend", "over time", "monthly", "weekly", "per month", "by month"]):
        return "trend", 0.85
    if any(k in t for k in ["top", "most", "highest", "rank"]):
        return "topk", 0.9
    if any(k in t for k in ["count", "how many", "number of", "total"]):
        return "count", 0.9
    if any(k in t for k in ["average", "avg", "mean"]):
        return "aggregate", 0.85
    if any(k in t for k in ["retention", "cohort", "cohort retention"]):
        return "cohort", 0.9
    # fallback
    return "select", 0.6

# heuristics to pick table from NL text
def pick_table_from_nl(text: str, default: str = "gold.feature_adoption") -> str:
    t = (text or "").lower()
    if "churn" in t or "churn rate" in t:
        return "gold.churn_rate"
    if "feature" in t or "adoption" in t:
        return "gold.feature_adoption"
    if "mrr" in t or "revenue" in t:
        return "gold.mrr"
    if "ltv predicted" in t or "predicted ltv" in t or "predicted" in t:
        return "gold.ltv_predicted"
    if "ltv" in t or "lifetime value" in t:
        return "gold.ltv"
    if "cohort" in t or "retention" in t:
        return "gold.user_retention_cohort"
    # fallback to default
    return default

# map alias -> real column using canonical map, else fallback to schema match (substring)
def map_alias_to_column(table: str, alias: str) -> str | None:
    if not alias or not table:
        return None
    cm = TABLE_CANONICAL_MAP.get(table, {})
    a = alias.lower().strip()
    # direct canonical lookup
    if a in cm:
        return cm[a]
    # try direct schema columns
    cols = get_table_columns(table) or []
    for c in cols:
        if c.lower() == a:
            return c
    # substring match
    for c in cols:
        if a in c.lower() or c.lower() in a:
            return c
    return None

# ----- Limit extraction (avoid getting year as limit) -----
def extract_ints_for_limit(text: str) -> List[int]:
    """
    Return integers that are likely to be limits (e.g., preceded by 'top', 'limit', 'first', 'last').
    Excludes year-like numbers unless explicitly asked as top/limit.
    """
    if not text:
        return []
    # common patterns for limits
    tokens = re.findall(r"\b(?:top|limit|first|last)\b\s+(\d+)\b", text, flags=re.I)
    if tokens:
        return [int(x) for x in tokens]
    # fallback: gather numbers but exclude year-like ones (1900-2100)
    nums = [int(x) for x in re.findall(r"\b(\d+)\b", text)]
    return [n for n in nums if n < 1900 or n > 2100]

# ----- Date / year extraction & range detection -----
def extract_date_range(text: str) -> Tuple[Any, Any, Any]:
    """
    Detects explicit years or simple date-range phrases.
    Returns (start_date_iso, end_date_iso, year_token) where any item may be None.
    """
    if not text:
        return (None, None, None)

    # explicit year
    year_match = re.search(r"\b(20\d{2})\b", text)
    if year_match:
        year_token = year_match.group(1)
        dt_start = dateparser.parse(f"first day of january {year_token}")
        dt_end = dateparser.parse(f"last day of december {year_token}")
        if dt_start and dt_end:
            return (dt_start.date().isoformat(), dt_end.date().isoformat(), year_token)

    # last N days
    m = re.search(r"last\s+(\d+)\s+day", text, re.I)
    if m:
        days = int(m.group(1))
        dt_end = dateparser.parse("today")
        dt_start = dateparser.parse(f"{days} days ago")
        if dt_start and dt_end:
            return (dt_start.date().isoformat(), dt_end.date().isoformat(), None)

    # month name + year e.g. "April 2025"
    m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})", text, re.I)
    if m:
        month_str = m.group(1) + " " + m.group(2)
        dt_start = dateparser.parse(f"first day of {month_str}")
        dt_end = dateparser.parse(f"last day of {month_str}")
        if dt_start and dt_end:
            return (dt_start.date().isoformat(), dt_end.date().isoformat(), None)

    # explicit from ... to ...
    m = re.search(r"from\s+([A-Za-z0-9\-\s,]+?)\s+to\s+([A-Za-z0-9\-\s,]+)", text, re.I)
    if m:
        a = dateparser.parse(m.group(1))
        b = dateparser.parse(m.group(2))
        if a and b:
            return (a.date().isoformat(), b.date().isoformat(), None)

    return (None, None, None)

# ----- Filters extraction (region/plan/year) -----
def extract_simple_filters_improved(text: str, table: str) -> List[Dict[str, Any]]:
    """
    Improved filters extractor:
      - Detect year/date tokens first (so they don't pollute region capture)
      - Then extract region using a tighter regex (stop at commas or prepositions)
      - Extract plan tokens
    """
    filters: List[Dict[str, Any]] = []
    if not text:
        return filters

    original_text = text
    t = text

    # 1) detect year / date range
    start_date, end_date, year_token = extract_date_range(t)
    if year_token:
        if map_alias_to_column(table, "year"):
            filters.append({"col": map_alias_to_column(table, "year"), "op": "=", "val": int(year_token)})
        # remove the year token from the working text to avoid contaminating region capture
        t = re.sub(r"\b" + re.escape(year_token) + r"\b", " ", t)

    # 2) map explicit date ranges to subscription_ts if present
    if start_date and end_date and map_alias_to_column(table, "subscription_ts"):
        filters.append({"col": map_alias_to_column(table, "subscription_ts"), "op": ">=", "val": start_date})
        filters.append({"col": map_alias_to_column(table, "subscription_ts"), "op": "<=", "val": end_date})
        # remove date phrases to avoid interference
        t = re.sub(r"\b(last|past)\s+\d+\s+day[s]?\b", " ", t, flags=re.I)
        t = re.sub(r"\b(in\s+[A-Za-z]+\s+\d{4})\b", " ", t, flags=re.I)

    # 3) region extraction: tighter - "in <region>" but stop at commas, prepositions or year
    m = re.search(r"\b(?:in|for|from)\s+([A-Za-z0-9\-\s]+?)(?:,|$|\b(?:in|for|from|on|at)\b)", t, re.I)
    if m:
        candidate = candidate.strip()
        candidate = re.sub(r"\s+(in|for|from|on|at)\s*$", "", candidate, flags=re.I)
        candidate = re.sub(r"\b(20\d{2})\b", "", candidate).strip()
        reg_col = map_alias_to_column(table, "region")
        if reg_col and candidate:
            normalized = normalize_region_token(candidate)
            filters.append({"col": reg_col, "op": "=", "val": normalized})
            
    # 4) fallback: look for well-known region tokens
    if not any(f.get('col','').lower().endswith("region") for f in filters):
        for token in ["uk", "ireland", "scotland", "glasgow", "dublin", "berlin", "bengaluru"]:
            if re.search(r"\b" + re.escape(token) + r"\b", original_text, re.I):
                reg_col = map_alias_to_column(table, "region")
                if reg_col:
                    filters.append({"col": reg_col, "op": "=", "val": token})
                    break

    # 5) subscription_plan / plan mentions
    m3 = re.search(r"\b(pro|enterprise|trial|free|basic|premium|team)\b", original_text, re.I)
    if m3:
        plan = m3.group(1).lower()
        plan_col = map_alias_to_column(table, "plan") or map_alias_to_column(table, "subscription_plan")
        if plan_col:
            filters.append({"col": plan_col, "op": "=", "val": plan})

    return filters

# Backwards-compatible small wrapper (keeps function name used elsewhere)
def extract_simple_filters(text: str, table: str) -> List[Dict[str, Any]]:
    return extract_simple_filters_improved(text, table)

# -------------------------
# Helper: extract "by <dim>" phrases
# -------------------------
def extract_by_dimensions(text: str) -> List[str]:
    if not text:
        return []
    matches = re.findall(r"\bby\s+([A-Za-z0-9_\-]+)\b", text, flags=re.I)
    return [m.lower() for m in matches]

# -------------------------
# Main parser: build structured payload
# -------------------------
def parse_nl_to_structured(nl: str, default_table: str = None) -> Dict[str, Any]:
    """
    Returns {"payload": {...}, "intent": str, "score": float}
    payload uses structure expected by /query_structured:
    { table, columns, filters, group_by, agg, order_by, limit }
    """
    nl = (nl or "").strip()
    intent, score = detect_intent(nl)
    table = pick_table_from_nl(nl, default_table or "gold.feature_adoption")

    # fetch table columns to confirm mappings
    cols = get_table_columns(table) or []

    payload: Dict[str, Any] = {"table": table}

    # ---- limit heuristics: prefer explicit top/limit mentions; fallback to DEFAULT_LIMIT ----
    limit_candidates = extract_ints_for_limit(nl)
    default_limit = getattr(config, "DEFAULT_LIMIT", 50)
    max_limit_per_query = getattr(config, "MAX_LIMIT_PER_QUERY", 5000)
    max_rows_return = getattr(config, "MAX_ROWS_RETURN", 1000)

    limit = limit_candidates[0] if limit_candidates else default_limit
    if limit <= 0:
        limit = default_limit
    if limit > max_limit_per_query:
        limit = max_rows_return
    # will assign payload["limit"] later when intent-specific logic sets it

    # ----- Early detection: year token and 'by <dimension>' -----
    start_date, end_date, year_token = extract_date_range(nl)
    year_filter = None
    if year_token and map_alias_to_column(table, "year"):
        year_filter = {"col": map_alias_to_column(table, "year"), "op": "=", "val": int(year_token)}

    by_dims = extract_by_dimensions(nl)

    # If user explicitly said "by <dim>", create a sensible group_by+agg for common metric tables
    if by_dims:
        dim_alias = by_dims[0]
        dim_col = map_alias_to_column(table, dim_alias) or map_alias_to_column(table, dim_alias + "s")
        if dim_col:
            if table in ("gold.mrr", "gold.ltv_predicted", "gold.ltv"):
                metric = map_alias_to_column(table, "amount") or map_alias_to_column(table, "total_revenue") or "total_revenue"
                payload["group_by"] = [dim_col]
                payload["agg"] = {f"{metric}_sum": metric}
                payload["order_by"] = [{"col": f"{metric}_sum", "dir": "desc"}]
                payload["limit"] = limit
            elif table == "gold.churn_rate":
                metric = map_alias_to_column(table, "churned_users") or "churned_users"
                payload["group_by"] = [dim_col]
                payload["agg"] = {f"{metric}_sum": metric}
                payload["order_by"] = [{"col": f"{metric}_sum", "dir": "desc"}]
                payload["limit"] = limit
            else:
                # generic fallback: group and count (attempt to find user count column)
                user_col = map_alias_to_column(table, "users") or map_alias_to_column(table, "user_count") or None
                if user_col:
                    payload["group_by"] = [dim_col]
                    payload["agg"] = {f"{user_col}_sum": user_col}
                    payload["order_by"] = [{"col": f"{user_col}_sum", "dir": "desc"}]
                else:
                    payload["group_by"] = [dim_col]
                    payload["agg"] = {"cnt": "1"}
                    payload["order_by"] = [{"col": "cnt", "dir": "desc"}]
                payload["limit"] = limit

    # Intent-specific building (keeps your existing logic; only augmenting behavior above)
    if intent == "topk":
        if table == "gold.feature_adoption":
            feat = map_alias_to_column(table, "feature") or "feature_name"
            metric = map_alias_to_column(table, "users") or "feature_users"
            payload.setdefault("group_by", [feat])
            payload["agg"] = {f"{metric}_sum": metric}
            payload["order_by"] = [{"col": f"{metric}_sum", "dir": "desc"}]
            payload.setdefault("limit", limit)
        elif table in ("gold.mrr", "gold.ltv_predicted"):
            group = map_alias_to_column(table, "region") or "region"
            payload.setdefault("group_by", [group])
            metric = map_alias_to_column(table, "amount") or "total_revenue"
            payload["agg"] = {f"{metric}_sum": metric}
            payload["order_by"] = [{"col": f"{metric}_sum", "dir": "desc"}]
            payload.setdefault("limit", limit)
        else:
            payload.setdefault("columns", cols[:6] if cols else ["*"])
            payload.setdefault("limit", limit)

    elif intent == "trend":
        gb = []
        if map_alias_to_column(table, "year"):
            gb.append(map_alias_to_column(table, "year"))
        if map_alias_to_column(table, "month"):
            gb.append(map_alias_to_column(table, "month"))
        if gb:
            if table == "gold.churn_rate":
                metric = map_alias_to_column(table, "churned_users") or "churned_users"
                payload.setdefault("group_by", gb)
                payload["agg"] = {f"{metric}_sum": metric}
                payload["order_by"] = [{"col": gb[0], "dir": "desc"}, {"col": gb[1] if len(gb)>1 else gb[0], "dir": "desc"}]
                payload.setdefault("limit", limit)
            elif table in ("gold.mrr", "gold.ltv_predicted"):
                metric = map_alias_to_column(table, "mrr") or map_alias_to_column(table, "amount") or "total_revenue"
                payload.setdefault("group_by", gb)
                payload["agg"] = {f"{metric}_sum": metric}
                payload["order_by"] = [{"col": gb[0], "dir": "desc"}, {"col": gb[1] if len(gb)>1 else gb[0], "dir": "desc"}]
                payload.setdefault("limit", limit)
            else:
                payload.setdefault("group_by", gb)
                payload["agg"] = {"count_users":"active_users"} if map_alias_to_column(table, "active_users") else {}
                payload.setdefault("limit", limit)
        else:
            payload.setdefault("columns", cols[:6] if cols else ["*"])
            payload.setdefault("limit", limit)

    elif intent == "count":
        if table == "gold.feature_adoption":
            metric_col = map_alias_to_column(table, "users") or "feature_users"
            payload["agg"] = {"user_count": metric_col}
            payload.setdefault("limit", 1)
        else:
            payload.setdefault("columns", ["COUNT(*) AS cnt"])
            payload.setdefault("limit", 1)

    elif intent == "cohort":
        if table != "gold.user_retention_cohort":
            table = "gold.user_retention_cohort"
            payload["table"] = table
            cols = get_table_columns(table) or []
        payload["group_by"] = [map_alias_to_column(table, "signup_month") or "signup_month",
                               map_alias_to_column(table, "activity_month") or "activity_month"]
        payload["agg"] = {"active_users_sum": map_alias_to_column(table, "active_users") or "active_users"}
        payload["order_by"] = [{"col": "signup_month", "dir": "desc"}, {"col": "activity_month", "dir": "desc"}]
        payload["limit"] = limit

    else:
        preferred = []
        for alias in ["feature","users","amount","region","year","month","active_users"]:
            cand = map_alias_to_column(table, alias)
            if cand and cand not in preferred:
                preferred.append(cand)
        if preferred:
            payload["columns"] = preferred[:6]
        else:
            payload["columns"] = cols[:6] if cols else ["*"]
        payload["limit"] = limit

    # Extract simple filters (region, year, plan) and append year_filter if we created it above
    filters = extract_simple_filters(nl, payload["table"]) or []
    if year_filter:
        # avoid duplicate year filters
        if not any(f.get("col") == year_filter["col"] for f in filters):
            filters.append(year_filter)

    if filters:
        payload["filters"] = filters

    return {"payload": payload, "intent": intent, "score": 0.75}