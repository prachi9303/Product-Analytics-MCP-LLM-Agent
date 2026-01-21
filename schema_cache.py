# schema_cache.py
import time
from typing import List, Optional
from config import SCHEMA_CACHE_TTL, ALLOWED_TABLES
from db.databricks_client import fetch_table_schema

# In-memory cache: { table_name: (timestamp_seconds, [column_names]) }
_CACHE = {}

def get_table_columns(table_fqn: str) -> Optional[List[str]]:
    """
    Return list of column names for `table_fqn`.
    Caches results for SCHEMA_CACHE_TTL seconds.
    Returns None if table is not allowed or schema fetch fails.
    """
    now = time.time()

    # Only handle allowed tables
    if table_fqn not in ALLOWED_TABLES:
        return None

    # If cached and fresh, return
    if table_fqn in _CACHE:
        ts, cols = _CACHE[table_fqn]
        if (now - ts) < SCHEMA_CACHE_TTL and cols is not None:
            return cols

    # Not cached or stale -> fetch fresh
    try:
        cols = fetch_table_schema(table_fqn)
        # normalize column names to the actual strings returned
        _CACHE[table_fqn] = (now, cols)
        return cols
    except Exception as e:
        # On failure, keep previous value if present (even if stale) to provide best-effort
        prev = _CACHE.get(table_fqn)
        if prev and prev[1] is not None:
            return prev[1]
        # otherwise return None so caller can handle error gracefully
        return None

def set_table_schema(table_fqn: str, columns: List[str]) -> None:
    """Manual cache setter (useful for tests)."""
    _CACHE[table_fqn] = (time.time(), columns)

def clear_cache() -> None:
    """Clear the entire schema cache (useful for debugging)."""
    _CACHE.clear()