# sql_validator.py
import re
from config import ALLOWED_TABLES, MAX_ROWS_RETURN

# Block dangerous keywords
BAD_KEYWORDS = [
    r"\b(insert|update|delete|drop|create|alter|truncate|merge|grant|revoke|replace|shutdown)\b",
    r";",                         # disallow multiple statements
    r"\binto\s+|outfile\b",       # disallow write/export style
]

SELECT_ONLY = re.compile(r"^\s*select\s+", re.IGNORECASE)

def is_safe_sql(sql: str) -> (bool, str):
    if not sql or not sql.strip():
        return False, "empty query"
    if not SELECT_ONLY.match(sql):
        return False, "only SELECT queries are allowed"
    for pat in BAD_KEYWORDS:
        if re.search(pat, sql, re.IGNORECASE):
            return False, "disallowed pattern found in SQL"
    # naive allowlist: if query references FROM <table>, ensure table is allowed
    lower = sql.lower()
    tables = re.findall(r"from\s+([a-z0-9_\.]+)", lower)
    for t in tables:
        if t not in [x.lower() for x in ALLOWED_TABLES]:
            return False, f"table '{t}' is not permitted"
    return True, "ok"