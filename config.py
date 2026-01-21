# config.py
# Edit these values directly (or set via env if you later prefer)
    # PAT you created
QUERY_TIMEOUT = 120   # seconds
# Allowed tables (canonical full names)
ALLOWED_TABLES = [
    "gold.churn_rate",
    "gold.feature_adoption",
    "gold.ltv",
    "gold.ltv_predicted",
    "gold.mrr",
    "gold.user_retention_cohort",
]

# Row/size limits to protect the warehouse
MAX_ROWS_RETURN = 2000
MAX_LIMIT_PER_QUERY = 5000
DEFAULT_LIMIT = 50

# Schema cache TTL (seconds)
SCHEMA_CACHE_TTL = 300