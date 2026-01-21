# models.py
# optional – simple container for responses, not required for the MVP
from dataclasses import dataclass
from typing import List, Any

@dataclass
class QueryResult:
    columns: List[str]
    rows: List[Any]