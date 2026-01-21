# main.py
from flask import Flask, request, jsonify
from db.databricks_client import run_query
from schema_cache import get_table_columns
from sql_validator import is_safe_sql
from query_templates import TEMPLATES
import config
import re, html
from nl_parser import parse_nl_to_structured
import json
from llm_adapter import call_llm_to_structured
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_\-]*$")  # allowed identifier tokens (no spaces/dots here)
FULL_TABLE_RE = re.compile(r"^[A-Za-z0-9_]+\.[A-Za-z0-9_]+$")  # e.g. gold.churn_rate

def sanitize_identifier(name: str):
    # allow column names like col or col_name; if dot present reject (we use table-qualified names only where needed)
    if not name:
        raise ValueError("empty identifier")
    if not re.match(r"^[A-Za-z0-9_]+$", name):
        raise ValueError(f"invalid identifier: {name}")
    return name

def sanitize_value(val):
    # If number, return as-is; else escape single quotes and wrap in quotes
    if isinstance(val, (int, float)):
        return str(val)
    if val is None:
        return "NULL"
    s = str(val)
    s = s.replace("'", "''")
    return f"'{s}'"

def build_structured_query(payload: dict) -> str:
    """
    payload keys:
      - table: full table fqn (must be in ALLOWED_TABLES)
      - columns: list of columns or None for *
      - filters: list of {col, op, val} where op in (=, !=, <, >, <=, >=, like, in)
      - group_by: list of cols
      - agg: dict like {"sum_col":"amount", "agg":"sum"}  (optional)  # here we expect {"alias": "source_col"}
      - order_by: list of {"col":"x","dir":"asc"|"desc"}
      - limit: int
    """
    table = payload.get("table")
    if not table or table not in config.ALLOWED_TABLES:
        raise ValueError("table not allowed or missing")
    # Get columns from schema cache
    cols = get_table_columns(table)
    if not cols:
        raise ValueError("could not fetch table schema for " + table)

    # columns selection
    sel_cols = payload.get("columns")
    if not sel_cols:
        select_clause = "*"
    else:
        safe_cols = []
        for c in sel_cols:
            # allow user to pass expressions (e.g. COUNT(*) AS cnt) only if they look safe
            if isinstance(c, str) and c.upper().startswith("COUNT(") or " AS " in c.upper():
                # simple permissive fallback for expressions provided by parser/templates
                safe_cols.append(c)
                continue
            if c not in cols:
                raise ValueError(f"column {c} not in table {table}")
            safe_cols.append(c)
        select_clause = ", ".join(safe_cols)

    # filters
    filters = payload.get("filters", [])
    where_fragments = []
    for f in filters:
        col = f.get("col")
        op = f.get("op", "=").lower()
        val = f.get("val")
        if col not in cols:
            raise ValueError(f"filter column {col} not in table")
        if op not in ["=", "==", "!=", "<>", "<", ">", "<=", ">=", "like", "in"]:
            raise ValueError("unsupported operator " + str(op))
        if op == "in":
            if not isinstance(val, (list, tuple)):
                raise ValueError("IN operator requires list value")
            safe_vals = ", ".join(sanitize_value(v) for v in val)
            where_fragments.append(f"{col} IN ({safe_vals})")
        else:
            safe_v = sanitize_value(val)
            if op in ["=", "=="]:
                where_fragments.append(f"{col} = {safe_v}")
            elif op in ["!=", "<>"]:
                where_fragments.append(f"{col} <> {safe_v}")
            else:
                where_fragments.append(f"{col} {op} {safe_v}")

    where_clause = ""
    if where_fragments:
        where_clause = "WHERE " + " AND ".join(where_fragments)

    # group by & agg
    group_by = payload.get("group_by", [])
    agg_clause = ""
    agg_aliases = set()
    if group_by:
        for g in group_by:
            if g not in cols:
                raise ValueError(f"group_by column {g} not in table")
        group_clause = ", ".join(group_by)
        # check if user provided aggregations
        agg_defs = payload.get("agg") or {}  # e.g. {"sum_amount":"amount"}
        agg_parts = []
        # if user didn't provide agg, default: COUNT(*) AS cnt
        if not agg_defs:
            agg_parts = ["COUNT(*) AS cnt"]
            agg_aliases.add("cnt")
        else:
            for alias, source_col in agg_defs.items():
                if source_col not in cols:
                    raise ValueError(f"agg source {source_col} not in table")
                # basic: SUM as default aggregation for numeric metrics (parser provides alias naming)
                safe_alias = sanitize_identifier(alias)
                agg_parts.append(f"SUM({source_col}) AS {safe_alias}")
                agg_aliases.add(safe_alias)
        select_clause = ", ".join(list(group_by) + agg_parts)
        agg_clause = f"GROUP BY {group_clause}"

    # order_by
    order_by = payload.get("order_by", [])
    order_frag = ""
    if order_by:
        order_parts = []
        for o in order_by:
            c = o.get("col")
            d = o.get("dir", "desc").lower()
            # allow ordering by actual table columns OR by aggregation aliases (including default "cnt")
            if c not in cols and c not in agg_aliases and c not in ("cnt",):
                raise ValueError(f"order_by column {c} not in table or aggregation")
            if d not in ("asc", "desc"):
                raise ValueError("order dir must be asc or desc")
            order_parts.append(f"{c} {d.upper()}")
        order_frag = "ORDER BY " + ", ".join(order_parts)

    # limit
    limit = payload.get("limit") or config.MAX_ROWS_RETURN
    limit = int(limit)
    if limit <= 0 or limit > config.MAX_LIMIT_PER_QUERY:
        raise ValueError("limit out of bounds")

    # final SQL assembly
    sql_parts = [
        f"SELECT {select_clause}",
        f"FROM {table}",
        where_clause,
        agg_clause,
        order_frag,
        f"LIMIT {limit}"
    ]
    sql = "\n".join(p for p in sql_parts if p and p.strip())
    return sql

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/query_structured", methods=["POST"])
def query_structured():
    payload = request.get_json(force=True)
    try:
        sql = build_structured_query(payload)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    # additional global safety validation
    ok, msg = is_safe_sql(sql)
    if not ok:
        return jsonify({"error": msg}), 403
    try:
        res = run_query(sql)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# existing endpoints: template + free query
@app.route("/template/<name>", methods=["POST"])
def template_run(name):
    tpl = TEMPLATES.get(name)
    if not tpl:
        return jsonify({"error":"unknown template"}), 404
    params = request.get_json(force=False) or {}
    try:
        sql = tpl.format(**params)
    except Exception as e:
        return jsonify({"error": "template format error: " + str(e)}), 400
    ok, msg = is_safe_sql(sql)
    if not ok:
        return jsonify({"error": msg}), 403
    try:
        res = run_query(sql)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/query", methods=["POST"])
def free_query():
    body = request.get_json(force=True)
    sql = body.get("sql")
    if not sql:
        return jsonify({"error":"sql required"}), 400
    ok, msg = is_safe_sql(sql)
    if not ok:
        return jsonify({"error": msg}), 403
    # if no limit present, append safe cap
    if "limit" not in sql.lower():
        sql = sql.strip() + f" LIMIT {config.MAX_ROWS_RETURN}"
    try:
        res = run_query(sql)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# -------------------------
# /nlq endpoint (place this BEFORE the "if __name__ == '__main__':" block)
# -------------------------
@app.route("/nlq", methods=["POST"])
def nlq():
    try:
        body = request.get_json(force=True)
        nl = body.get("nl")
        default_table = body.get("default_table")
        use_llm = bool(body.get("use_llm", False))

        if not nl:
            return jsonify({"error":"Missing field: nl"}), 400

        # parsed should be a dict with keys: payload, intent, score
        parsed = None

        if use_llm:
            try:
                # ensure OPENAI_API_KEY present in env OR pass explicit key here
                api_key = os.environ.get("OPENAI_API_KEY")
                parsed = call_llm_to_structured(nl, default_table=default_table, api_key=api_key)
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
        else:
            parsed = parse_nl_to_structured(nl, default_table=default_table)

        # ensure parsed is present and has payload
        if not parsed or not isinstance(parsed, dict):
            return jsonify({"error": "Parsing failed"}), 500

        payload = parsed.get("payload", {})
        intent = parsed.get("intent")
        score = parsed.get("score")

        if not payload.get("table"):
            return jsonify({"error":"Parser failed to identify a table"}), 400

        # build SQL
        try:
            sql = build_structured_query(payload)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        # validate SQL
        ok, msg = is_safe_sql(sql)
        if not ok:
            return jsonify({"error":"Unsafe SQL blocked", "reason": msg}), 403

        # run
        result = run_query(sql)

        # audit
        with open("mcp_audit.log", "a") as fh:
            fh.write(json.dumps({
                "nl": nl,
                "parsed": parsed,
                "sql": sql
            }) + "\n")

        return jsonify({
            "nl": nl,
            "intent": intent,
            "score": score,
            "payload": payload,
            "sql": sql,
            "result": result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

# near bottom of main.py, before app.run(...)
if __name__ == "__main__":
    # debug: print all registered routes
    print("Registered routes:")
    for r in sorted([rule.rule for rule in app.url_map.iter_rules()]):
        print(" ", r)
    app.run(host="0.0.0.0", port=8000)
