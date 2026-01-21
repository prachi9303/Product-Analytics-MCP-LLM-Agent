# db/databricks_client.py
from databricks.sql import connect
from config import DATABRICKS_HOST, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN, QUERY_TIMEOUT

def run_query(sql, params=None, timeout=QUERY_TIMEOUT):
    conn = connect(server_hostname=DATABRICKS_HOST, http_path=DATABRICKS_HTTP_PATH, access_token=DATABRICKS_TOKEN, timeout=timeout)
    cur = conn.cursor()
    try:
        if params:
            cur.execute(sql, params)
        else:
            cur.execute(sql)
        cols = [c[0] for c in cur.description] if cur.description else []
        rows = cur.fetchall() if cur.description else []
        return {"columns": cols, "rows": rows}
    finally:
        try: cur.close()
        except: pass
        conn.close()

def fetch_table_schema(table_fqn, sample_limit=1):
    """
    Return list of column names for table_fqn (e.g. 'gold.churn_rate').
    We run SELECT * LIMIT 1 to get schema. This is fast for small result.
    """
    sql = f"SELECT * FROM {table_fqn} LIMIT {sample_limit}"
    res = run_query(sql)
    return res["columns"]
