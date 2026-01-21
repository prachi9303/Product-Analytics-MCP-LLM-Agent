# query_templates.py
# Templated SQL fragments targeted to the six gold.* metric tables.
# These templates produce SQL that the MCP's existing validator will check.

TEMPLATES = {
    # Generic utility templates (table must be replaced)
    "sample": "SELECT * FROM {table} LIMIT {limit}",

    # churn_rate: monthly churn by region / plan
    "churn_by_month": """
        SELECT year, month, region,
               SUM(churned_users) AS churned_users,
               SUM(active_users) AS active_users,
               AVG(churn_rate) AS avg_churn_rate
        FROM {table}
        {where_clause}
        GROUP BY year, month, region
        ORDER BY year DESC, month DESC
        LIMIT {limit}
    """,

    # feature_adoption: top features by users
    "top_features_by_users": """
        SELECT feature_name,
               SUM(feature_users) AS total_feature_users,
               SUM(total_active_users) AS total_active_users,
               AVG(adoption_rate) AS avg_adoption_rate,
               year, month, region
        FROM {table}
        {where_clause}
        GROUP BY feature_name, year, month, region
        ORDER BY total_feature_users DESC
        LIMIT {limit}
    """,

    # mrr: MRR by region / plan
    "mrr_by_region": """
        SELECT year, month, region, subscription_plan,
               SUM(total_revenue) AS total_revenue,
               AVG(avg_revenue_per_user) AS avg_revenue_per_user,
               SUM(mrr) AS mrr
        FROM {table}
        {where_clause}
        GROUP BY year, month, region, subscription_plan
        ORDER BY year DESC, month DESC, mrr DESC
        LIMIT {limit}
    """,

    # ltv: LTV by region and plan
    "ltv_by_region_plan": """
        SELECT region, subscription_plan, ltv_type,
               AVG(avg_ltv) AS avg_ltv,
               SUM(user_count) AS user_count
        FROM {table}
        {where_clause}
        GROUP BY region, subscription_plan, ltv_type
        ORDER BY avg_ltv DESC
        LIMIT {limit}
    """,

    # ltv_predicted: forecasted metrics by month/region/plan
    "ltv_predicted_by_month": """
        SELECT year, month, region, subscription_plan,
               SUM(total_revenue) AS total_revenue,
               AVG(avg_revenue_per_user) AS avg_revenue_per_user,
               SUM(mrr) AS mrr,
               AVG(predicted_ltv) AS predicted_ltv
        FROM {table}
        {where_clause}
        GROUP BY year, month, region, subscription_plan
        ORDER BY year DESC, month DESC
        LIMIT {limit}
    """,

    # user_retention_cohort: cohort retention table queries
    "cohort_retention_summary": """
        SELECT region, signup_month, activity_month,
               SUM(active_users) AS active_users,
               SUM(cohort_size) AS cohort_size,
               AVG(retention_rate) AS retention_rate,
               AVG(months_since_signup) AS avg_months_since_signup
        FROM {table}
        {where_clause}
        GROUP BY region, signup_month, activity_month
        ORDER BY signup_month DESC, activity_month DESC
        LIMIT {limit}
    """
}

# short alias to template mapping for easier use in MCP:
TEMPLATE_BY_TABLE = {
    "gold.churn_rate": "churn_by_month",
    "gold.feature_adoption": "top_features_by_users",
    "gold.mrr": "mrr_by_region",
    "gold.ltv": "ltv_by_region_plan",
    "gold.ltv_predicted": "ltv_predicted_by_month",
    "gold.user_retention_cohort": "cohort_retention_summary",
}