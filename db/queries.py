"""
db/queries.py — SQL helpers for metadata filtering and analytics.
All functions return plain dicts/lists so they're easy to inject into prompts.
"""

from __future__ import annotations
import os
from typing import Optional
import psycopg2
import psycopg2.extras
from contextlib import contextmanager


# ------------------------------------------------------------------ connection
@contextmanager
def get_conn():
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    try:
        yield conn
    finally:
        conn.close()


def _fetchall(sql: str, params: tuple = ()) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]


def _fetchone(sql: str, params: tuple = ()) -> Optional[dict]:
    rows = _fetchall(sql, params)
    return rows[0] if rows else None


# ------------------------------------------------------------------ documents

def get_active_categories() -> list[str]:
    """All distinct document categories currently in the DB."""
    rows = _fetchall(
        "SELECT DISTINCT category FROM documents WHERE is_active = TRUE ORDER BY category"
    )
    return [r["category"] for r in rows]


def get_document_metadata(doc_id: int) -> Optional[dict]:
    """Full metadata for a single document — used to build citation context."""
    return _fetchone(
        """
        SELECT d.id, d.title, d.filename, d.category, d.last_updated,
               d.owner_email, e.department, e.name AS owner_name
        FROM documents d
        LEFT JOIN employees e ON d.owner_email = e.email
        WHERE d.id = %s
        """,
        (doc_id,),
    )


def get_stale_documents(days: int = 180) -> list[dict]:
    """
    Documents that haven't been updated in `days` days.
    Used by the chatbot to warn users when citing old content.

    SQL skill practiced:
      - INTERVAL arithmetic
      - JOIN with employees
      - Filtering on is_active
    """
    return _fetchall(
        """
        SELECT d.id, d.title, d.category,
               d.last_updated,
               EXTRACT(DAY FROM NOW() - d.last_updated)::INT AS days_old,
               d.owner_email,
               e.department
        FROM documents d
        LEFT JOIN employees e ON d.owner_email = e.email
        WHERE d.last_updated < NOW() - INTERVAL '%s days'
          AND d.is_active = TRUE
        ORDER BY d.last_updated ASC
        """,
        (days,),
    )


def get_documents_by_category(category: str) -> list[dict]:
    """All active docs in a category — used to build metadata filter sets."""
    return _fetchall(
        """
        SELECT id, title, filename, last_updated, owner_email
        FROM documents
        WHERE category = %s AND is_active = TRUE
        ORDER BY last_updated DESC
        """,
        (category,),
    )


# ------------------------------------------------------------------ query log

def log_query(
    *,
    session_id: str,
    user_email: str,
    question: str,
    answer: str,
    source_doc_ids: list[int],
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
    was_refused: bool = False,
    refusal_reason: Optional[str] = None,
) -> None:
    """Insert a completed query into the audit log."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO query_log
                    (session_id, user_email, question, answer, source_doc_ids,
                     input_tokens, output_tokens, latency_ms,
                     was_refused, refusal_reason)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    session_id, user_email, question, answer,
                    source_doc_ids, input_tokens, output_tokens,
                    latency_ms, was_refused, refusal_reason,
                ),
            )
        conn.commit()


def get_cost_summary(days: int = 30) -> dict:
    """
    Aggregate token usage and estimated USD cost for the last N days.

    SQL skill practiced:
      - Conditional aggregation
      - Cost arithmetic inside SQL
      - Date filtering with INTERVAL
    """
    return _fetchone(
        """
        SELECT
            COUNT(*)                                   AS total_queries,
            SUM(input_tokens)                          AS total_input_tokens,
            SUM(output_tokens)                         AS total_output_tokens,
            ROUND(AVG(latency_ms))                     AS avg_latency_ms,
            COUNT(*) FILTER (WHERE was_refused)        AS refused_queries,
            -- Claude Sonnet: $3/M input, $15/M output
            ROUND(
                (SUM(input_tokens)  * 0.000003 +
                 SUM(output_tokens) * 0.000015)::NUMERIC, 4
            )                                          AS estimated_cost_usd
        FROM query_log
        WHERE created_at > NOW() - INTERVAL '%s days'
        """,
        (days,),
    )


def get_top_questions(limit: int = 10) -> list[dict]:
    """Most frequently asked questions (simple keyword grouping)."""
    return _fetchall(
        """
        SELECT question,
               COUNT(*) AS frequency,
               ROUND(AVG(latency_ms)) AS avg_latency_ms
        FROM query_log
        WHERE was_refused = FALSE
        GROUP BY question
        ORDER BY frequency DESC
        LIMIT %s
        """,
        (limit,),
    )
