"""
app/retriever.py — pgvector similarity search with SQL metadata filters.

Two retrieval modes:
  1. similarity_search(query)          — pure vector search
  2. filtered_search(query, category)  — vector search scoped to a category

The SQL filter is pushed down to Postgres before vector scoring,
which is faster than post-filtering in Python.
"""

from __future__ import annotations
import os
from dataclasses import dataclass

import psycopg2
import psycopg2.extras
from langchain_huggingface import HuggingFaceEmbeddings

embeddings_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},   # swap to "cuda" if you have a GPU
)

TOP_K = 5   # how many chunks to retrieve per query


@dataclass
class RetrievedChunk:
    chunk_id:     int
    document_id:  int
    doc_title:    str
    doc_category: str
    last_updated: str
    content:      str
    score:        float   # cosine similarity (1.0 = identical)
    is_stale:     bool    # True if doc is > 180 days old


def _get_conn():
    return psycopg2.connect(os.environ["DATABASE_URL"])


def _embed_query(query: str) -> list[float]:
    return embeddings_model.embed_query(query)


# ------------------------------------------------------------------ core search

def similarity_search(
    query: str,
    top_k: int = TOP_K,
    category: str | None = None,
) -> list[RetrievedChunk]:
    """
    Retrieve the top_k most relevant chunks for a query.
    Optionally filter by document category.

    SQL skill practiced:
      - pgvector <=> cosine distance operator
      - JOIN across chunks + documents + employees
      - Optional WHERE filter injected at query time
      - Staleness flag computed inline with INTERVAL arithmetic
    """
    query_vec = _embed_query(query)
    vec_str = "[" + ",".join(str(v) for v in query_vec) + "]"

    category_filter = "AND d.category = %(category)s" if category else ""

    sql = f"""
        SELECT
            c.id                                              AS chunk_id,
            c.document_id,
            d.title                                           AS doc_title,
            d.category                                        AS doc_category,
            TO_CHAR(d.last_updated, 'YYYY-MM-DD')            AS last_updated,
            c.content,
            1 - (c.embedding <=> %(vec)s::vector)            AS score,
            d.last_updated < NOW() - INTERVAL '180 days'     AS is_stale
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE d.is_active = TRUE
          {category_filter}
        ORDER BY c.embedding <=> %(vec)s::vector
        LIMIT %(top_k)s
    """

    params = {"vec": vec_str, "top_k": top_k, "category": category}

    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    finally:
        conn.close()

    return [
        RetrievedChunk(
            chunk_id=r["chunk_id"],
            document_id=r["document_id"],
            doc_title=r["doc_title"],
            doc_category=r["doc_category"],
            last_updated=r["last_updated"],
            content=r["content"],
            score=float(r["score"]),
            is_stale=bool(r["is_stale"]),
        )
        for r in rows
    ]


def format_context_for_prompt(chunks: list[RetrievedChunk]) -> tuple[str, list[int]]:
    """
    Convert retrieved chunks into a formatted context string for the LLM prompt,
    and return the list of source document IDs for logging.

    Returns:
        context_str  — the formatted string to inject into the system prompt
        doc_ids      — unique document IDs cited (for query_log)
    """
    parts = []
    doc_ids = []

    for i, chunk in enumerate(chunks, 1):
        stale_note = " ⚠️ NOTE: this document may be out of date." if chunk.is_stale else ""
        parts.append(
            f"[SOURCE {i}] {chunk.doc_title} "
            f"(Category: {chunk.doc_category}, Last updated: {chunk.last_updated})"
            f"{stale_note}\n{chunk.content}"
        )
        if chunk.document_id not in doc_ids:
            doc_ids.append(chunk.document_id)

    context_str = "\n\n---\n\n".join(parts)
    return context_str, doc_ids
