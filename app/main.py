"""
app/main.py — FastAPI backend exposing the chat endpoint and a health check.

Endpoints:
  POST /chat          — main Q&A endpoint
  GET  /health        — liveness probe
  GET  /docs-meta     — list all active documents (for admin UI)
  GET  /cost-summary  — token usage and cost report (for PM dashboard)
"""

from __future__ import annotations
import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.rag_chain import chat, ChatResponse
from db.queries import get_cost_summary, get_active_categories, get_stale_documents


app = FastAPI(
    title="KB Chatbot API",
    description="RAG-powered internal knowledge base assistant",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ schemas

class ChatRequest(BaseModel):
    question:        str
    category_filter: Optional[str] = None
    session_id:      Optional[str] = None


class ChatApiResponse(BaseModel):
    answer:          str
    citations:       list[dict]
    confidence:      str
    source_doc_ids:  list[int]
    input_tokens:    int
    output_tokens:   int
    latency_ms:      int
    was_refused:     bool
    # Computed cost estimate (for per-query cost display in the UI)
    estimated_cost_usd: float


# ------------------------------------------------------------------ routes

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatApiResponse)
def chat_endpoint(
    req: ChatRequest,
    x_user_email: str = Header(default="anonymous"),
):
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")

    result: ChatResponse = chat(
        req.question,
        category_filter=req.category_filter,
        session_id=req.session_id,
        user_email=x_user_email,
    )

    # Claude Sonnet pricing: $3/M input, $15/M output
    cost = (result.input_tokens * 0.000003) + (result.output_tokens * 0.000015)

    return ChatApiResponse(
        answer=result.answer,
        citations=result.citations,
        confidence=result.confidence,
        source_doc_ids=result.source_doc_ids,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=result.latency_ms,
        was_refused=result.was_refused,
        estimated_cost_usd=round(cost, 6),
    )


@app.get("/categories")
def list_categories():
    """Return all active document categories (for the category filter dropdown)."""
    return {"categories": get_active_categories()}


@app.get("/stale-documents")
def stale_documents(days: int = 180):
    """Documents that haven't been updated in `days` days — for admin review."""
    return {"documents": get_stale_documents(days=days)}


@app.get("/cost-summary")
def cost_summary(days: int = 30):
    """Token usage and estimated cost for the last N days."""
    summary = get_cost_summary(days=days)
    return {"period_days": days, "summary": summary}
