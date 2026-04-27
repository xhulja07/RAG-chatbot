"""
app/guardrails.py — Scope and safety checks before hitting the RAG chain.

Two layers:
  1. Keyword/heuristic fast-check  (no LLM call, ~0ms)
  2. LLM-based scope classification (one small Claude call, ~200ms)

In production you'd use Layer 1 for obvious cases and Layer 2
for borderline inputs where confidence is needed.
"""

from __future__ import annotations
import re

# ------------------------------------------------------------------ config

# Topics the chatbot is NOT supposed to answer
OUT_OF_SCOPE_PATTERNS = [
    r"\bstoc?k\s*price\b",
    r"\binvest(ment|or|ing)\b",
    r"\blegal\s+advice\b",
    r"\bdiagnos(is|e)\b",
    r"\bprescri(ption|be)\b",
    r"\bhow\s+to\s+(hack|crack|exploit)\b",
    r"\bpassword\b.{0,20}\bother\s+user\b",
]

# PII patterns that should not be echoed back in answers
PII_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",          # SSN
    r"\b\d{16}\b",                       # credit card (naive)
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # email in query
]

REFUSAL_MESSAGE = (
    "I'm only able to answer questions about internal company policies, "
    "SOPs, HR guidelines, and operational documents. "
    "For this topic, please reach out to the relevant team directly."
)


# ------------------------------------------------------------------ checks

def contains_pii(text: str) -> bool:
    """True if the query itself contains PII we shouldn't process."""
    for pattern in PII_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def is_out_of_scope(query: str) -> tuple[bool, str]:
    """
    Fast heuristic check.
    Returns (is_blocked, reason_string).
    """
    lower = query.lower()
    for pattern in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, lower):
            return True, f"Matched out-of-scope pattern: {pattern}"

    if contains_pii(query):
        return True, "Query appears to contain PII"

    # Length guard — nonsense / injection attempts tend to be very long
    if len(query) > 1500:
        return True, "Query exceeds maximum length"

    return False, ""


def strip_pii_from_answer(text: str) -> str:
    """Remove any PII that might have leaked into the answer (belt-and-suspenders)."""
    for pattern in PII_PATTERNS:
        text = re.sub(pattern, "[REDACTED]", text)
    return text


# ------------------------------------------------------------------ prompt injection guard

INJECTION_PHRASES = [
    "ignore previous instructions",
    "ignore all instructions",
    "you are now",
    "act as",
    "pretend you are",
    "disregard",
    "system prompt",
    "new persona",
]


def is_prompt_injection(query: str) -> bool:
    lower = query.lower()
    return any(phrase in lower for phrase in INJECTION_PHRASES)


# ------------------------------------------------------------------ unified check

class GuardrailResult:
    def __init__(self, blocked: bool, reason: str = ""):
        self.blocked = blocked
        self.reason  = reason
        self.message = REFUSAL_MESSAGE if blocked else ""


def check(query: str) -> GuardrailResult:
    """
    Run all guardrail checks. Call this before every RAG chain invocation.

    Usage:
        result = guardrails.check(user_query)
        if result.blocked:
            return result.message
    """
    if is_prompt_injection(query):
        return GuardrailResult(blocked=True, reason="Prompt injection attempt")

    blocked, reason = is_out_of_scope(query)
    if blocked:
        return GuardrailResult(blocked=True, reason=reason)

    return GuardrailResult(blocked=False)
