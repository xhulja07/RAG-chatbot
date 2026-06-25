"""
streamlit_app.py — Chat UI and admin document upload page.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations
import os
import time
import tempfile
import uuid
from pathlib import Path

import requests
import streamlit as st

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Company Knowledge Assistant",
    page_icon="📚",
    layout="wide",
)

# ------------------------------------------------------------------ session state

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0


# ------------------------------------------------------------------ sidebar

with st.sidebar:
    st.title("📚 KB Assistant")
    st.caption("Powered by Claude + pgvector")

    st.divider()

    # Category filter
    try:
        cats = requests.get(f"{API_BASE}/categories", timeout=5).json()["categories"]
    except Exception:
        cats = []

    category = st.selectbox(
        "Filter by document category",
        options=["All categories"] + cats,
        index=0,
    )
    category_filter = None if category == "All categories" else category

    st.divider()
    st.metric("Session cost (est.)", f"${st.session_state.total_cost:.4f}")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.total_cost = 0.0
        st.rerun()


# ------------------------------------------------------------------ tabs

tab_chat, tab_admin, tab_costs = st.tabs(["💬 Chat", "📂 Admin", "💰 Cost report"])


# ================================================================== CHAT TAB
with tab_chat:
    st.subheader("Ask a question")

    # Render message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander("Sources", expanded=False):
                    for c in msg["citations"]:
                        st.markdown(
                            f"**[{c['source_number']}] {c['document_title']}**  \n"
                            f"> {c['relevant_excerpt']}"
                        )
            if msg.get("meta"):
                m = msg["meta"]
                st.caption(
                    f"Confidence: {m['confidence']} · "
                    f"{m['input_tokens']}↑ {m['output_tokens']}↓ tokens · "
                    f"{m['latency_ms']} ms · est. ${m['estimated_cost_usd']:.5f}"
                )

    # Chat input
    if prompt := st.chat_input("e.g. What is the remote work policy?"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Call API
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base…"):
                try:
                    resp = requests.post(
                        f"{API_BASE}/chat",
                        json={
                            "question": prompt,
                            "category_filter": category_filter,
                            "session_id": st.session_state.session_id,
                        },
                        timeout=30,
                    ).json()
                except Exception as e:
                    st.error(f"API error: {e}")
                    st.stop()

            answer = resp.get("answer", "No answer returned.")
            citations = resp.get("citations", [])
            was_refused = resp.get("was_refused", False)

            if was_refused:
                st.warning(answer)
            else:
                st.markdown(answer)

            if citations:
                with st.expander("Sources", expanded=True):
                    for c in citations:
                        st.markdown(
                            f"**[{c['source_number']}] {c['document_title']}**  \n"
                            f"> {c['relevant_excerpt']}"
                        )

            meta = {
                "confidence":          resp.get("confidence", "—"),
                "input_tokens":        resp.get("input_tokens", 0),
                "output_tokens":       resp.get("output_tokens", 0),
                "latency_ms":          resp.get("latency_ms", 0),
                "estimated_cost_usd":  resp.get("estimated_cost_usd", 0),
            }
            st.caption(
                f"Confidence: {meta['confidence']} · "
                f"{meta['input_tokens']}↑ {meta['output_tokens']}↓ tokens · "
                f"{meta['latency_ms']} ms · est. ${meta['estimated_cost_usd']:.5f}"
            )

            st.session_state.total_cost += meta["estimated_cost_usd"]

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "citations": citations,
            "meta": meta,
        })


# ================================================================== ADMIN TAB
with tab_admin:
    st.subheader("Upload documents")
    st.caption("Upload PDF or Markdown files to add them to the knowledge base.")

    upload_category = st.selectbox(
        "Assign category",
        ["HR Policy", "SOP", "Finance", "Legal", "IT", "General"],
        key="upload_cat",
    )
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
    )

    if st.button("Ingest documents", disabled=not uploaded_files):
        with st.spinner("Ingesting…"):
            with tempfile.TemporaryDirectory() as tmpdir:
                for f in uploaded_files:
                    Path(tmpdir, f.name).write_bytes(f.read())

                # Call ingestion directly (in production: queue a background job)
                import subprocess
                result = subprocess.run(
                    [
                        "python", "ingestion/ingest.py",
                        "--source", tmpdir,
                        "--category", upload_category,
                    ],
                    capture_output=True, text=True,
                )
                if result.returncode == 0:
                    st.success(f"Ingested {len(uploaded_files)} file(s) successfully.")
                    st.text(result.stdout)
                else:
                    st.error("Ingestion failed.")
                    st.text(result.stderr)

    st.divider()
    st.subheader("Stale documents")
    st.caption("Documents not updated in the last 180 days.")

    try:
        stale = requests.get(f"{API_BASE}/stale-documents", timeout=5).json()["documents"]
        if stale:
            st.dataframe(stale, use_container_width=True)
        else:
            st.success("No stale documents found.")
    except Exception as e:
        st.warning(f"Could not fetch stale documents: {e}")


# ================================================================== COST TAB
with tab_costs:
    st.subheader("Usage and cost report")

    period = st.slider("Last N days", min_value=7, max_value=90, value=30)

    try:
        data = requests.get(
            f"{API_BASE}/cost-summary",
            params={"days": period},
            timeout=5,
        ).json()
        s = data.get("summary", {})

        if s:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total queries",       s.get("total_queries", 0))
            col2.metric("Refused queries",     s.get("refused_queries", 0))
            col3.metric("Avg latency",         f"{s.get('avg_latency_ms', 0)} ms")
            col4.metric("Est. total cost",     f"${s.get('estimated_cost_usd', 0):.4f}")

            st.divider()
            per_query = (
                float(s.get("estimated_cost_usd", 0)) / max(int(s.get("total_queries", 1)), 1)
            )
            st.info(
                f"**Cost per query:** ${per_query:.5f}  |  "
                f"**Projected monthly cost at 500 queries/day:** "
                f"${per_query * 500 * 30:.2f}"
            )
        else:
            st.info("No query data yet.")
    except Exception as e:
        st.warning(f"Could not fetch cost data: {e}")
