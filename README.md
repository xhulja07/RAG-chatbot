# Project 1 — Internal Knowledge Base Chatbot

A RAG-powered assistant built with LangChain, pgvector, and the Anthropic Claude API.
User ask questions; the system retrieves relevant policy docs and answers with citations.

## Project structure

```
kb-chatbot/
├── app/
│   ├── main.py            # FastAPI backend (chat endpoint + health)
│   ├── rag_chain.py       # LangChain RAG chain with citation tool use
│   ├── retriever.py       # pgvector retriever + SQL metadata filters
│   └── guardrails.py      # Scope check + PII refusal logic
├── ingestion/
│   ├── ingest.py          # PDF/Markdown loader → chunk → embed → store
│   └── chunking.py        # Custom chunking strategies
├── db/
│   ├── schema.sql         # Postgres + pgvector schema
│   └── queries.py         # SQL helpers (stale docs, metadata filters)
├── evals/
│   ├── eval_set.json      # 10 Q&A pairs for regression testing
│   └── run_evals.py       # LangSmith eval runner
├── streamlit_app.py       # Chat UI + admin upload page
├── docker-compose.yml     # Postgres + app services
├── Dockerfile
├── requirements.txt
└── .env.example
```

## Quickstart

```bash
cp .env.example .env          # fill in your keys
docker-compose up -d          # start Postgres with pgvector
pip install -r requirements.txt
python ingestion/ingest.py --source ./docs_sample   # ingest sample docs
uvicorn app.main:app --reload  # start API
streamlit run streamlit_app.py # start UI
```

## Environment variables

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `DATABASE_URL` | Postgres connection string |
| `LANGCHAIN_API_KEY` | LangSmith tracing (optional) |
| `LANGCHAIN_TRACING_V2` | Set to `true` to enable LangSmith |
