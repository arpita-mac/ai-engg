# 📄 Handbook RAG Assistant

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-009688)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector%20store-orange)
![Groq](https://img.shields.io/badge/LLM-Groq%20Llama%203.3-purple)

A Retrieval-Augmented Generation (RAG) system that answers natural language questions from a company handbook PDF — built from scratch with real semantic search, grounded answers, and source citations.

> Ask *"Can I work from home on Fridays?"* — get an accurate answer pulled directly from the document, with the exact source text shown alongside it.

---

## 🧠 What it does

1. **Embeds** the question into a vector
2. **Searches** ChromaDB for the most semantically similar chunks of the handbook
3. **Grounds** an LLM's answer using only those retrieved chunks
4. **Returns** the answer *and* the source chunks used — fully auditable, not a black box

If the answer isn't in the document, it says so — it doesn't guess.

## 🤔 Why this project

Most RAG tutorials use clean, well-organized sample documents. This one intentionally uses a **messy, real-world-style handbook** — mixed topics within paragraphs, informal asides, information scattered across sections — to stress-test retrieval and chunking under realistic conditions instead of a tutorial best-case.

## 🏗️ Architecture

```
PDF document
    │
    ▼
PyPDFLoader → RecursiveCharacterTextSplitter (chunk_size=300, overlap=50)
    │
    ▼
sentence-transformers (embed each chunk)
    │
    ▼
ChromaDB (store embeddings + chunk text)

──────────────────────────────────────────

User question
    │
    ▼
Embed question → ChromaDB similarity search (top-3 chunks)
    │
    ▼
ChatPromptTemplate (inject retrieved chunks as context)
    │
    ▼
LLM (Groq · Llama 3.3 70B) → grounded answer + cited sources
```

## ⚙️ Tech stack

| Layer | Tool |
|---|---|
| LLM | Groq — Llama 3.3 70B |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector store | ChromaDB |
| Orchestration | LangChain (prompt templates, chains, loaders, splitters) |
| API | FastAPI |
| PDF parsing | PyPDF |

## 🚀 Running it

```bash
# install dependencies
pip install -r requirements.txt

# add your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# run the API
uvicorn main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for interactive API docs, or call it directly:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many sick leave days do I get?"}'
```

```json
{
  "answer": "You get 7 days of sick leave per year. If taken for more than 2 consecutive days, a medical certificate is required.",
  "sources": [
    "sick leave is separate, you get 7 days sick leave, need a doctor note if its more than 2 days in a row...",
    "..."
  ]
}
```

## 🎯 Design decisions worth noting

**Chunking with overlap** — chunks overlap by 50 characters so details near a chunk boundary aren't lost; they appear in full in the adjacent chunk too.

**Grounded refusal** — the system prompt explicitly instructs the model to say *"I don't have information about that"* rather than fall back on general knowledge. Tested against edge cases (out-of-scope questions, prompt injection attempts) — held up correctly.

**Source citations on every answer** — retrieval returns the closest chunks by vector similarity, not necessarily the *correct* ones. Returning sources makes the system's reasoning auditable instead of a black box.

**One-time setup, cheap per-request work** — PDF loading, chunking, and embedding happen once at server startup. Each `/ask` request only embeds the question and queries the existing vector store.

## ⚠️ Known limitations

- Character-based chunking has no topic awareness — a paragraph spanning multiple topics may fragment a topic across chunks. Semantic chunking would address this at added complexity.
- ChromaDB is in-memory and resets on server restart; production would use persistent storage.
- No formal evaluation suite yet — testing has been manual, covering both answerable and intentionally unanswerable/edge-case questions.

## 🔭 What's next

- [ ] A small evaluation set (question/expected-answer pairs) to measure retrieval and answer quality systematically
- [ ] Index/cache invalidation strategy for handling document updates without serving stale answers
- [ ] Persistent vector storage
