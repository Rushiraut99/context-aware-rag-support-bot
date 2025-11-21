# Context-Aware RAG Support Bot

A small end-to-end **Retrieval-Augmented Generation (RAG)** microservice built with **FastAPI**, **LangChain**, and **OpenAI**.

The service ingests local text documents, builds a vector store using **FAISS**, and exposes HTTP APIs to:
- Rebuild / load the document index
- Ask natural language questions over the indexed knowledge base

## ✨ Features

- **RAG pipeline**: Retrieve relevant chunks from your own documents before calling the LLM
- **FastAPI service**: Clean, typed endpoints (`/health`, `/ingest`, `/query`)
- **Vector search** with FAISS
- Configurable **LLM + embedding models**
- Minimal **tests** with `pytest`
- Ready to be extended with:
  - authentication
  - multi-tenant indexes
  - Docker + CI/CD

## 🧱 Architecture Overview

**High-level flow:**

1. Documents in `data/sample_docs/*.txt` are loaded.
2. They are split into smaller chunks using a recursive character splitter.
3. Chunks are embedded via OpenAI embeddings and stored in a FAISS vector store.
4. At query time:
   - The service retrieves the top-k similar chunks for the user’s query.
   - It passes those chunks + question to the LLM.
   - The LLM responds based only on the given context.

This design mirrors typical **production RAG systems** used in internal knowledge bots and support assistants.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/context-aware-rag-support-bot.git
cd context-aware-rag-support-bot
