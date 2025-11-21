from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    HealthResponse,
    IngestRequest,
    QueryRequest,
    QueryResponse,
    Chunk,
)
from .rag_pipeline import RAGPipeline

app = FastAPI(
    title="Context-Aware RAG Support Bot",
    version="0.1.0",
    description="A simple RAG microservice built with FastAPI + LangChain",
)

# Allow local frontend / tools to call this API easily
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can tighten this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGPipeline()

@app.on_event("startup")
def startup_event():
    # Build or load vector store when the API starts
    try:
        rag.build_vector_store(rebuild=False)
    except Exception as e:
        # In a real app, use proper logging instead of print
        print(f"[startup] Failed to build vector store: {e}")


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok", detail="Service is up and running.")


@app.post("/ingest", response_model=HealthResponse)
def ingest_docs(payload: IngestRequest):
    try:
        rag.build_vector_store(rebuild=payload.rebuild)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return HealthResponse(
        status="ok",
        detail="Vector store built successfully." if payload.rebuild else "Vector store loaded or created.",
    )

@app.post("/query", response_model=QueryResponse)
def query_docs(payload: QueryRequest):
    try:
        answer, hits = rag.answer(
            query=payload.query,
            top_k=payload.top_k,
            debug=payload.debug,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    chunks = None
    if payload.debug:
        chunks = [
            Chunk(content=doc.page_content, score=float(score))
            for doc, score in hits
        ]

    return QueryResponse(answer=answer, context=chunks)
