import os
from pathlib import Path
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document

from .config import settings

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "sample_docs"
VECTOR_DB_DIR = BASE_DIR / "vectorstore"

class RAGPipeline:
    """
    A small RAG helper class:
    - Loads documents from data/sample_docs
    - Builds a FAISS vector store
    - Answers questions using an LLM with retrieved context
    """

    def __init__(self) -> None:
        self._vector_store = None
        self._embeddings = OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            model=settings.embedding_model_name,
        )
        self._llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.model_name,
            temperature=0.2,
        )

    def _load_documents(self) -> List[Document]:
        if not DATA_DIR.exists():
            raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

        docs: List[Document] = []
        for file_path in DATA_DIR.glob("*.txt"):
            # You can later add PDF loaders or others
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs.extend(loader.load())
        return docs

    def _split_documents(self, docs: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len,
        )
        return splitter.split_documents(docs)

    def build_vector_store(self, rebuild: bool = False) -> None:
        """
        Build or load a FAISS vector store.
        If rebuild=True, it ignores any saved store and rebuilds from scratch.
        """
        VECTOR_DB_DIR.mkdir(exist_ok=True)

        index_path = VECTOR_DB_DIR / "index.faiss"
        store_path = VECTOR_DB_DIR / "store.pkl"

        if not rebuild and index_path.exists() and store_path.exists():
            # Load existing store
            self._vector_store = FAISS.load_local(
                VECTOR_DB_DIR,
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
            return

        docs = self._load_documents()
        if not docs:
            raise ValueError(f"No documents found in {DATA_DIR}")

        split_docs = self._split_documents(docs)

        self._vector_store = FAISS.from_documents(split_docs, self._embeddings)
        self._vector_store.save_local(VECTOR_DB_DIR)

    def _similar_chunks(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        if self._vector_store is None:
            raise RuntimeError("Vector store not initialized. Call build_vector_store() first.")
        return self._vector_store.similarity_search_with_score(query, k=k)

    def answer(self, query: str, top_k: int = 4, debug: bool = False) -> Tuple[str, List[Tuple[Document, float]]]:
        """
        Main RAG call:
        1. Retrieve relevant chunks
        2. Ask the LLM using those chunks as context
        """
        hits = self._similar_chunks(query, k=top_k)

        context_text = "\n\n".join(
            f"Chunk {idx+1}:\n{doc.page_content}"
            for idx, (doc, _score) in enumerate(hits)
        )

        prompt = (
            "You are a helpful assistant answering questions based on the provided context.\n"
            "If the answer is not in the context, say you are not sure instead of hallucinating.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n"
            "Answer in a concise, clear way.\n"
        )

        llm_response = self._llm.invoke(prompt)

        # LangChain ChatOpenAI usually returns an object with 'content'
        answer_text = getattr(llm_response, "content", str(llm_response))

        return answer_text, hits
