import os
import hashlib
import logging
from time import monotonic
from typing import List
import asyncio

from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt, RetryError, AsyncRetrying

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("RAG_LOG_LEVEL", "INFO"))

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_chroma import Chroma as ChromaClass
except Exception:
    from langchain_community.vectorstores import Chroma as ChromaClass


class RAGSystem:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            temperature=0.3,
            max_retries=0,
            max_tokens=2048,
        )

        self._min_retry_delay = int(os.getenv("RAG_MIN_RETRY_DELAY", "1"))
        self._max_retry_delay = int(os.getenv("RAG_MAX_RETRY_DELAY", "30"))
        self._max_retries = int(os.getenv("RAG_MAX_RETRIES", "3"))

        self._bucket_capacity = float(os.getenv("RAG_RATE_LIMIT_CAPACITY", "3"))
        self._refill_rate = float(os.getenv("RAG_RATE_LIMIT_REFILL_RATE", "0.25"))
        self._tokens = float(self._bucket_capacity)
        self._last_refill_time = monotonic()
        self._async_lock = asyncio.Lock()

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder="data/embedding_cache",
        )

        os.makedirs("data/chroma_db", exist_ok=True)
        os.makedirs("data/embedding_cache", exist_ok=True)
        
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        """Initialize or recreate the vector store and retriever"""
        try:
            self.vector_store = ChromaClass(
                collection_name="docs",
                persist_directory="data/chroma_db",
                embedding_function=self.embeddings,
            )
            
            doc_count = self.vector_store._collection.count()
            logger.info(f"Vector store initialized with {doc_count} documents")
            
        except Exception as e:
            logger.warning(f"Failed to initialize vector store: {e}. Creating new collection.")
            try:
                self.vector_store = ChromaClass(
                    collection_name="docs",
                    persist_directory="data/chroma_db",
                    embedding_function=self.embeddings,
                )
            except Exception as e2:
                logger.error(f"Failed to create new collection: {e2}")
                raise RuntimeError("Could not initialize Chroma vector store") from e2
        
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 15})

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=300,
        )

        self.prompt = ChatPromptTemplate.from_template(
            """Answer the question using ONLY the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}
"""
        )

        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        self.rag_chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
        )

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
    def add_document(self, text: str, source: str = "unknown"):
        """Synchronous wrapper for backward compatibility"""
        asyncio.run(self.add_document_async(text, source))

    async def add_document_async(self, text: str, source: str = "unknown"):
        """Async version for adding documents with deduplication"""
        chunks = self.text_splitter.split_text(text)

        existing = self.vector_store.get(include=["metadatas"])
        existing_hashes = {
            meta.get("hash")
            for meta in existing.get("metadatas", [])
            if meta
        }

        new_docs = []
        for chunk in chunks:
            h = self._hash_text(chunk)
            if h not in existing_hashes:
                new_docs.append(
                    Document(
                        page_content=chunk,
                        metadata={"source": source, "hash": h},
                    )
                )

        if new_docs:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.vector_store.add_documents, new_docs)
            logger.info("Added %d new document chunks", len(new_docs))

    async def query_async(self, question: str):
        doc_count = self.vector_store._collection.count()
        if doc_count == 0:
            return {
                "answer": "No documents in the database. Please add documents first using add_document().",
                "source_documents": []
            }
        
        async def _attempt(q: str):
            while True:
                async with self._async_lock:
                    now = monotonic()
                    elapsed = now - self._last_refill_time
                    if elapsed > 0:
                        refill = elapsed * self._refill_rate
                        self._tokens = min(self._bucket_capacity, self._tokens + refill)
                        self._last_refill_time = now

                    if self._tokens >= 1.0:
                        self._tokens -= 1.0
                        logger.debug(
                            "Token-bucket: consuming token (remaining=%.2f)", self._tokens
                        )
                        break
                    else:
                        needed = 1.0 - self._tokens
                        wait_time = needed / max(self._refill_rate, 1e-6)
                        logger.debug("Token-bucket: no tokens, sleeping %.3fs", wait_time)
                await asyncio.sleep(wait_time)

            logger.info("Invoking RAG chain for question (rate-limited at 15 RPM)")

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self.rag_chain.invoke, q)
            retrieved = await loop.run_in_executor(None, self.retriever.invoke, q)
            return response, retrieved

        try:
            async for attempt in AsyncRetrying(
                wait=wait_exponential(min=self._min_retry_delay, max=self._max_retry_delay),
                stop=stop_after_attempt(self._max_retries),
                reraise=True,
            ):
                with attempt:
                    response, docs = await _attempt(question)

        except RetryError as e:
            exc = e.last_attempt.exception()
            msg = str(exc or "")

            if "insufficient_quota" in msg or "quota" in msg or "429" in msg:
                raise RuntimeError(
                    "API returned 429 / insufficient_quota. "
                    "Check your API plan, billing, and usage limits."
                ) from exc

            raise RuntimeError(f"Query failed after retries: {exc}") from exc
        except Exception as e:
            raise RuntimeError(f"Query failed: {e}") from e

        return {
            "answer": response.content,
            "source_documents": list(
                {doc.metadata.get("source", "unknown") for doc in docs}
            ),
        }

    def query(self, question: str):
        """Synchronous wrapper for backward compatibility"""
        return asyncio.run(self.query_async(question))

    def clear_database(self):
        """Clear all documents and recreate the collection"""
        try:
            self.vector_store.delete_collection()
            logger.info("Deleted existing collection")
        except Exception as e:
            logger.warning(f"Could not delete collection: {e}")
        
        self._initialize_vector_store()
        self._tokens = self._bucket_capacity
        self._last_refill_time = monotonic()
        logger.info("Database cleared and reinitialized")