from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from app.config import settings

COLLECTION_NAME = "real_estate_docs"


class VectorStoreService:
    """Handles text chunking, embedding generation, and Qdrant storage."""

    def __init__(self):
        # Initialize Google Generative AI Embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2", 
            google_api_key=settings.GEMINI_API_KEY
        )
        
        # Initialize local Qdrant Client (in-memory mode for easy development/testing)
        self.client = QdrantClient(":memory:")
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Creates the Qdrant vector collection if it doesn't already exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Splits raw documents into smaller chunks with overlap."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", " ", ""]
        )
        return text_splitter.split_documents(documents)

    def index_documents(self, documents: List[Document]) -> QdrantVectorStore:
        """Chunks documents and stores them inside the Qdrant vector database."""
        chunks = self.chunk_documents(documents)
        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=COLLECTION_NAME,
            embedding=self.embeddings,
        )
        vector_store.add_documents(chunks)
        return vector_store

    def get_retriever(self, k: int = 4):
        """Returns a retriever object to fetch top 'k' matching property chunks."""
        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=COLLECTION_NAME,
            embedding=self.embeddings,
        )
        return vector_store.as_retriever(search_kwargs={"k": k})