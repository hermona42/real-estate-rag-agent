from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from app.config import settings
from app.services.pdf_parser import process_pdf_document
from app.services.vector_store import VectorStoreService
from app.agents.rag_graph import RealEstateAgentGraph

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="LangGraph Multi-Agent RAG API for Real Estate Documents"
)

# Initialize persistent services in memory
vector_store_service = VectorStoreService()
agent_graph = RealEstateAgentGraph(vector_store_service=vector_store_service)


# Request schema for chat
class ChatRequest(BaseModel):
    question: str


# Response schema for chat
class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    extracted_data: Optional[Dict[str, Any]] = None


@app.get("/")
def read_root():
    return {"status": "online", "app_name": settings.APP_NAME}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Uploads a PDF real estate document, parses it with OCR, and indexes vectors into Qdrant."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        contents = await file.read()
        # Step 1: Parse PDF text & fallback OCR
        documents = process_pdf_document(contents, file.filename)

        if not documents:
            raise HTTPException(status_code=400, detail="Could not extract text from document.")

        # Step 2: Index into Qdrant vector database
        vector_store_service.index_documents(documents)

        return {
            "status": "success",
            "filename": file.filename,
            "pages_processed": len(documents),
            "message": "Document indexed successfully into vector store."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
def chat_with_agent(request: ChatRequest):
    """Processes a query using the LangGraph state machine agent."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = agent_graph.run(request.question)
        
        # Extract metadata sources for citations
        sources = [
            {
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
                "snippet": doc.page_content[:150] + "..."
            }
            for doc in result.get("documents", [])
        ]

        return ChatResponse(
            answer=result.get("answer", "No answer generated."),
            sources=sources,
            extracted_data=result.get("extracted_data")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent workflow error: {str(e)}")