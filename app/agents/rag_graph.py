from typing import List, TypedDict, Optional, Any, Dict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from app.config import settings
from app.services.vector_store import VectorStoreService


# 1. Define the Graph State (what data flows through our agents)
class AgentState(TypedDict):
    question: str
    documents: List[Document]
    answer: str
    route: str
    extracted_data: Optional[Dict[str, Any]]


class RouteResponse(BaseModel):
    route: str = Field(description="Must be either 'document_qa' or 'property_extraction'")


class PropertyExtraction(BaseModel):
    rent: Optional[str] = Field(None, description="Rent amount or terms")
    deposit: Optional[str] = Field(None, description="Security deposit details")
    pet_fees: Optional[str] = Field(None, description="Pet fee terms")
    notice_period: Optional[str] = Field(None, description="Notice period duration")
    parties_involved: Optional[List[str]] = Field(None, description="Parties involved in the lease (e.g., landlord, tenant)")


class RealEstateAgentGraph:
    """Manages the LangGraph compilation and execution for property RAG."""

    def __init__(self, vector_store_service: VectorStoreService):
        self.vector_store_service = vector_store_service
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=settings.GEMINI_API_KEY
        )
        self.graph = self._build_graph()

    def _supervisor_node(self, state: AgentState) -> AgentState:
        """Evaluates and routes the query."""
        question = state["question"]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a routing assistant. Classify the user's input into either 'document_qa' (questions about specific clauses or details) or 'property_extraction' (requests to summarize key property metadata, financial terms, or structured key-value pairs)."),
            ("human", "{question}")
        ])
        
        llm_with_structured = self.llm.with_structured_output(RouteResponse)
        chain = prompt | llm_with_structured
        response = chain.invoke({"question": question})
        
        return {"route": response.route}

    def _rag_node(self, state: AgentState) -> AgentState:
        """Executes vector retrieval and answers document questions."""
        question = state["question"]
        retriever = self.vector_store_service.get_retriever(k=4)
        retrieved_docs = retriever.invoke(question)
        
        context = "\n\n".join([f"Source: {doc.metadata.get('source')} (Page {doc.metadata.get('page')})\n{doc.page_content}" for doc in retrieved_docs])
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an expert AI real estate assistant. Answer the user's question accurately using ONLY the provided property document context. If the answer cannot be found in the context, state that clearly.\n\nContext:\n{context}"),
            ("human", "{question}")
        ])
        
        chain = prompt_template | self.llm
        response = chain.invoke({"context": context, "question": question})
        
        return {"documents": retrieved_docs, "answer": response.content}
        
    def _extraction_node(self, state: AgentState) -> AgentState:
        """Uses structured outputs to parse specific details into JSON format."""
        question = state["question"]
        retriever = self.vector_store_service.get_retriever(k=6)
        retrieved_docs = retriever.invoke(question)
        
        context = "\n\n".join([f"Source: {doc.metadata.get('source')} (Page {doc.metadata.get('page')})\n{doc.page_content}" for doc in retrieved_docs])
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an expert AI real estate data extractor. Extract the following property details from the provided context. If a field is not found in the context, leave it null.\n\nContext:\n{context}"),
            ("human", "{question}")
        ])
        
        llm_with_structured = self.llm.with_structured_output(PropertyExtraction)
        chain = prompt_template | llm_with_structured
        response = chain.invoke({"context": context, "question": question})
        
        return {"documents": retrieved_docs, "extracted_data": response.model_dump(), "answer": "Property data extracted successfully."}

    def _build_graph(self):
        """Compiles the LangGraph state machine workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("rag_node", self._rag_node)
        workflow.add_node("extraction_node", self._extraction_node)

        # Connect edges
        workflow.set_entry_point("supervisor")
        
        def route_condition(state: AgentState) -> str:
            if state.get("route") == "property_extraction":
                return "extraction_node"
            return "rag_node"
            
        workflow.add_conditional_edges(
            "supervisor",
            route_condition,
            {
                "rag_node": "rag_node",
                "extraction_node": "extraction_node"
            }
        )
        
        workflow.add_edge("rag_node", END)
        workflow.add_edge("extraction_node", END)

        return workflow.compile()

    def run(self, question: str) -> dict:
        """Executes the graph workflow for a given user question."""
        initial_state = {"question": question, "documents": [], "answer": "", "route": "", "extracted_data": None}
        result = self.graph.invoke(initial_state)
        return result