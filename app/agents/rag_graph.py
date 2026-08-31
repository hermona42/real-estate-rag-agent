from typing import List, TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from app.config import settings
from app.services.vector_store import VectorStoreService


# 1. Define the Graph State (what data flows through our agents)
class AgentState(TypedDict):
    question: str
    documents: List[Document]
    answer: str


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

    def _retrieve_node(self, state: AgentState) -> AgentState:
        """Node 1: Retrieves relevant real estate docs from Qdrant."""
        question = state["question"]
        retriever = self.vector_store_service.get_retriever(k=4)
        retrieved_docs = retriever.invoke(question)
        return {"documents": retrieved_docs}

    def _generate_node(self, state: AgentState) -> AgentState:
        """Node 2: Generates an answer using the LLM and retrieved documents."""
        question = state["question"]
        docs = state["documents"]

        # Format context from retrieved document chunks
        context = "\n\n".join([f"Source: {doc.metadata.get('source')} (Page {doc.metadata.get('page')})\n{doc.page_content}" for doc in docs])

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an expert AI real estate assistant. Answer the user's question accurately using ONLY the provided property document context. If the answer cannot be found in the context, state that clearly.\n\nContext:\n{context}"),
            ("human", "{question}")
        ])

        chain = prompt_template | self.llm
        response = chain.invoke({"context": context, "question": question})

        return {"answer": response.content}

    def _build_graph(self):
        """Compiles the LangGraph state machine workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self._generate_node)

        # Connect edges (Retrieve -> Generate -> END)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    def run(self, question: str) -> dict:
        """Executes the graph workflow for a given user question."""
        initial_state = {"question": question, "documents": [], "answer": ""}
        result = self.graph.invoke(initial_state)
        return result