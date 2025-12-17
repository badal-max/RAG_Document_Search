"""LangGraph nodes for RAG workflow + ReAct Agent"""

from typing import List, Optional
from src.states.rag_state import RAGState

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

# Wikipedia tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun


class RAGNodes:
    """Contains node functions for RAG workflow"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None  # lazy init

    # ---------------- Retrieval ---------------- #

    def retrieve_docs(self, state: RAGState) -> RAGState:
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )

    # ---------------- Tools ---------------- #

    def _build_tools(self) -> List[Tool]:
        """Build retriever + Wikipedia tools"""

        def retriever_tool_fn(query: str) -> str:
            docs: List[Document] = self.retriever.invoke(query)
            if not docs:
                return "No documents found."

            chunks = []
            for i, d in enumerate(docs[:8], start=1):
                meta = d.metadata or {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                chunks.append(f"[{i}] {title}\n{d.page_content}")

            return "\n\n".join(chunks)

        retriever_tool = Tool(
            name="retriever",
            description="Fetch passages from the indexed document corpus.",
            func=retriever_tool_fn,
        )

        wiki = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=3, lang="en")
        )

        wikipedia_tool = Tool(
            name="wikipedia",
            description="Search Wikipedia for general world knowledge.",
            func=wiki.run,
        )

        return [retriever_tool, wikipedia_tool]

    # ---------------- Agent ---------------- #

    def _build_agent(self):
        tools = self._build_tools()

        system_message = SystemMessage(
            content=(
                "You are a helpful RAG assistant.\n"
                "- Prefer the `retriever` tool for user-provided documents.\n"
                "- Use `wikipedia` only for general background knowledge.\n"
                "- Use tools when helpful.\n"
                "- Return only the final helpful answer."
            )
        )

        self._agent = create_react_agent(
            self.llm,
            tools=tools,
        )

        # store system message for reuse
        self._system_message = system_message

    # ---------------- Generation ---------------- #

    def generate_answer(self, state: RAGState) -> RAGState:
        if self._agent is None:
            self._build_agent()

        result = self._agent.invoke(
            {
                "messages": [
                    self._system_message,
                    HumanMessage(content=state.question),
                ]
            }
        )

        messages = result.get("messages", [])
        answer: Optional[str] = None

        if messages:
            answer = getattr(messages[-1], "content", None)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer or "Could not generate an answer."
        )
