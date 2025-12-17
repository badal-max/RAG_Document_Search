"""LangGraph nodes for RAG workflow"""

from src.states.rag_state import RAGState


class RAGNodes:
    """Contains node functions for RAG workflow"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """Retrieve relevant documents"""
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )

    def generate_answer(self, state: RAGState) -> RAGState:
        """Generate answer using Groq LLM"""

        if not state.retrieved_docs:
            context = "No relevant documents found."
        else:
            context = "\n\n".join(doc.page_content for doc in state.retrieved_docs)

        prompt = f"""
You are a helpful assistant.
Answer the question strictly using the provided context.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{state.question}
"""

        response = self.llm.invoke(prompt)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=response.content.strip()
        )
