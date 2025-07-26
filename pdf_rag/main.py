"""Main application for PDF question answering."""

from typing import List

from config import CHAT_TEMPLATE, Config
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from vector import retriever


class PDFChatBot:
    """PDF question answering chatbot."""

    def __init__(self) -> None:
        self.model = ChatGroq(
            groq_api_key=Config.GROQ_API_KEY,
            model_name=Config.LLM_MODEL,
        )

        self.prompt = ChatPromptTemplate.from_template(CHAT_TEMPLATE)
        self.chain = self.prompt | self.model
        self.retriever = retriever

    def format_docs(self, docs: List[Document]) -> str:
        """Format retrieved documents for the prompt."""
        formatted = []
        for doc in docs:
            source = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            content = doc.page_content.strip()
            formatted.append(f"[Source: {source}, Page: {page}]\n{content}")
        return "\n\n".join(formatted)

    def get_unique_sources(self, docs: List[Document]) -> List[str]:
        """Get unique sources from retrieved documents."""
        sources = set()
        for doc in docs:
            source_file = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            sources.add(f"- {source_file} (Page {page})")
        return sorted(sources)

    def answer_question(self, question: str) -> tuple[str, List[str]]:
        """Answer a question based on PDF content."""
        # Retrieve relevant documents
        docs = self.retriever.invoke(question)
        context = self.format_docs(docs)

        # Generate response
        result = self.chain.invoke({"context": context, "question": question})

        # Get sources
        sources = self.get_unique_sources(docs)

        return result.content, sources

    def run_chat_loop(self) -> None:
        """Run the main chat loop."""
        print("PDF Question Answering System")
        print("Type 'q' to quit")

        while True:
            print("\n" + "-" * 80)
            question = input("Ask your question about the PDFs (q to quit): ")
            print()

            if question.lower() == "q":
                break

            answer, _ = self.answer_question(question)

            print("Answer:")
            print(answer)
            print("\n" + "-" * 80)


if __name__ == "__main__":
    chatbot = PDFChatBot()
    chatbot.run_chat_loop()
