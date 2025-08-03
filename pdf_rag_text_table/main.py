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
            content_type = doc.metadata.get("content_type", "text")
            content = doc.page_content.strip()

            if content_type == "table":
                table_idx = doc.metadata.get("table_index", "Unknown")
                header = f"[Table {table_idx} from {source}, Page {page}]"
            else:
                header = f"[Source: {source}, Page: {page}]"

            formatted.append(f"{header}\n{content}")
        return "\n\n".join(formatted)

    def get_unique_sources(self, docs: List[Document]) -> List[str]:
        """Get unique sources from retrieved documents."""
        sources = set()
        for doc in docs:
            source_file = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            content_type = doc.metadata.get("content_type", "text")

            if content_type == "table":
                table_idx = doc.metadata.get("table_index", "Unknown")
                source_str = (
                    f"- {source_file} (Page {page}, Table {table_idx})"
                )
            else:
                source_str = f"- {source_file} (Page {page})"

            sources.add(source_str)
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
        print("PDF Question Answering System (with Table Support)")
        print("Type 'q' to quit")

        while True:
            print("\n" + "-" * 80)
            question = input("Ask your question about the PDFs (q to quit): ")
            print()

            if question.lower() == "q":
                break

            answer, sources = self.answer_question(question)

            print("Answer:")
            print(answer)

            if sources:
                print("\nSources:")
                for source in sources:
                    print(source)

            print("\n" + "-" * 80)


if __name__ == "__main__":
    chatbot = PDFChatBot()
    chatbot.run_chat_loop()
