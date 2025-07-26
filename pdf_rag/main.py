import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from vector import retriever

load_dotenv()

model = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
)

template = """
You are an expert assistant that answers questions based on the provided
PDF documents.

Here are relevant excerpts from the documents: {context}

Question: {question}

Please provide a comprehensive answer based on the context above.
If the answer cannot be found in the provided context, please say so.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


def format_docs(docs):
    """Format retrieved documents for the prompt"""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        content = doc.page_content.strip()
        formatted.append(f"[Source: {source}, Page: {page}]\n{content}")
    return "\n\n".join(formatted)


while True:
    print("\n" + "-" * 80)
    question = input("Ask your question about the PDFs (q to quit): ")
    print("\n")
    if question == "q":
        break

    # Retrieve relevant documents
    docs = retriever.invoke(question)
    context = format_docs(docs)

    # Generate response
    result = chain.invoke({"context": context, "question": question})
    print("Answer:")
    print(result.content)

    # Optionally show sources
    print("\n" + "-" * 40)
    print("Sources:")
    sources = set()
    for doc in docs:
        source_file = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        sources.add(f"- {source_file} (Page {page})")

    for source in sorted(sources):
        print(source)
