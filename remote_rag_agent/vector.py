import os

import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()

df = pd.read_csv("./remote_rag_agent/realistic_restaurant_reviews.csv")

db_location = "./remote_rag_agent/chrome_langchain_db"

add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for row_index, row in df.iterrows():
        document = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id=str(row_index),
        )
        ids.append(str(row_index))
        documents.append(document)

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
)

vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings,
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
