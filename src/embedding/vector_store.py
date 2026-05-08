import json
import os
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from src.embedding.embedder import load_embedding_model


def create_vector_db():

    # Load processed documents
    with open("data/processed/documents.json", "r") as f:
        texts = json.load(f)

    print(f"Loaded {len(texts)} documents")

    # Convert to LangChain format
    documents = [Document(page_content=text) for text in texts]

    # Load embedding model
    embedding_model = load_embedding_model()

    # Create vector DB
    db = FAISS.from_documents(documents, embedding_model)

    # Save locally
    os.makedirs("vector_db", exist_ok=True)
    db.save_local("vector_db/")

    print("✅ Vector DB created and saved!")


if __name__ == "__main__":
    create_vector_db()