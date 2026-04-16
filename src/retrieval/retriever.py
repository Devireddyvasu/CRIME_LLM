from langchain.vectorstores import FAISS
from src.embedding.embedder import load_embedding_model

def load_retriever():
    embedding_model = load_embedding_model()

    db = FAISS.load_local(
        "vector_db/",
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return db.as_retriever(search_kwargs={"k": 5})