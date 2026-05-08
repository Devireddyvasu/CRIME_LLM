from src.retrieval.retriever import load_retriever

retriever = load_retriever()

query = "theft in cambridgeshire in 2025"

docs = retriever.invoke(query)

for i, doc in enumerate(docs):
    print(f"\nResult {i+1}:")
    print(doc.page_content[:300])