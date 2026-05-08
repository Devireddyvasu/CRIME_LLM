from src.retrieval.retriever import load_retriever
from src.reranking.reranker import rerank
from src.routing.router import route_query

retriever = load_retriever()

query = "Which area has the most theft cases?"

# Step 1: Routing
query_type = route_query(query)
print(f"\nQuery Type: {query_type}")

# Step 2: Retrieval
docs = retriever.invoke(query)

# Step 3: Re-ranking
docs = rerank(query, docs)

# Step 4: Show results
for i, doc in enumerate(docs[:3]):
    print("query:", query)
    print(f"\nResult {i+1}:")
    print(doc.page_content[:300])