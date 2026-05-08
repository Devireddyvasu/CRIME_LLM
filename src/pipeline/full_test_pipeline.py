from src.pipeline.rag_pipeline import CrimeRAGPipeline
from src.evaluation.evaluate_rag import evaluate_response

# 🔑 Add your HF token if using API
HF_TOKEN = "your_token_here"

# Initialize pipeline
rag = CrimeRAGPipeline()


def run_full_pipeline(query):

    print("\n" + "="*60)
    print(f"🔍 Query: {query}")

    # 🔥 Use FULL RAG pipeline (IMPORTANT)
    answer, docs, _ = rag.run(query)

    print("\n🧠 Final Answer:\n")
    print(answer)
    metrics = evaluate_response(query, answer, docs)

    print("\n📊 Evaluation:\n")
    print(metrics)

    print("\n📚 Top Retrieved Docs:\n")
    for i, doc in enumerate(docs[:2]):
        print(f"\n--- Doc {i+1} ---")
        print(doc.page_content[:300])


# 🔥 TEST QUERIES
queries = [
    "Which area has the most theft cases?",
    "What are the most common crime types?",
    "Explain crime patterns in Cambridge",
    "What outcomes are most frequent?"
]

for q in queries:
    run_full_pipeline(q)