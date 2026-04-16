from src.pipeline.rag_pipeline import run_rag

query = "Which location has the most theft cases?"

response = run_rag(query)

print("\nFinal Answer:\n")
print(response)