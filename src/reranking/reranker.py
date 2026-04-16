from sentence_transformers import CrossEncoder

# Load once (important for performance)
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query, docs):
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker_model.predict(pairs)

    ranked_docs = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, score in ranked_docs]