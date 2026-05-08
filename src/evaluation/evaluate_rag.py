# import pandas as pd
# from datasets import Dataset
# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy
# from src.pipeline.rag_pipeline import run_rag
# from src.llm.finetuned_llm import llm_manager
# import os

# # Sample Test Dataset
# test_questions = [
#     "What type of crime occurred on or near Dunbar Gardens?",
#     "Explain the outcome of the crime on Staploe Lane.",
#     "Which area has the most crimes reported?",
# ]

# def run_evaluation(model_choice="VasuReddy07/Llama-3.2-Crime-QA"):
#     print(f"Loading model: {model_choice}...")
#     llm_manager.load_model(model_choice)
    
#     data = {"question": [], "answer": [], "contexts": []}
    
#     print("Generating responses for test set...")
#     for q in test_questions:
#         res = run_rag(q)
#         data["question"].append(q)
#         data["answer"].append(res["answer"])
#         data["contexts"].append([res["context"]])
        
#     dataset = Dataset.from_dict(data)
    
#     print("Running Ragas Evaluation (Faithfulness & Answer Relevancy)...")
#     try:
#         # Note: Ragas requires an OpenAI API key or a local judge model configured to run.
#         # Ensure your environment variables are set if using OpenAI for judging.
#         result = evaluate(
#             dataset=dataset,
#             metrics=[faithfulness, answer_relevancy]
#         )
        
#         df = result.to_pandas()
#         df["model"] = model_choice
        
#         output_path = "evaluation_results.csv"
#         if os.path.exists(output_path):
#             existing_df = pd.read_csv(output_path)
#             df = pd.concat([existing_df, df], ignore_index=True)
            
#         df.to_csv(output_path, index=False)
#         print(f"Evaluation complete. Results saved to {output_path}")
#         print(result)
        
#     except Exception as e:
#         print(f"Evaluation failed (did you set up your LLM judge API key for Ragas?): {e}")

# if __name__ == "__main__":
#     # Example usage: Evaluate Llama first, then Qwen
#     # run_evaluation("VasuReddy07/Llama-3.2-Crime-QA")
#     # run_evaluation("VasuReddy07/Qwen-2.5-Crime-QA")
#     print("Evaluation script initialized. Uncomment the models in main to run.")
import re
from sentence_transformers import SentenceTransformer, util

# Load once (important)
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

# ==============================
# BASIC TEXT METRICS
# ==============================
def basic_metrics(answer):
    return {
        "length": len(answer),
        "word_count": len(answer.split()),
        "has_numbers": bool(re.search(r"\d", answer)),
        "has_crime_keyword": any(
            word in answer.lower()
            for word in ["crime", "theft", "robbery", "violence"]
        )
    }


# ==============================
# RELEVANCE (simple proxy)
# ==============================
def relevance_score(query, answer):
    query_words = set(query.lower().split())
    answer_words = set(answer.lower().split())

    overlap = query_words.intersection(answer_words)

    return len(overlap) / len(query_words) if query_words else 0


# ==============================
# CONTEXT MATCH (faithfulness)
# ==============================
def context_match(answer, docs):
    context_text = " ".join([d.page_content.lower() for d in docs])

    matched_words = sum(
        1 for word in answer.lower().split()
        if word in context_text
    )

    return matched_words / len(answer.split()) if answer else 0

def dynamic_keyword(query):
    q = query.lower()

    if "theft" in q:
        return "theft"
    elif "outcome" in q:
        return "outcome"
    elif "violence" in q:
        return "violence"
    elif "pattern" in q or "explain" in q:
        return None   # 👈 important
    else:
        return None
# ==============================
# FINAL EVALUATION FUNCTION
# ==============================
def evaluate_response(query, answer, docs):
    metrics = {}

    metrics.update(basic_metrics(answer))

    # semantic relevance
    metrics["semantic_relevance"] = semantic_relevance(query, answer)

    # faithfulness
    metrics["faithfulness"] = round(context_match(answer, docs), 3)

    # 🔥 dynamic retrieval precision
    keyword = dynamic_keyword(query)

    if keyword:
        metrics["retrieval_precision"] = round(retrieval_precision(docs, keyword), 3 )
    else:
     metrics["retrieval_precision"] = "N/A"

    return metrics
def retrieval_precision(docs, keyword="theft"):
    relevant = sum(
        1 for d in docs
        if keyword in d.page_content.lower()
    )
    return relevant / len(docs) if docs else 0
def semantic_relevance(query, answer):
    q_emb = semantic_model.encode(query, convert_to_tensor=True)
    a_emb = semantic_model.encode(answer, convert_to_tensor=True)

    score = util.cos_sim(q_emb, a_emb).item()
    return round(score, 3)