from src.pipeline.rag_pipeline import run_rag
from src.evaluation.evaluate import evaluate_answer, normalize_answer
from src.evaluation.ground_truth import get_ground_truth_most_common_lsoa

query = "Which location has the most theft cases?"

# 🔥 Automatically computed ground truth
expected_answer = get_ground_truth_most_common_lsoa()

generated = run_rag(query)

score = evaluate_answer(generated, expected_answer)
exact = normalize_answer(generated) == normalize_answer(expected_answer)

print("\nQuery:", query)
print("\nGenerated Answer:", generated)
print("\nGround Truth:", expected_answer)
print("\nSimilarity Score:", score)
print("Exact Match:", exact)