from src.pipeline.rag_pipeline import run_rag
from src.evaluation.evaluate import evaluate_answer, normalize_answer
from src.evaluation.ground_truth import get_ground_truth_location


# 🔥 Get ground truth dynamically
expected = get_ground_truth_location()

test_queries = [
    "Which location has the most theft cases?",
    "Where do most theft crimes occur?",
    "Top area for theft incidents?",
]

total_score = 0
exact_matches = 0

for query in test_queries:

    generated = run_rag(query)

    score = evaluate_answer(generated, expected)
    exact = normalize_answer(generated) == normalize_answer(expected)

    total_score += score
    if exact:
        exact_matches += 1

    print("\n---------------------------")
    print("Query:", query)
    print("Generated:", generated)
    print("Ground Truth:", expected)
    print("Score:", score)
    print("Exact Match:", exact)


# Final results
avg_score = total_score / len(test_queries)
accuracy = exact_matches / len(test_queries)

print("\n===========================")
print("Average Similarity:", avg_score)
print("Exact Match Accuracy:", accuracy)