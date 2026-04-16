import csv
import time

from src.pipeline.rag_pipeline import run_rag
from src.evaluation.evaluate import evaluate_answer, normalize_answer
from src.analysis.global_analysis import (
    get_most_common_location,
    get_most_common_crime_type,
    get_most_common_outcome
)

# 🔹 Queries to evaluate
queries = [
    "Which location has the most theft cases?",
    "Where do crimes occur most?",
    "Top area for theft incidents?",
    "What is the most common crime type?",
    "What is the most common outcome of crimes?",
    "Where do crimes occur most and what type of crimes are common?"
]

# 🔹 Accuracy counters
total_queries = len(queries)
correct_matches = 0

# 🔹 Open CSV file
with open("evaluation_results.csv", "w", newline="", encoding="utf-8") as file:

    writer = csv.writer(file)

    # 🔹 Header
    writer.writerow([
        "Query",
        "Generated_Answer",
        "Ground_Truth",
        "Similarity_Score",
        "Exact_Match",
        "Response_Time_Seconds"
    ])

    # 🔥 Process each query
    for query in queries:

        # 🔹 Start timing
        start_time = time.time()

        generated = run_rag(query)

        # 🔥 Clean debug text if present
        generated = generated.replace("# 🔥 add this", "").strip()

        # 🔹 End timing
        end_time = time.time()
        response_time = round(end_time - start_time, 3)

        q_lower = query.lower()

        # 🔥 Assign correct ground truth

        # Multi-intent query
        if "and" in q_lower:
            loc = get_most_common_location()
            crime = get_most_common_crime_type()
            ground_truth = f"{loc} and {crime}"

        # Outcome queries
        elif "outcome" in q_lower or "result" in q_lower:
            ground_truth = get_most_common_outcome()

        # Crime type queries
        elif any(phrase in q_lower for phrase in [
            "crime type", "type of crime", "which crime"
        ]):
            ground_truth = get_most_common_crime_type()

        # Location queries
        elif any(word in q_lower for word in ["where", "location", "area"]):
            ground_truth = get_most_common_location()

        # Default fallback
        else:
            ground_truth = get_most_common_location()

        # 🔥 Clean ground truth debug text if present
        ground_truth = ground_truth.replace("# 🔥 ADD THIS", "").strip()

        # 🔹 Evaluate
        score = evaluate_answer(generated, ground_truth)
        exact = normalize_answer(generated) == normalize_answer(ground_truth)

        # 🔹 Count accuracy
        if exact:
            correct_matches += 1

        # 🔹 Write row
        writer.writerow([
            query,
            generated,
            ground_truth,
            round(score, 3),
            exact,
            response_time
        ])

# 🔹 Final Accuracy
accuracy = correct_matches / total_queries

print("✅ CSV file generated successfully: evaluation_results.csv")
print(f"🎯 Overall Accuracy: {accuracy:.2%}")