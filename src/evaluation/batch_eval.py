import os
import sys
import json
import random
import pandas as pd
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.pipeline.rag_pipeline import CrimeRAGPipeline
from src.evaluation.evaluate_rag import evaluate_response

def load_real_test_cases(file_path, num_samples=10):
    from collections import Counter
    with open(file_path, 'r', encoding='utf-8') as f:
        docs = json.load(f)
    
    # We will build a dictionary to group crimes by location
    location_data = {}
    
    for doc in docs:
        lines = doc.split("\n")
        location = ""
        crime_type = ""
        outcome = ""
        
        for line in lines:
            if "Location:" in line:
                location = line.split("Location:")[-1].strip()
            elif "Crime Type:" in line:
                crime_type = line.split("Crime Type:")[-1].strip()
            elif "Outcome:" in line:
                outcome = line.split("Outcome:")[-1].strip()
        
        if location and location.lower() not in ["no location", "unknown"]:
            if location not in location_data:
                location_data[location] = {"crime_types": [], "outcomes": []}
            if crime_type and crime_type.lower() != "unknown":
                location_data[location]["crime_types"].append(crime_type)
            if outcome and outcome.lower() != "unknown":
                location_data[location]["outcomes"].append(outcome)
                
    # Filter for locations with a decent amount of data (e.g., > 3 crimes) to test aggregation properly
    valid_locations = [loc for loc in location_data if len(location_data[loc]["crime_types"]) > 3]
    
    random.seed(42) # For reproducibility
    random.shuffle(valid_locations)
    
    test_cases = []
    
    for loc in valid_locations:
        if len(test_cases) >= num_samples:
            break
            
        crime_counts = Counter(location_data[loc]["crime_types"])
        outcome_counts = Counter(location_data[loc]["outcomes"])
        
        # Randomly choose between testing Crime Type aggregation or Outcome aggregation
        if random.choice([True, False]) and crime_counts:
            # The mathematical true most common crime
            true_most_common = crime_counts.most_common(1)[0][0]
            query = f"What is the most common crime type {loc.lower()}?"
            ground_truth = true_most_common
        else:
            if outcome_counts:
                true_most_common = outcome_counts.most_common(1)[0][0]
                query = f"What is the most frequent outcome {loc.lower()}?"
                ground_truth = true_most_common
            else:
                continue
                
        test_cases.append({
            "query": query,
            "ground_truth": ground_truth
        })
        
    return test_cases

def run_batch_evaluation():
    doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/documents.json'))
    
    print("Loading real test cases from dataset...")
    test_cases = load_real_test_cases(doc_path, num_samples=10)
    
    print("Initializing RAG Pipeline for Batch Evaluation...")
    pipeline = CrimeRAGPipeline()
    
    results = []
    
    for tc in test_cases:
        query = tc["query"]
        ground_truth = tc["ground_truth"]
        
        print(f"\nProcessing query: '{query}'...")
        answer, docs, metas = pipeline.run(query)
        
        metrics = evaluate_response(query, answer, docs)
        
        # Calculate Exact Match
        exact_match = False
        if answer and ground_truth.lower() in answer.lower():
            exact_match = True
            
        # Build result row mapped to the old headers + new rich metrics
        row = {
            "Query": query,
            "Generated_Answer": answer,
            "Ground_Truth": ground_truth,
            "Similarity_Score": metrics.get("semantic_relevance", 0),
            "Exact_Match": exact_match,
            "Faithfulness": metrics.get("faithfulness", 0),
            "Retrieval_Precision": metrics.get("retrieval_precision", "N/A"),
            "Word_Count": metrics.get("word_count", 0),
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        results.append(row)
        
    df = pd.DataFrame(results)
    
    # Overwrite the evaluation_results.csv directly
    output_path = os.path.join(os.path.dirname(__file__), "evaluation_results.csv")
    
    try:
        df.to_csv(output_path, index=False)
        print(f"\nBatch evaluation complete! Results successfully saved to: {output_path}")
    except PermissionError:
        # If the file is open in Excel or another program, save it with a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = os.path.join(os.path.dirname(__file__), f"evaluation_results_{timestamp}.csv")
        df.to_csv(fallback_path, index=False)
        print(f"\n⚠️ Permission Denied: Could not overwrite {output_path} because it is open in another program.")
        print(f"✅ Results successfully saved to alternative file: {fallback_path}")

if __name__ == "__main__":
    run_batch_evaluation()