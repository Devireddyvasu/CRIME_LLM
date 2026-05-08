import sys
import os
import pandas as pd
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.pipeline.rag_pipeline import CrimeRAGPipeline
from src.evaluation.evaluate_rag import evaluate_response

queries = [
    "1. What is the number of ASB reports in Petersfield Ward for the months of January, February and March 2025",
    "2. What is the number of ASB reports in Castle Ward for 2025",
    "3. Which LSOA in Cambridgeshire has the most reports of shoplifting in 2025",
    "4. Rank the top 10 LSOA in Cambridgeshire for Shoplifting in 2025",
    "5. Which Output Area in Market Ward has the most reports of shoplifting",
    "6. Identify the top Output Area in Huntingdon East Ward for Shoplifting",
    "7. Rank the top LSOA in Cambridgeshire for Burglary offences in April, May and June 2025",
    "8. How many bicycle thefts occurred in East Chesterton Ward from January to June 2025",
    "9. Which LSOA had the most robbery offences in 2025",
    "10. Which Ward had the most robbery offences in 2025",
    "11. Which Output Area had the most robbery offences in 2025",
    "12. Which Local Authority had the most robbery offences in 2025",
    "13. Which LSOA had the most vehicle crime in 2025",
    "14. Which Ward had the most vehicle crime in 2025",
    "15. Which Output Area had the most vehicle crime in 2025",
    "16. Which Local Authority had the most vehicle crime in 2025",
    "17. Which LSOA had the most public order offences in 2025",
    "18. Which Ward had the most public order offences in 2025",
    "19. Which Output Area had the most public order offences in 2025",
    "20. Which Local Authority had the most public order offences in 2025",
    "21. Which LSOA had the most drug offences in 2023",
    "22. Which Ward had the most drug offences in 2023",
    "23. Which Output Area had the most drug offences in 2023",
    "24. Which Local Authority had the most drug offences in 2023",
    "25. Which LSOA had the most possession of weapons crime in 2023",
    "26. Which Ward had the most possession of weapons crime in 2023",
    "27. Which Output Area had the most possession of weapons crime in 2023",
    "28. Which Local Authority had the most possession of weapons crime in 2023",
    "29. Which LSOA had the most burglary offences in 2023",
    "30. Which Ward had the most burglary offences in 2023",
    "31. Which Output Area had the most burglary offences in 2023",
    "32. Which Local Authority had the most burglary offences in 2023",
    "33. Which LSOA had the most criminal damage and arson offences in 2022",
    "34. Which Ward had the most criminal damage and arson offences in 2022",
    "35. Which Output Area had the most criminal damage and arson offences in 2022",
    "36. Which Local Authority had the most criminal damage and arson offences in 2022",
    "37. Which LSOA had the most violence and sexual offences in 2024",
    "38. Which Ward had the most violence and sexual offences in 2024",
    "39. Which Output Area had the most violence and sexual offences in 2024",
    "40. Which Local Authority had the most violence and sexual offences in 2024"
]

def run_odin_evaluation():
    print("Initializing RAG Pipeline for ODIN Evaluation...")
    pipeline = CrimeRAGPipeline()
    results = []
    
    for q in queries:
        print(f"\nProcessing query: '{q}'...")
        ans, docs, metas = pipeline.run(q)
        
        # Calculate full metrics
        metrics = evaluate_response(q, ans, docs)
        
        row = {
            "Query": q,
            "Generated_Answer": ans,
            "Ground_Truth": "",  # Left empty for manual user input
            "Similarity_Score": metrics.get("semantic_relevance", 0),
            "Exact_Match": "N/A", # Can't calculate without ground truth
            "Faithfulness": metrics.get("faithfulness", 0),
            "Retrieval_Precision": metrics.get("retrieval_precision", "N/A"),
            "Word_Count": metrics.get("word_count", 0),
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        results.append(row)
        
    df = pd.DataFrame(results)
    
    output_path = os.path.join(os.path.dirname(__file__), "custom_evaluation_results.csv")
    
    try:
        df.to_csv(output_path, index=False)
        print(f"\nODIN evaluation complete! Results successfully saved to: {output_path}")
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = os.path.join(os.path.dirname(__file__), f"custom_evaluation_results_{timestamp}.csv")
        df.to_csv(fallback_path, index=False)
        print(f"\nPermission Denied: Could not overwrite {output_path}.")
        print(f"Results successfully saved to alternative file: {fallback_path}")

if __name__ == "__main__":
    run_odin_evaluation()
