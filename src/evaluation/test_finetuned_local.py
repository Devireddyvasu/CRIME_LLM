import sys
import os
import pandas as pd
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.pipeline.rag_pipeline import CrimeRAGPipeline
from src.evaluation.evaluate_rag import evaluate_response
from src.llm.finetuned_llm import llm_manager
from src.evaluation.odintrainqueries import queries

def run_local_finetuned_eval(model_choice="VasuReddy07/Llama-3.2-Crime-QA"):
    print(f"\n--- Starting Safe Local Evaluation for {model_choice} ---")
    print("This script uses 4-bit quantization to prevent OOM errors.")
    
    # Load the heavy model using the fixed memory-safe loader
    llm_manager.load_model(model_choice)
    
    # Initialize the RAG Pipeline (which still defaults to pure math/FLAN-T5 for general app usage)
    print("Initializing RAG context retriever...")
    pipeline = CrimeRAGPipeline()
    results = []
    
    for q in queries:
        print(f"\nProcessing query: '{q}'...")
        
        # 1. Retrieve the context via the pipeline (force_llm=True skips math engine)
        ans_fallback, docs, metas = pipeline.run(q, force_llm=True)
        
        # 2. Extract the context text
        context_text = "\n\n".join([d.page_content for d in docs])
        
        # 3. Generate the response physically using the memory-safe LLaMA/Qwen model
        ans_finetuned = llm_manager.generate_response(q, context_text)
        
        print(f"Generated Answer: {ans_finetuned[:100]}...")
        
        # Calculate full metrics
        metrics = evaluate_response(q, ans_finetuned, docs)
        
        row = {
            "Query": q,
            "Generated_Answer": ans_finetuned,
            "Ground_Truth": "",  # Left empty for manual user input
            "Similarity_Score": metrics.get("semantic_relevance", 0),
            "Exact_Match": "N/A", 
            "Faithfulness": metrics.get("faithfulness", 0),
            "Retrieval_Precision": metrics.get("retrieval_precision", "N/A"),
            "Word_Count": metrics.get("word_count", 0),
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        results.append(row)
        
    df = pd.DataFrame(results)
    
    # Save to a dedicated file so we don't overwrite the math engine's CSV
    output_path = os.path.join(os.path.dirname(__file__), "llm_finetuned_evaluation.csv")
    
    try:
        df.to_csv(output_path, index=False)
        print(f"\n✅ Local LLM evaluation complete! Results successfully saved to: {output_path}")
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = os.path.join(os.path.dirname(__file__), f"llm_finetuned_evaluation_{timestamp}.csv")
        df.to_csv(fallback_path, index=False)
        print(f"\n⚠️ Permission Denied: Could not overwrite {output_path}.")
        print(f"✅ Results successfully saved to alternative file: {fallback_path}")

if __name__ == "__main__":
    # You can change this to "VasuReddy07/Qwen-2.5-Crime-QA" to test the other model
    run_local_finetuned_eval("VasuReddy07/Llama-3.2-Crime-QA")
