# from src.retrieval.retriever import load_retriever
# from src.reranking.reranker import rerank
# from src.routing.router import route_query, detect_intents
# from src.llm.finetuned_llm import llm_manager
# from src.analysis.global_analysis import (
#     get_most_common_location,
#     get_most_common_crime_type,
#     get_most_common_outcome
# )

# # Load retriever once
# retriever = load_retriever()

# def run_rag(query):
#     result = {
#         "query_type": "Unknown",
#         "context": "",
#         "answer": ""
#     }

#     # Step 1: Detect query type
#     query_type = route_query(query)
#     result["query_type"] = query_type
#     print("Query Type:", query_type)

#     #  STEP 2: HANDLE ANALYTICAL QUERY (MULTI-INTENT)
#     if query_type == "analytical_query":
#         print("[INFO] USING GLOBAL ANALYSIS")
#         intents = detect_intents(query)
#         answers = []

#         # 🔹 Location intent
#         if "location" in intents:
#             location = get_most_common_location()
#             answers.append(f"the area with the most crimes is {location}")

#         # 🔹 Crime type intent
#         if "crime_type" in intents:
#             crime_type = get_most_common_crime_type()
#             answers.append(f"the most common crime type is {crime_type}")

#         # 🔹 Outcome intent
#         if "outcome" in intents:
#             outcome = get_most_common_outcome()
#             answers.append(f"the most common outcome is {outcome}")

#         if answers:
#             ans_str = " and ".join(answers).capitalize() + "."
#             result["answer"] = ans_str
#             result["context"] = "Global Dataset Analysis"
#             return result

#         result["answer"] = "Not enough data."
#         return result

#     # 🔹 STEP 3: RETRIEVAL + RERANKING (RAG)
#     docs = retriever.invoke(query)
#     docs = rerank(query, docs)

#     # 🔹 STEP 4: BUILD CONTEXT
#     context = "\n".join([doc.page_content for doc in docs[:3]])
#     result["context"] = context

#     # 🔹 STEP 5 & 6: GENERATE RESPONSE using the active model
#     # (The active model is loaded from app.py before calling run_rag)
#     if not llm_manager.current_model_id:
#         result["answer"] = "Error: No LLM loaded. Please select a model."
#         return result

#     final_answer = llm_manager.generate_response(query, context)
#     result["answer"] = final_answer

#     return result

# from src.retrieval.retriever import load_retriever
# from src.llm.api_llm import APILLM


# class CrimeRAGPipeline:
#     def __init__(self, data_path, hf_token):
#         print("Initializing RAG Pipeline...")

#         self.retriever = load_retriever()
#         self.llm = APILLM(hf_token)

#         print("Pipeline Ready ✅")

#     def build_prompt(self, query, docs):
#         context = "\n\n".join(docs)

#         return f"""
# You are an expert crime analyst.

# Answer clearly based on the provided context.

# Context:
# {context}

# Question:
# {query}

# Answer:
# """


#     def run(self, query, model_type="llama", k=5):

#      results = self.retriever.invoke(query)

#      docs = [doc.page_content for doc in results]
#      metas = [doc.metadata for doc in results]

#      if not docs:
#         return "No relevant documents found.", [], []

#      prompt = self.build_prompt(query, docs)

#      answer = self.llm.generate(prompt, model_type=model_type)

#      return answer, docs, metas

from src.retrieval.retriever import load_retriever
from src.llm.base_llm import LocalLLM
from src.routing.router import route_query, detect_intents

from collections import Counter


# ==============================
# CLEAN DOCUMENT (FOR LLM)
# ==============================
def clean_doc(doc):
    text = doc.page_content

    important = []
    for line in text.split("\n"):
        if any(k in line for k in ["Crime Type", "Location", "Outcome"]):
            important.append(line.strip())

    return "\n".join(important)


# ==============================
# EXTRACT STATS FROM TEXT
# ==============================
def extract_stats(docs, query=""):
    crime_types = []
    outcomes = []
    locations = []
    lsoas = []
    wards = []
    output_areas = []
    
    q_lower = query.lower()
    filtered_docs = []
    
    for doc in docs:
        text_lower = doc.page_content.lower()
        
        # Simple heuristic filtering: if query specifically mentions a crime type, only include docs that have it.
        # This prevents "What is the number of shoplifting" from counting everything.
        skip = False
        for ct in ["shoplifting", "burglary", "robbery", "bicycle theft", "anti-social behaviour", "public order", "drugs", "possession of weapons", "criminal damage", "violence"]:
            if ct in q_lower and ct not in text_lower:
                skip = True
                break
        
        if "2025" in q_lower and "2025" not in text_lower: skip = True
        if "2024" in q_lower and "2024" not in text_lower: skip = True
        if "2023" in q_lower and "2023" not in text_lower: skip = True
        if "2022" in q_lower and "2022" not in text_lower: skip = True
        
        if not skip:
            filtered_docs.append(doc)
            
    # Fallback if filters are too strict
    if not filtered_docs:
        filtered_docs = docs
        
    for d in filtered_docs:
        text = d.page_content.lower()
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("crime type:"):
                val = line.split("crime type:")[-1].strip()
                if val and val not in ["unknown", "nan"]: crime_types.append(val)
            elif line.startswith("outcome:"):
                val = line.split("outcome:")[-1].strip()
                if val and val not in ["unknown", "nan"]: outcomes.append(val)
            elif line.startswith("location:") or "on or near" in line:
                val = line.split(":")[-1].strip()
                if val and val not in ["unknown", "no location", "nan"]: locations.append(val)
            elif line.startswith("lsoa:"):
                val = line.split("lsoa:")[-1].strip()
                if val and val not in ["unknown", "nan"]: lsoas.append(val)
            elif line.startswith("ward:"):
                val = line.split("ward:")[-1].strip()
                if val and val not in ["unknown", "nan"]: wards.append(val)
            elif line.startswith("output area:"):
                val = line.split("output area:")[-1].strip()
                if val and val not in ["unknown", "nan"]: output_areas.append(val)

    return Counter(crime_types), Counter(outcomes), Counter(locations), Counter(lsoas), Counter(wards), Counter(output_areas), filtered_docs


# ==============================
# MAIN PIPELINE CLASS
# ==============================
class CrimeRAGPipeline:
    def __init__(self):
        print("Initializing RAG Pipeline...")

        self.retriever = load_retriever()
        self.llm = LocalLLM()

        print("Pipeline Ready")

    def set_model(self, model_choice, hf_token=None):
        if "API Inference" in model_choice:
            repo_id = model_choice.split(" ")[0]
            from src.llm.api_llm import APILLM
            self.llm = APILLM(repo_id, hf_token)
            print(f"Switched to API Model: {repo_id}")
        else:
            from src.llm.base_llm import LocalLLM
            if not isinstance(self.llm, LocalLLM):
                self.llm = LocalLLM()
            print("Switched to Local Model (FLAN-T5)")

    # ==============================
    # PROMPT BUILDER (IMPROVED)
    # ==============================
    def build_prompt(self, query, docs):
        context = "\n\n".join([clean_doc(d) for d in docs])

        return f"""
You are a UK crime data analyst.

Using ONLY the data below:

- Identify most common crime types
- Identify most frequent outcomes
- Identify high-crime locations
- Describe overall crime patterns


Do NOT copy raw text.
Do NOT include IDs or coordinates.

Crime Data:
{context}

Question:
{query}

Answer:
"""

    # ==============================
    # MAIN RUN FUNCTION
    # ==============================
    def run(self, query, force_llm=False):

        print(f"\nQuery: {query}")

        # --------------------------
        # Step 1: Routing
        # --------------------------
        query_type = route_query(query)
        print(f"Query Type: {query_type}")

        # --------------------------
        # Step 2: Dynamic k
        # --------------------------
        if query_type == "analytical":
            k = 100
        else:
            k = 10

        # --------------------------
        # Step 3: Retrieval
        # --------------------------
        docs = self.retriever.invoke(query)
        docs = [d for d in docs if d.page_content.strip()][:k]

        if not docs:
            return "❌ No relevant documents found.", [], []

        print(f"Retrieved: {len(docs)} docs")

        # --------------------------
        # Step 4: ANALYTICAL (NO LLM)
        # --------------------------
        if query_type == "analytical" and not force_llm:
            intents = detect_intents(query)
            crime_counts, outcome_counts, location_counts, lsoa_counts, ward_counts, oa_counts, filtered_docs = extract_stats(docs, query)

            answers = []
            
            # COUNTING INTENT
            if "count" in intents:
                answers.append(f"Based on the data retrieved, there are {len(filtered_docs)} matching records.")
            
            # RANKING INTENT
            elif "rank" in intents:
                if "lsoa" in intents and lsoa_counts:
                    top_n = lsoa_counts.most_common(10)
                    ranks = "\n".join([f"{i+1}. {x[0].title()} ({x[1]} cases)" for i, x in enumerate(top_n)])
                    answers.append(f"Top LSOAs:\n{ranks}")
                elif "ward" in intents and ward_counts:
                    top_n = ward_counts.most_common(10)
                    ranks = "\n".join([f"{i+1}. {x[0].title()} ({x[1]} cases)" for i, x in enumerate(top_n)])
                    answers.append(f"Top Wards:\n{ranks}")
                elif "crime_type" in intents and crime_counts:
                    top_n = crime_counts.most_common(10)
                    ranks = "\n".join([f"{i+1}. {x[0].title()} ({x[1]} cases)" for i, x in enumerate(top_n)])
                    answers.append(f"Top Crime Types:\n{ranks}")
                elif location_counts:
                    top_n = location_counts.most_common(10)
                    ranks = "\n".join([f"{i+1}. {x[0].title()} ({x[1]} cases)" for i, x in enumerate(top_n)])
                    answers.append(f"Top Locations:\n{ranks}")
                else:
                    answers.append("Insufficient data to rank the requested entity.")
            
            # STANDARD ANALYTICAL INTENTS
            else:
                top_crime = crime_counts.most_common(1) if crime_counts else [("N/A", 0)]
                top_location = location_counts.most_common(1) if location_counts else [("N/A", 0)]
                top_outcome = outcome_counts.most_common(1) if outcome_counts else [("N/A", 0)]
                top_lsoa = lsoa_counts.most_common(1) if lsoa_counts else [("N/A", 0)]
                
                if "lsoa" in intents:
                    answers.append(f"The LSOA with the highest reports is {top_lsoa[0][0].title()} with {top_lsoa[0][1]} cases.")
                elif "ward" in intents:
                    answers.append(f"The Ward with the highest reports is {ward_counts.most_common(1)[0][0].title()} with {ward_counts.most_common(1)[0][1]} cases.") if ward_counts else answers.append("No ward data found.")
                elif "location" in intents or not intents:
                    answers.append(f"The area with the highest crimes is {top_location[0][0].title()} with {top_location[0][1]} incidents.")
                
                if "crime_type" in intents or not intents:
                    answers.append(f"The most common crime type is {top_crime[0][0].title()} with {top_crime[0][1]} cases.")
                if "outcome" in intents or not intents:
                    answers.append(f"The most frequent outcome is '{top_outcome[0][0].title()}' with {top_outcome[0][1]} cases.")
                
            answer = "\n".join(answers)

            return answer, filtered_docs, [doc.metadata for doc in filtered_docs]
        # --------------------------
        # Step 5: LOCATION
        # --------------------------
        elif query_type == "location":

            _, _, location_counts, _, _, _, _ = extract_stats(docs, query)

            answer = f"""
Top locations:
{location_counts.most_common(3)}
"""

            return answer, docs, [doc.metadata for doc in docs]

        # --------------------------
        # Step 6: GENERAL → LLM
        # --------------------------
        else:
            # --------------------------
            # Step 6: CONTEXT INJECTION
            # --------------------------
            prompt = self.build_prompt(query, docs)

            print("CONTEXT INJECTION INTO LLM")
            print("="*80)
            print(prompt[:1200])   # limit for clean screenshot

            # --------------------------
            # Step 7: FINAL RESPONSE
            # --------------------------
            answer = self.llm.generate(prompt)

            print("FINAL SYSTEM RESPONSE")
            print("="*80)
            print(answer)

            return answer, docs, [doc.metadata for doc in docs]