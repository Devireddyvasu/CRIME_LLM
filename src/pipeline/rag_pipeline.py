from src.retrieval.retriever import load_retriever
from src.reranking.reranker import rerank
from src.routing.router import route_query, detect_intents
from src.llm.base_llm import load_base_llm
from src.analysis.global_analysis import (
    get_most_common_location,
    get_most_common_crime_type,
    get_most_common_outcome
)

# Load components once
retriever = load_retriever()
llm = load_base_llm()


def run_rag(query):

    # Step 1: Detect query type
    query_type = route_query(query)
    print("Query Type:", query_type)

    #  STEP 2: HANDLE ANALYTICAL QUERY (MULTI-INTENT)
    if query_type == "analytical_query":
        print("✅ USING GLOBAL ANALYSIS")

        intents = detect_intents(query)
        answers = []

        # 🔹 Location intent
        if "location" in intents:
            location = get_most_common_location()
            answers.append(f"the area with the most crimes is {location}")

        # 🔹 Crime type intent
        if "crime_type" in intents:
            crime_type = get_most_common_crime_type()
            answers.append(f"the most common crime type is {crime_type}")

        # 🔹 Outcome intent
        if "outcome" in intents:
            outcome = get_most_common_outcome()
            answers.append(f"the most common outcome is {outcome}")

        # 🔹 Combine answers
        if answers:
            return " and ".join(answers).capitalize() + "."

        return "Not enough data."

    # 🔹 STEP 3: RETRIEVAL + RERANKING (RAG)
    docs = retriever.get_relevant_documents(query)
    docs = rerank(query, docs)

    # 🔹 STEP 4: BUILD CONTEXT
    context = "\n".join([doc.page_content for doc in docs[:3]])

    # 🔹 STEP 5: PROMPT
    prompt = f"""
You are an intelligent crime analysis assistant.

Use ONLY the provided data.
If the answer is not found, say "Not enough data".

Context:
{context}

Question: {query}

Answer:
"""

    # 🔹 STEP 6: GENERATE RESPONSE
    response = llm(prompt)[0]["generated_text"]

    # 🔹 STEP 7: CLEAN OUTPUT
    final_answer = response.replace(prompt, "").strip()

    return final_answer