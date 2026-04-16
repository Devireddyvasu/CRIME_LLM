import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from src.pipeline.rag_pipeline import run_rag
from src.routing.router import route_query
st.set_page_config(page_title="Crime Data QA System", layout="wide")

st.title("🔍 Intelligent Crime Data QA System")
st.markdown("Ask questions about crime data using RAG + LLM")

# Input box
query = st.text_input("Enter your question:")

if query:

    # Detect query type
    query_type = route_query(query)

    # Get answer
    answer = run_rag(query)

    # Display results
    st.subheader("📌 Query Type")
    st.write(query_type)

    st.subheader("💡 Answer")
    st.success(answer)