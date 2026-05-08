import sys
import os
import torch

# Fix PyTorch Memory Fragmentation and HF Warnings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from src.pipeline.rag_pipeline import CrimeRAGPipeline
from src.evaluation.evaluate_rag import evaluate_response

st.set_page_config(page_title="Crime Data QA System", layout="wide")

# Sidebar for Model Selection
st.sidebar.title("⚙️ System Configuration")
st.sidebar.markdown("Select the model to run the RAG pipeline.")

model_choice = st.sidebar.selectbox(
    "Active Model",
    (
        "Local LLM (FLAN-T5) - Memory Efficient",
        "VasuReddy07/Llama-3.2-Crime-QA (API Inference)",
        "VasuReddy07/Qwen-2.5-Crime-QA (API Inference)"
    )
)

st.sidebar.divider()
hf_token = st.sidebar.text_input("🔑 HuggingFace API Key", type="password", help="Required to use the LLaMA or Qwen finetuned models via Cloud API.")

# VRAM Monitor in Sidebar
if torch.cuda.is_available():
    st.sidebar.divider()
    st.sidebar.subheader("📊 System Resources")
    vram_used = torch.cuda.memory_allocated() / 1024**3
    vram_reserved = torch.cuda.memory_reserved() / 1024**3
    st.sidebar.write(f"**VRAM Allocated:** {vram_used:.2f} GB")
    st.sidebar.write(f"**VRAM Reserved:** {vram_reserved:.2f} GB")
    if vram_used > 6:
        st.sidebar.warning("High VRAM usage detected!")

st.title("🔍 Intelligent Crime Data QA System")
st.markdown("Ask questions about crime data using Hybrid RAG + Custom Fine-Tuned ODIN Models")

# Initialize Pipeline — no caching so model choice is always applied fresh
pipeline = CrimeRAGPipeline()

# Input box
query = st.text_input("Enter your question about crime data:")

if query:
    if "API Inference" in model_choice and not hf_token:
        st.warning("⚠️ **HuggingFace API Key** is required to use the Finetuned Models. Please enter it in the sidebar. This ensures your local system does not crash!")
    else:
        with st.spinner("Processing RAG pipeline..."):
            pipeline.set_model(model_choice, hf_token)
            answer, docs, metas = pipeline.run(query)
            
            # Evaluate the response
            metrics = evaluate_response(query, answer, docs)

        # Display results
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("📌 Retrieved Context")
            with st.expander("View Retrieved Sources"):
                for i, doc in enumerate(docs[:5]):
                    content = doc.page_content.strip()
                    
                    # Extract location dynamically from the text since metadata is empty
                    display_loc = "Unknown"
                    if "Location:" in content:
                        try:
                            display_loc = content.split("Location:")[1].split(" LSOA:")[0].split(" Crime Type:")[0].strip()
                        except IndexError:
                            pass
                    
                    st.markdown(f"**Source {i+1} (Location: {display_loc})**")
                    if content:
                        st.write(content)
                    else:
                        st.write("No textual content available.")
                    st.divider()
                
                if len(docs) > 5:
                    st.info(f"...and {len(docs) - 5} more sources used for this analysis.")

            st.subheader("📊 Evaluation Metrics")
            with st.expander("View Quality Metrics"):
                st.json(metrics)

        with col2:
            st.subheader(f"💡 AI Response")
            st.success(answer)