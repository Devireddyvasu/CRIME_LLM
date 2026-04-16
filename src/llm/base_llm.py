from transformers import pipeline

def load_base_llm():
    generator = pipeline(
        "text-generation",
        model="distilgpt2",
        max_new_tokens=80,
        do_sample=False,
        temperature=0.1,   
    )
    return generator