from transformers import T5ForConditionalGeneration, AutoTokenizer

class LocalLLM:
    def __init__(self):
        print("Loading local fallback LLM (FLAN-T5-base)...")
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
        print("Fallback LLM loaded successfully.")

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=150, 
            temperature=0.1, 
            do_sample=False
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)