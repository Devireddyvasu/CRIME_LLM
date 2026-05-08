import torch
import gc
import os
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import pipeline

# Set HuggingFace cache directory to D: drive 
os.environ["HF_HOME"] = "D:/hf_cache"
class LLMManager:
    def __init__(self):
        self.current_model_id = None
        self.model = None
        self.tokenizer = None
        self.pipeline = None

    def load_model(self, model_choice: str):
        """Loads the requested model, unloading the previous one if necessary to save VRAM."""
        if self.current_model_id == model_choice:
            return # Already loaded

        print(f"Unloading previous model: {self.current_model_id}")
        
        # Explicitly delete objects
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        if self.pipeline is not None:
            del self.pipeline
            
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        # Force garbage collection
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect() # Additional cleanup for IPC

        print(f"Loading new model: {model_choice}")
        if model_choice == "baseline_t5":
            self.pipeline = pipeline(
                model="google/flan-t5-base",
                max_new_tokens=100,
                do_sample=False,
                temperature=0.1,   
            )
        else:
            # Determine base model based on choice
            if "Llama" in model_choice:
                base_model_id = "meta-llama/Llama-3.2-3B-Instruct"
            elif "Qwen" in model_choice:
                base_model_id = "Qwen/Qwen2.5-7B-Instruct"
            else:
                base_model_id = "meta-llama/Llama-3.2-3B-Instruct" # Fallback

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_enable_fp32_cpu_offload=True
            )

            print(f"Loading base model {base_model_id} in 4-bit...")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id, 
                quantization_config=bnb_config,
                device_map="auto", 
                low_cpu_mem_usage=True,
            )
            
            print(f"Loading LoRA weights {model_choice}...")
            self.model = PeftModel.from_pretrained(base_model, model_choice)
            
        self.current_model_id = model_choice

    def generate_response(self, query: str, context: str) -> str:
        """Generates a response using the active model."""
        if self.current_model_id == "baseline_t5":
            prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            return self.pipeline(prompt)[0]["generated_text"].replace(prompt, "").strip()
        
        # ODIN Logic for Fine-Tuned Models
        messages = [
            {
                "role": "system",
                "content": f"You are ODIN (Operational Decision Intelligence Network), an AI assistant specialized in UK policing and crime analysis. You provide accurate, professional responses based ONLY on the provided police reports and location data.\n\n### Context:\n{context}"
            },
            {"role": "user", "content": query}
        ]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.3,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

# Global singleton to be used across the app
llm_manager = LLMManager()
