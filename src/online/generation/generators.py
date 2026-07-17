from src.online.generation import BaseGenerator
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc

SYSTEM_PROMPT = "Du bist ein hilfreicher Assistent. Beantworte die Frage basierend auf dem gegebenen Kontext. Halte dich kurz und gib nur die Informationen, nach denen gefragt wurde. Wenn die Antwort nicht im Kontext zu finden ist, antworte: 'Dazu enthalten die bereitgestellten Dokumente keine Informationen.'"

class HuggingfaceGenerator(BaseGenerator):
    def __init__(self, model_name: str):
        # 1. Properly detect and isolate the GPU device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 2. Force model entirely into VRAM at FP16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16
        ).to(self.device)
        
        # 3. Configure tokenizer globally for batching safety
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    @property
    def name(self) -> str:
        return "huggingface"

    def generate(self, query: str, context) -> str:
        context_str = "\n".join([f"Source {i+1}:\n{result.chunk.text}" for i, result in enumerate(context)])
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"Frage: {query}\n\nKontext:\n{context_str}"
            }
        ]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,   
                do_sample=False
            )
            
        generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    


def generate_batch(self, queries: list[str], contexts: list[list]) -> list[str]:
    if not queries:
        return []
    
    context_strings = [
        "\n".join([f"Source {i+1}:\n{result.chunk.text}" for i, result in enumerate(context)]) 
        for context in contexts
    ]
    
    batch_messages = []
    for query, context_str in zip(queries, context_strings):
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"Frage: {query}\n\nKontext:\n{context_str}"
            }
        ]
        batch_messages.append(messages)
    
    prompts = self.tokenizer.apply_chat_template(batch_messages, tokenize=False, add_generation_prompt=True)
    
    try:
        # Ensure padding side is left for batch generation if not already set
        # (Right padding can cause models to generate gibberish or ignore system prompts)
        if self.tokenizer.padding_side != "left":
            self.tokenizer.padding_side = "left"
            
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=False  
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,   
                do_sample=False       
            )
        
        results = []
        input_len = inputs.input_ids.shape[-1] # Global width of input batch
        
        for i in range(len(prompts)):
            # Safely slice everything after the padded prompt window
            generated_tokens = outputs[i][input_len:]
            decoded_output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            results.append(decoded_output)
            
        return results

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        # Check if it is indeed a CUDA OOM error
        is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()
        
        if is_oom:
            # 1. Clear up GPU memory fragmentation
            del inputs
            if 'outputs' in locals():
                del outputs
            torch.cuda.empty_cache()
            gc.collect()
            
            # 2. If a single item is OOMing, we can't split further
            if len(queries) == 1:
                raise RuntimeError(
                    f"CUDA OOM occurred even with a batch size of 1. "
                    f"The context length or max_new_tokens is too large for your GPU memory."
                ) from e
            
            # 3. Recursively split the batch in half
            mid = len(queries) // 2
            
            left_results = self.generate_batch(queries[:mid], contexts[:mid])
            right_results = self.generate_batch(queries[mid:], contexts[mid:])
            
            return left_results + right_results
        else:
            # Re-raise if it was a different RuntimeError (e.g. device mismatches, tensor shape issues)
            raise e


class PassthroughGenerator(BaseGenerator):
    @property
    def name(self) -> str:
        return "passthrough"

    def generate(self, query: str, context):
        context_str = "\n".join([f"Source {i+1}:\n{result.chunk.text}" for i, result in enumerate(context)])
        return context_str