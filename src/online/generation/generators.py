from src.online.generation import BaseGenerator
from transformers import AutoModelForCausalLM, AutoTokenizer

class HuggingfaceGenerator(BaseGenerator):
    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    @property
    def name(self) -> str:
        return "huggingface"

    def generate(self, query: str, context):
        context_str = "\n".join([f"Source {i+1}:\n{result.chunk.text}" for i, result in enumerate(context)])
        messages = [
            {
                "role": "system",
                "content": "Du bist ein hilfreicher Assistent. Beantworte die Frage basierend auf dem gegebenen Kontext. Wenn die Antwort nicht im Kontext zu finden ist, antworte: 'Dazu enthalten die bereitgestellten Dokumente keine Informationen.'"
            },
            {
                "role": "user",
                "content": f"Frage: {query}\n\nKontext:\n{context_str}"
            }
        ]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,   
            temperature=0.1,      
            do_sample=False       
        )
        generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    def generate_batch(self, queries: list[str], contexts: list[list]) -> list[str]:
        # 1. Build context strings exactly as you started
        context_strings = [
            "\n".join([f"Source {i+1}:\n{result.chunk.text}" for i, result in enumerate(context)]) 
            for context in contexts
        ]
        
        # 2. Construct the messages list for each item in the batch
        batch_messages = []
        for query, context_str in zip(queries, context_strings):
            messages = [
                {
                    "role": "system",
                    "content": "Du bist ein hilfreicher Assistent. Beantworte die Frage basierend auf dem gegebenen Kontext. Wenn die Antwort nicht im Kontext zu finden ist, antworte: 'Dazu enthalten die bereitgestellten Dokumente keine Informationen.'"
                },
                {
                    "role": "user",
                    "content": f"Frage: {query}\n\nKontext:\n{context_str}"
                }
            ]
            batch_messages.append(messages)
        
        # 3. Apply chat template to all prompts in the batch
        prompts = self.tokenizer.apply_chat_template(batch_messages, tokenize=False, add_generation_prompt=True)
        
        # 4. CRITICAL: Configure tokenizer for left-padding
        # (Ensure the tokenizer has a pad_token set; if not, use the eos_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.tokenizer.padding_side = "left"
        
        # 5. Tokenize the batch with padding and attention masks
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=False  # Set to True + max_length if you want to hard-cap long contexts
        ).to(self.model.device)
        
        # 6. Generate outputs for the whole batch
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,   
            temperature=0.1,      
            do_sample=False       
        )
        
        # 7. Decode only the newly generated tokens for each sequence
        results = []
        for i in range(len(prompts)):
            input_len = inputs.input_ids[i].shape[-1]
            # Because we used left-padding, the generated tokens are strictly 
            # everything *after* the original input length.
            generated_tokens = outputs[i][input_len:]
            decoded_output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            results.append(decoded_output)
            
        return results

class PassthroughGenerator(BaseGenerator):
    @property
    def name(self) -> str:
        return "passthrough"
    
    def generate(self, query: str, context):
        context_str = "\n".join([f"Source {i+1}:\n{result.chunk.text}" for i, result in enumerate(context)])
        return context_str
 


        
    