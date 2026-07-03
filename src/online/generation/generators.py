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

class PassthroughGenerator(BaseGenerator):
    @property
    def name(self) -> str:
        return "passthrough"
    
    def generate(self, query: str, context):
        context_str = "\n".join([f"Source {i+1}:\n{result.chunk.text}" for i, result in enumerate(context)])
        return context_str
 


        
    