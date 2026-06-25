from src.online.generation import BaseGenerator
from transformers import AutoModelForCausalLM, AutoTokenizer

class SentenceTransformerGenerator(BaseGenerator):
    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    @property
    def name(self) -> str:
        return "sentence_transformer"

    def generate(self, query: str, context):
        prompt = self.construct_prompt(query, context)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0])

class PassthroughGenerator(BaseGenerator):
    @property
    def name(self) -> str:
        return "passthrough"
    
    def generate(self, query: str, context):
        context_str = "\n".join([f"Source {i+1}:\n{result.chunk.text}" for i, result in enumerate(context)])
        return context_str
 


        
    