"""
Qwen/Qwen3-4B-Instruct-2507
microsoft/Phi-4-mini-instruct
Qwen/Qwen3.5-2B
HuggingFaceTB/SmolLM-1.7B
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
 
torch.random.manual_seed(0)

model_path = "microsoft/Phi-4-mini-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=False,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
 
messages = [
    {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
    {"role": "user", "content": "Was ist das schönste Land der Welt?"}
]
 
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)
 
generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}
 
output = pipe(messages, **generation_args)
print(output[0]['generated_text'])
