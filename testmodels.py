import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --- CONFIGURATION ---
# Swap this out to test different models:
# 1. "microsoft/Phi-4-mini-instruct"
# 2. "Qwen/Qwen2.5-3B-Instruct" 
MODEL_ID = "microsoft/Phi-4-mini-instruct"

print(f"Loading tokenizer and model for: {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto"
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# --- SIMULATED GERMAN RAG CONTEXT ---
mock_context = """
Firmenrichtlinie Reisekosten (Stand 2026):
1. Bahnfahrten innerhalb Deutschlands werden standardmäßig in der 2. Klasse gebucht. Executive-Manager dürfen ab einer Distanz von 300 km die 1. Klasse nutzen.
2. Die Pauschale für das tägliche Verpflegungsgeld (Spesen) bei einer Abwesenheit von mehr als 8 Stunden beträgt 28 Euro im Inland.
3. Hotelbuchungen müssen über das interne Portal 'KramTravel' vorgenommen werden. Ausgaben über 150 Euro pro Nacht bedürfen der vorherigen Freigabe durch den Teamleiter.
"""

# --- TEST CASES ---
test_cases = [
    {
        "name": "Positive RAG Test (Direct Fact)",
        "query": "Wie hoch ist die Spesenpauschale bei einer Abwesenheit von mehr als 8 Stunden im Inland?"
    },
    {
        "name": "Negative RAG Test (Absence of Fact / Hallucination Check)",
        "query": "Wie viel Geld bekomme ich zurückerstattet, wenn ich ein Taxi in Berlin nutze?"
    }
]

# --- EVALUATION LOOP ---
print("\n=== Starte RAG-Modell-Evaluierung ===\n")

for test in test_cases:
    print(f"Test-Szenario: {test['name']}")
    print(f"Frage: {test['query']}")
    
    # Constructing a rigid RAG constraint prompt in German
    prompt_messages = [
        {
            "role": "system",
            "content": (
                "Du bist ein präziser RAG-Assistent. Beantworte die Frage ausschließlich basierend auf dem "
                "bereitgestellten Kontext. Wenn die Antwort nicht im Kontext direkt zu finden ist, erfinde "
                "nichts, sondern antworte exakt mit: 'Diese Information ist im bereitgestellten Kontext nicht enthalten.'"
            )
        },
        {
            "role": "user",
            "content": f"Kontext:\n{mock_context}\n\nFrage:\n{test['query']}"
        }
    ]
    
    # Generate response
    outputs = pipe(
        prompt_messages, 
        max_new_tokens=150, 
        temperature=0.1,  # Low temperature is vital for deterministic RAG behavior
        top_p=0.9
    )
    
    response = outputs[0]["generated_text"][-1]["content"].strip()
    print(f"Modell-Antwort:\n{response}")
    print("-" * 50)