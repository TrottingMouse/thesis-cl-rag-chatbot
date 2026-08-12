from src.online.generation import BaseGenerator
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc

SYSTEM_PROMPT = (
    "Du bist ein hilfreicher Assistent. Beantworte die Frage basierend auf dem gegebenen Kontext. "
    "Halte dich kurz und gib nur die Informationen, nach denen gefragt wurde. "
    "Wenn die Antwort nicht im Kontext zu finden ist, antworte: "
    "'Dazu enthalten die bereitgestellten Dokumente keine Informationen.'"
)


class HuggingfaceGenerator(BaseGenerator):
    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        ).to(self.device)
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
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Frage: {query}\n\nKontext:\n{context_str}"}
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

        batch_messages = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Frage: {query}\n\nKontext:\n{context_str}"}
            ]
            for query, context_str in zip(queries, context_strings)
        ]

        prompts = self.tokenizer.apply_chat_template(batch_messages, tokenize=False, add_generation_prompt=True)

        try:
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

            input_len = inputs.input_ids.shape[-1]  # padded prompt width
            return [
                self.tokenizer.decode(outputs[i][input_len:], skip_special_tokens=True).strip()
                for i in range(len(prompts))
            ]

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()

            if is_oom:
                del inputs
                if "outputs" in locals():
                    del outputs
                torch.cuda.empty_cache()
                gc.collect()

                if len(queries) == 1:
                    raise RuntimeError(
                        "CUDA OOM occurred even with a batch size of 1. "
                        "The context length or max_new_tokens is too large for your GPU memory."
                    ) from e

                # Recursively split the batch in half
                mid = len(queries) // 2
                return (
                    self.generate_batch(queries[:mid], contexts[:mid])
                    + self.generate_batch(queries[mid:], contexts[mid:])
                )
            else:
                raise


class PassthroughGenerator(BaseGenerator):
    @property
    def name(self) -> str:
        return "passthrough"

    def generate(self, query: str, context) -> str:
        return "\n".join([f"Source {i+1}:\n{result.chunk.text}" for i, result in enumerate(context)])