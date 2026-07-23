from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models import AugmentedQuery
from src.online.query import BaseQueryProcessor

_DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


class NoProcessingProcessor(BaseQueryProcessor):
    @property
    def name(self) -> str:
        return "no_processing"

    def process(self, query: str) -> AugmentedQuery:
        return AugmentedQuery(
            original_query=query,
            processed_queries=[query]
        )


class HyDEQueryProcessor(BaseQueryProcessor):
    """
    Hypothetical Document Embeddings (HyDE) query processor.

    Instead of embedding the raw user query, this processor asks an LLM to
    generate a *hypothetical* document that would answer the query.  The
    synthetic document is then embedded and used for retrieval.

    This shifts the query-embedding from the "short question" space into the
    "relevant document chunk" space, which typically improves dense-retrieval
    quality – especially for longer, domain-specific corpora.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier to use for hypothesis generation.
        Defaults to ``mistralai/Mistral-7B-Instruct-v0.2``.
    max_new_tokens:
        Maximum number of tokens to generate for the hypothetical document.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        max_new_tokens: int = 200,
    ) -> None:
        self._max_new_tokens = max_new_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    @property
    def name(self) -> str:
        return "hyde"

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _build_messages(self, query: str) -> list[dict]:
        return [
            {
                "role": "system",
                "content": (
                    "Du bist ein hilfreicher Assistent. "
                    "Schreibe einen kurzen, sachlichen Textabschnitt (2–4 Sätze), "
                    "der die folgende Frage beantwortet. "
                    "Antworte ausschließlich mit dem Textabschnitt – ohne Einleitung, "
                    "Überschrift oder sonstige Erklärungen."
                ),
            },
            {
                "role": "user",
                "content": f"Frage: {query}\n\nHypothetischer Antwortabschnitt:",
            },
        ]

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _generate_hypothesis(self, query: str) -> str:
        """Run the local HuggingFace model and return the generated hypothetical document text."""
        messages = self._build_messages(query)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )

        generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    def process(self, query: str) -> AugmentedQuery:
        """
        Generate a hypothetical document for *query* and return it as the
        sole processed query with ``query_type='document'``.
        """
        hypothesis = self._generate_hypothesis(query)
        return AugmentedQuery(
            original_query=query,
            processed_queries=[hypothesis],
            processor_name=self.name,
            query_type="document",
        )


class CoTQueryProcessor(BaseQueryProcessor):
    """
    Chain-of-Thought (CoT) query processor.

    Prompts the LLM to reason step-by-step before answering the query.
    The resulting rationale (the full CoT output) is then used as a rich
    retrieval document by prepending the original query five times so that
    the embedding is anchored to the question while benefiting from the
    expanded reasoning context.

    Prompt template
    ---------------
    Answer the following query:
    {query}
    Give the rationale before answering.

    Retrieval document
    ------------------
    ``"{query} " * 5 + rationale``

    Parameters
    ----------
    model_name:
        HuggingFace model identifier to use for CoT generation.
        Defaults to ``mistralai/Mistral-7B-Instruct-v0.2``.
    max_new_tokens:
        Maximum number of tokens to generate for the rationale.
    query_repeats:
        How many times the original query is prepended to the rationale.
        Defaults to ``5``.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        max_new_tokens: int = 300,
        query_repeats: int = 5,
    ) -> None:
        self._max_new_tokens = max_new_tokens
        self._query_repeats = query_repeats
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    @property
    def name(self) -> str:
        return "cot"

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _build_messages(self, query: str) -> list[dict]:
        return [
            {
                "role": "user",
                "content": (
                    f"Beantworte die folgende Frage:\n{query}\n"
                    "Erkläre deine Gedankengänge, bevor du antwortest."
                ),
            }
        ]

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _generate_rationale(self, query: str) -> str:
        """Run the local HuggingFace model and return the full CoT output."""
        messages = self._build_messages(query)
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )

        generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    def process(self, query: str) -> AugmentedQuery:
        """
        Generate a CoT rationale for *query*, then build the retrieval
        document by prepending the original query ``query_repeats`` times.
        """
        rationale = self._generate_rationale(query)
        retrieval_doc = " ".join([query] * self._query_repeats) + " " + rationale
        return AugmentedQuery(
            original_query=query,
            processed_queries=[retrieval_doc],
            processor_name=self.name,
            query_type="document",
        )