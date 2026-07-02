"""
online.generation – Answer generation

Produces the final natural-language answer from the augmented query
(user question + retrieved context).

Exports:
    BaseGenerator                – abstract interface all generators must implement
    HuggingfaceGenerator         – generates answers using a Huggingface model
    PassthroughGenerator         – returns the context directly without generation
"""

from .base import BaseGenerator
from .generators import HuggingfaceGenerator, PassthroughGenerator

__all__ = ["BaseGenerator", "HuggingfaceGenerator", "PassthroughGenerator"]
