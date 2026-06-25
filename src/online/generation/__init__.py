"""
online.generation – Answer generation

Produces the final natural-language answer from the augmented query
(user question + retrieved context).

Exports:
    BaseGenerator                – abstract interface all generators must implement
    SentenceTransformerGenerator – generates answers using a SentenceTransformer model
    PassthroughGenerator         – returns the context directly without generation
"""

from .base import BaseGenerator
from .generators import SentenceTransformerGenerator, PassthroughGenerator

__all__ = ["BaseGenerator", "SentenceTransformerGenerator", "PassthroughGenerator"]
