"""
Abstract base class for query processors (Online Pipeline – Step 1).

A query processor transforms the user's raw query string into an
:class:`~src.models.AugmentedQuery` that may contain one or more query
variants to retrieve against.

Possible concrete implementations
----------------------------------
* ``PassthroughQueryProcessor``  – returns the query unchanged (baseline)
* ``HyDEQueryProcessor``         – Hypothetical Document Embeddings: generates
                                   a synthetic answer via LLM, retrieves against
                                   the answer embedding
* ``QueryExpansionProcessor``    – generates N paraphrases / synonyms via LLM
* ``MultiQueryProcessor``        – decomposes a complex query into sub-queries
* ``StepBackQueryProcessor``     – generates an abstract "step-back" question
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models.data_models import AugmentedQuery


class BaseQueryProcessor(ABC):
    """
    Contract for all query processors.

    Subclasses **must** implement :meth:`process`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Short human-readable identifier for this query processor.

        Example: ``'hyde'``.
        """

    @abstractmethod
    def process(self, query: str) -> AugmentedQuery:
        """
        Process a raw user query.

        Parameters
        ----------
        query:
            The raw query string from the user.

        Returns
        -------
        AugmentedQuery
            Contains the original query plus one or more processed query
            strings to retrieve against.
        """
