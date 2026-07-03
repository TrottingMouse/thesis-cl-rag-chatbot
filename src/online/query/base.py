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

from src.models import AugmentedQuery


class BaseQueryProcessor(ABC):
    """
    Contract for all query processors.

    Subclasses **must** implement :meth:`process`.  For throughput-sensitive
    workloads, subclasses may also override :meth:`process_batch`.
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

    def process_batch(self, queries: list[str]) -> list[AugmentedQuery]:
        """
        Process a batch of raw user queries.

        The default implementation calls :meth:`process` for each query
        sequentially.  Subclasses that can exploit true batch-level
        parallelism (e.g. a single LLM call for all queries) should
        override this method.

        Parameters
        ----------
        queries:
            A list of raw query strings from the user.

        Returns
        -------
        list[AugmentedQuery]
            One :class:`~src.models.AugmentedQuery` per input query,
            in the same order.
        """
        return [self.process(q) for q in queries]
