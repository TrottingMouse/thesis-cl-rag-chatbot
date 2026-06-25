from src.models import AugmentedQuery
from src.online.query import BaseQueryProcessor

class NoProcessingProcessor(BaseQueryProcessor):
    @property
    def name(self) -> str:
        return "no_processing"

    def process(self, query: str) -> AugmentedQuery:
        return AugmentedQuery(
            original_query=query,
            processed_queries=[query]
        )