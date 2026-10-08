from typing import List, Dict


class DegradedRetriever:

    """
    Intentionally degraded retriever used to test
    whether the regression system can detect a
    significant retrieval-quality drop.

    It removes the expected source documents from
    the candidate results and therefore simulates
    a retrieval system that is failing to retrieve
    the correct course material.
    """

    def __init__(self, retriever):

        self.retriever = retriever

    def search(
        self,
        query: str,
        top_k: int = 5,
        exclude_sources: List[str] | None = None,
    ) -> List[Dict]:

        exclude_sources = (
            exclude_sources or []
        )

        # Get a larger candidate pool from the
        # functioning retriever.
        candidate_results = self.retriever.search(
            query,
            top_k=50,
        )

        degraded_results = []

        for result in candidate_results:

            source = result["metadata"]["source"]

            if source in exclude_sources:
                continue

            degraded_results.append(result)

            if len(degraded_results) >= top_k:
                break

        return degraded_results