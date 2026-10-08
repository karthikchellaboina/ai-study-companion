from typing import List, Dict


def source_matches(
    result: Dict,
    expected_sources: List[str],
) -> bool:
    """
    Check whether a retrieved result comes from
    one of the expected source documents.
    """

    actual_source = result["metadata"]["source"]

    return actual_source in expected_sources


def recall_at_k(
    results: List[Dict],
    expected_sources: List[str],
) -> float:
    """
    Returns 1 if at least one relevant source appears
    in the retrieved results, otherwise 0.
    """

    for result in results:

        if source_matches(
            result,
            expected_sources,
        ):
            return 1.0

    return 0.0


def reciprocal_rank(
    results: List[Dict],
    expected_sources: List[str],
) -> float:
    """
    Calculate Reciprocal Rank.

    Example:

    Relevant result at position 1 -> 1.0
    Relevant result at position 2 -> 0.5
    Relevant result at position 3 -> 0.333
    """

    for index, result in enumerate(
        results,
        start=1,
    ):

        if source_matches(
            result,
            expected_sources,
        ):

            return 1.0 / index

    return 0.0


def average_similarity(
    results: List[Dict],
) -> float:
    """
    Calculate the average similarity score
    of retrieved results.
    """

    if not results:
        return 0.0

    scores = [
        result["score"]
        for result in results
    ]

    return sum(scores) / len(scores)


def calculate_metrics(
    evaluation_results: List[Dict],
) -> Dict:
    """
    Calculate overall evaluation metrics.
    """

    if not evaluation_results:

        return {
            "questions": 0,
            "recall_at_5": 0.0,
            "mrr_at_5": 0.0,
            "average_similarity": 0.0,
        }

    recall_scores = [
        result["recall_at_5"]
        for result in evaluation_results
    ]

    reciprocal_ranks = [
        result["reciprocal_rank"]
        for result in evaluation_results
    ]

    similarity_scores = [
        result["average_similarity"]
        for result in evaluation_results
    ]

    return {
        "questions": len(
            evaluation_results
        ),
        "recall_at_5": (
            sum(recall_scores)
            / len(recall_scores)
        ),
        "mrr_at_5": (
            sum(reciprocal_ranks)
            / len(reciprocal_ranks)
        ),
        "average_similarity": (
            sum(similarity_scores)
            / len(similarity_scores)
        ),
    }