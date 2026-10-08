import json
from pathlib import Path

from rag.retriever import Retriever

from evaluation.metrics import (
    recall_at_k,
    reciprocal_rank,
    average_similarity,
    calculate_metrics,
)

from evaluation.degraded_retriever import (
    DegradedRetriever,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GOLD_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "gold_questions.json"
)

BASELINE_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "baseline_metrics.json"
)


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path: Path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# EVALUATE RETRIEVER
# ============================================================

def evaluate_retriever(
    retriever,
    questions,
    degraded: bool = False,
):

    results = []

    for item in questions:

        question = item["question"]

        expected_sources = (
            item["expected_sources"]
        )

        if degraded:

            retrieved = retriever.search(
                query=question,
                top_k=5,
                exclude_sources=expected_sources,
            )

        else:

            retrieved = retriever.search(
                query=question,
                top_k=5,
            )

        recall = recall_at_k(
            retrieved,
            expected_sources,
        )

        reciprocal_rank_score = (
            reciprocal_rank(
                retrieved,
                expected_sources,
            )
        )

        similarity = average_similarity(
            retrieved,
        )

        results.append(
            {
                "question": question,
                "recall_at_5": recall,
                "reciprocal_rank": (
                    reciprocal_rank_score
                ),
                "average_similarity": similarity,
            }
        )

    return results


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(
    title,
    metrics,
):

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)

    print(
        f"Questions: "
        f"{metrics['questions']}"
    )

    print(
        f"Recall@5: "
        f"{metrics['recall_at_5'] * 100:.2f}%"
    )

    print(
        f"MRR@5: "
        f"{metrics['mrr_at_5']:.4f}"
    )

    print(
        f"Average similarity: "
        f"{metrics['average_similarity']:.4f}"
    )


# ============================================================
# REGRESSION TEST
# ============================================================

def run_regression_test():

    print()
    print("=" * 60)
    print("🧪 RAG REGRESSION TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Load baseline
    # --------------------------------------------------------

    baseline = load_json(
        BASELINE_FILE
    )

    # --------------------------------------------------------
    # Load evaluation questions
    # --------------------------------------------------------

    questions = load_json(
        GOLD_FILE
    )

    # --------------------------------------------------------
    # Load retriever
    # --------------------------------------------------------

    retriever = Retriever()

    if not retriever.load_index():

        raise RuntimeError(
            "Search index not found.\n"
            "Run the Streamlit application once "
            "to build the index."
        )

    print()
    print(
        f"Knowledge base: "
        f"{len(retriever.chunks)} chunks"
    )

    # --------------------------------------------------------
    # Current retriever
    # --------------------------------------------------------

    print()
    print(
        "Running current retriever..."
    )

    current_results = evaluate_retriever(
        retriever,
        questions,
        degraded=False,
    )

    current_metrics = calculate_metrics(
        current_results
    )

    # --------------------------------------------------------
    # Degraded retriever
    # --------------------------------------------------------

    print(
        "Running degraded retriever..."
    )

    degraded_retriever = (
        DegradedRetriever(
            retriever
        )
    )

    degraded_results = evaluate_retriever(
        degraded_retriever,
        questions,
        degraded=True,
    )

    degraded_metrics = calculate_metrics(
        degraded_results
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print_metrics(
        "📌 BASELINE",
        {
            "questions": len(questions),
            "recall_at_5": baseline["recall_at_5"],
            "mrr_at_5": baseline["mrr_at_5"],
            "average_similarity": (
                baseline["average_similarity"]
            ),
        },
    )

    print_metrics(
        "🚀 CURRENT RETRIEVER",
        current_metrics,
    )

    print_metrics(
        "⚠️ DEGRADED RETRIEVER",
        degraded_metrics,
    )

    # ========================================================
    # REGRESSION CHECK
    # ========================================================

    print()
    print("=" * 60)
    print("🔍 REGRESSION ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------------
    # Baseline thresholds
    # --------------------------------------------------------

    recall_threshold = (
        baseline["recall_at_5"] * 0.90
    )

    mrr_threshold = (
        baseline["mrr_at_5"] * 0.90
    )

    # --------------------------------------------------------
    # Current system check
    # --------------------------------------------------------

    current_recall_pass = (
        current_metrics["recall_at_5"]
        >= recall_threshold
    )

    current_mrr_pass = (
        current_metrics["mrr_at_5"]
        >= mrr_threshold
    )

    # --------------------------------------------------------
    # Degraded system check
    # --------------------------------------------------------

    degraded_recall_failed = (
        degraded_metrics["recall_at_5"]
        < recall_threshold
    )

    degraded_mrr_failed = (
        degraded_metrics["mrr_at_5"]
        < mrr_threshold
    )

    # --------------------------------------------------------
    # Current system status
    # --------------------------------------------------------

    if (
        current_recall_pass
        and current_mrr_pass
    ):

        print(
            "Current retriever: ✅ PASS"
        )

    else:

        print(
            "Current retriever: ❌ REGRESSION DETECTED"
        )

    # --------------------------------------------------------
    # Degraded-system detection
    # --------------------------------------------------------

    if (
        degraded_recall_failed
        and degraded_mrr_failed
    ):

        print(
            "Regression detector: ✅ PASS"
        )

        print(
            "The test successfully detected "
            "the degraded retriever."
        )

    else:

        print(
            "Regression detector: ❌ FAIL"
        )

        print(
            "The degraded retriever was not "
            "detected strongly enough."
        )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    overall_pass = (
        current_recall_pass
        and current_mrr_pass
        and degraded_recall_failed
        and degraded_mrr_failed
    )

    print()
    print("=" * 60)

    if overall_pass:

        print(
            "🎉 REGRESSION TEST: PASS"
        )

    else:

        print(
            "❌ REGRESSION TEST: FAIL"
        )

    print("=" * 60)
    print()

    return overall_pass


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_regression_test()