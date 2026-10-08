import json
from pathlib import Path

from rag.retriever import Retriever

from evaluation.metrics import (
    recall_at_k,
    reciprocal_rank,
    average_similarity,
    calculate_metrics,
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


# ============================================================
# LOAD GOLD QUESTIONS
# ============================================================

def load_gold_questions():

    with open(
        GOLD_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# RUN EVALUATION
# ============================================================

def run_evaluation():

    print()
    print("=" * 60)
    print("🎓 AI STUDY COMPANION")
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 60)
    print()

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

    print(
        f"Knowledge base: "
        f"{len(retriever.chunks)} chunks"
    )

    print()

    # --------------------------------------------------------
    # Load evaluation questions
    # --------------------------------------------------------

    questions = load_gold_questions()

    print(
        f"Evaluation questions: "
        f"{len(questions)}"
    )

    print()

    evaluation_results = []

    # --------------------------------------------------------
    # Evaluate every question
    # --------------------------------------------------------

    for number, item in enumerate(
        questions,
        start=1,
    ):

        question = item["question"]

        expected_sources = (
            item["expected_sources"]
        )

        print(
            f"[{number}/{len(questions)}] "
            f"{question}"
        )

        # ----------------------------------------------------
        # Retrieve top 5
        # ----------------------------------------------------

        results = retriever.search(
            question,
            top_k=5,
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        recall = recall_at_k(
            results,
            expected_sources,
        )

        rank = reciprocal_rank(
            results,
            expected_sources,
        )

        similarity = average_similarity(
            results,
        )

        # ----------------------------------------------------
        # First matching source
        # ----------------------------------------------------

        matched_source = None

        for result in results:

            source = result["metadata"]["source"]

            if source in expected_sources:

                matched_source = source

                break

        if recall == 1.0:

            status = "✅ PASS"

        else:

            status = "❌ FAIL"

        print(
            f"    {status} | "
            f"Recall@5={recall:.2f} | "
            f"RR={rank:.2f} | "
            f"Avg Similarity={similarity:.4f}"
        )

        if matched_source:

            print(
                f"    Source: {matched_source}"
            )

        evaluation_results.append(
            {
                "question": question,
                "expected_sources": expected_sources,
                "matched_source": matched_source,
                "recall_at_5": recall,
                "reciprocal_rank": rank,
                "average_similarity": similarity,
            }
        )

        print()

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        evaluation_results
    )

    print("=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)

    print(
        f"Questions evaluated: "
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

    print("=" * 60)
    print()

    return metrics


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_evaluation()