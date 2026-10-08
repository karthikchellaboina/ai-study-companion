def calculate_score(
    correct_answers: int,
    total_questions: int,
) -> float:
    """
    Calculate quiz score as a percentage.
    """

    if total_questions <= 0:
        return 0.0

    return (
        correct_answers / total_questions
    ) * 100


def recommend_difficulty(
    score: float,
) -> str:
    """
    Recommend the next quiz difficulty based
    on the student's latest score.
    """

    if score <= 40:
        return "Easy"

    elif score <= 70:
        return "Medium"

    return "Hard"


def get_feedback(
    score: float,
) -> str:
    """
    Provide simple learning feedback.
    """

    if score <= 40:

        return (
            "Let's strengthen the fundamentals. "
            "The next quiz will be easier."
        )

    elif score <= 70:

        return (
            "You're making progress. "
            "Let's continue with medium-level questions."
        )

    return (
        "Excellent work! "
        "You're ready for harder questions."
    )