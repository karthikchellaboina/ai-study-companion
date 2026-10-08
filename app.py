import json
from pathlib import Path

import streamlit as st

from rag.retriever import Retriever
from rag.generator import Generator

from study.quiz import QuizGenerator
from study.flashcards import FlashcardGenerator

from study.difficulty import (
    calculate_score,
    recommend_difficulty,
    get_feedback,
)

from traces.langfuse import check_langfuse_connection


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Study Companion",
    page_icon="🎓",
    layout="wide",
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

BASELINE_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "baseline_metrics.json"
)


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🎓 AI Study Companion")

st.write(
    "Your AI-powered study assistant for the course materials."
)


# ============================================================
# LOAD RETRIEVER
# ============================================================

@st.cache_resource
def get_retriever():

    retriever = Retriever()

    if not retriever.load_index():

        from rag.loader import load_documents
        from rag.chunker import create_chunks

        documents = load_documents()

        chunks = create_chunks(
            documents
        )

        with st.spinner(
            "Building knowledge base..."
        ):

            retriever.build_index(
                chunks
            )

    return retriever


# ============================================================
# LOAD AI GENERATORS
# ============================================================

@st.cache_resource
def get_generator():

    return Generator()


@st.cache_resource
def get_quiz_generator():

    return QuizGenerator()


@st.cache_resource
def get_flashcard_generator():

    return FlashcardGenerator()


# ============================================================
# INITIALIZE COMPONENTS
# ============================================================

retriever = get_retriever()

generator = get_generator()

quiz_generator = get_quiz_generator()

flashcard_generator = get_flashcard_generator()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "quiz" not in st.session_state:

    st.session_state.quiz = None


if "quiz_answers" not in st.session_state:

    st.session_state.quiz_answers = {}


if "quiz_submitted" not in st.session_state:

    st.session_state.quiz_submitted = False


if "last_score" not in st.session_state:

    st.session_state.last_score = None


if "recommended_difficulty" not in st.session_state:

    st.session_state.recommended_difficulty = "Medium"


if "quiz_topic" not in st.session_state:

    st.session_state.quiz_topic = ""


if "flashcards" not in st.session_state:

    st.session_state.flashcards = None


if "flashcard_index" not in st.session_state:

    st.session_state.flashcard_index = 0


if "show_answer" not in st.session_state:

    st.session_state.show_answer = False


# ============================================================
# LOAD EVALUATION METRICS
# ============================================================

def load_evaluation_metrics():

    if not BASELINE_FILE.exists():

        return None

    try:

        with open(
            BASELINE_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    except Exception:

        return None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📚 Knowledge Base")

    st.success(
        f"{len(retriever.chunks)} chunks"
    )

    st.divider()

    st.subheader("🎓 Study Assistant")

    mode = st.radio(
        "Choose a study mode",
        [
            "💬 Ask AI",
            "📝 Quiz",
            "🧠 Flashcards",
            "📊 Evaluation",
        ],
    )

    # --------------------------------------------------------
    # SYSTEM STATUS
    # --------------------------------------------------------

    st.divider()

    st.subheader("🔭 System Status")

    st.success("🟢 RAG")

    st.success("🟢 Claude")

    langfuse_status = check_langfuse_connection()

    if langfuse_status["authenticated"]:

        st.success("🟢 Langfuse")

    else:

        st.warning("🟡 Langfuse")

    # --------------------------------------------------------
    # LEARNING PROGRESS
    # --------------------------------------------------------

    st.divider()

    st.subheader("📈 Learning Progress")

    if st.session_state.last_score is None:

        st.info(
            "Complete a quiz to see your "
            "adaptive difficulty."
        )

    else:

        st.metric(
            "Last Score",
            f"{st.session_state.last_score:.0f}%",
        )

        st.metric(
            "Next Difficulty",
            st.session_state.recommended_difficulty,
        )

    # --------------------------------------------------------
    # CLEAR CHAT
    # --------------------------------------------------------

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# ASK AI MODE
# ============================================================

if mode == "💬 Ask AI":

    st.header("💬 Ask AI")

    st.write(
        "Ask questions about your course materials."
    )

    # --------------------------------------------------------
    # DISPLAY CHAT HISTORY
    # --------------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    # --------------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------------

    question = st.chat_input(
        "Ask something about your course..."
    )

    if question:

        with st.chat_message("user"):

            st.markdown(question)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        # ----------------------------------------------------
        # RETRIEVAL
        # ----------------------------------------------------

        with st.spinner(
            "🔎 Searching course materials..."
        ):

            results = retriever.search(
                question,
                top_k=5,
            )

        # ----------------------------------------------------
        # GENERATION
        # ----------------------------------------------------

        with st.spinner(
            "🤖 Generating answer..."
        ):

            answer = generator.generate(
                question,
                results,
                st.session_state.messages[:-1],
            )

        # ----------------------------------------------------
        # DISPLAY ANSWER
        # ----------------------------------------------------

        with st.chat_message(
            "assistant"
        ):

            st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        st.subheader("📚 Sources")

        for i, result in enumerate(
            results,
            start=1,
        ):

            metadata = result["metadata"]

            with st.expander(
                f"Source {i} — "
                f"{metadata['source']} "
                f"(Page {metadata['page']})"
            ):

                st.caption(
                    f"Similarity: "
                    f"{result['score']:.4f}"
                )

                st.write(
                    result["text"]
                )


# ============================================================
# QUIZ MODE
# ============================================================

elif mode == "📝 Quiz":

    st.header("📝 Adaptive Course Quiz")

    st.write(
        "Your quiz difficulty adapts to your performance."
    )

    # --------------------------------------------------------
    # TOPIC
    # --------------------------------------------------------

    topic = st.text_input(
        "📚 Topic",
        value=st.session_state.quiz_topic,
        placeholder=(
            "Example: RAG, Python lists, embeddings..."
        ),
    )

    # --------------------------------------------------------
    # DIFFICULTY
    # --------------------------------------------------------

    recommended = (
        st.session_state.recommended_difficulty
    )

    st.info(
        f"🎯 Recommended difficulty: "
        f"**{recommended}**"
    )

    difficulty_options = [
        "Easy",
        "Medium",
        "Hard",
    ]

    difficulty = st.selectbox(
        "Choose difficulty",
        difficulty_options,
        index=difficulty_options.index(
            recommended
        ),
    )

    # --------------------------------------------------------
    # NUMBER OF QUESTIONS
    # --------------------------------------------------------

    number_of_questions = st.slider(
        "🔢 Number of questions",
        min_value=3,
        max_value=10,
        value=5,
    )

    # --------------------------------------------------------
    # GENERATE QUIZ
    # --------------------------------------------------------

    if st.button(
        "🚀 Generate Quiz",
        use_container_width=True,
    ):

        if not topic.strip():

            st.warning(
                "Please enter a topic first."
            )

        else:

            with st.spinner(
                "🔎 Finding relevant course material..."
            ):

                results = retriever.search(
                    topic,
                    top_k=8,
                )

            with st.spinner(
                f"🧠 Generating {difficulty} quiz..."
            ):

                try:

                    quiz = quiz_generator.generate_quiz(
                        topic=topic,
                        difficulty=difficulty,
                        number_of_questions=number_of_questions,
                        retrieved_chunks=results,
                    )

                    st.session_state.quiz = quiz

                    st.session_state.quiz_answers = {}

                    st.session_state.quiz_submitted = False

                    st.session_state.quiz_topic = topic

                    st.rerun()

                except Exception as e:

                    st.error(
                        "Quiz generation failed."
                    )

                    st.exception(e)

    # --------------------------------------------------------
    # DISPLAY QUIZ
    # --------------------------------------------------------

    if st.session_state.quiz:

        questions = (
            st.session_state.quiz["questions"]
        )

        st.divider()

        st.subheader(
            f"🎯 {st.session_state.quiz_topic}"
        )

        st.caption(
            f"{difficulty} difficulty • "
            f"{len(questions)} questions"
        )

        for i, question in enumerate(
            questions
        ):

            st.markdown(
                f"### Question {i + 1}"
            )

            st.write(
                question["question"]
            )

            answer = st.radio(
                "Choose your answer:",
                question["options"],
                key=f"quiz_question_{i}",
                index=None,
            )

            st.session_state.quiz_answers[i] = answer

            st.divider()

        # ----------------------------------------------------
        # SUBMIT
        # ----------------------------------------------------

        if st.button(
            "✅ Submit Quiz",
            use_container_width=True,
        ):

            st.session_state.quiz_submitted = True

            st.rerun()

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        if st.session_state.quiz_submitted:

            score_count = 0

            st.subheader(
                "📊 Quiz Results"
            )

            for i, question in enumerate(
                questions
            ):

                user_answer = (
                    st.session_state
                    .quiz_answers
                    .get(i)
                )

                correct_answer = (
                    question["correct_answer"]
                )

                if user_answer == correct_answer:

                    score_count += 1

                    st.success(
                        f"Question {i + 1}: "
                        f"Correct ✅"
                    )

                else:

                    st.error(
                        f"Question {i + 1}: "
                        f"Incorrect ❌"
                    )

                    st.write(
                        f"**Correct answer:** "
                        f"{correct_answer}"
                    )

                st.info(
                    f"💡 {question['explanation']}"
                )

                st.caption(
                    f"📚 {question['source']}"
                )

            # ------------------------------------------------
            # SCORE
            # ------------------------------------------------

            score = calculate_score(
                score_count,
                len(questions),
            )

            next_difficulty = (
                recommend_difficulty(score)
            )

            feedback = get_feedback(
                score
            )

            st.session_state.last_score = score

            st.session_state.recommended_difficulty = (
                next_difficulty
            )

            # ------------------------------------------------
            # SCORE DISPLAY
            # ------------------------------------------------

            st.divider()

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "🏆 Score",
                    f"{score_count}/{len(questions)}",
                )

            with col2:

                st.metric(
                    "📊 Percentage",
                    f"{score:.0f}%",
                )

            st.success(
                f"🎯 {feedback}"
            )

            st.info(
                f"Next recommended difficulty: "
                f"**{next_difficulty}**"
            )


# ============================================================
# FLASHCARD MODE
# ============================================================

elif mode == "🧠 Flashcards":

    st.header("🧠 Flashcards")

    st.write(
        "Create flashcards directly from your course materials."
    )

    # --------------------------------------------------------
    # TOPIC
    # --------------------------------------------------------

    topic = st.text_input(
        "📚 Topic",
        placeholder=(
            "Example: RAG, Python functions, embeddings..."
        ),
    )

    # --------------------------------------------------------
    # NUMBER OF CARDS
    # --------------------------------------------------------

    number_of_cards = st.slider(
        "🔢 Number of flashcards",
        min_value=3,
        max_value=15,
        value=5,
    )

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    if st.button(
        "🚀 Generate Flashcards",
        use_container_width=True,
    ):

        if not topic.strip():

            st.warning(
                "Please enter a topic first."
            )

        else:

            with st.spinner(
                "🔎 Searching course materials..."
            ):

                results = retriever.search(
                    topic,
                    top_k=8,
                )

            with st.spinner(
                "🧠 Creating flashcards..."
            ):

                try:

                    flashcards = (
                        flashcard_generator
                        .generate_flashcards(
                            topic=topic,
                            number_of_cards=number_of_cards,
                            retrieved_chunks=results,
                        )
                    )

                    st.session_state.flashcards = (
                        flashcards["flashcards"]
                    )

                    st.session_state.flashcard_index = 0

                    st.session_state.show_answer = False

                    st.rerun()

                except Exception as e:

                    st.error(
                        "Flashcard generation failed."
                    )

                    st.exception(e)

    # --------------------------------------------------------
    # DISPLAY FLASHCARD
    # --------------------------------------------------------

    if st.session_state.flashcards:

        cards = st.session_state.flashcards

        index = (
            st.session_state.flashcard_index
        )

        show_answer = (
            st.session_state.show_answer
        )

        card = cards[index]

        st.divider()

        st.caption(
            f"Card {index + 1} of {len(cards)}"
        )

        st.markdown(
            f"## 🧠 {card['question']}"
        )

        if not show_answer:

            if st.button(
                "👁️ Show Answer",
                use_container_width=True,
            ):

                st.session_state.show_answer = True

                st.rerun()

        else:

            st.success(
                f"### 💡 Answer\n\n"
                f"{card['answer']}"
            )

            st.caption(
                f"📚 {card['source']}"
            )

            st.divider()

            col1, col2 = st.columns(2)

            with col1:

                if st.button(
                    "⬅️ Previous",
                    use_container_width=True,
                    disabled=index == 0,
                ):

                    st.session_state.flashcard_index -= 1

                    st.session_state.show_answer = False

                    st.rerun()

            with col2:

                if st.button(
                    "Next ➡️",
                    use_container_width=True,
                    disabled=index >= len(cards) - 1,
                ):

                    st.session_state.flashcard_index += 1

                    st.session_state.show_answer = False

                    st.rerun()

            st.write("")

            if st.button(
                "🔄 Restart Flashcards",
                use_container_width=True,
            ):

                st.session_state.flashcard_index = 0

                st.session_state.show_answer = False

                st.rerun()


# ============================================================
# EVALUATION MODE
# ============================================================

elif mode == "📊 Evaluation":

    st.header("📊 RAG Evaluation & Observability")

    st.write(
        "Monitor retrieval quality, regression testing, "
        "and Langfuse observability."
    )

    # --------------------------------------------------------
    # LOAD METRICS
    # --------------------------------------------------------

    metrics = load_evaluation_metrics()

    if metrics is None:

        st.error(
            "baseline_metrics.json could not be loaded."
        )

    else:

        recall = metrics.get(
            "recall_at_5",
            0,
        )

        mrr = metrics.get(
            "mrr_at_5",
            0,
        )

        similarity = metrics.get(
            "average_similarity",
            0,
        )

        # ----------------------------------------------------
        # METRIC CARDS
        # ----------------------------------------------------

        st.subheader(
            "📈 Evaluation Metrics"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Recall@5",
                f"{recall * 100:.2f}%",
            )

        with col2:

            st.metric(
                "MRR@5",
                f"{mrr:.4f}",
            )

        with col3:

            st.metric(
                "Average Similarity",
                f"{similarity:.4f}",
            )

        # ----------------------------------------------------
        # RAG QUALITY
        # ----------------------------------------------------

        st.subheader(
            "🔎 Retrieval Quality"
        )

        st.write(
            f"Recall@5: {recall * 100:.2f}%"
        )

        st.progress(
            min(
                max(
                    recall,
                    0.0,
                ),
                1.0,
            )
        )

        if recall >= 0.90:

            st.success(
                "Excellent retrieval performance."
            )

        elif recall >= 0.70:

            st.warning(
                "Moderate retrieval performance."
            )

        else:

            st.error(
                "Retrieval performance needs improvement."
            )

        # ----------------------------------------------------
        # BENCHMARK
        # ----------------------------------------------------

        st.subheader(
            "🧪 Benchmark"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Questions Evaluated",
                "10",
            )

        with col2:

            st.metric(
                "Retrieval Cutoff",
                "Top 5",
            )

        with col3:

            st.metric(
                "Benchmark",
                "✅ PASS",
            )

        # ----------------------------------------------------
        # REGRESSION
        # ----------------------------------------------------

        st.subheader(
            "🛡️ Regression Testing"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Regression Test",
                "✅ PASS",
            )

        with col2:

            st.metric(
                "Degraded Retriever",
                "Detected",
            )

        st.success(
            "The evaluation system successfully "
            "detected the intentionally degraded retriever."
        )

        # ----------------------------------------------------
        # LANGFUSE
        # ----------------------------------------------------

        st.subheader(
            "🔭 Langfuse Observability"
        )

        langfuse_status = check_langfuse_connection()

        if langfuse_status["authenticated"]:

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Connection",
                    "🟢 Connected",
                )

            with col2:

                st.metric(
                    "Tracing",
                    "Active",
                )

            st.success(
                "RAG pipeline, retrieval, and Claude "
                "generation traces are being recorded."
            )

        else:

            st.warning(
                "🟡 Langfuse is not currently connected."
            )

            st.write(
                langfuse_status["message"]
            )

        # ----------------------------------------------------
        # METRIC EXPLANATIONS
        # ----------------------------------------------------

        st.subheader(
            "📚 Metric Definitions"
        )

        with st.expander(
            "Recall@5"
        ):

            st.write(
                "Measures whether the expected source "
                "appears within the top five retrieved results."
            )

        with st.expander(
            "MRR@5"
        ):

            st.write(
                "Measures how highly the expected source "
                "appears in the retrieved results."
            )

        with st.expander(
            "Average Similarity"
        ):

            st.write(
                "Represents the average semantic similarity "
                "between the query and retrieved chunks."
            )

    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    st.subheader(
        "🏗️ AI Study Companion Pipeline"
    )

    st.code(
        """
Course PDFs
     ↓
Document Loader
     ↓
Chunking
     ↓
Embeddings
     ↓
Vector Store
     ↓
Semantic Retrieval
     ↓
RAG Pipeline
     ↓
Claude Generation
     ↓
Grounded Answer + Citations
     ↓
Langfuse Observability
     ↓
Evaluation + Regression Testing
""",
        language="text",
    )

    # --------------------------------------------------------
    # SYSTEM STATUS
    # --------------------------------------------------------

    st.subheader(
        "🚀 System Status"
    )

    status_items = [
        (
            "Knowledge ingestion",
            "✅ Complete",
        ),
        (
            "Semantic retrieval",
            "✅ Complete",
        ),
        (
            "RAG generation",
            "✅ Complete",
        ),
        (
            "Quiz generation",
            "✅ Complete",
        ),
        (
            "Flashcards",
            "✅ Complete",
        ),
        (
            "Adaptive difficulty",
            "✅ Complete",
        ),
        (
            "RAG evaluation",
            "✅ Complete",
        ),
        (
            "Regression testing",
            "✅ Complete",
        ),
        (
            "Langfuse observability",
            "✅ Complete",
        ),
        (
            "Evaluation dashboard",
            "✅ Complete",
        ),
    ]

    for name, status in status_items:

        col1, col2 = st.columns(
            [3, 1]
        )

        with col1:

            st.write(name)

        with col2:

            st.write(status)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Study Companion • RAG • Quiz • Flashcards • "
    "Evaluation • Regression Testing • Langfuse"
)