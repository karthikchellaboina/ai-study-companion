from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from traces.langfuse import get_langfuse_client


@contextmanager
def trace_rag_pipeline(
    query: str,
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
):
    """
    Create a Langfuse trace for one RAG question.

    This traces:
        User Question
             ↓
        Retrieved Chunks
             ↓
        RAG Generation
    """

    langfuse = get_langfuse_client()

    # If Langfuse is not configured, continue normally.
    if langfuse is None:
        yield None
        return

    retrieved_chunks = retrieved_chunks or []

    # Prepare retrieval information for Langfuse.
    retrieval_output = []

    for index, chunk in enumerate(retrieved_chunks, start=1):

        metadata = chunk.get("metadata", {})

        retrieval_output.append(
            {
                "rank": index,
                "source": metadata.get("source", "Unknown"),
                "page": metadata.get("page", "Unknown"),
                "score": chunk.get("score"),
                "text_preview": chunk.get("text", "")[:500],
            }
        )

    try:

        with langfuse.start_as_current_observation(
            as_type="span",
            name="rag-pipeline",
            input={
                "query": query,
            },
        ) as trace:

            # Record retrieved documents.
            with langfuse.start_as_current_observation(
                as_type="span",
                name="retrieval",
                input={
                    "query": query,
                    "top_k": len(retrieved_chunks),
                },
            ) as retrieval_span:

                retrieval_span.update(
                    output={
                        "chunks": retrieval_output,
                        "count": len(retrieved_chunks),
                    }
                )

            yield trace

            # Mark the overall trace as completed.
            trace.update(
                output={
                    "status": "completed",
                }
            )

        langfuse.flush()

    except Exception:
        # Never allow observability to break the application.
        yield None