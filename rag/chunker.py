from typing import List, Dict


CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping chunks.

    chunk_size:
        Maximum approximate number of characters per chunk.

    overlap:
        Number of characters shared between consecutive chunks.
    """

    text = text.strip()

    if not text:
        return []

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks


def create_chunks(documents: List[Dict]) -> List[Dict]:
    """
    Convert extracted PDF pages into chunks while preserving metadata.
    """

    all_chunks = []

    for document in documents:

        text = document["text"]
        metadata = document["metadata"]

        chunks = chunk_text(text)

        for chunk_index, chunk in enumerate(chunks):

            all_chunks.append(
                {
                    "text": chunk,
                    "metadata": {
                        "source": metadata["source"],
                        "page": metadata["page"],
                        "chunk_id": chunk_index,
                    },
                }
            )

    return all_chunks