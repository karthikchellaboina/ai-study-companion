from pathlib import Path
from pypdf import PdfReader

from config import DOCUMENTS_DIR


def load_documents():
    """
    Load and extract text from all PDF course materials.
    """

    documents = []

    pdf_files = sorted(Path(DOCUMENTS_DIR).glob("*.pdf"))

    for pdf_path in pdf_files:
        reader = PdfReader(str(pdf_path))

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            if text.strip():
                documents.append({
                    "text": text,
                    "metadata": {
                        "source": pdf_path.name,
                        "page": page_number,
                    },
                })

    return documents