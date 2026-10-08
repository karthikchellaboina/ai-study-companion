import os
from pathlib import Path

from dotenv import load_dotenv


# Load .env
load_dotenv()


# Project paths
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
EVALUATION_DIR = DATA_DIR / "evaluation"

STORAGE_DIR = BASE_DIR / "storage"


# Create required directories
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_DIR.mkdir(parents=True, exist_ok=True)


# Anthropic configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

ANTHROPIC_MODEL = os.getenv(
    "ANTHROPIC_MODEL",
    "claude-sonnet-4-5"
)


# RAG configuration
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 5


if not ANTHROPIC_API_KEY:
    raise ValueError(
        "ANTHROPIC_API_KEY is missing. "
        "Add it to your .env file."
    )