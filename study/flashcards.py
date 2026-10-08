import json
import os

from anthropic import Anthropic
from dotenv import load_dotenv


load_dotenv()


class FlashcardGenerator:

    def __init__(self):

        api_key = os.getenv("ANTHROPIC_API_KEY")

        model = os.getenv(
            "ANTHROPIC_MODEL",
            "claude-sonnet-4-5"
        )

        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set in .env"
            )

        self.client = Anthropic(
            api_key=api_key
        )

        self.model = model

    def generate_flashcards(
        self,
        topic: str,
        number_of_cards: int,
        retrieved_chunks: list,
    ):

        context_parts = []

        for i, result in enumerate(
            retrieved_chunks,
            start=1
        ):

            metadata = result["metadata"]

            context_parts.append(
                f"""
[SOURCE {i}]
Document: {metadata['source']}
Page: {metadata['page']}

Content:
{result['text']}
"""
            )

        context = "\n".join(context_parts)

        system_prompt = """
You are a flashcard generator for an AI Study Companion.

Create flashcards ONLY from the supplied course material.

Do not use outside knowledge.

Create exactly the requested number of flashcards.

Each flashcard must contain:

- question
- answer
- source

The question should test an important concept.

The answer should be concise but useful for studying.

The source must identify the source number and page.

Return ONLY valid JSON.

Required format:

{
  "flashcards": [
    {
      "question": "...",
      "answer": "...",
      "source": "Source 1, Page 10"
    }
  ]
}
"""

        user_prompt = f"""
COURSE MATERIAL:

{context}

FLASHCARD REQUIREMENTS:

Topic:
{topic}

Number of flashcards:
{number_of_cards}

Create exactly {number_of_cards} useful flashcards.
"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=3000,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
        )

        raw_response = response.content[0].text.strip()

        if raw_response.startswith("```"):

            raw_response = (
                raw_response
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

        flashcards = json.loads(raw_response)

        return flashcards