import json
import os

from anthropic import Anthropic
from dotenv import load_dotenv


load_dotenv()


class QuizGenerator:

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

    def generate_quiz(
        self,
        topic: str,
        difficulty: str,
        number_of_questions: int,
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
You are a quiz generator for an AI Study Companion.

Generate questions ONLY from the supplied course material.

Do not use outside knowledge.

Create exactly the requested number of questions.

Every question must contain:

- question
- four options
- correct_answer
- explanation
- source

The correct_answer must exactly match one of the
four options.

The source must identify the source number and page.

Questions should match the requested difficulty.

Return ONLY valid JSON.

Required format:

{
  "questions": [
    {
      "question": "...",
      "options": [
        "...",
        "...",
        "...",
        "..."
      ],
      "correct_answer": "...",
      "explanation": "...",
      "source": "Source 1, Page 123"
    }
  ]
}
"""

        user_prompt = f"""
COURSE MATERIAL:

{context}

QUIZ REQUIREMENTS:

Topic:
{topic}

Difficulty:
{difficulty}

Number of questions:
{number_of_questions}

Create exactly {number_of_questions} questions.

Make every question directly supported by the
provided course material.
"""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
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

        quiz = json.loads(raw_response)

        return quiz