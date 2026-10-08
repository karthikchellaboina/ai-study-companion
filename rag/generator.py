import os

from dotenv import load_dotenv
from anthropic import Anthropic
from langfuse import get_client


load_dotenv()


class Generator:

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

        # --------------------------------------------------------
        # LANGFUSE
        # --------------------------------------------------------

        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")

        if public_key and secret_key:
            self.langfuse = get_client()
        else:
            self.langfuse = None


    def generate(
        self,
        question: str,
        retrieved_chunks: list,
        chat_history: list | None = None,
    ) -> str:

        # --------------------------------------------------------
        # BUILD CONTEXT
        # --------------------------------------------------------

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

        # --------------------------------------------------------
        # CHAT HISTORY
        # --------------------------------------------------------

        history_text = ""

        if chat_history:

            history_text = "\nPrevious conversation:\n"

            for message in chat_history[-6:]:

                role = message["role"]
                content = message["content"]

                history_text += (
                    f"{role.upper()}: {content}\n"
                )

        # --------------------------------------------------------
        # SYSTEM PROMPT
        # --------------------------------------------------------

        system_prompt = """
You are an AI Study Companion.

Your knowledge for this answer comes ONLY from
the supplied course-material context.

IMPORTANT RULES:

1. Answer using the provided course material.
2. Do not invent facts that are not supported by it.
3. If the material does not contain enough information,
   clearly say that the course material does not provide
   enough information.
4. Keep explanations useful for a student.
5. You may simplify the wording, but preserve the
   meaning of the source material.
6. Cite supporting sources using [Source 1],
   [Source 2], etc.
7. Only use source numbers that actually exist.
8. Do not create fake page numbers or sources.
"""

        # --------------------------------------------------------
        # USER PROMPT
        # --------------------------------------------------------

        user_prompt = f"""
COURSE MATERIAL:

{context}

{history_text}

CURRENT STUDENT QUESTION:

{question}

Answer the question clearly.

When making claims based on the course material,
include citations such as [Source 1] or [Source 2].
"""

        # --------------------------------------------------------
        # LANGFUSE GENERATION TRACE
        # --------------------------------------------------------

        if self.langfuse:

            with self.langfuse.start_as_current_observation(
                as_type="generation",
                name="claude-study-companion",
                input={
                    "question": question,
                    "retrieved_chunks": len(retrieved_chunks),
                },
                model=self.model,
            ) as generation:

                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1500,
                    system=system_prompt,
                    messages=[
                        {
                            "role": "user",
                            "content": user_prompt,
                        }
                    ],
                )

                answer = response.content[0].text

                # Record useful generation information
                generation.update(
                    output={
                        "answer": answer,
                        "sources_used": len(retrieved_chunks),
                    },
                    metadata={
                        "provider": "Anthropic",
                        "model": self.model,
                        "max_tokens": 1500,
                    },
                )

                # Record token usage when available
                if hasattr(response, "usage"):

                    usage = response.usage

                    generation.update(
                        metadata={
                            "provider": "Anthropic",
                            "model": self.model,
                            "max_tokens": 1500,
                            "input_tokens": getattr(
                                usage,
                                "input_tokens",
                                None,
                            ),
                            "output_tokens": getattr(
                                usage,
                                "output_tokens",
                                None,
                            ),
                        }
                    )

                return answer

        # --------------------------------------------------------
        # NORMAL GENERATION
        # --------------------------------------------------------
        # Langfuse is optional. If credentials are not configured,
        # the application continues working normally.
        # --------------------------------------------------------

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
        )

        return response.content[0].text