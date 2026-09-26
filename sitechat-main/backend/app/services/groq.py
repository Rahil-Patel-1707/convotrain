"""
Groq Cloud LLM service for generating responses.
"""

from typing import AsyncGenerator, Optional

from groq import AsyncGroq
from loguru import logger

from app.config import settings


_groq_service: Optional["GroqService"] = None


class GroqService:
    """Service for interacting with the Groq Cloud LLM API."""

    def __init__(self):
        if not settings.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not configured. "
                "Set GROQ_API_KEY in the environment."
            )

        self.client = AsyncGroq(
            api_key=settings.GROQ_API_KEY,
        )

        self.model = settings.GROQ_MODEL or settings.LLM_MODEL

    def _build_messages(
        self,
        prompt: str,
        system_prompt: str = None,
    ) -> list:
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        messages.append({
            "role": "user",
            "content": prompt,
        })

        return messages

    async def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """Generate a complete response from Groq."""

        messages = self._build_messages(
            prompt,
            system_prompt,
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )

            content = response.choices[0].message.content

            return content or ""

        except Exception as e:
            logger.error(f"Groq generation error: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> AsyncGenerator[str, None]:
        """Stream a response from Groq."""

        messages = self._build_messages(
            prompt,
            system_prompt,
        )

        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            async for chunk in stream:
                if not chunk.choices:
                    continue

                content = chunk.choices[0].delta.content

                if content:
                    yield content

        except Exception as e:
            logger.error(f"Groq streaming error: {e}")
            raise

    async def check_health(self) -> bool:
        """Check whether Groq credentials/API are usable."""

        try:
            response = await self.client.models.list()

            return bool(response.data)

        except Exception as e:
            logger.warning(f"Groq health check failed: {e}")
            return False

    async def list_models(self) -> list:
        """List models available to the configured Groq account."""

        try:
            response = await self.client.models.list()

            return [
                model.id
                for model in response.data
            ]

        except Exception as e:
            logger.error(f"Error listing Groq models: {e}")
            return []


def get_groq_service() -> GroqService:
    """Get or create the singleton GroqService."""

    global _groq_service

    if _groq_service is None:
        _groq_service = GroqService()

    return _groq_service