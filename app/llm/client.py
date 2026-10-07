"""LLM client abstraction for Gemini."""

from typing import TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel


StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


class LLMClient:
    """Small abstraction around the Gemini chat model."""

    def __init__(self, api_key: str, model: str) -> None:
        self._model = ChatGoogleGenerativeAI(
            model=model,
            api_key=api_key,
            max_retries=2,
        )

    def generate(
        self,
        system_prompt: str,
        user_message: str,
    ) -> str:
        """Generate a plain-text response."""

        response = self._model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ]
        )

        return response.text

    def structured(
        self,
        schema: type[StructuredModel],
        system_prompt: str,
        user_message: str,
    ) -> StructuredModel:
        """Generate a response constrained to a Pydantic schema."""

        structured_model = self._model.with_structured_output(
            schema.model_json_schema(),
            method="json_schema",
        )

        response = structured_model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ]
        )

        return schema.model_validate(response)
