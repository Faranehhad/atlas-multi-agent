"""LLM-based query analysis and task decomposition."""

from app.agents.schemas import QueryPlan
from app.llm.client import LLMClient


QUERY_ANALYZER_PROMPT = """
You are the query analyzer for Atlas, a multi-agent travel assistant.

Your job is to understand the user's request and convert it into one or more
independent tasks that specialized agents can execute.

Rules:

1. Identify every distinct question or actionable request in the user's message.
2. If the user asks multiple questions, create multiple tasks.
3. Never merge unrelated questions into one task.
4. Rewrite each task as a self-contained question.
5. Resolve references such as "there", "it", "that city", "the second one",
   or similar references using the conversation history.
6. Preserve important entities such as city names, countries, repository names,
   dates, and other constraints.
7. Choose the most appropriate task type:
   - weather
   - city_info
   - github
8. Add dependencies when a task requires the result of another task.
9. Independent tasks should have an empty dependencies list.
10. Do not answer the user's questions. Only produce the structured task plan.
"""


class QueryAnalyzer:
    """Analyze a user request using an LLM and produce a query plan."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def analyze(
        self,
        user_message: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> QueryPlan:
        """Convert a user request and conversation history into a QueryPlan."""

        history = conversation_history or []

        history_text = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in history
        )

        prompt = f"""
CONVERSATION HISTORY:
<conversation_history>
{history_text}
</conversation_history>

CURRENT USER MESSAGE:
<user_message>
{user_message}
</user_message>
"""

        return self._llm.structured(
            schema=QueryPlan,
            system_prompt=QUERY_ANALYZER_PROMPT,
            user_message=prompt,
        )
