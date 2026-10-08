from unittest.mock import Mock

from app.agents.query_analyzer import QueryAnalyzer
from app.agents.schemas import QueryPlan, Task


def test_analyzer_returns_query_plan():
    llm = Mock()
    llm.structured.return_value = QueryPlan(
        tasks=[
            Task(
                id="task_1",
                type="weather",
                question="What is the weather in Prague tomorrow?",
            ),
            Task(
                id="task_2",
                type="city_info",
                question="What are the best places to visit in Prague?",
            ),
        ]
    )

    analyzer = QueryAnalyzer(llm)

    result = analyzer.analyze(
        "What is the weather in Prague tomorrow and what should I visit there?"
    )

    assert isinstance(result, QueryPlan)
    assert len(result.tasks) == 2
    assert result.tasks[0].type == "weather"
    assert result.tasks[1].type == "city_info"
    llm.structured.assert_called_once()


def test_analyzer_passes_conversation_history():
    llm = Mock()
    llm.structured.return_value = QueryPlan(
        tasks=[
            Task(
                id="task_1",
                type="city_info",
                question="What are the best places to visit in Prague?",
            )
        ]
    )

    analyzer = QueryAnalyzer(llm)

    analyzer.analyze(
        "What should I visit there?",
        conversation_history=[
            {"role": "user", "content": "I am planning a trip to Prague."},
            {"role": "assistant", "content": "Prague has many historic attractions."},
        ],
    )

    llm.structured.assert_called_once()

    _, kwargs = llm.structured.call_args

    prompt = kwargs["user_message"]

    assert "I am planning a trip to Prague." in prompt
    assert "What should I visit there?" in prompt
