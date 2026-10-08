from unittest.mock import Mock

from app.agents.schemas import QueryPlan, Task
from app.graph.workflow import build_graph


def test_graph_stores_query_plan():
    query_analyzer = Mock()

    query_analyzer.analyze.return_value = QueryPlan(
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

    graph = build_graph(query_analyzer)

    result = graph.invoke(
        {
            "user_message": (
                "What's the weather in Prague tomorrow "
                "and what should I visit there?"
            ),
            "conversation_history": [],
            "query_plan": None,
            "task_results": [],
            "final_answer": None,
        }
    )

    assert result["query_plan"] is not None
    assert len(result["query_plan"].tasks) == 2
    assert result["query_plan"].tasks[0].type == "weather"
    assert result["query_plan"].tasks[1].type == "city_info"

    query_analyzer.analyze.assert_called_once_with(
        user_message=(
            "What's the weather in Prague tomorrow "
            "and what should I visit there?"
        ),
        conversation_history=[],
    )
