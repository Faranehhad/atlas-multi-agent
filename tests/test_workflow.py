from unittest.mock import Mock

from app.agents.schemas import QueryPlan, Task
from app.graph.workflow import build_graph


def test_graph_fans_out_multiple_tasks():
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
            Task(
                id="task_3",
                type="github",
                question="Is apache/airflow active on GitHub recently?",
            ),
        ]
    )

    graph = build_graph(query_analyzer)

    result = graph.invoke(
        {
            "user_message": (
                "What's the weather in Prague tomorrow, "
                "what should I visit there, "
                "and is apache/airflow active on GitHub recently?"
            ),
            "conversation_history": [],
            "query_plan": None,
            "task_results": [],
            "final_answer": None,
        }
    )

    assert result["query_plan"] is not None
    assert len(result["query_plan"].tasks) == 3

    assert len(result["task_results"]) == 3

    result_by_id = {
        item["task_id"]: item
        for item in result["task_results"]
    }

    assert result_by_id["task_1"]["task_type"] == "weather"
    assert result_by_id["task_2"]["task_type"] == "city_info"
    assert result_by_id["task_3"]["task_type"] == "github"


def test_graph_handles_single_task():
    query_analyzer = Mock()

    query_analyzer.analyze.return_value = QueryPlan(
        tasks=[
            Task(
                id="task_1",
                type="weather",
                question="What is the weather in Prague tomorrow?",
            )
        ]
    )

    graph = build_graph(query_analyzer)

    result = graph.invoke(
        {
            "user_message": "What is the weather in Prague tomorrow?",
            "conversation_history": [],
            "query_plan": None,
            "task_results": [],
            "final_answer": None,
        }
    )

    assert len(result["task_results"]) == 1
    assert result["task_results"][0]["task_id"] == "task_1"
