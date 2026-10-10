from unittest.mock import Mock

from app.agents.schemas import QueryPlan, Task
from app.graph.workflow import build_graph


def test_graph_fans_out_tasks_and_synthesizes_results():
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

    weather_agent = Mock()
    weather_agent.answer.return_value = "Mock weather forecast."

    synthesis_agent = Mock()
    synthesis_agent.synthesize.return_value = (
        "Combined answer covering all three questions."
    )

    graph = build_graph(
        query_analyzer=query_analyzer,
        weather_agent=weather_agent,
        synthesis_agent=synthesis_agent,
    )

    user_message = (
        "What's the weather in Prague tomorrow, "
        "what should I visit there, "
        "and is apache/airflow active on GitHub recently?"
    )

    result = graph.invoke(
        {
            "user_message": user_message,
            "conversation_history": [],
            "query_plan": None,
            "task_results": [],
            "final_answer": None,
        }
    )

    assert result["query_plan"] is not None
    assert len(result["query_plan"].tasks) == 3
    assert len(result["task_results"]) == 3

    results_by_id = {
        item["task_id"]: item
        for item in result["task_results"]
    }

    assert results_by_id["task_1"]["task_type"] == "weather"
    assert results_by_id["task_1"]["answer"] == "Mock weather forecast."
    assert results_by_id["task_2"]["task_type"] == "city_info"
    assert results_by_id["task_3"]["task_type"] == "github"

    weather_agent.answer.assert_called_once_with(
        user_message="What is the weather in Prague tomorrow?",
        conversation_history=[],
    )

    synthesis_agent.synthesize.assert_called_once()
    synthesis_call = synthesis_agent.synthesize.call_args.kwargs

    assert synthesis_call["user_message"] == user_message
    assert len(synthesis_call["task_results"]) == 3
    assert result["final_answer"] == (
        "Combined answer covering all three questions."
    )


def test_graph_synthesizes_single_weather_task():
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

    weather_agent = Mock()
    weather_agent.answer.return_value = "Mock weather forecast."

    synthesis_agent = Mock()
    synthesis_agent.synthesize.return_value = (
        "The weather forecast for Prague is available."
    )

    graph = build_graph(
        query_analyzer=query_analyzer,
        weather_agent=weather_agent,
        synthesis_agent=synthesis_agent,
    )

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
    assert result["final_answer"] == (
        "The weather forecast for Prague is available."
    )
    synthesis_agent.synthesize.assert_called_once()
