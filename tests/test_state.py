from app.agents.schemas import QueryPlan, Task
from app.graph.state import TaskResult


def test_task_result_has_expected_fields():
    result: TaskResult = {
        "task_id": "task_1",
        "task_type": "weather",
        "answer": "Sunny in Prague.",
    }

    assert result["task_id"] == "task_1"
    assert result["task_type"] == "weather"
    assert result["answer"] == "Sunny in Prague."


def test_query_plan_can_be_stored_in_graph_state():
    plan = QueryPlan(
        tasks=[
            Task(
                id="task_1",
                type="weather",
                question="What is the weather in Prague tomorrow?",
            )
        ]
    )

    state = {
        "user_message": "What is the weather in Prague tomorrow?",
        "conversation_history": [],
        "query_plan": plan,
        "task_results": [],
        "final_answer": None,
    }

    assert state["query_plan"] == plan
    assert state["task_results"] == []
    assert state["final_answer"] is None
