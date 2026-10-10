"""LangGraph workflow for Atlas."""

from app.agents.query_analyzer import QueryAnalyzer
from app.graph.state import AgentState, TaskState
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


def build_graph(query_analyzer: QueryAnalyzer):
    """Build and compile the Atlas graph."""

    def analyze_query(state: AgentState) -> dict:
        """Analyze the user request and store the resulting query plan."""

        query_plan = query_analyzer.analyze(
            user_message=state["user_message"],
            conversation_history=state["conversation_history"],
        )

        return {"query_plan": query_plan}

    def route_tasks(state: AgentState) -> list[Send]:
        """Fan out each planned task to the task worker."""

        query_plan = state["query_plan"]

        if query_plan is None:
            return []

        return [
            Send(
                "execute_task",
                {"task": task},
            )
            for task in query_plan.tasks
        ]

    def execute_task(state: TaskState) -> dict:
        """Execute a single task placeholder."""

        task = state["task"]

        return {
            "task_results": [
                {
                    "task_id": task.id,
                    "task_type": task.type,
                    "answer": f"Task '{task.id}' routed successfully.",
                }
            ]
        }

    builder = StateGraph(AgentState)

    builder.add_node("analyze_query", analyze_query)
    builder.add_node("execute_task", execute_task)

    builder.add_edge(START, "analyze_query")
    builder.add_conditional_edges(
        "analyze_query",
        route_tasks,
        ["execute_task"],
    )
    builder.add_edge("execute_task", END)

    return builder.compile()
