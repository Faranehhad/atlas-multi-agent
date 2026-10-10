"""LangGraph workflow for Atlas."""

from app.agents.query_analyzer import QueryAnalyzer
from app.agents.weather_agent import WeatherAgent
from app.graph.state import AgentState, TaskState
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


def build_graph(
    query_analyzer: QueryAnalyzer,
    weather_agent: WeatherAgent,
):
    """Build and compile the Atlas graph."""

    def analyze_query(state: AgentState) -> dict:
        """Analyze the request and store the resulting query plan."""

        query_plan = query_analyzer.analyze(
            user_message=state["user_message"],
            conversation_history=state["conversation_history"],
        )

        return {"query_plan": query_plan}

    def route_tasks(state: AgentState) -> list[Send]:
        """Dispatch each planned task to an independent worker."""

        query_plan = state["query_plan"]

        if query_plan is None:
            return []

        return [
            Send(
                "execute_task",
                {
                    "task": task,
                    "conversation_history": state["conversation_history"],
                },
            )
            for task in query_plan.tasks
        ]

    def execute_task(state: TaskState) -> dict:
        """Dispatch a task to its specialized agent."""

        task = state["task"]

        if task.type == "weather":
            answer = weather_agent.answer(
                user_message=task.question,
                conversation_history=state["conversation_history"],
            )
        else:
            answer = (
                f"Task '{task.id}' has type '{task.type}', "
                "but its specialized agent is not implemented yet."
            )

        return {
            "task_results": [
                {
                    "task_id": task.id,
                    "task_type": task.type,
                    "answer": answer,
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
