"""LangGraph workflow for Atlas."""

from app.agents.query_analyzer import QueryAnalyzer
from app.graph.state import AgentState
from langgraph.graph import END, START, StateGraph


def build_graph(query_analyzer: QueryAnalyzer):
    """Build and compile the Atlas graph."""

    def analyze_query(state: AgentState) -> dict:
        """Analyze the user request and store the resulting query plan."""

        query_plan = query_analyzer.analyze(
            user_message=state["user_message"],
            conversation_history=state["conversation_history"],
        )

        return {"query_plan": query_plan}

    builder = StateGraph(AgentState)

    builder.add_node("analyze_query", analyze_query)

    builder.add_edge(START, "analyze_query")
    builder.add_edge("analyze_query", END)

    return builder.compile()
