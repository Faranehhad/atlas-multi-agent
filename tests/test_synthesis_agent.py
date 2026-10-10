from unittest.mock import Mock

from app.agents.synthesis_agent import SynthesisAgent


def test_synthesis_combines_all_task_results():
    llm = Mock()
    llm.generate.return_value = (
        "Prague will be cool tomorrow. "
        "Monday will also be cool. "
        "Brno currently has mild conditions."
    )

    agent = SynthesisAgent(llm)

    results = [
        {
            "task_id": "task_1",
            "task_type": "weather",
            "answer": "Prague tomorrow: maximum 14 C.",
        },
        {
            "task_id": "task_2",
            "task_type": "weather",
            "answer": "Prague Monday: maximum 15 C.",
        },
        {
            "task_id": "task_3",
            "task_type": "weather",
            "answer": "Brno now: 15 C.",
        },
    ]

    question = (
        "How is the weather tomorrow in Prague, "
        "how is it next Monday there, and how is it now in Brno?"
    )

    answer = agent.synthesize(question, results)

    assert "Prague will be cool tomorrow." in answer
    llm.generate.assert_called_once()

    prompt = llm.generate.call_args.kwargs["user_message"]

    assert question in prompt
    assert "Prague tomorrow: maximum 14 C." in prompt
    assert "Prague Monday: maximum 15 C." in prompt
    assert "Brno now: 15 C." in prompt


def test_synthesis_handles_empty_results_without_calling_llm():
    llm = Mock()
    agent = SynthesisAgent(llm)

    answer = agent.synthesize(
        user_message="What's the weather?",
        task_results=[],
    )

    assert "couldn't obtain any results" in answer
    llm.generate.assert_not_called()
