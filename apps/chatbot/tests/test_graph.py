import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from apps.chatbot.app.graph import ChatGraph, State, build_graph, trim_history

from .fakes import TEST_SYSTEM_PROMPT as SYSTEM_PROMPT
from .fakes import FakeChatModel, fake_llm


def user(text: str, message_id: str | None = None) -> State:
    return {"messages": [HumanMessage(text, id=message_id)]}


def thread(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def graph_with_memory(llm: FakeChatModel, max_history_messages: int = 7) -> ChatGraph:
    return build_graph(llm, InMemorySaver(), system_prompt=SYSTEM_PROMPT, max_history_messages=max_history_messages)


async def test_chatbot_node_appends_ai_message() -> None:
    graph = build_graph(fake_llm("Hi there!"), system_prompt=SYSTEM_PROMPT)

    result = await graph.ainvoke(user("Hello"))

    assert [type(m) for m in result["messages"]] == [HumanMessage, AIMessage]
    assert result["messages"][-1].text == "Hi there!"


async def test_system_prompt_is_sent_to_the_model_but_not_stored() -> None:
    llm = fake_llm("Hi there!")
    graph = graph_with_memory(llm)

    result = await graph.ainvoke(user("Hello"), thread("t1"))

    assert [(type(m), m.text) for m in llm.calls[0]] == [(SystemMessage, SYSTEM_PROMPT), (HumanMessage, "Hello")]
    assert not any(isinstance(m, SystemMessage) for m in result["messages"])


async def test_follow_up_sees_earlier_messages_in_the_same_thread() -> None:
    llm = fake_llm("Nice to meet you, Ada.", "Your name is Ada.")
    graph = graph_with_memory(llm)

    await graph.ainvoke(user("My name is Ada."), thread("t1"))
    result = await graph.ainvoke(user("What is my name?"), thread("t1"))

    assert [m.text for m in llm.calls[1]] == [
        SYSTEM_PROMPT,
        "My name is Ada.",
        "Nice to meet you, Ada.",
        "What is my name?",
    ]
    assert result["messages"][-1].text == "Your name is Ada."


async def test_threads_are_isolated() -> None:
    llm = fake_llm("Reply A", "Reply B")
    graph = graph_with_memory(llm)

    await graph.ainvoke(user("Secret for A"), thread("a"))
    await graph.ainvoke(user("Hello from B"), thread("b"))

    assert [m.text for m in llm.calls[1]] == [SYSTEM_PROMPT, "Hello from B"]


async def test_without_a_checkpointer_each_call_starts_fresh() -> None:
    llm = fake_llm("First", "Second")
    graph = build_graph(llm, system_prompt=SYSTEM_PROMPT)

    await graph.ainvoke(user("One"))
    await graph.ainvoke(user("Two"))

    assert [m.text for m in llm.calls[1]] == [SYSTEM_PROMPT, "Two"]


async def test_model_only_sees_the_most_recent_messages() -> None:
    llm = fake_llm("r1", "r2", "r3")
    graph = graph_with_memory(llm, max_history_messages=3)

    for text in ("q1", "q2", "q3"):
        result = await graph.ainvoke(user(text), thread("t1"))

    assert [m.text for m in llm.calls[-1]] == [SYSTEM_PROMPT, "q2", "r2", "q3"]
    assert len(result["messages"]) == 6  # the thread itself keeps everything


async def test_retrying_a_message_id_replaces_the_failed_message() -> None:
    llm = fake_llm("Recovered", errors=[RuntimeError("provider down")])
    graph = graph_with_memory(llm)

    with pytest.raises(RuntimeError):
        await graph.ainvoke(user("Hello", message_id="m1"), thread("t1"))
    result = await graph.ainvoke(user("Hello", message_id="m1"), thread("t1"))

    assert [(m.type, m.text) for m in result["messages"]] == [("human", "Hello"), ("ai", "Recovered")]


def test_trim_history_starts_on_a_user_message() -> None:
    messages = [HumanMessage("q1"), AIMessage("r1"), HumanMessage("q2"), AIMessage("r2"), HumanMessage("q3")]

    # The last 4 would start with AIMessage("r1"), so that one is dropped too.
    assert [m.text for m in trim_history(messages, max_messages=4)] == ["q2", "r2", "q3"]
