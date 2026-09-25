"""The LangGraph chatbot: a single `chatbot` node over a list of messages."""

from collections.abc import Sequence
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, BaseMessage, SystemMessage, trim_messages
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph

from .config import DEFAULT_SYSTEM_PROMPT

CHATBOT_NODE = "chatbot"


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


type ChatGraph = CompiledStateGraph[State, None, State, State]


def build_graph(
    llm: BaseChatModel,
    checkpointer: BaseCheckpointSaver[str] | None = None,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    max_history_messages: int = 7,
) -> ChatGraph:
    """Compile `START -> chatbot -> END`.

    The checkpointer saves each conversation's state under its `thread_id`, so
    follow-up questions can see earlier messages. Without one, every call starts fresh.
    """

    async def chatbot(state: State) -> State:
        # The system prompt is added per call rather than kept in the state.
        recent = trim_history(state["messages"], max_history_messages)
        reply = await llm.ainvoke([SystemMessage(system_prompt), *recent])
        return {"messages": [reply]}

    builder = StateGraph(State)
    builder.add_node(CHATBOT_NODE, chatbot)
    builder.add_edge(START, CHATBOT_NODE)
    builder.add_edge(CHATBOT_NODE, END)
    return builder.compile(checkpointer=checkpointer)


def trim_history(messages: Sequence[BaseMessage], max_messages: int) -> list[BaseMessage]:
    """Keep at most the last `max_messages` messages, starting on a user message."""
    return trim_messages(
        messages,
        max_tokens=max_messages,
        token_counter=len,  # counts messages, so max_tokens acts as a message limit
        strategy="last",
        start_on="human",
    )
