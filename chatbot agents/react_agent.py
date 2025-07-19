from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()


class AgentState(TypedDict):
    """Agent state with annotated messages."""

    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a (int): First number to add.
        b (int): Second number to add.

    Returns:
        int: Sum of a and b.
    """
    return a + b


@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers.

    Args:
        a (int): Number to subtract from.
        b (int): Number to subtract.

    Returns:
        int: Difference of a and b.
    """
    return a - b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.

    Args:
        a (int): First number to multiply.
        b (int): Second number to multiply.

    Returns:
        int: Product of a and b.
    """
    return a * b


tools = [add, subtract, multiply]

model = init_chat_model("groq:llama3-70b-8192").bind_tools(tools)


def model_call(state: AgentState) -> AgentState:
    """Invoke the chat model with system prompt and state messages.

    Args:
        state (AgentState): Current agent state.

    Returns:
        AgentState: Updated agent state with model response.
    """
    system_prompt = SystemMessage(
        content="You are my AI assistant, "
        "please answer my query to the best of your ability."
    )
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Determine if the conversation should continue.

    Args:
        state (AgentState): Current agent state.

    Returns:
        str: 'continue' or 'end' based on tool calls in last message.
    """
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

graph.add_edge("tools", "our_agent")

app = graph.compile()


def print_stream(stream):
    """Print the messages in a stream.

    Args:
        stream: Stream of agent states.
    """
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


inputs = {
    "messages": [
        (
            "user",
            "Add 40 + 12. Sustract 6. Multiply by 10."
            "Also tell me a joke please.",
        )
    ]
}
print_stream(app.stream(inputs, stream_mode="values"))
