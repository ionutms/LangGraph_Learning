"""Module for initializing and interacting with a chatbot."""

from typing import Annotated

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()

llm = init_chat_model("groq:llama3-70b-8192")


# Define the message schema
class State(TypedDict):
    """State for the graph, containing conversation messages.

    Attributes:
        messages: List of messages in the conversation.
    """

    messages: Annotated[list, add_messages]


def init_state_graph() -> StateGraph:
    """Initialize the state graph for conversation management.

    Returns:
        StateGraph: The initialized state graph.
    """
    return StateGraph(State)


# Initialize the state graph
graph_builder = init_state_graph()


def chatboot(state: State) -> dict:
    """Handle chat interaction by invoking the LLM.

    Args:
        state: The current state of the conversation.

    Returns:
        dict: The updated state with the LLM's response.
    """
    return {"messages": [llm.invoke(state["messages"])]}


def add_nodes_and_edges(graph_builder: StateGraph) -> None:
    """Define the nodes and edges of the graph.

    Args:
        graph_builder: The state graph builder.
    """
    graph_builder.add_edge(START, "chatboot")
    graph_builder.add_node("chatboot", chatboot)
    graph_builder.add_edge("chatboot", END)


# Define the nodes and edges of the graph
add_nodes_and_edges(graph_builder)


def compile_graph(graph_builder: StateGraph) -> StateGraph:
    """Compile the graph to finalize its structure.

    Args:
        graph_builder: The state graph builder.

    Returns:
        StateGraph: The compiled state graph.
    """
    return graph_builder.compile()


# Compile the graph to finalize its structure
graph = compile_graph(graph_builder)


if __name__ == "__main__":
    # Print the ASCII representation of the graph for debugging
    print(graph.get_graph().draw_ascii())

    # Main loop to interact with the chatboot
    user_input = input("Add a message: ")

    # Invoke the graph with the user's input
    def invoke_graph(graph: StateGraph, user_input: str) -> State:
        """Invoke the graph with the user's input.

        Args:
            graph: The compiled state graph.
            user_input: The user's input message.

        Returns:
            State: The updated state with the LLM's response.
        """
        return graph.invoke({
            "messages": [{"role": "user", "content": user_input}]
        })

    state = invoke_graph(graph, user_input)

    # Print the last message from the state
    def print_last_message(state: State) -> None:
        """Print the last message from the state.

        Args:
            state: The current state of the conversation.
        """
        print(state["messages"][-1].content)

    print_last_message(state)
