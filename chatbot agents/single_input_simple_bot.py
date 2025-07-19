"""Module for initializing and running a chat agent."""

from typing import List, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

load_dotenv()


class AgentState(TypedDict):
    """Represents the state of a chat agent.

    Attributes:
        messages (List[HumanMessage]): List of human messages.
    """

    messages: List[HumanMessage]


llm = init_chat_model("groq:llama3-70b-8192")


def process(state: AgentState) -> AgentState:
    """Processes the current state of the chat agent.

    Args:
        state (AgentState): Current state of the chat agent.

    Returns:
        AgentState: Updated state of the chat agent.
    """
    response = llm.invoke(state["messages"])
    print(f"AI: {response.content}")
    return state


graph = StateGraph(AgentState)

graph.add_edge(START, "process")
graph.add_node("process", process)
graph.add_edge("process", END)

agent = graph.compile()

user_input = input("Enter: ")
agent.invoke({"messages": [HumanMessage(content=user_input)]})
