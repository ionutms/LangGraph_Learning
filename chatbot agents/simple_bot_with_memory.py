"""Module for conversational AI using LangChain."""

from typing import List, TypedDict, Union

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

load_dotenv()


class AgentState(TypedDict):
    """Represents the state of a conversational agent.

    Attributes:
        messages: List of messages exchanged between human and AI.
    """

    messages: List[Union[HumanMessage, AIMessage]]


llm = init_chat_model("groq:llama3-70b-8192")


def process(state: AgentState) -> AgentState:
    """Processes the current state of the conversational agent.

    Args:
        state: Current state of the agent.

    Returns:
        Updated state with AI response.
    """
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    print(f"AI: {response.content}")
    print("CURRENT STATE: ", state["messages"])
    return state


graph = StateGraph(AgentState)

graph.add_edge(START, "process")
graph.add_node("process", process)
graph.add_edge("process", END)

agent = graph.compile()


conversation_history = []
user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    user_input = input("Enter: ")
