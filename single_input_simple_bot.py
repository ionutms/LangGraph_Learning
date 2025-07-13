from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv


load_dotenv()


class AgentState(TypedDict):
    messages: List[HumanMessage]


llm = init_chat_model("groq:llama3-70b-8192")


def process(state: AgentState) -> AgentState:
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
