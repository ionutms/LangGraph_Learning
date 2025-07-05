from dotenv import load_dotenv
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict

load_dotenv()

llm = init_chat_model("groq:llama3-70b-8192")


# Define the message schema
class State(TypedDict):
    """State for the graph, containing the messages."""

    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


def chatboot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder.add_node("chatboot", chatboot)
graph_builder.add_edge(START, "chatboot")
graph_builder.add_edge("chatboot", END)


graph = graph_builder.compile()


if __name__ == "__main__":
    print(graph.get_graph().draw_ascii())

    user_input = input("Add a message: ")

    state = graph.invoke({
        "messages": [{"role": "user", "content": user_input}]
    })
    print(state["messages"][-1].content)
