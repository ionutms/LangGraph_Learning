from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

# This is the global variable to store document content
document_content = ""


class AgentState(TypedDict):
    """
    A dictionary representing the state of the agent.

    Attributes:
        messages (Annotated[Sequence[BaseMessage], add_messages]):
            A sequence of messages.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def update(content: str) -> str:
    """
    Updates the document with the provided content.

    Args:
        content (str): The new content for the document.

    Returns:
        str: A success message with the updated document content.
    """
    global document_content
    document_content = content
    return (
        "Document has been updated successfully! "
        "The current content is:\n{document_content}"
    )


@tool
def save(filename: str) -> str:
    """
    Save the current document to a text file and finish the process.

    Args:
        filename (str): Name for the text file.

    Returns:
        str: A success message with the filename or an error message.
    """

    global document_content

    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"

    try:
        with open(filename, "w") as file:
            file.write(document_content)
        print(f"\n💾 Document has been saved to: {filename}")
        return f"Document has been saved successfully to '{filename}'."

    except Exception as e:
        return f"Error saving document: {str(e)}"


tools = [update, save]

model = init_chat_model("groq:llama3-70b-8192").bind_tools(tools)


def our_agent(state: AgentState) -> AgentState:
    """
    The main agent function that handles user input and responds accordingly.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        AgentState: The updated state of the agent.
    """
    system_prompt = SystemMessage(
        content=f"""
    You are Drafter, a helpful writing assistant.
    You are going to help the user update and modify documents.

    - If the user wants to update or modify content,
        use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.

    The current document content is:{document_content}
    """
    )

    if not state["messages"]:
        user_input = (
            "I'm ready to help you update a document. "
            "What would you like to create?"
        )
        user_message = HumanMessage(content=user_input)

    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)

    print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}


def should_continue(state: AgentState) -> str:
    """
    Determine if we should continue or end the conversation.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        str: 'continue' or 'end' based on the conversation state.
    """

    messages = state["messages"]

    if not messages:
        return "continue"

    # This looks for the most recent tool message....
    for message in reversed(messages):
        # ... and checks if this is a ToolMessage resulting from save
        if (
            isinstance(message, ToolMessage)
            and "saved" in message.content.lower()
            and "document" in message.content.lower()
        ):
            return "end"  # goes to the end edge which leads to the endpoint

    return "continue"


def print_messages(messages):
    """
    Print the messages in a more readable format.

    Args:
        messages: A sequence of messages.
    """
    if not messages:
        return

    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")


graph = StateGraph(AgentState)

graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")

graph.add_edge("agent", "tools")


graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    },
)

app = graph.compile()


def run_document_agent():
    """
    Run the document agent.

    This function initializes the agent and starts the conversation.
    """
    print("\n ===== DRAFTER =====")

    state = {"messages": []}

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])

    print("\n ===== DRAFTER FINISHED =====")


if __name__ == "__main__":
    run_document_agent()
