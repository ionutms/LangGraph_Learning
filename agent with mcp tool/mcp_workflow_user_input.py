import asyncio
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()

first_server_path = Path(__file__).resolve().parent / "first_mcp_server.py"
second_server_path = Path(__file__).resolve().parent / "second_mcp_server.py"
python_server_path = (
    Path(__file__).resolve().parent / "python_tools_mcp_server.py"
)

# Define system context
SYSTEM_PROMPT = """
You are a helpful AI assistant capable of answering a wide range of questions
and performing tasks. Use your tools for math-related queries when needed,
and provide clear, concise answers for all other questions."""


@tool
def divide(a: int, b: int) -> float:
    """Divide two numbers.

    Args:
        a (int): First number (dividend).
        b (int): Second number (divisor).

    Returns:
        float: Division of a by b.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


async def main():
    # Initialize the model (assumed synchronous)
    model = init_chat_model("groq:llama-3.3-70b-versatile")

    # Set up MCP client (assumed synchronous initialization)
    client = MultiServerMCPClient({
        "math": {
            "command": "python",
            "args": [str(first_server_path)],
            "transport": "stdio",
        },
        "math2": {
            "command": "python",
            "args": [str(second_server_path)],
            "transport": "stdio",
        },
        "python_tools": {
            "command": "python",
            "args": [str(python_server_path)],
            "transport": "stdio",
        },
    })

    # Asynchronously get tools from MCP servers
    mcp_tools = await client.get_tools()

    # Combine MCP tools with local tools
    all_tools = mcp_tools + [divide]

    # Bind all tools to model
    model_with_tools = model.bind_tools(all_tools)

    # Create ToolNode with all tools
    tool_node = ToolNode(all_tools)

    # Define should_continue function (synchronous)
    def should_continue(state: MessagesState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END

    # Define call_model function (already async)
    async def call_model(state: MessagesState):
        messages = state["messages"]
        response = await model_with_tools.ainvoke(messages)
        return {"messages": [response]}

    # Build and compile the graph (synchronous)
    builder = StateGraph(MessagesState)

    builder.add_node("call_model", call_model)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", should_continue)
    builder.add_edge("tools", "call_model")

    graph = builder.compile()

    # Print the ASCII representation of the graph for debugging
    print(graph.get_graph().draw_ascii())

    while True:
        # Request user input for any question or command
        user_input = input("Ask me anything (or type 'q' to exit): ")

        # Check if user wants to quit
        if user_input.lower() == "q":
            print("Exiting...")
            break

        # Test the graph asynchronously with user input
        math_response = await graph.ainvoke({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ]
        })

        for msg in math_response["messages"]:
            if msg.type == "ai":
                print(f"Assistant: {msg.content}")
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        print(
                            f"  Tool Call: {tool_call['name']} "
                            f"with args: {tool_call['args']}"
                        )
            elif msg.type == "tool":
                print(f"Tool Response: {msg.content}")
        print()  # Add a blank line for readability between iterations


if __name__ == "__main__":
    asyncio.run(main())
