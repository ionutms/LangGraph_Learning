import asyncio

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

load_dotenv()


# Define the async main function to contain all await calls
async def main():
    # Initialize the model (assumed synchronous)
    model = init_chat_model("groq:llama3-70b-8192")

    # Set up MCP client (assumed synchronous initialization)
    client = MultiServerMCPClient({
        "math": {
            "command": "python",
            "args": [
                "C:/Users/Mihai_Ionut/Documents/GitHub/LangGraph_Learning/"
                "chatbot agents/math_mcp_server.py"
            ],
            "transport": "stdio",
        },
    })

    # Asynchronously get tools
    tools = await client.get_tools()

    # Bind tools to model (assumed synchronous)
    model_with_tools = model.bind_tools(tools)

    # Create ToolNode (synchronous)
    tool_node = ToolNode(tools)

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

    # Test the graph asynchronously
    math_response = await graph.ainvoke({
        "messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]
    })

    for msg in math_response["messages"]:
        if msg.type == "ai":
            print(f"AI: {msg.content}")
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    print(
                        f"  Tool Call: {tool_call['name']} "
                        f"with args {tool_call['args']}"
                    )
        elif msg.type == "tool":
            print(f"Tool Response: {msg.content}")


# Run the async main function when the script is executed
if __name__ == "__main__":
    asyncio.run(main())
