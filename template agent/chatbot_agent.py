from typing import Annotated, Optional

from chatbot_handlers import ChatbotHandlers
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()

# Available LLM models for selection
LLM_MODELS = [
    "groq:llama-3.3-70b-versatile",
    "groq:deepseek-r1-distill-llama-70b",
    "groq:qwen/qwen3-32b",
    "groq:moonshotai/kimi-k2-instruct",
]

# Global LLM instructions for chatbot
LLM_INSTRUCTIONS = """
You are a helpful AI assistant chatbot.
Provide clear, concise, and meaningful responses to user queries.
Be conversational and friendly in your responses.

User message: {user_input}
"""


class AgentState(TypedDict):
    """State for chatbot workflow.

    Attributes:
        user_input: User's input message.
        messages: List of chat messages for LLM interaction.
        error: Error message if processing fails, None otherwise.
        response: AI assistant's response.
        selected_model: The selected LLM model.
        continue_chatting: Whether to continue in interactive mode.
        user_choice: User's choice for continuing.
    """

    user_input: str
    messages: Annotated[list, add_messages]
    error: Optional[str]
    response: str
    selected_model: str
    continue_chatting: bool
    user_choice: str


class ChatbotApp:
    """Simple chatbot application using LangGraph.

    Manages a workflow for chatting with an AI assistant.

    Attributes:
        llm: Initialized language model.
        selected_model: Selected LLM model identifier.
        chat_prompt: Prompt template for LLM.
        graph: Compiled LangGraph workflow.
        handler: Instance of ChatbotHandlers.
    """

    def __init__(self, model: str = None):
        """Initialize the ChatbotApp.

        Args:
            model: LLM model identifier (optional).
        """
        self.llm = None
        self.selected_model = model
        self.chat_prompt = LLM_INSTRUCTIONS
        self.handler = ChatbotHandlers(None, self.chat_prompt, LLM_MODELS)
        self.graph = self.create_workflow()

    def initialize_llm(self, model: str):
        """Initialize LLM with selected model.

        Args:
            model: LLM model identifier to initialize.
        """
        self.llm = init_chat_model(model, temperature=0.0, max_tokens=4000)
        self.selected_model = model
        self.handler.llm = self.llm
        self.handler.chat_prompt = ChatPromptTemplate.from_messages([
            ("system", self.chat_prompt),
            ("user", "User message: {user_input}"),
        ])

    def create_workflow(self) -> StateGraph:
        """Create and compile the LangGraph workflow for chatbot.

        Returns:
            StateGraph: Compiled workflow for chatting.
        """
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("select_model", self.handler.select_model)
        workflow.add_node("get_user_input", self.handler.get_user_input)
        workflow.add_node("chat_response", self.handler.chat_response)
        workflow.add_node("ask_continue", self.handler.ask_continue)

        # Define edges
        workflow.add_edge(START, "select_model")
        workflow.add_edge("select_model", "get_user_input")
        workflow.add_edge("get_user_input", "chat_response")
        workflow.add_edge("chat_response", "ask_continue")

        # Conditional edge for continuing
        workflow.add_conditional_edges(
            "ask_continue",
            self._should_continue,
            {"continue": "select_model", "end": END},
        )

        return workflow.compile()

    def _should_continue(self, state: dict) -> str:
        """Determine whether to continue chatting or end.

        Args:
            state: Agent state with continue_chatting flag.

        Returns:
            str: "continue" or "end" based on user choice.
        """
        return "continue" if state.get("continue_chatting", False) else "end"

    def run_interactive_mode(self):
        """Run the chatbot app."""
        print("🤖 Simple Chatbot")
        print("=" * 60)

        initial_state = {
            "user_input": "",
            "messages": [],
            "error": None,
            "response": "",
            "selected_model": self.selected_model or "",
            "continue_chatting": True,
            "user_choice": "",
        }

        try:
            # Set the handler's app reference for LLM initialization
            self.handler.app = self
            self.graph.invoke(initial_state)
            print("👋 Goodbye!")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        except Exception as e:
            print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    app = ChatbotApp()
    print("Graph structure:")
    print(app.graph.get_graph().draw_ascii())
    print("\n" + "=" * 60 + "\n")

    app.run_interactive_mode()
