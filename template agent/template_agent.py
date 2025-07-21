from typing import Annotated, Optional

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from template_handlers import TemplateHandlers
from typing_extensions import TypedDict

load_dotenv()

# Available LLM models for selection
LLM_MODELS = [
    "groq:llama-3.3-70b-versatile",
    "groq:deepseek-r1-distill-llama-70b",
    "groq:qwen/qwen3-32b",
    "groq:moonshotai/kimi-k2-instruct",
]

# Global LLM instructions for template operations
LLM_INSTRUCTIONS = """
You are a helpful AI assistant for processing user input.
Process the provided input in a helpful and informative way.
Provide clear, concise, and meaningful responses.

**Input Format**:
- User input: {input_data}

**Output Format**:
- Return your processed response.
- Be helpful and informative.
"""


class AgentState(TypedDict):
    """State for template processing workflow.

    Attributes:
        input_data: Input data to process.
        processed_data: Data after processing.
        messages: List of chat messages for LLM interaction.
        error: Error message if processing fails, None otherwise.
        result: Final result of the processing.
        selected_model: The selected LLM model for processing.
        continue_processing: Whether to continue in interactive mode.
        user_choice: User's choice for continuing or model selection.
    """

    input_data: str
    processed_data: str
    messages: Annotated[list, add_messages]
    error: Optional[str]
    result: str
    selected_model: str
    continue_processing: bool
    user_choice: str


class TemplateApp:
    """Orchestrates the template processing workflow using LangGraph.

    Manages a workflow for processing input data with interactive
    capabilities through nodes and handlers.

    Attributes:
        llm: Initialized language model for processing.
        selected_model: Selected LLM model identifier.
        template_prompt: Prompt template for LLM processing.
        graph: Compiled LangGraph workflow for processing.
        handler: Instance of TemplateHandlers for node operations.
    """

    def __init__(self, model: str = None):
        """Initialize the TemplateApp with workflow.

        Args:
            model: LLM model identifier (optional, set when needed).
        """
        self.llm = None
        self.selected_model = model
        self.template_prompt = LLM_INSTRUCTIONS
        self.handler = TemplateHandlers(
            None, self.template_prompt, LLM_MODELS
        )
        self.graph = self.create_workflow()

    def initialize_llm(self, model: str):
        """Initialize LLM with selected model.

        Args:
            model: LLM model identifier to initialize.
        """
        self.llm = init_chat_model(model, temperature=0.0, max_tokens=4000)
        self.selected_model = model
        self.handler.llm = self.llm
        # Initialize the prompt template now that LLM is available
        self.handler.template_prompt = ChatPromptTemplate.from_messages([
            ("system", self.template_prompt),
            ("user", "User input: {input_data}"),
        ])

    def create_workflow(self) -> StateGraph:
        """Create and compile the LangGraph workflow for template processing.

        Returns:
            StateGraph: Compiled workflow for processing input data.
        """
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("select_model", self.handler.select_model)
        workflow.add_node("get_user_input", self.handler.get_user_input)
        workflow.add_node("process", self.handler.process_input)
        workflow.add_node("display_results", self.handler.display_results)
        workflow.add_node("ask_continue", self.handler.ask_continue)

        # Define edges
        workflow.add_edge(START, "select_model")
        workflow.add_edge("select_model", "get_user_input")
        workflow.add_edge("get_user_input", "process")
        workflow.add_edge("process", "display_results")
        workflow.add_edge("display_results", "ask_continue")

        # Conditional edge for continuing
        workflow.add_conditional_edges(
            "ask_continue",
            self._should_continue,
            {"continue": "get_user_input", "end": END},
        )

        return workflow.compile()

    def _should_continue(self, state: dict) -> str:
        """Determine whether to continue processing or end.

        Args:
            state: Agent state with continue_processing flag.

        Returns:
            str: "continue" or "end" based on user choice.
        """
        return (
            "continue" if state.get("continue_processing", False) else "end"
        )

    def run_interactive_mode(self):
        """Run the template app in interactive mode using the workflow."""
        print("🚀 Template App - Interactive Mode")
        print("=" * 60)

        initial_state = {
            "input_data": "",
            "processed_data": "",
            "messages": [],
            "error": None,
            "result": "",
            "selected_model": self.selected_model or "",
            "continue_processing": True,
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
    app = TemplateApp()
    print("Graph structure:")
    print(app.graph.get_graph().draw_ascii())
    print("\n" + "=" * 60 + "\n")

    app.run_interactive_mode()
