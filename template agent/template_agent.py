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
    """

    input_data: str
    processed_data: str
    messages: Annotated[list, add_messages]
    error: Optional[str]
    result: str
    selected_model: str


class TemplateApp:
    """Orchestrates the template processing workflow using LangGraph.

    Manages a simple workflow for processing input data with a single node
    and helper class.

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
        self.handler = TemplateHandlers(None, self.template_prompt)
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
        workflow.add_node("process", self.handler.process_input)
        workflow.add_edge(START, "process")
        workflow.add_edge("process", END)
        return workflow.compile()

    def select_model(self) -> str:
        """Prompt user to select an LLM model from available options.

        Returns:
            str: Selected LLM model identifier.
        """
        print("\nAvailable LLM Models:")
        for model_index, model in enumerate(LLM_MODELS, 1):
            current_indicator = (
                " (current)" if model == self.selected_model else ""
            )
            print(f"{model_index}. {model}{current_indicator}")

        while True:
            choice = input(
                f"\nSelect a model (1-{len(LLM_MODELS)}): "
            ).strip()
            try:
                selected_index = int(choice) - 1
                if 0 <= selected_index < len(LLM_MODELS):
                    return LLM_MODELS[selected_index]
                print(f"Select a number between 1 and {len(LLM_MODELS)}.")
            except ValueError:
                print("Invalid input.")

    def process_input(self, input_data: str) -> dict:
        """Process input data.

        Args:
            input_data: Data to process.

        Returns:
            dict:
                Result containing success status, processed data,
                and messages.
        """
        print("🔧 Processing input data")
        print(f"📊 Input length: {len(input_data)} characters")
        print("-" * 50)

        initial_state = {
            "input_data": input_data,
            "processed_data": "",
            "messages": [],
            "error": None,
            "result": "",
            "selected_model": self.selected_model,
        }

        try:
            result = self.graph.invoke(initial_state)
            success = not bool(result.get("error"))
            if success:
                print("✅ Successfully processed input")
            else:
                print(f"❌ Error: {result['error']}")
            print("✨ Done!")
            return {
                "success": success,
                "processed_data": result["processed_data"],
                "result": result["result"],
                "messages": result["messages"],
                "error": result["error"],
            }
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return {
                "success": False,
                "processed_data": "",
                "result": "",
                "messages": [],
                "error": str(e),
            }

    def interactive_mode(self):
        """Run the template app in interactive mode."""
        print("🚀 Template App - Interactive Mode")
        print("=" * 60)

        # Select model first
        selected_model = self.select_model()
        self.initialize_llm(selected_model)
        print(f"\n🤖 Selected LLM Model: {selected_model}")
        print("=" * 60)

        while True:
            try:
                # Get input data from user
                print("\n📝 Enter your input:")
                input_data = input("> ").strip()

                if not input_data:
                    print("❌ Please provide some input.")
                    continue

                # Process the input
                result = self.process_input(input_data)

                if result["success"]:
                    print("\n📊 Result:")
                    print(f"{result['result']}")

                # Ask if user wants to continue
                choice = (
                    input("\n🔄 Process more input? (y/n): ").lower().strip()
                )
                if choice not in ["y", "yes"]:
                    print("👋 Goodbye!")
                    return

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                return
            except Exception as e:
                print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    app = TemplateApp()
    print("Graph structure:")
    print(app.graph.get_graph().draw_ascii())
    print("\n" + "=" * 60 + "\n")

    app.interactive_mode()
