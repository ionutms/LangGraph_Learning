from typing import Annotated, Optional

from chatbot_handlers import DocstringForgeHandlers
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()

LLM_MODELS = [
    "groq:llama-3.3-70b-versatile",
    "groq:deepseek-r1-distill-llama-70b",
    "groq:qwen/qwen3-32b",
    "groq:moonshotai/kimi-k2-instruct",
]

LLM_INSTRUCTIONS = """
You are an expert Python docstring generator.
- Improve or add docstrings in the provided Python code, following Google
docstring conventions without the example section.
- Keep maximum 79 chars per line.
- Remove trailing whitespaces from docstrings.
- Add docstrings to all functions and classes that lack them.
- Do not modify other parts of the code or refactor it.

Requirements:
1. Clear, concise descriptions.
2. Document all parameters and return values.
3. Use consistent formatting.
4. Follow Google docstring style without examples.
5. Ensure docstrings are added for undocumented functions/classes.
"""


class AgentState(TypedDict):
    """State for chatbot workflow.

    Attributes:
        user_input: User input message.
        messages: Chat messages for LLM.
        error: Error message, if any.
        response: AI assistant response.
        selected_model: Selected LLM model.
        continue_chatting: Continue interactive mode.
        user_choice: User continue choice.
        python_files: List of Python file paths.
        selected_file: Selected file path.
        action: Action (remove, update, quit).
        original_code: Original code content.
        processed_code: Processed code content.
        docstring_info: Docstring details list.
        output_dir: Output directory for files.
        saved_file: Saved processed file path.
    """

    user_input: str
    messages: Annotated[list, add_messages]
    error: Optional[str]
    response: str
    selected_model: str
    continue_chatting: bool
    user_choice: str
    python_files: list[str]
    selected_file: str
    action: Optional[str]
    original_code: str
    processed_code: str
    docstring_info: list[dict]
    output_dir: str
    saved_file: str


class DocstringForge:
    """Chatbot app with docstring processing using LangGraph.

    Manages file selection, action selection, and docstring processing.

    Attributes:
        graph: Compiled LangGraph workflow.
        handler: DocstringForgeHandlers instance.
    """

    def __init__(self):
        """Initialize DocstringForge."""
        self.handler = DocstringForgeHandlers(LLM_INSTRUCTIONS, LLM_MODELS)
        self.graph = self.create_workflow()

    def create_workflow(self) -> StateGraph:
        """Create LangGraph workflow for chatbot and docstring processing.

        Returns:
            StateGraph: Compiled workflow.
        """
        workflow = StateGraph(AgentState)

        workflow.add_node("find_files", self.handler.find_files)
        workflow.add_node("select_file", self.handler.select_file)
        workflow.add_node("select_action", self.handler.select_action)
        workflow.add_node("select_model", self.handler.select_model)
        workflow.add_node("load", self.handler.load_file)
        workflow.add_node("analyze", self.handler.analyze_docstrings)
        workflow.add_node("process", self.handler.process_docstrings)
        workflow.add_node("llm", self.handler.llm_process)
        workflow.add_node("save", self.handler.save_result)

        workflow.add_edge(START, "find_files")
        workflow.add_edge("find_files", "select_file")
        workflow.add_edge("select_file", "select_action")
        workflow.add_edge("select_action", "load")
        workflow.add_edge("load", "process")
        workflow.add_conditional_edges(
            "process",
            self.handler.should_use_llm,
            {"use_llm": "analyze", "skip_llm": "save"},
        )
        workflow.add_edge("analyze", "select_model")
        workflow.add_edge("select_model", "llm")
        workflow.add_edge("llm", "save")
        workflow.add_edge("save", END)

        return workflow.compile()

    def run_interactive_mode(self):
        """Run the chatbot in interactive mode."""
        print("🤖 Chatbot with Docstring Processing")
        print("=" * 60)
        state = {
            "user_input": "",
            "messages": [],
            "error": None,
            "response": "",
            "selected_model": "",
            "continue_chatting": True,
            "user_choice": "",
            "python_files": [],
            "selected_file": "",
            "action": None,
            "original_code": "",
            "processed_code": "",
            "docstring_info": [],
            "output_dir": "",
            "saved_file": "",
        }
        try:
            self.graph.invoke(state)
            print("👋 Goodbye!")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    app = DocstringForge()
    print("Graph structure:")
    print(app.graph.get_graph().draw_ascii())
    print("\n" + "=" * 60 + "\n")
    app.run_interactive_mode()
