from pathlib import Path
from typing import Annotated, List, Optional

from docstring_forge_handlers import DocstringForgeHandlers
from docstring_forge_tools import find_python_files_tool
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
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

# Global LLM instructions for docstring operations
LLM_INSTRUCTIONS = """
You are an expert Python docstring generator.
- Improve or add docstrings in the provided Python code, following Google
docstring conventions without the example section.
- Keep maximum 79 chars per line.
- Remove trailing whitespaces from docstrings.
- Add docstrings to all functions and classes that lack them.
- Do not modify other parts of the code or refactor it.

**Requirements**:
1. Clear, concise descriptions.
2. Document all parameters and return values.
3. Use consistent formatting.
4. Follow Google docstring style without examples.
5. Ensure docstrings are added for undocumented functions/classes.

**Input Format**:
- Current docstrings found: {docstrings_info}
- Original code:
```python
{original_code}
```

**Output Format**:
- Return the complete updated Python code with improved/added docstrings.
- Wrap the output in a single ```python code block.
- Do not include other code blocks or markers.
"""


class AgentState(TypedDict):
    """State for docstring and comment processing workflow.

    Attributes:
        file_path: Path to the Python file to process.
        original_code: Original Python code content.
        processed_code: Code after docstring/comment processing.
        action: Action to perform ('remove' or 'update').
        docstring_info: List of dictionaries with docstring details.
        messages: List of chat messages for LLM interaction.
        error: Error message if processing fails, None otherwise.
        output_dir: Directory to save processed files.
        saved_file: Path to the saved processed file.
        selected_model: The selected LLM model for docstring updates.
        available_models: List of available LLM model identifiers.
    """

    file_path: str
    original_code: str
    processed_code: str
    action: str
    docstring_info: List[dict]
    messages: Annotated[list, add_messages]
    error: Optional[str]
    output_dir: str
    saved_file: str
    selected_model: str
    available_models: List[str]


class DocstringForge:
    """Orchestrates the docstring processing workflow using LangGraph.

    Manages the workflow for finding, loading, analyzing, processing, and
    saving Python files with updated or removed docstrings and comments.

    Attributes:
        llm: Initialized language model for docstring updates.
        selected_model: Selected LLM model identifier.
        sv_prompt: Prompt template for LLM docstring generation.
        graph: Compiled LangGraph workflow for processing.
        handler: Instance of DocstringForgeHandlers for node operations.
    """

    def __init__(self, model: str = LLM_MODELS[0]):
        """Initialize the DocstringForge with LLM and workflow.

        Args:
            model: LLM model identifier (default: first model in LLM_MODELS).
        """
        self.llm = init_chat_model(model, temperature=0.0, max_tokens=12000)
        self.selected_model = model
        self.sv_prompt = LLM_INSTRUCTIONS
        self.handler = DocstringForgeHandlers(self.llm, self.sv_prompt)
        self.graph = self.create_workflow()

    def create_workflow(self) -> StateGraph:
        """Create and compile the LangGraph workflow for docstring processing.

        Returns:
            StateGraph: Compiled workflow for processing Python files.
        """
        workflow = StateGraph(AgentState)

        # Add nodes with descriptive names
        workflow.add_node("load_file", self.handler.load_file)
        workflow.add_node(
            "analyze_docstrings", self.handler.analyze_docstrings
        )
        workflow.add_node(
            "process_docstrings", self.handler.process_docstrings
        )
        workflow.add_node("llm_update", self.handler.llm_process)
        workflow.add_node("save_file", self.handler.save_result)
        workflow.add_node("handle_error", self.handler.handle_error)

        # Start workflow
        workflow.add_edge(START, "load_file")

        # Conditional routing after file loading
        workflow.add_conditional_edges(
            "load_file",
            self.handler.check_load_success,
            {"success": "analyze_docstrings", "error": "handle_error"},
        )

        # Conditional routing after docstring analysis
        workflow.add_conditional_edges(
            "analyze_docstrings",
            self.handler.check_analysis_success,
            {"success": "process_docstrings", "error": "handle_error"},
        )

        # Main routing decision after processing
        workflow.add_conditional_edges(
            "process_docstrings",
            self.handler.route_processing_outcome,
            {
                "update_with_llm": "llm_update",
                "save_directly": "save_file",
                "error": "handle_error",
            },
        )

        # LLM processing conditional routing
        workflow.add_conditional_edges(
            "llm_update",
            self.handler.check_llm_success,
            {"success": "save_file", "error": "handle_error"},
        )

        # End workflow paths
        workflow.add_edge("save_file", END)
        workflow.add_edge("handle_error", END)

        return workflow.compile()

    def process_file(
        self, action: str, file_path: Path, output_dir: str
    ) -> dict:
        """Process a single Python file with the specified action.

        Args:
            action: Action to perform ('remove' or 'update').
            file_path: Path to the Python file to process.
            output_dir: Directory to save the processed file.

        Returns:
            dict:
                Result containing success status, processed code,
                and messages.
        """
        rel_path = file_path.relative_to(Path.cwd())
        print(f"🔧 Processing: {rel_path}")
        print(f"📝 Action: {action}")
        print(f"📂 Output: {output_dir}")
        print("-" * 50)

        initial_state = {
            "file_path": str(file_path),
            "action": action,
            "original_code": "",
            "processed_code": "",
            "docstring_info": [],
            "messages": [],
            "error": None,
            "output_dir": output_dir,
            "saved_file": "",
            "selected_model": self.selected_model,
            "available_models": LLM_MODELS,
        }

        try:
            result = self.graph.invoke(initial_state)
            success = not bool(result.get("error"))
            if success:
                print(
                    f"✅ {'Removed' if action == 'remove' else 'Updated'} "
                    f"{len(result['docstring_info'])} docstrings"
                )
            else:
                print(f"❌ Error: {result['error']}")
            print("✨ Done!")
            return {
                "success": success,
                "processed_code": result["processed_code"],
                "messages": result["messages"],
                "error": result["error"],
                "saved_file": result["saved_file"],
                "docstring_info": result["docstring_info"],
            }
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return {
                "success": False,
                "processed_code": "",
                "messages": [],
                "error": str(e),
                "saved_file": "",
                "docstring_info": [],
            }

    def create_model_selection_workflow(self) -> StateGraph:
        """Create a simple workflow for model selection.

        Returns:
            StateGraph: Compiled workflow for model selection.
        """
        workflow = StateGraph(AgentState)

        workflow.add_node("select_model", self.handler.select_model_node)
        workflow.add_node("handle_error", self.handler.handle_error)

        workflow.add_edge(START, "select_model")
        workflow.add_conditional_edges(
            "select_model",
            self.handler.check_model_selection_success,
            {"success": END, "error": "handle_error"},
        )
        workflow.add_edge("handle_error", END)

        return workflow.compile()

    def select_model_with_workflow(self) -> str:
        """Use workflow to select LLM model.

        Returns:
            str: Selected LLM model identifier.
        """
        model_workflow = self.create_model_selection_workflow()

        initial_state = {
            "file_path": "",
            "action": "",
            "original_code": "",
            "processed_code": "",
            "docstring_info": [],
            "messages": [],
            "error": None,
            "output_dir": "",
            "saved_file": "",
            "selected_model": "",
            "available_models": LLM_MODELS,
        }

        try:
            result = model_workflow.invoke(initial_state)
            if result.get("error"):
                print(f"Error in model selection: {result['error']}")
                return LLM_MODELS[0]  # Return default model
            return result.get("selected_model", LLM_MODELS[0])
        except Exception as e:
            print(f"Error in model selection workflow: {str(e)}")
            return LLM_MODELS[0]  # Return default model

    def run(self):
        """Run the docstring forge in interactive mode."""
        current_dir = Path.cwd()
        print("🔥 Docstring Forge - Interactive Mode")
        print(f"📂 Scanning: {current_dir}")
        print(f"🤖 Using Model: {self.selected_model}")
        print("=" * 60)

        while True:
            try:
                result = find_python_files_tool.invoke({
                    "directory": str(current_dir)
                })
                if result["error"]:
                    print(f"❌ Error: {result['error']}")
                    return

                python_files = [Path(f) for f in result["python_files"]]
                if not python_files:
                    print("❌ No Python files found.")
                    return

                print(f"✅ Found {len(python_files)} Python files")

                # First select the file
                self.handler.display_files_menu(python_files)
                selected_file = self.handler.get_file_choice(python_files)

                # Then select the action
                action = self.handler.get_action_choice()
                output_dir = self.handler.get_output_directory()

                max_repeats = 5
                repeat_count = 0
                while repeat_count < max_repeats:
                    self.process_file(action, selected_file, output_dir)
                    repeat_count += 1
                    if repeat_count < max_repeats:
                        print(f"🔄 Repeated {repeat_count} of {max_repeats}")
                        choice = (
                            input(
                                f"🔄 Repeat on {selected_file.name}? (y/n): "
                            )
                            .lower()
                            .strip()
                        )
                        if choice not in ["y", "yes"]:
                            break
                    else:
                        print(f"🔄 Max repeats ({max_repeats}) reached")

                choice = (
                    input("\n🔄 Process another file? (y/n): ")
                    .lower()
                    .strip()
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
    forge = DocstringForge()
    print("Graph structure:")
    print(forge.graph.get_graph().draw_ascii())
    print("\n" + "=" * 60 + "\n")

    # Select model using workflow
    selected_model = forge.select_model_with_workflow()
    forge.llm = init_chat_model(
        selected_model, temperature=0.0, max_tokens=12000
    )
    forge.selected_model = selected_model
    forge.handler.llm = forge.llm
    print(f"\nSelected LLM Model: {selected_model}")
    print("\n" + "=" * 60 + "\n")

    forge.run()
